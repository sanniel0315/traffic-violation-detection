"""API Key 認證 dependency + 速率限制"""
from __future__ import annotations

import collections
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from api.models import ApiKey, get_db, verify_password

# In-memory rate limiter: key_id -> deque of timestamps
_rate_windows: dict[int | str, collections.deque] = {}

# 固定 API Key（從 .env 讀取）
_STATIC_API_KEY = os.getenv("EXTERNAL_API_KEY", "").strip()

# 固定 key 的虛擬 ApiKey 物件
class _StaticApiKey:
    """模擬 ApiKey model，供固定 key 使用"""
    id = "static"
    name = "環境變數固定 Key"
    scopes = ["vd_report", "congestion_report"]
    rate_limit_per_min = 120
    enabled = True
    expires_at = None
    last_used_at = None

_STATIC_KEY_OBJ = _StaticApiKey()


def _check_rate_limit(key_id, limit: int = 60) -> None:
    now = time.time()
    window = _rate_windows.setdefault(key_id, collections.deque())
    while window and window[0] < now - 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(
            status_code=429,
            detail={"status": "error", "error": {"code": "RATE_LIMITED", "message": f"超過速率限制 ({limit} req/min)"}},
        )
    window.append(now)


async def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> ApiKey | _StaticApiKey:
    """驗證 X-API-Key header，支援固定 key 和資料庫 key"""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "error": {"code": "MISSING_API_KEY", "message": "請提供 X-API-Key header"}},
        )

    # 1. 先比對固定 key
    if _STATIC_API_KEY and secrets.compare_digest(x_api_key, _STATIC_API_KEY):
        _check_rate_limit("static", _STATIC_KEY_OBJ.rate_limit_per_min)
        return _STATIC_KEY_OBJ

    # 2. 再比對資料庫 key
    now = datetime.now(timezone.utc)
    candidates = db.query(ApiKey).filter(ApiKey.enabled == True).all()  # noqa: E712

    matched: ApiKey | None = None
    for candidate in candidates:
        if candidate.expires_at and candidate.expires_at < now:
            continue
        if verify_password(x_api_key, candidate.key_hash):
            matched = candidate
            break

    if not matched:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "error": {"code": "INVALID_API_KEY", "message": "API Key 無效或已過期"}},
        )

    _check_rate_limit(matched.id, matched.rate_limit_per_min or 60)

    try:
        matched.last_used_at = now
        db.commit()
    except Exception:
        db.rollback()

    return matched


def require_scope(scope: str):
    """回傳一個 dependency 檢查 ApiKey 是否擁有指定 scope"""
    async def _check(api_key=Depends(get_api_key)):
        scopes = api_key.scopes or []
        if scope not in scopes:
            raise HTTPException(
                status_code=403,
                detail={"status": "error", "error": {"code": "INSUFFICIENT_SCOPE", "message": f"此 API Key 無 '{scope}' 權限"}},
            )
        return api_key
    return _check
