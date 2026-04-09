"""API Key 管理端點（需 admin 權限）"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.models import ApiKey, get_db, hash_password
from api.routes.auth import get_admin_user

router = APIRouter(prefix="/api/auth/api-keys", tags=["API Key 管理"])


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., max_length=100)
    scopes: list[str] = Field(default=["vd_report", "congestion_report"])
    rate_limit_per_min: int = Field(default=60, ge=1, le=600)
    expires_at: Optional[str] = None


class UpdateApiKeyRequest(BaseModel):
    name: Optional[str] = None
    scopes: Optional[list[str]] = None
    enabled: Optional[bool] = None
    rate_limit_per_min: Optional[int] = Field(default=None, ge=1, le=600)
    expires_at: Optional[str] = None


@router.post("")
async def create_api_key(
    data: CreateApiKeyRequest,
    admin=Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    raw_key = "tvd_" + secrets.token_urlsafe(32)
    prefix = raw_key[:8]
    key_hash = hash_password(raw_key)

    expires = None
    if data.expires_at:
        try:
            expires = datetime.fromisoformat(data.expires_at)
        except ValueError:
            raise HTTPException(status_code=400, detail="expires_at 格式錯誤，請使用 ISO 8601")

    api_key = ApiKey(
        name=data.name,
        key_prefix=prefix,
        key_hash=key_hash,
        scopes=data.scopes,
        rate_limit_per_min=data.rate_limit_per_min,
        expires_at=expires,
        created_by=admin.username,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return {
        "status": "success",
        "item": {
            "id": api_key.id,
            "name": api_key.name,
            "api_key": raw_key,
            "key_prefix": prefix,
            "scopes": api_key.scopes,
            "rate_limit_per_min": api_key.rate_limit_per_min,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
        },
        "warning": "此 API Key 僅顯示一次，請妥善保存",
    }


@router.get("")
async def list_api_keys(
    _admin=Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    keys = db.query(ApiKey).order_by(ApiKey.created_at.desc()).all()
    return {
        "status": "success",
        "items": [
            {
                "id": k.id,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "scopes": k.scopes,
                "enabled": k.enabled,
                "rate_limit_per_min": k.rate_limit_per_min,
                "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                "created_by": k.created_by,
                "created_at": k.created_at.isoformat() if k.created_at else None,
            }
            for k in keys
        ],
    }


@router.put("/{key_id}")
async def update_api_key(
    key_id: int,
    data: UpdateApiKeyRequest,
    _admin=Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API Key 不存在")

    if data.name is not None:
        key.name = data.name
    if data.scopes is not None:
        key.scopes = data.scopes
    if data.enabled is not None:
        key.enabled = data.enabled
    if data.rate_limit_per_min is not None:
        key.rate_limit_per_min = data.rate_limit_per_min
    if data.expires_at is not None:
        try:
            key.expires_at = datetime.fromisoformat(data.expires_at) if data.expires_at else None
        except ValueError:
            raise HTTPException(status_code=400, detail="expires_at 格式錯誤")

    db.commit()
    return {"status": "success", "message": "已更新"}


@router.delete("/{key_id}")
async def delete_api_key(
    key_id: int,
    _admin=Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API Key 不存在")
    db.delete(key)
    db.commit()
    return {"status": "success", "message": "已刪除"}
