#!/usr/bin/env python3
"""推播通知:App 註冊 Expo push token,偵測到新違規時自動推播。

部署(main.py 已幫你接好):
  from api.routes import push
  app.include_router(push.router)
  # lifespan 啟動時:push.start_poller()

流程:
  App 開啟推播 → POST /api/push/register {token}
  背景 poller 每 20 秒查 Violation 表,有新的就呼叫 Expo Push API 推給所有 token。
Token 存在專案根目錄 push_tokens.json(純檔案,不動 DB schema)。
"""
import asyncio
import json
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/push", tags=["推播"])

_TOKENS_FILE = Path(__file__).resolve().parent.parent.parent / "push_tokens.json"
_EXPO_URL = "https://exp.host/--/api/v2/push/send"
_POLL_SEC = 20


def _load_tokens() -> List[str]:
    try:
        return list(json.loads(_TOKENS_FILE.read_text(encoding="utf-8")))
    except Exception:
        return []


def _save_tokens(tokens: List[str]) -> None:
    try:
        _TOKENS_FILE.write_text(json.dumps(sorted(set(tokens))), encoding="utf-8")
    except Exception:
        pass


class TokenBody(BaseModel):
    token: str


@router.post("/register")
def register(body: TokenBody) -> dict:
    """App 註冊 push token"""
    tokens = _load_tokens()
    if body.token and body.token not in tokens:
        tokens.append(body.token)
        _save_tokens(tokens)
    return {"ok": True, "count": len(tokens)}


@router.post("/unregister")
def unregister(body: TokenBody) -> dict:
    """App 關閉推播時移除 token"""
    tokens = [t for t in _load_tokens() if t != body.token]
    _save_tokens(tokens)
    return {"ok": True, "count": len(tokens)}


@router.get("/status")
def status() -> dict:
    """目前註冊了幾個裝置"""
    return {"tokens": len(_load_tokens()), "poll_sec": _POLL_SEC}


def _send_expo(tokens: List[str], title: str, body: str, data: Optional[dict] = None) -> None:
    """呼叫 Expo Push API(同步,poller 用 to_thread 呼叫)。分批每 90 筆。"""
    if not tokens:
        return
    for i in range(0, len(tokens), 90):
        chunk = tokens[i:i + 90]
        msgs = [{
            "to": tk,
            "title": title,
            "body": body,
            "sound": "default",
            "priority": "high",
            "data": data or {},
        } for tk in chunk]
        try:
            requests.post(_EXPO_URL, json=msgs, timeout=10,
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
        except Exception:
            pass


def push_alert(title: str, body: str, data: Optional[dict] = None) -> None:
    """給其他模組用的即時告警推播(如 Hanwha 球機追蹤事件)。
    同步、非阻塞失敗吞掉;沒有註冊 token 就直接跳過。"""
    try:
        tokens = _load_tokens()
        if tokens:
            _send_expo(tokens, title, body, data)
    except Exception:
        pass


async def _poller() -> None:
    """每 _POLL_SEC 秒查新違規並推播。起始時記住目前最大 id,不對歷史狂推。"""
    from api.models import SessionLocal, Violation

    last_id = 0
    try:
        db = SessionLocal()
        row = db.query(Violation).order_by(Violation.id.desc()).first()
        last_id = int(row.id) if row else 0
        db.close()
    except Exception:
        last_id = 0

    while True:
        await asyncio.sleep(_POLL_SEC)
        try:
            db = SessionLocal()
            news = (db.query(Violation)
                    .filter(Violation.id > last_id)
                    .order_by(Violation.id.asc())
                    .all())
            db.close()
        except Exception:
            continue
        if not news:
            continue
        last_id = int(news[-1].id)
        tokens = _load_tokens()
        if not tokens:
            continue
        # 最多推最近 10 筆,避免一次湧入洗版
        for v in news[-10:]:
            title = getattr(v, "violation_name", None) or getattr(v, "violation_type", None) or "違規"
            speed = getattr(v, "speed_kmh", None)
            speed_txt = f" {round(speed)} km/h" if speed else ""
            plate = getattr(v, "license_plate", None) or "未辨識車牌"
            cam = getattr(v, "camera_id", None)
            cam_txt = f" · cam_{cam}" if cam is not None else ""
            await asyncio.to_thread(
                _send_expo, tokens,
                f"⚠ {title}{speed_txt}",
                f"{plate}{cam_txt}",
                {"violation_id": int(v.id)},
            )


_task: Optional[asyncio.Task] = None


def start_poller() -> None:
    """在 FastAPI lifespan startup 呼叫一次"""
    global _task
    if _task is None:
        _task = asyncio.create_task(_poller())
