#!/usr/bin/env python3
"""推播通知:App 註冊 Expo push token,偵測到新違規時自動推播。

部署(main.py 已幫你接好):
  from api.routes import push
  app.include_router(push.router)
  # lifespan 啟動時:push.start_poller()

流程:
  App 開啟推播 → POST /api/push/register {token, types}
  types 訂閱類別:violation=違規警報、lpr=車牌辨識(含球機追蹤);不帶=全訂(相容舊版 App)。
  背景 poller 每 20 秒查 Violation 表,有新的就推給訂閱 violation 的 token。
Token 存在專案根目錄 push_tokens.json(純檔案,不動 DB schema);
舊格式(token 字串陣列)自動視為全訂,第一次存檔時升級成 {token: [types]}。
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


_ALL_TYPES = ["violation", "lpr"]


def _load_token_map() -> dict:
    """token → 訂閱類別列表;相容舊格式(字串陣列=全訂)"""
    try:
        raw = json.loads(_TOKENS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, list):
        return {t: list(_ALL_TYPES) for t in raw if t}
    if isinstance(raw, dict):
        return {t: ([x for x in v if x in _ALL_TYPES] or list(_ALL_TYPES)) for t, v in raw.items() if t}
    return {}


def _save_token_map(m: dict) -> None:
    try:
        _TOKENS_FILE.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _tokens_for(category: str) -> List[str]:
    """訂閱了該類別的 token"""
    return [t for t, types in _load_token_map().items() if category in types]


class TokenBody(BaseModel):
    token: str
    types: Optional[List[str]] = None   # 訂閱類別 violation/lpr;不帶=全訂(舊版 App)


@router.post("/register")
def register(body: TokenBody) -> dict:
    """App 註冊 push token(重複註冊=更新訂閱類別)"""
    m = _load_token_map()
    if body.token:
        types = [t for t in (body.types or _ALL_TYPES) if t in _ALL_TYPES]
        m[body.token] = types or list(_ALL_TYPES)
        _save_token_map(m)
    return {"ok": True, "count": len(m)}


@router.post("/unregister")
def unregister(body: TokenBody) -> dict:
    """App 關閉推播時移除 token"""
    m = _load_token_map()
    m.pop(body.token, None)
    _save_token_map(m)
    return {"ok": True, "count": len(m)}


@router.get("/status")
def status() -> dict:
    """目前註冊了幾個裝置(含各類別訂閱數)"""
    m = _load_token_map()
    return {"tokens": len(m),
            "violation": sum(1 for v in m.values() if "violation" in v),
            "lpr": sum(1 for v in m.values() if "lpr" in v),
            "poll_sec": _POLL_SEC}


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


def push_alert(title: str, body: str, data: Optional[dict] = None, category: str = "lpr") -> None:
    """給其他模組用的即時告警推播(如 Hanwha 球機追蹤事件,歸類 lpr)。
    只推給有訂閱該 category 的 token;同步、失敗吞掉。"""
    try:
        tokens = _tokens_for(category)
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
        tokens = _tokens_for("violation")
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
