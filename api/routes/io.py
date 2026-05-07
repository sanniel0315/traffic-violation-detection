"""
I/O module API routes.

GET  /api/io/status         - DI/DO snapshot + system state
POST /api/io/do/{ch}        - manually drive a relay (ch 0..2)
WS   /api/io/events         - real-time DI edge events (JSON)
GET  /api/io/sync/config    - return current sync URL + token
POST /api/io/sync/config    - save sync URL + token to io_settings.json
POST /api/io/sync/trigger   - trigger config sync immediately
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from services.io_service import get_service

router = APIRouter(prefix="/api/io", tags=["io"])


def _log(level: str, msg: str) -> None:
    try:
        from api.routes.logs import add_log
        add_log(level, msg, "io")
    except Exception:
        pass


@router.get("/status")
async def io_status():
    return get_service().status()


class DOCommand(BaseModel):
    on: bool


@router.post("/do/{ch}")
async def set_do(ch: int, cmd: DOCommand):
    if ch not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="ch must be 0, 1, or 2")
    svc = get_service()
    try:
        svc.set_relay(ch, cmd.on)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    colors = {0: "紅", 1: "綠", 2: "白"}
    _log("info", f"網頁手動控制 DO{ch}({colors.get(ch,'')}) → {'ON' if cmd.on else 'OFF'}")
    return {"ch": ch, "on": cmd.on}


@router.websocket("/events")
async def di_events_ws(ws: WebSocket):
    await ws.accept()
    svc = get_service()
    last_seq = svc.di_event_seq  # 連線當下的 seq，跳過先前事件
    try:
        while True:
            if svc.di_event_seq > last_seq:
                # snapshot 後再過濾，避免迭代中 deque 被修改
                for evt in list(svc.di_events):
                    if evt.get("seq", 0) > last_seq:
                        await ws.send_json(evt)
                        last_seq = evt["seq"]
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


# ── Config Sync endpoints ─────────────────────────────────────────────────────

class SyncConfig(BaseModel):
    url: str
    token: str = ""


@router.get("/sync/config")
async def get_sync_config():
    from services.config_sync import get_sync_url, get_sync_token
    return {"url": get_sync_url(), "token": get_sync_token()}


@router.post("/sync/config")
async def save_sync_config(cfg: SyncConfig):
    from services.config_sync import save_sync_settings
    url = cfg.url.strip()
    save_sync_settings(url, cfg.token.strip())
    _log("info", f"網頁設定遠端 config 同步 URL: {url or '(清除)'}")
    return {"ok": True, "url": url}


@router.post("/sync/trigger")
async def trigger_sync():
    from services import config_sync
    if config_sync.is_running():
        return {"ok": False, "msg": "已在執行中"}
    if not config_sync.get_sync_url():
        raise HTTPException(status_code=400, detail="CONFIG_SYNC_URL 未設定")
    ok = config_sync.trigger()
    return {"ok": ok, "msg": "已觸發" if ok else "已在執行中"}
