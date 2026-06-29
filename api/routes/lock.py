"""
E-1507 電子鎖 API routes (獨立於 IO 燈號面板)。

GET  /api/lock/status        - 即時狀態快照 (門磁/手柄/鑰匙/動作)
GET  /api/lock/events        - 刷卡/開鎖歷史記錄 (DB, 分頁)
WS   /api/lock/live          - 即時刷卡/開鎖事件推送
"""
from __future__ import annotations

import asyncio
from datetime import timezone

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import desc
from sqlalchemy.orm import Session

from api.models import LockEvent, get_db
from services.io_service import get_service

router = APIRouter(prefix="/api/lock", tags=["lock"])


@router.get("/status")
def lock_status():
    """電子鎖即時狀態 (從 IO daemon 透傳的 lock 區塊)。"""
    full = get_service().status()
    return full.get("lock", {"enabled": False, "connected": False, "status": {}})


@router.post("/add-card")
def add_card():
    """讓電子鎖進入加卡模式,使用者隨後在鎖上刷要新增的卡即學習錄入。"""
    return get_service().add_lock_card()


@router.post("/remove-card")
def remove_card():
    """讓電子鎖進入刪卡模式,使用者隨後在鎖上刷要刪除的卡即移除。"""
    return get_service().remove_lock_card()


@router.get("/cards")
def list_cards(db: Session = Depends(get_db)):
    """卡片庫:加卡時記錄的卡號清單 (含持有人/部門)。"""
    from api.models import LockCard
    rows = (db.query(LockCard)
            .filter(LockCard.active == True)
            .order_by(LockCard.added_at.desc()).all())
    return {"items": [
        {"id": r.id, "card_no": r.card_no, "holder_name": r.holder_name, "dept": r.dept,
         "added_at": r.added_at.replace(tzinfo=timezone.utc).isoformat() if r.added_at else None}
        for r in rows
    ]}


class CardUpdate(BaseModel):
    holder_name: str = ""
    dept: str = ""


@router.put("/cards/{card_id}")
def update_card(card_id: int, body: CardUpdate, db: Session = Depends(get_db)):
    """編輯卡片持有人/部門。"""
    from api.models import LockCard
    r = db.query(LockCard).filter(LockCard.id == card_id).first()
    if not r:
        return {"ok": False, "msg": "卡片不存在"}
    r.holder_name = body.holder_name.strip() or None
    r.dept = body.dept.strip() or None
    db.commit()
    return {"ok": True}


@router.delete("/cards/{card_id}")
def delete_card(card_id: int, db: Session = Depends(get_db)):
    """從記錄內刪除卡片:用卡號 FC10 直接刪鎖內該卡(不需現場刷),並標記庫為停用。"""
    from api.models import LockCard
    r = db.query(LockCard).filter(LockCard.id == card_id).first()
    if not r:
        return {"ok": False, "msg": "卡片不存在"}
    res = get_service().delete_lock_card_by_no(r.card_no)
    if res.get("ok"):
        r.active = False
        db.commit()
    return res


@router.get("/events")
def lock_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """刷卡/開鎖歷史記錄 (DB 持久化, 最新在前, 分頁)。"""
    q = db.query(LockEvent).order_by(desc(LockEvent.created_at))
    total = q.count()
    rows = q.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": r.id,
                "addr": r.lock_addr,
                "action": r.action_code,
                "label": r.action_label,
                "door_closed": r.door_closed,
                "handle_in_place": r.handle_in_place,
                "key_in_place": r.key_in_place,
                "time": (
                    r.created_at.replace(tzinfo=timezone.utc).isoformat()
                    if r.created_at else None
                ),
            }
            for r in rows
        ],
    }


@router.websocket("/live")
async def lock_live_ws(ws: WebSocket):
    """即時推送刷卡/開鎖事件 (連線後才發生的)。"""
    await ws.accept()
    svc = get_service()
    last_seq = svc.lock_event_seq  # 連線當下 seq, 跳過先前事件
    try:
        while True:
            if svc.lock_event_seq > last_seq:
                for evt in list(svc.lock_events):
                    if evt.get("seq", 0) > last_seq:
                        await ws.send_json(evt)
                        last_seq = evt["seq"]
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
