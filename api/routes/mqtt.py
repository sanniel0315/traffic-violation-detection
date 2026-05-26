#!/usr/bin/env python3
"""MQTT 設定 + 訊息監控 + 手動發佈 API"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.mqtt_bridge import bridge

router = APIRouter(prefix="/api/mqtt", tags=["MQTT"])


class MqttSettings(BaseModel):
    mode: Optional[str] = None           # embedded | external | off
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: Optional[str] = None
    base_topic: Optional[str] = None
    subscribe_patterns: Optional[List[str]] = None
    qos: Optional[int] = None
    retain: Optional[bool] = None


class MqttPublish(BaseModel):
    topic: str
    payload: Any
    qos: int = 0
    retain: bool = False


@router.get("/status")
async def get_mqtt_status():
    """MQTT bridge 狀態 + 統計"""
    return bridge.status()


@router.get("/settings")
async def get_mqtt_settings():
    """讀取設定（不回 password）"""
    s = dict(bridge.settings)
    s["password"] = "***" if s.get("password") else ""
    return s


@router.put("/settings")
async def update_mqtt_settings(body: MqttSettings):
    """更新設定 + 自動重連"""
    data = {k: v for k, v in body.dict().items() if v is not None}
    # 如果 password 是 "***" 表示沒變，移除不覆寫
    if data.get("password") == "***":
        data.pop("password", None)
    merged = bridge.save_settings(data)
    return {"status": "ok", "settings": {k: v for k, v in merged.items() if k != "password"}}


@router.post("/reconnect")
async def mqtt_reconnect():
    ok = bridge.reconnect()
    return {"status": "ok" if ok else "failed", "connected": bridge.connected(), "error": bridge._conn_err}


@router.post("/publish")
async def mqtt_publish(body: MqttPublish):
    """手動發佈測試訊息"""
    if not bridge.connected():
        st = bridge.status()
        err = (st.get("error") or "").strip()
        mode = st.get("mode") or "embedded"
        host = st.get("host") or "localhost"
        port = st.get("port") or 1883
        if err:
            detail = f"MQTT 未連線：{err}"
        elif mode == "embedded":
            detail = (
                f"本機 broker (localhost:{port}) 未啟動。"
                f"請在 Jetson 安裝並啟動 mosquitto：sudo apt install -y mosquitto && sudo systemctl enable --now mosquitto"
            )
        else:
            detail = f"外部 broker {host}:{port} 連線失敗 — 請檢查網路與 broker 服務"
        raise HTTPException(status_code=503, detail=detail)
    ok = bridge.publish(body.topic, body.payload, qos=body.qos, retain=body.retain)
    if not ok:
        raise HTTPException(status_code=500, detail="publish 失敗 (broker 可能剛斷線)")
    return {"status": "ok"}


@router.get("/messages/sent")
async def get_sent_messages(limit: int = 100):
    """已發送訊息列表（最近 N 筆）"""
    return {"messages": bridge.recent_sent(limit=limit)}


@router.get("/messages/recv")
async def get_received_messages(limit: int = 100):
    """已接收訊息列表（最近 N 筆）"""
    return {"messages": bridge.recent_recv(limit=limit)}


@router.delete("/messages")
async def clear_messages():
    bridge.log_sent.clear()
    bridge.log_recv.clear()
    return {"status": "ok"}
