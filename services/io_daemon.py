"""Standalone IO daemon — independent systemd unit (traffic-io.service).

完全獨占 /dev/ttyTHS2 + 跑 IOService 全部邏輯。pyserial / Tegra UART 偶發
native SEGV 由 systemd Restart=always 拉回，**不會牽連 traffic-api**。

traffic-api 內 io routes 透過 HTTP 127.0.0.1:8011 跟本 daemon 通訊。

HTTP API:
  GET  /status               → io_service.status()
  POST /do/{ch}  body {on}   → set_relay
  POST /set_auto_mode {auto} → set_auto_mode
  POST /set_downloading {active}
  POST /set_comm_fault {fault}
  GET  /events?since=N       → long-poll DI rising edge events
"""
from __future__ import annotations

import os
# daemon 內部要跑 _di_loop，不論 traffic-api 端 env 怎麼設都 force 開
os.environ["IO_DI_DISABLED"] = "0"
os.environ.pop("IO_DAEMON_URL", None)  # daemon 本身不要走 client mode
# 標記 daemon host：跳過 reset/download callback wire (client 端去接 events)
os.environ["IO_DAEMON_HOST"] = "1"

import time
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from services.io_service import get_service

io = get_service()
io.start()

app = FastAPI(title="IO Daemon")


class OnBody(BaseModel):
    on: bool


class AutoBody(BaseModel):
    auto: bool


class ActiveBody(BaseModel):
    active: bool


class FaultBody(BaseModel):
    fault: bool


@app.get("/status")
def get_status():
    return io.status()


@app.post("/do/{ch}")
def set_do(ch: int, body: OnBody):
    if ch not in (0, 1, 2):
        return {"error": "ch must be 0..2"}
    io.set_relay(ch, body.on)
    return {"ch": ch, "on": body.on}


@app.post("/set_auto_mode")
def set_auto(body: AutoBody):
    io.set_auto_mode(body.auto)
    return {"auto": body.auto}


@app.post("/set_downloading")
def set_dl(body: ActiveBody):
    io.set_downloading(body.active)
    return {"active": body.active}


@app.post("/set_comm_fault")
def set_fault(body: FaultBody):
    io.set_comm_fault(body.fault)
    return {"fault": body.fault}


@app.get("/events")
def get_events(since: int = 0):
    """Long-poll: 最多等 5 秒拿到 seq > since 的 DI events。沒有就回空 list。"""
    deadline = time.time() + 5.0
    while time.time() < deadline:
        events = [e for e in list(io.di_events) if int(e.get("seq", 0)) > since]
        if events:
            return {"events": events, "latest_seq": io.di_event_seq}
        time.sleep(0.1)
    return {"events": [], "latest_seq": io.di_event_seq}


@app.get("/health")
def health():
    return {"ok": True, "connected": io._mod.ok, "di_seq": io.di_event_seq}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8011, log_level="warning")
