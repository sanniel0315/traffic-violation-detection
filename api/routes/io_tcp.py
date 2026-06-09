"""
Modbus TCP IO endpoints — 跟現有 USB PD3R3 (/api/io/*) 並存。
多 station 管理,每個 station 一台 tPET / tET 等 Modbus TCP IO module。

endpoints:
- GET    /api/io_tcp/modules                列已配置 modules + 當前狀態
- POST   /api/io_tcp/modules                新增/更新一個 module config
- DELETE /api/io_tcp/modules/{id}           刪除 module
- GET    /api/io_tcp/{id}/status            單一 module DI/DO snapshot
- POST   /api/io_tcp/{id}/do/{ch}           寫單一 DO (body: {on: bool})
- POST   /api/io_tcp/{id}/reconnect         手動重連
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.modbus_tcp_io import (
    ModbusTcpConfig, ModbusTcpIO,
    list_configs, get_module, upsert_config, delete_config,
)

router = APIRouter(prefix="/api/io_tcp", tags=["io-tcp"])


def _log(level: str, msg: str) -> None:
    try:
        from api.routes.logs import add_log
        add_log(level, msg, "io_tcp")
    except Exception:
        pass


class ModuleConfigBody(BaseModel):
    id: str = Field(..., description="站號 (e.g. junction_1)")
    name: str = Field(..., description="顯示名稱")
    ip: str = Field(..., description="IP address")
    port: int = 502
    unit_id: int = 1
    di_count: int = 2
    do_count: int = 2
    di_addr: int = 0
    do_addr: int = 0
    timeout: float = 1.5
    enabled: bool = True


class DOCommand(BaseModel):
    on: bool


class PulseCommand(BaseModel):
    duration_ms: int = 1000


def _module_status(mod: ModbusTcpIO) -> dict:
    di = mod.read_inputs()
    do = mod.read_outputs()
    return {
        "id": mod.cfg.id,
        "name": mod.cfg.name,
        "ip": mod.cfg.ip,
        "port": mod.cfg.port,
        "unit_id": mod.cfg.unit_id,
        "di_count": mod.cfg.di_count,
        "do_count": mod.cfg.do_count,
        "ok": mod.ok,
        "error": mod.error,
        "di": di,
        "do": do,
    }


@router.get("/modules")
def list_modules():
    """列出所有已配置 modules + 當前 DI/DO snapshot"""
    cfgs = list_configs()
    out = []
    for cfg in cfgs:
        mod = get_module(cfg.id)
        if mod is None:
            out.append({
                "id": cfg.id, "name": cfg.name, "ip": cfg.ip, "port": cfg.port,
                "unit_id": cfg.unit_id, "di_count": cfg.di_count, "do_count": cfg.do_count,
                "ok": False, "error": "module not initialised",
                "di": [], "do": [],
            })
        else:
            out.append(_module_status(mod))
    return {"modules": out}


@router.post("/modules")
def add_or_update_module(body: ModuleConfigBody):
    cfg = ModbusTcpConfig(**body.model_dump())
    upsert_config(cfg)
    _log("info", f"Modbus TCP IO module 更新: {cfg.id} ({cfg.ip}:{cfg.port})")
    mod = get_module(cfg.id)
    return _module_status(mod) if mod else {"id": cfg.id, "enabled": cfg.enabled}


@router.delete("/modules/{module_id}")
def remove_module(module_id: str):
    ok = delete_config(module_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"module {module_id} not found")
    _log("info", f"Modbus TCP IO module 已刪除: {module_id}")
    return {"id": module_id, "deleted": True}


@router.get("/{module_id}/status")
def module_status(module_id: str):
    mod = get_module(module_id)
    if mod is None:
        raise HTTPException(status_code=404, detail=f"module {module_id} not found")
    return _module_status(mod)


@router.post("/{module_id}/do/{ch}")
def set_do(module_id: str, ch: int, cmd: DOCommand):
    mod = get_module(module_id)
    if mod is None:
        raise HTTPException(status_code=404, detail=f"module {module_id} not found")
    if not mod.write_output(ch, cmd.on):
        raise HTTPException(status_code=503, detail=mod.error or "write failed")
    _log("info", f"[{module_id}] DO{ch} → {'ON' if cmd.on else 'OFF'}")
    return {"id": module_id, "ch": ch, "on": cmd.on}


@router.post("/{module_id}/pulse/{ch}")
def pulse_do(module_id: str, ch: int, cmd: PulseCommand):
    """DO 短脈衝 — ON N 毫秒後自動 OFF (違規警示燈 / 蜂鳴器 用途)"""
    mod = get_module(module_id)
    if mod is None:
        raise HTTPException(status_code=404, detail=f"module {module_id} not found")
    if cmd.duration_ms <= 0 or cmd.duration_ms > 60000:
        raise HTTPException(status_code=400, detail="duration_ms must be 1..60000")
    if not mod.pulse_output(ch, cmd.duration_ms):
        raise HTTPException(status_code=503, detail=mod.error or "pulse failed")
    _log("info", f"[{module_id}] DO{ch} pulse {cmd.duration_ms}ms")
    return {"id": module_id, "ch": ch, "duration_ms": cmd.duration_ms, "ok": True}


@router.post("/{module_id}/reconnect")
def reconnect_module(module_id: str):
    mod = get_module(module_id)
    if mod is None:
        raise HTTPException(status_code=404, detail=f"module {module_id} not found")
    ok = mod.connect()
    return {"id": module_id, "ok": ok, "error": mod.error}
