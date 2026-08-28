#!/usr/bin/env python3
"""推播通知:App 註冊 Expo push token,後端依「裝置訂閱規則」推播告警。

部署(main.py 已幫你接好):
  from api.routes import push
  app.include_router(push.router)
  # lifespan 啟動時:push.start_poller()

裝置訂閱規則(網頁「告警通知」頁設定,存 push_tokens.json):
  每個 token 一筆:{enabled, note, categories, cameras, window{start,end}}
  - categories 告警分類:tracking 球機追蹤 / speeding 超速 / red_line_stop 紅線臨停
    / lpr 車牌辨識 / other 其他;空=全部
  - cameras 攝影機 id 列表;空=全部
  - window 推播時段 HH:MM~HH:MM(可跨夜);空=全天
App 端只負責 register/unregister(帶 token),細節規則在網頁設定。
舊格式(字串陣列 / token→types)自動遷移,不動 DB schema。
背景 poller 每 20 秒查 Violation 表推違規;球機追蹤走 push_alert()。
"""
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.routes.auth import get_current_user

router = APIRouter(prefix="/api/push", tags=["推播"])

_TOKENS_FILE = Path(__file__).resolve().parent.parent.parent / "push_tokens.json"
_EVENTS_FILE = Path(__file__).resolve().parent.parent.parent / "push_events.json"
_EVENTS_MAX = 200
_EXPO_URL = "https://exp.host/--/api/v2/push/send"
_POLL_SEC = 20

# 告警分類(key → 顯示名稱)。「不以違規為主力」:球機追蹤、車牌辨識是一級分類
CATEGORIES: Dict[str, str] = {
    "tracking": "球機追蹤",
    "speeding": "超速",
    "red_line_stop": "紅線臨停",
    "lpr": "車牌辨識",
    "other": "其他告警",
}


def _default_device() -> dict:
    return {"enabled": True, "note": "", "categories": [], "cameras": [], "window": {"start": "", "end": ""}}


def _load_devices() -> Dict[str, dict]:
    """token → 訂閱規則;相容舊格式(字串陣列=全訂 / token→types 列表)"""
    try:
        raw = json.loads(_TOKENS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("devices"), dict):
        out = {}
        for t, d in raw["devices"].items():
            if t and isinstance(d, dict):
                out[t] = {**_default_device(), **d}
        return out
    if isinstance(raw, list):   # v1:token 字串陣列
        return {t: _default_device() for t in raw if t}
    if isinstance(raw, dict):   # v2:token → types 列表(violation/lpr)
        return {t: _default_device() for t in raw.keys() if t}
    return {}


def _save_devices(devices: Dict[str, dict]) -> None:
    try:
        _TOKENS_FILE.write_text(json.dumps({"v": 3, "devices": devices}, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def _in_window(win: dict, now_hm: str) -> bool:
    """推播時段判斷;start/end 空=全天。支援跨夜(如 22:00~07:00)。"""
    s = str(win.get("start") or "").strip()
    e = str(win.get("end") or "").strip()
    if not s or not e:
        return True
    if s <= e:
        return s <= now_hm <= e
    return now_hm >= s or now_hm <= e   # 跨夜

def _targets(category: str, camera_id: Optional[int] = None) -> List[str]:
    """依規則篩出要推的 token:啟用 + 分類符合 + 攝影機符合 + 在時段內"""
    now_hm = datetime.now().strftime("%H:%M")
    out: List[str] = []
    for tk, d in _load_devices().items():
        if not d.get("enabled", True):
            continue
        cats = [c for c in (d.get("categories") or []) if c in CATEGORIES]
        if cats and category not in cats:
            continue
        cams = d.get("cameras") or []
        if cams and camera_id is not None:
            try:
                if int(camera_id) not in [int(x) for x in cams]:
                    continue
            except Exception:
                pass
        if not _in_window(d.get("window") or {}, now_hm):
            continue
        out.append(tk)
    return out


class TokenBody(BaseModel):
    token: str
    types: Optional[List[str]] = None   # 舊版 App 相容欄位,現已不使用


@router.post("/register")
def register(body: TokenBody) -> dict:
    """App 註冊 push token。已存在的裝置保留網頁上設定的規則,只當作心跳。"""
    devices = _load_devices()
    if body.token:
        if body.token not in devices:
            devices[body.token] = _default_device()
        devices[body.token]["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_devices(devices)
    return {"ok": True, "count": len(devices)}


@router.post("/unregister")
def unregister(body: TokenBody) -> dict:
    """App 關閉推播時移除 token(該裝置的規則一併移除)"""
    devices = _load_devices()
    devices.pop(body.token, None)
    _save_devices(devices)
    return {"ok": True, "count": len(devices)}


@router.get("/status")
def status() -> dict:
    """目前註冊裝置數與啟用數"""
    devices = _load_devices()
    return {"tokens": len(devices),
            "enabled": sum(1 for d in devices.values() if d.get("enabled", True)),
            "poll_sec": _POLL_SEC}


# ── 網頁「告警通知」設定頁用的管理 API(需登入)──────────────────────────

@router.get("/devices")
def list_devices(_user=Depends(get_current_user)) -> dict:
    """列出所有裝置與訂閱規則(給網頁設定頁)"""
    devices = _load_devices()
    rows = []
    for tk, d in devices.items():
        rows.append({"token": tk, **{k: d.get(k) for k in ("enabled", "note", "categories", "cameras", "window", "last_seen")}})
    return {"devices": rows, "categories": CATEGORIES}


class DeviceBody(BaseModel):
    token: str
    enabled: Optional[bool] = None
    note: Optional[str] = None
    categories: Optional[List[str]] = None   # 空列表=全部分類
    cameras: Optional[List[int]] = None      # 空列表=全部攝影機
    window: Optional[Dict[str, str]] = None  # {start:"22:00", end:"07:00"};空字串=全天


@router.put("/devices")
def update_device(body: DeviceBody, _user=Depends(get_current_user)) -> dict:
    """更新某裝置的訂閱規則(網頁設定頁)"""
    devices = _load_devices()
    if body.token not in devices:
        raise HTTPException(status_code=404, detail="裝置不存在(可能已解除註冊)")
    d = devices[body.token]
    if body.enabled is not None:
        d["enabled"] = bool(body.enabled)
    if body.note is not None:
        d["note"] = str(body.note)[:60]
    if body.categories is not None:
        d["categories"] = [c for c in body.categories if c in CATEGORIES]
    if body.cameras is not None:
        d["cameras"] = [int(x) for x in body.cameras]
    if body.window is not None:
        d["window"] = {"start": str(body.window.get("start") or ""), "end": str(body.window.get("end") or "")}
    _save_devices(devices)
    return {"ok": True, "device": {"token": body.token, **d}}


@router.post("/devices/delete")
def delete_device(body: TokenBody, _user=Depends(get_current_user)) -> dict:
    """從網頁移除裝置(踢掉不再使用的手機)"""
    devices = _load_devices()
    devices.pop(body.token, None)
    _save_devices(devices)
    return {"ok": True, "count": len(devices)}


@router.post("/devices/test")
def test_device(body: TokenBody, _user=Depends(get_current_user)) -> dict:
    """對單一裝置發測試推播(網頁「測試」按鈕)"""
    devices = _load_devices()
    if body.token not in devices:
        raise HTTPException(status_code=404, detail="裝置不存在")
    _send_expo([body.token], "🔔 測試通知", "告警通知設定連通測試", {"type": "test"})
    return {"ok": True}


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


def _record_event(title: str, body: str, data: Optional[dict], category: str) -> None:
    """把即時告警事件寫進 push_events.json(環形,最多 _EVENTS_MAX 筆),給 App 告警面板查歷史"""
    try:
        try:
            events = json.loads(_EVENTS_FILE.read_text(encoding="utf-8"))
            if not isinstance(events, list):
                events = []
        except Exception:
            events = []
        events.append({
            "id": int(time.time() * 1000),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "body": body,
            "data": data or {},
            "category": category,
        })
        _EVENTS_FILE.write_text(json.dumps(events[-_EVENTS_MAX:], ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


@router.get("/events")
def events(limit: int = 40) -> dict:
    """最近的即時告警事件(球機追蹤等),新到舊;App 告警面板用"""
    try:
        evs = json.loads(_EVENTS_FILE.read_text(encoding="utf-8"))
        if not isinstance(evs, list):
            evs = []
    except Exception:
        evs = []
    return {"events": list(reversed(evs))[:max(1, min(limit, _EVENTS_MAX))]}


def push_alert(title: str, body: str, data: Optional[dict] = None, category: str = "tracking") -> None:
    """給其他模組用的即時告警推播(如 Hanwha 球機追蹤事件,歸類 tracking)。
    依裝置訂閱規則(分類/攝影機/時段)篩選對象;同步、失敗吞掉。
    無論有沒有對象都會記進事件檔,App 告警面板才查得到歷史。"""
    try:
        _record_event(title, body, data, category)
        cam = (data or {}).get("camera_id")
        tokens = _targets(category, cam if isinstance(cam, int) else None)
        if tokens:
            _send_expo(tokens, title, body, data)
    except Exception:
        pass


def _violation_category(violation_type: str) -> str:
    vt = (violation_type or "").strip().lower()
    return vt if vt in CATEGORIES else "other"


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
        # 最多推最近 10 筆,避免一次湧入洗版;每筆依自己的分類/攝影機篩對象
        for v in news[-10:]:
            title = getattr(v, "violation_name", None) or getattr(v, "violation_type", None) or "違規"
            speed = getattr(v, "speed_kmh", None)
            speed_txt = f" {round(speed)} km/h" if speed else ""
            plate = getattr(v, "license_plate", None) or "未辨識車牌"
            cam = getattr(v, "camera_id", None)
            cam_txt = f" · cam_{cam}" if cam is not None else ""
            category = _violation_category(getattr(v, "violation_type", "") or "")
            tokens = _targets(category, int(cam) if cam is not None else None)
            if not tokens:
                continue
            await asyncio.to_thread(
                _send_expo, tokens,
                f"⚠ {title}{speed_txt}",
                f"{plate}{cam_txt}",
                {"violation_id": int(v.id), "type": category, "camera_id": cam},
            )


_task: Optional[asyncio.Task] = None


def start_poller() -> None:
    """在 FastAPI lifespan startup 呼叫一次"""
    global _task
    if _task is None:
        _task = asyncio.create_task(_poller())
