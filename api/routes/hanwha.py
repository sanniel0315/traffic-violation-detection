#!/usr/bin/env python3
"""Hanwha SUNAPI 球機追蹤與車牌放大 API。"""
from __future__ import annotations

import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from api.models import Camera, get_db
from api.routes.auth import get_current_user
from services.hanwha_sunapi import (
    BBox,
    DEFAULT_CHANNEL,
    DEFAULT_PROFILE,
    PtzStopWindow,
    SunapiError,
    build_plate_lpr_workflow,
    build_sunapi_client_from_camera,
)


router = APIRouter(prefix="/api/hanwha", tags=["hanwha-sunapi"])


class BBoxRequest(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    def to_bbox(self) -> BBox:
        return BBox(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2)


class TrackingControlRequest(BaseModel):
    channel: int = DEFAULT_CHANNEL
    profile: int = DEFAULT_PROFILE


class PlateZoomRequest(BaseModel):
    plate_bbox: BBoxRequest
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    channel: int = DEFAULT_CHANNEL
    profile: Optional[int] = DEFAULT_PROFILE
    padding_ratio: float = Field(default=0.35, ge=0.0, le=2.0)


class PtzStopWindowRequest(BaseModel):
    pan: Optional[float] = None
    tilt: Optional[float] = None
    zoom: Optional[float] = None
    pan_tolerance: float = Field(default=2.0, ge=0.0)
    tilt_tolerance: float = Field(default=2.0, ge=0.0)
    zoom_tolerance: float = Field(default=1.0, ge=0.0)

    def to_stop_window(self) -> PtzStopWindow:
        return PtzStopWindow(
            pan=self.pan,
            tilt=self.tilt,
            zoom=self.zoom,
            pan_tolerance=self.pan_tolerance,
            tilt_tolerance=self.tilt_tolerance,
            zoom_tolerance=self.zoom_tolerance,
        )


class WorkflowStepRequest(BaseModel):
    """單次追蹤流程判斷。

    plate_bbox 是攝影機 metadata 或 YOLO 找到的車牌座標。
    stop_zone 是你指定的畫面停止區，車牌中心進入後會停止追蹤。
    """

    plate_bbox: BBoxRequest
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    stop_zone: Optional[BBoxRequest] = None
    ptz_stop_window: Optional[PtzStopWindowRequest] = None
    channel: int = DEFAULT_CHANNEL
    profile: int = DEFAULT_PROFILE
    min_lpr_plate_width: int = Field(default=160, gt=0)
    min_lpr_plate_height: int = Field(default=48, gt=0)
    zoom_padding_ratio: float = Field(default=0.35, ge=0.0, le=2.0)
    execute: bool = True


def _get_camera_or_404(db: Session, camera_id: int) -> Camera:
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="找不到攝影機")
    return camera


def _client_for_camera(db: Session, camera_id: int):
    camera = _get_camera_or_404(db, camera_id)
    try:
        return camera, build_sunapi_client_from_camera(camera)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{camera_id}/ptz")
def get_ptz_position(
    camera_id: int,
    channel: int = DEFAULT_CHANNEL,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """讀取球機目前 Pan/Tilt/Zoom 座標。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        position = client.query_position(channel=channel)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "camera_id": camera.id,
        "camera_name": camera.name,
        "position": position.as_dict(),
    }


@router.get("/{camera_id}/supported-ptz-actions")
def get_supported_ptz_actions(
    camera_id: int,
    channel: int = DEFAULT_CHANNEL,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """查詢攝影機支援的 PTZ 子功能，確認是否有 query、areazoom、digitalautotracking。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        actions = client.supported_ptz_actions(channel=channel)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "camera_name": camera.name, "supported": actions}


@router.post("/{camera_id}/tracking/start")
def start_tracking(
    camera_id: int,
    req: TrackingControlRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """啟動攝影機端 digital auto tracking。"""
    camera, client = _client_for_camera(db, camera_id)
    print(f"[Hanwha] cam{camera_id} 手動啟動追蹤(面板/API)", flush=True)
    try:
        result = client.start_digital_autotracking(channel=req.channel, profile=req.profile)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "camera_name": camera.name, "result": result}


@router.post("/{camera_id}/tracking/stop")
def stop_tracking(
    camera_id: int,
    req: TrackingControlRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """停止攝影機端 digital auto tracking，並送出 PTZ stop。"""
    camera, client = _client_for_camera(db, camera_id)
    print(f"[Hanwha] cam{camera_id} 手動停止追蹤(面板/API)", flush=True)
    try:
        result = client.stop_digital_autotracking(channel=req.channel, profile=req.profile)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "camera_name": camera.name, "result": result}


class ContinuousMoveRequest(BaseModel):
    """連續移動速度,-100..100,0=不動。放開按鈕請打 /ptz/stop。"""

    pan: int = Field(default=0, ge=-100, le=100)
    tilt: int = Field(default=0, ge=-100, le=100)
    zoom: int = Field(default=0, ge=-100, le=100)
    channel: int = DEFAULT_CHANNEL


class AbsoluteMoveRequest(BaseModel):
    pan: Optional[float] = None
    tilt: Optional[float] = None
    zoom: Optional[float] = None
    channel: int = DEFAULT_CHANNEL


class FocusRequest(BaseModel):
    mode: str = Field(pattern="^(Near|Far|Stop|Auto|near|far|stop|auto)$")
    channel: int = DEFAULT_CHANNEL


class ChannelRequest(BaseModel):
    channel: int = DEFAULT_CHANNEL


@router.post("/{camera_id}/ptz/move")
def ptz_continuous_move(
    camera_id: int,
    req: ContinuousMoveRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """連續 PTZ 移動(按住方向鍵),全 0 等同 stop。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        if req.pan == 0 and req.tilt == 0 and req.zoom == 0:
            result = client.stop(channel=req.channel)
        else:
            result = client.continuous_move(
                pan=req.pan, tilt=req.tilt, zoom=req.zoom, channel=req.channel
            )
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "result": result}


@router.post("/{camera_id}/ptz/stop")
def ptz_stop(
    camera_id: int,
    req: ChannelRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """停止所有 PTZ 移動。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        result = client.stop(channel=req.channel)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "result": result}


@router.post("/{camera_id}/ptz/absolute")
def ptz_absolute_move(
    camera_id: int,
    req: AbsoluteMoveRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """絕對座標移動(只送有給值的軸)。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        result = client.absolute_move(
            pan=req.pan, tilt=req.tilt, zoom=req.zoom, channel=req.channel
        )
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "result": result}


@router.post("/{camera_id}/ptz/focus")
def ptz_focus(
    camera_id: int,
    req: FocusRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """對焦 Near / Far / Stop / Auto。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        result = client.focus_move(mode=req.mode, channel=req.channel)
    except (SunapiError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "result": result}


@router.post("/{camera_id}/ptz/home")
def ptz_home(
    camera_id: int,
    req: ChannelRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """回 Home 位置。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        result = client.goto_home(channel=req.channel)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "result": result}


@router.get("/{camera_id}/presets")
def list_presets(
    camera_id: int,
    channel: int = DEFAULT_CHANNEL,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """列出預置點。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        presets = client.list_presets(channel=channel)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "presets": presets}


@router.post("/{camera_id}/presets/{preset_no}/goto")
def goto_preset(
    camera_id: int,
    preset_no: int,
    req: ChannelRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """呼叫預置點。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        result = client.goto_preset(preset=preset_no, channel=req.channel)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "result": result}


class StopWindowRequest(BaseModel):
    """「追到這個座標就停止追蹤」:至少填一軸,公差內視為到達。"""

    pan: Optional[float] = None
    tilt: Optional[float] = None
    zoom: Optional[float] = None
    pan_tolerance: float = Field(default=2.0, ge=0.0)
    tilt_tolerance: float = Field(default=2.0, ge=0.0)
    zoom_tolerance: float = Field(default=1.0, ge=0.0)
    enabled: bool = True
    channel: int = DEFAULT_CHANNEL
    # 停止後行為:等 N 秒 → 回預置點 → 重新啟用追蹤(preset 給 None/0 = 不回)
    return_preset: Optional[int] = 1
    return_delay_sec: float = Field(default=5.0, ge=0.0)
    resume_tracking: bool = True


# ── 到點停追背景監看:每秒查 PTZ,進入停止窗(外→內轉態)就停止追蹤 ──────────
_stop_watchers: dict[int, threading.Event] = {}
_stop_watchers_lock = threading.Lock()


def _stop_window_watch_loop(camera_id: int, client, window: PtzStopWindow, stop_evt: threading.Event,
                            return_preset: Optional[int] = 1, return_delay_sec: float = 5.0,
                            resume_tracking: bool = True):
    inside_prev = False
    keepalive_n = 0
    _keepalive_backoff_until = [0.0]
    print(f"[Hanwha] cam{camera_id} 到點停追監看啟動 window={window.as_dict()} "
          f"回預置={return_preset} 延遲={return_delay_sec}s", flush=True)
    while not stop_evt.is_set():
        try:
            # keepalive:每 10 秒確認追蹤引擎還開著。這款球機「追完一台車
            # (目標離開視野/追丟)就自動把 Enable 關掉」(2026-08-28 整晚實測,
            # 與鎖定/手動操作無關) → 發現關了就重新啟用。新車出現時 LPR gate
            # 的 _ensure 也會即時重開,這裡是後備。
            if resume_tracking and stop_evt.wait(0) is False:
                keepalive_n += 1
                if keepalive_n >= 10 and time.monotonic() >= _keepalive_backoff_until[0]:
                    keepalive_n = 0
                    try:
                        if client.get_autotracking_enabled() is False:
                            # 🛑 set 是切換語意且 view 可能短暫過期 → 設定後必須回讀驗證,
                            #    失敗再補一次;兩次都失敗就退避 5 分鐘,避免跟相機互相切換
                            #    震盪(2026-08-28 實測每 12 秒關一次的循環就是這樣來的)。
                            ok = False
                            for _try in range(2):
                                client.start_digital_autotracking()
                                stop_evt.wait(1.2)
                                if client.get_autotracking_enabled() is True:
                                    ok = True
                                    break
                            if ok:
                                print(f"[Hanwha] cam{camera_id} 追蹤引擎被關閉,keepalive 已重新啟用(驗證通過)", flush=True)
                            else:
                                _keepalive_backoff_until[0] = time.monotonic() + 300
                                print(f"[Hanwha] cam{camera_id} keepalive 無法維持引擎開啟,退避 5 分鐘(可能有其他端在控制)", flush=True)
                    except Exception:
                        pass
            pos = client.query_position()
            inside = window.contains(pos)
            if inside and not inside_prev:
                client.stop_digital_autotracking()
                print(f"[Hanwha] cam{camera_id} 到達停止座標 {pos.as_dict()},已停止追蹤", flush=True)
                # 停止後等 N 秒 → 回預置點 → 重新啟用追蹤等下一台
                if return_preset:
                    stop_evt.wait(max(0.0, float(return_delay_sec)))
                    if not stop_evt.is_set():
                        try:
                            client.goto_preset(int(return_preset))
                            print(f"[Hanwha] cam{camera_id} 已回預置點 {return_preset}", flush=True)
                        except Exception as e:
                            print(f"[Hanwha] cam{camera_id} 回預置點失敗: {e}", flush=True)
                        if resume_tracking:
                            stop_evt.wait(2.0)   # 等相機走到定位再開追蹤
                            try:
                                client.start_digital_autotracking()
                                print(f"[Hanwha] cam{camera_id} 追蹤已重新啟用,等待下一台", flush=True)
                            except Exception as e:
                                print(f"[Hanwha] cam{camera_id} 追蹤重啟失敗: {e}", flush=True)
            inside_prev = inside
        except Exception as e:
            print(f"[Hanwha] cam{camera_id} 到點停追監看錯誤: {e}", flush=True)
            stop_evt.wait(3.0)
        stop_evt.wait(1.0)
    print(f"[Hanwha] cam{camera_id} 到點停追監看結束", flush=True)


def _disarm_stop_watcher(camera_id: int) -> None:
    with _stop_watchers_lock:
        evt = _stop_watchers.pop(camera_id, None)
    if evt:
        evt.set()


def _arm_stop_watcher(camera_id: int, client, window: PtzStopWindow,
                      return_preset: Optional[int] = 1, return_delay_sec: float = 5.0,
                      resume_tracking: bool = True) -> None:
    _disarm_stop_watcher(camera_id)
    evt = threading.Event()
    with _stop_watchers_lock:
        _stop_watchers[camera_id] = evt
    threading.Thread(
        target=_stop_window_watch_loop,
        args=(camera_id, client, window, evt, return_preset, return_delay_sec, resume_tracking),
        name=f"hanwha-stopwin-{camera_id}", daemon=True,
    ).start()


@router.post("/{camera_id}/tracking/stop-window")
def set_tracking_stop_window(
    camera_id: int,
    req: StopWindowRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """設定/啟停「追到指定座標就停止追蹤」。設定存進 camera detection_config,
    LPR 追蹤工作流與背景監看共用同一組停止窗。"""
    camera, client = _client_for_camera(db, camera_id)
    if req.enabled and req.pan is None and req.tilt is None and req.zoom is None:
        raise HTTPException(status_code=400, detail="至少要輸入一個座標(P/T/Z)")

    window_cfg = {
        "pan": req.pan, "tilt": req.tilt, "zoom": req.zoom,
        "pan_tolerance": req.pan_tolerance,
        "tilt_tolerance": req.tilt_tolerance,
        "zoom_tolerance": req.zoom_tolerance,
        "return_preset": req.return_preset,
        "return_delay_sec": req.return_delay_sec,
        "resume_tracking": req.resume_tracking,
    }
    cfg = dict(camera.detection_config or {})
    tracking_cfg = dict(cfg.get("hanwha_lpr_tracking") or {})
    tracking_cfg["ptz_stop_window"] = window_cfg
    tracking_cfg["ptz_stop_watch"] = bool(req.enabled)
    cfg["hanwha_lpr_tracking"] = tracking_cfg
    camera.detection_config = cfg
    flag_modified(camera, "detection_config")
    db.commit()

    if req.enabled:
        window = PtzStopWindow(
            pan=req.pan, tilt=req.tilt, zoom=req.zoom,
            pan_tolerance=req.pan_tolerance,
            tilt_tolerance=req.tilt_tolerance,
            zoom_tolerance=req.zoom_tolerance,
        )
        _arm_stop_watcher(camera.id, client, window,
                          return_preset=req.return_preset,
                          return_delay_sec=req.return_delay_sec,
                          resume_tracking=req.resume_tracking)
    else:
        _disarm_stop_watcher(camera.id)

    return {"camera_id": camera.id, "enabled": bool(req.enabled), "stop_window": window_cfg}


@router.get("/{camera_id}/tracking/stop-window")
def get_tracking_stop_window(
    camera_id: int,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """讀取目前的停止窗設定與監看狀態。"""
    camera = _get_camera_or_404(db, camera_id)
    cfg = (camera.detection_config or {}).get("hanwha_lpr_tracking") or {}
    with _stop_watchers_lock:
        armed = camera_id in _stop_watchers
    return {
        "camera_id": camera.id,
        "enabled": bool(cfg.get("ptz_stop_watch")),
        "armed": armed,
        "stop_window": cfg.get("ptz_stop_window"),
    }


@router.on_event("startup")
def _rearm_stop_watchers_on_startup() -> None:
    """服務重啟後,把 DB 裡標記啟用的到點停追監看重新掛回來。"""
    try:
        from api.models import SessionLocal
        db = SessionLocal()
        try:
            cams = db.query(Camera).all()
            for cam in cams:
                cfg = (cam.detection_config or {}).get("hanwha_lpr_tracking") or {}
                win = cfg.get("ptz_stop_window") or {}
                if not cfg.get("ptz_stop_watch") or not win:
                    continue
                try:
                    client = build_sunapi_client_from_camera(cam)
                except Exception:
                    continue
                window = PtzStopWindow(
                    pan=win.get("pan"), tilt=win.get("tilt"), zoom=win.get("zoom"),
                    pan_tolerance=float(win.get("pan_tolerance", 2.0)),
                    tilt_tolerance=float(win.get("tilt_tolerance", 2.0)),
                    zoom_tolerance=float(win.get("zoom_tolerance", 1.0)),
                )
                _arm_stop_watcher(cam.id, client, window,
                                  return_preset=win.get("return_preset", 1),
                                  return_delay_sec=float(win.get("return_delay_sec", 5.0)),
                                  resume_tracking=bool(win.get("resume_tracking", True)))
        finally:
            db.close()
    except Exception as e:
        print(f"[Hanwha] 到點停追監看重掛失敗: {e}", flush=True)


class TargetLockRequest(BaseModel):
    """點畫面鎖定目標:x/y 為相對整個影像內容的比例(0~1)。"""

    x_ratio: float = Field(ge=0.0, le=1.0)
    y_ratio: float = Field(ge=0.0, le=1.0)
    channel: int = DEFAULT_CHANNEL


class ObjectFilterRequest(BaseModel):
    vehicle_only: bool = True
    channel: int = DEFAULT_CHANNEL


@router.post("/{camera_id}/tracking/target-lock")
def tracking_target_lock(
    camera_id: int,
    req: TargetLockRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """在畫面上點一個位置,鎖定該處目標開始追蹤。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        result = client.target_lock(x_ratio=req.x_ratio, y_ratio=req.y_ratio, channel=req.channel)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "result": result}


@router.post("/{camera_id}/tracking/object-filter")
def tracking_object_filter(
    camera_id: int,
    req: ObjectFilterRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """追蹤物件過濾:只追車(Vehicle)或人車都追。"""
    camera, client = _client_for_camera(db, camera_id)
    try:
        result = client.set_tracking_object_filter(vehicle_only=req.vehicle_only, channel=req.channel)
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "result": result}


@router.post("/{camera_id}/plate/zoom")
def zoom_plate_area(
    camera_id: int,
    req: PlateZoomRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """用車牌 bbox 執行 SUNAPI areazoom，讓後續 LPR 拿到較大的車牌圖。"""
    camera, client = _client_for_camera(db, camera_id)
    bbox = req.plate_bbox.to_bbox().padded(req.frame_width, req.frame_height, req.padding_ratio)
    try:
        result = client.area_zoom(
            bbox=bbox,
            frame_width=req.frame_width,
            frame_height=req.frame_height,
            channel=req.channel,
            profile=req.profile,
        )
    except SunapiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"camera_id": camera.id, "camera_name": camera.name, "result": result}


@router.post("/{camera_id}/workflow/step")
def run_tracking_lpr_step(
    camera_id: int,
    req: WorkflowStepRequest,
    db: Session = Depends(get_db),
    _user=Depends(get_current_user),
):
    """執行一次「追蹤 -> zoom -> LPR ready -> 到停止區停止」流程。"""
    camera, client = _client_for_camera(db, camera_id)
    ptz_position = None
    if req.ptz_stop_window:
        try:
            ptz_position = client.query_position(channel=req.channel)
        except SunapiError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    workflow = build_plate_lpr_workflow(
        plate_bbox=req.plate_bbox.to_bbox(),
        frame_width=req.frame_width,
        frame_height=req.frame_height,
        stop_zone=req.stop_zone.to_bbox() if req.stop_zone else None,
        ptz_position=ptz_position,
        ptz_stop_window=req.ptz_stop_window.to_stop_window() if req.ptz_stop_window else None,
        min_lpr_plate_width=req.min_lpr_plate_width,
        min_lpr_plate_height=req.min_lpr_plate_height,
        zoom_padding_ratio=req.zoom_padding_ratio,
    )

    executed: list[dict] = []
    if req.execute:
        try:
            if "stop_tracking" in workflow["actions"]:
                executed.append(client.stop_digital_autotracking(channel=req.channel, profile=req.profile))
            elif "area_zoom" in workflow["actions"]:
                executed.append(
                    client.area_zoom(
                        bbox=BBox(*workflow["zoom_bbox"]),
                        frame_width=req.frame_width,
                        frame_height=req.frame_height,
                        channel=req.channel,
                        profile=req.profile,
                    )
                )
        except SunapiError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "camera_id": camera.id,
        "camera_name": camera.name,
        "workflow": workflow,
        "executed": executed,
        "next_step": "run_lpr" if workflow["lpr_ready"] else workflow["state"],
    }
