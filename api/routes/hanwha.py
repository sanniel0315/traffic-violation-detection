#!/usr/bin/env python3
"""Hanwha SUNAPI 球機追蹤與車牌放大 API。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

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
