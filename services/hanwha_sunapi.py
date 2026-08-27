#!/usr/bin/env python3
"""Hanwha SUNAPI 球機追蹤與車牌放大控制。

使用方式：
- 先用 `start_digital_autotracking()` 讓攝影機端追蹤 Vehicle。
- 每次取得車牌 bbox 後呼叫 `build_plate_lpr_workflow()`，決定是否需要 areazoom。
- 當目標進入指定停止區時呼叫 `stop_digital_autotracking()`，後續交給 LPR/OCR。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from requests.auth import HTTPDigestAuth


SUNAPI_TIMEOUT_SEC = 4.0
DEFAULT_CHANNEL = 0
DEFAULT_PROFILE = 2


@dataclass(frozen=True)
class BBox:
    """影像座標框，格式為左上與右下座標。"""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def padded(self, frame_width: int, frame_height: int, ratio: float) -> "BBox":
        pad_x = int(round(self.width * max(0.0, ratio)))
        pad_y = int(round(self.height * max(0.0, ratio)))
        return BBox(
            x1=max(0, self.x1 - pad_x),
            y1=max(0, self.y1 - pad_y),
            x2=min(frame_width, self.x2 + pad_x),
            y2=min(frame_height, self.y2 + pad_y),
        )

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class PtzPosition:
    pan: Optional[float]
    tilt: Optional[float]
    zoom: Optional[float]
    zoom_pulse: Optional[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pan": self.pan,
            "tilt": self.tilt,
            "zoom": self.zoom,
            "zoom_pulse": self.zoom_pulse,
        }


@dataclass(frozen=True)
class PtzStopWindow:
    """PTZ 停止條件，目標位置進入容許誤差範圍就停止追蹤。"""

    pan: Optional[float] = None
    tilt: Optional[float] = None
    zoom: Optional[float] = None
    pan_tolerance: float = 2.0
    tilt_tolerance: float = 2.0
    zoom_tolerance: float = 1.0

    def contains(self, position: Optional[PtzPosition]) -> bool:
        if position is None:
            return False
        checks: list[bool] = []
        if self.pan is not None:
            checks.append(position.pan is not None and abs(position.pan - self.pan) <= self.pan_tolerance)
        if self.tilt is not None:
            checks.append(position.tilt is not None and abs(position.tilt - self.tilt) <= self.tilt_tolerance)
        if self.zoom is not None:
            checks.append(position.zoom is not None and abs(position.zoom - self.zoom) <= self.zoom_tolerance)
        return bool(checks) and all(checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pan": self.pan,
            "tilt": self.tilt,
            "zoom": self.zoom,
            "pan_tolerance": self.pan_tolerance,
            "tilt_tolerance": self.tilt_tolerance,
            "zoom_tolerance": self.zoom_tolerance,
        }


class SunapiError(RuntimeError):
    """SUNAPI 呼叫失敗。"""


class HanwhaSunapiClient:
    """SUNAPI HTTP client，支援 Digest Auth 與基本 key=value 回應解析。"""

    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = SUNAPI_TIMEOUT_SEC,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        if not self.base_url:
            raise ValueError("SUNAPI base_url 不可為空")
        self.username = username or ""
        self.password = password or ""
        self.timeout = float(timeout)
        # 🛑 共用 Session:沒有的話每次請求都重新 TCP 交握 + Digest 401 挑戰,
        #    延遲直接翻倍;分析迴圈裡每個車牌幀都可能打,累積很可觀。
        self._session = requests.Session()
        if self.username:
            self._session.auth = HTTPDigestAuth(self.username, self.password)

    def _request(self, path: str, params: dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise SunapiError(f"SUNAPI 連線失敗: {exc}") from exc
        if resp.status_code >= 400:
            raise SunapiError(f"SUNAPI 回應 HTTP {resp.status_code}: {resp.text[:200]}")
        return resp

    @staticmethod
    def _parse_text_response(text: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
        return parsed

    @staticmethod
    def _as_float(data: dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _as_int(data: dict[str, Any], *keys: str) -> Optional[int]:
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
        return None

    def query_position(self, channel: int = DEFAULT_CHANNEL) -> PtzPosition:
        resp = self._request(
            "/stw-cgi/ptzcontrol.cgi",
            {
                "msubmenu": "query",
                "action": "view",
                "Channel": int(channel),
                "Query": "Pan,Tilt,Zoom",
            },
        )
        data: dict[str, Any]
        try:
            body = resp.json()
            items = body.get("Query") if isinstance(body, dict) else None
            data = dict(items[0]) if isinstance(items, list) and items else {}
        except ValueError:
            data = self._parse_text_response(resp.text)
        return PtzPosition(
            pan=self._as_float(data, "Pan", "pan"),
            tilt=self._as_float(data, "Tilt", "tilt"),
            zoom=self._as_float(data, "Zoom", "zoom"),
            zoom_pulse=self._as_int(data, "ZoomPulse", "zoom_pulse"),
        )

    def start_digital_autotracking(
        self,
        profile: int = DEFAULT_PROFILE,
        channel: int = DEFAULT_CHANNEL,
    ) -> dict[str, Any]:
        return self._digital_autotracking("Start", profile=profile, channel=channel)

    def stop_digital_autotracking(
        self,
        profile: int = DEFAULT_PROFILE,
        channel: int = DEFAULT_CHANNEL,
    ) -> dict[str, Any]:
        result = self._digital_autotracking("Stop", profile=profile, channel=channel)
        self.stop(channel=channel)
        return result

    def _digital_autotracking(self, mode: str, profile: int, channel: int) -> dict[str, Any]:
        resp = self._request(
            "/stw-cgi/ptzcontrol.cgi",
            {
                "msubmenu": "digitalautotracking",
                "action": "control",
                "Channel": int(channel),
                "Profile": int(profile),
                "Mode": mode,
            },
        )
        return {"ok": True, "mode": mode, "status_code": resp.status_code}

    def configure_tracking_object_filter(
        self,
        object_types: list[str],
        channel: int = DEFAULT_CHANNEL,
    ) -> dict[str, Any]:
        resp = self._request(
            "/stw-cgi/ptzconfig.cgi",
            {
                "msubmenu": "digitalautotracking",
                "action": "set",
                "Channel": int(channel),
                "ObjectTypeFilter": ",".join(object_types),
            },
        )
        return {"ok": True, "object_types": object_types, "status_code": resp.status_code}

    def area_zoom(
        self,
        bbox: BBox,
        frame_width: int,
        frame_height: int,
        channel: int = DEFAULT_CHANNEL,
        profile: Optional[int] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "msubmenu": "areazoom",
            "action": "control",
            "Channel": int(channel),
            "Type": "ZoomIn",
            "X1": int(bbox.x1),
            "Y1": int(bbox.y1),
            "X2": int(bbox.x2),
            "Y2": int(bbox.y2),
            "TileWidth": int(frame_width),
            "TileHeight": int(frame_height),
        }
        if profile is not None:
            params["Profile"] = int(profile)
        resp = self._request("/stw-cgi/ptzcontrol.cgi", params)
        return {
            "ok": True,
            "bbox": bbox.as_list(),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "status_code": resp.status_code,
        }

    def continuous_move(
        self,
        pan: int = 0,
        tilt: int = 0,
        zoom: int = 0,
        channel: int = DEFAULT_CHANNEL,
    ) -> dict[str, Any]:
        """連續移動(搖桿式)。速度 -100..100,0=該軸不動;全部 0 請改用 stop()。"""
        params: dict[str, Any] = {
            "msubmenu": "continuous",
            "action": "control",
            "Channel": int(channel),
        }
        if int(pan):
            params["Pan"] = max(-100, min(100, int(pan)))
        if int(tilt):
            params["Tilt"] = max(-100, min(100, int(tilt)))
        if int(zoom):
            params["Zoom"] = max(-100, min(100, int(zoom)))
        resp = self._request("/stw-cgi/ptzcontrol.cgi", params)
        return {"ok": True, "pan": pan, "tilt": tilt, "zoom": zoom, "status_code": resp.status_code}

    def absolute_move(
        self,
        pan: Optional[float] = None,
        tilt: Optional[float] = None,
        zoom: Optional[float] = None,
        channel: int = DEFAULT_CHANNEL,
    ) -> dict[str, Any]:
        """絕對座標移動,只送有給值的軸。"""
        params: dict[str, Any] = {
            "msubmenu": "absolute",
            "action": "control",
            "Channel": int(channel),
        }
        if pan is not None:
            params["Pan"] = float(pan)
        if tilt is not None:
            params["Tilt"] = float(tilt)
        if zoom is not None:
            params["Zoom"] = float(zoom)
        resp = self._request("/stw-cgi/ptzcontrol.cgi", params)
        return {"ok": True, "pan": pan, "tilt": tilt, "zoom": zoom, "status_code": resp.status_code}

    def focus_move(self, mode: str, channel: int = DEFAULT_CHANNEL) -> dict[str, Any]:
        """對焦控制:Near / Far / Stop(部分機型支援 Auto)。"""
        mode = str(mode).strip().capitalize()
        if mode not in {"Near", "Far", "Stop", "Auto"}:
            raise ValueError(f"不支援的 focus mode: {mode}")
        resp = self._request(
            "/stw-cgi/ptzcontrol.cgi",
            {"msubmenu": "focus", "action": "control", "Channel": int(channel), "Focus": mode},
        )
        return {"ok": True, "focus": mode, "status_code": resp.status_code}

    def goto_home(self, channel: int = DEFAULT_CHANNEL) -> dict[str, Any]:
        """回 Home 位置。"""
        resp = self._request(
            "/stw-cgi/ptzcontrol.cgi",
            {"msubmenu": "home", "action": "control", "Channel": int(channel)},
        )
        return {"ok": True, "status_code": resp.status_code}

    def list_presets(self, channel: int = DEFAULT_CHANNEL) -> Any:
        """列出預置點。"""
        resp = self._request(
            "/stw-cgi/ptzcontrol.cgi",
            {"msubmenu": "preset", "action": "view", "Channel": int(channel)},
        )
        try:
            return resp.json()
        except ValueError:
            return self._parse_text_response(resp.text)

    def goto_preset(self, preset: int, channel: int = DEFAULT_CHANNEL) -> dict[str, Any]:
        """呼叫預置點。"""
        resp = self._request(
            "/stw-cgi/ptzcontrol.cgi",
            {
                "msubmenu": "preset",
                "action": "control",
                "Channel": int(channel),
                "Preset": int(preset),
            },
        )
        return {"ok": True, "preset": int(preset), "status_code": resp.status_code}

    def stop(self, channel: int = DEFAULT_CHANNEL) -> dict[str, Any]:
        resp = self._request(
            "/stw-cgi/ptzcontrol.cgi",
            {"msubmenu": "stop", "action": "control", "Channel": int(channel)},
        )
        return {"ok": True, "status_code": resp.status_code}

    def supported_ptz_actions(self, channel: int = DEFAULT_CHANNEL) -> dict[str, Any]:
        resp = self._request(
            "/stw-cgi/ptzcontrol.cgi",
            {
                "msubmenu": "supportedptzactions",
                "action": "view",
                "Channel": int(channel),
            },
        )
        try:
            return resp.json()
        except ValueError:
            return self._parse_text_response(resp.text)


def _camera_host(camera: Any) -> str:
    ip = str(getattr(camera, "ip", "") or "").strip()
    if ip:
        return ip
    source = str(getattr(camera, "source", "") or "").strip()
    if source:
        parsed = urlparse(source)
        if parsed.hostname:
            return parsed.hostname
    raise ValueError("Camera 缺少 ip，且 source 無法解析主機")


def build_sunapi_client_from_camera(camera: Any) -> HanwhaSunapiClient:
    config = getattr(camera, "detection_config", None) or {}
    sunapi_config = config.get("sunapi") if isinstance(config, dict) else {}
    sunapi_config = sunapi_config if isinstance(sunapi_config, dict) else {}
    base_url = (
        sunapi_config.get("base_url")
        or (config.get("sunapi_base_url") if isinstance(config, dict) else None)
        or ""
    )
    if not base_url:
        scheme = str(sunapi_config.get("scheme") or "http").strip() or "http"
        http_port = sunapi_config.get("http_port") or (
            config.get("sunapi_http_port") if isinstance(config, dict) else None
        )
        host = _camera_host(camera)
        port_suffix = f":{int(http_port)}" if http_port else ""
        base_url = f"{scheme}://{host}{port_suffix}"

    return HanwhaSunapiClient(
        base_url=base_url,
        username=str(sunapi_config.get("username") or getattr(camera, "username", "") or ""),
        password=str(sunapi_config.get("password") or getattr(camera, "password", "") or ""),
        timeout=float(sunapi_config.get("timeout") or SUNAPI_TIMEOUT_SEC),
    )


def build_plate_lpr_workflow(
    *,
    plate_bbox: BBox,
    frame_width: int,
    frame_height: int,
    stop_zone: Optional[BBox] = None,
    ptz_position: Optional[PtzPosition] = None,
    ptz_stop_window: Optional[PtzStopWindow] = None,
    min_lpr_plate_width: int = 160,
    min_lpr_plate_height: int = 48,
    zoom_padding_ratio: float = 0.35,
) -> dict[str, Any]:
    """判斷追蹤、放大、LPR、停止追蹤的下一步。

    回傳的 `actions` 給路由決定是否實際呼叫 SUNAPI；這樣單元測試不用連攝影機。
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width/frame_height 必須大於 0")
    if plate_bbox.width <= 0 or plate_bbox.height <= 0:
        raise ValueError("plate_bbox 無效")

    center_x, center_y = plate_bbox.center
    reached_stop_zone = bool(stop_zone and stop_zone.contains_point(center_x, center_y))
    reached_ptz_stop = bool(ptz_stop_window and ptz_stop_window.contains(ptz_position))
    lpr_ready = plate_bbox.width >= min_lpr_plate_width and plate_bbox.height >= min_lpr_plate_height
    zoom_bbox = plate_bbox.padded(frame_width, frame_height, zoom_padding_ratio)
    actions: list[str] = []

    if reached_stop_zone or reached_ptz_stop:
        actions.append("stop_tracking")
    elif not lpr_ready:
        actions.append("area_zoom")
    else:
        actions.append("lpr_ready")

    return {
        "state": "stopped" if (reached_stop_zone or reached_ptz_stop) else ("lpr_ready" if lpr_ready else "zooming"),
        "actions": actions,
        "lpr_ready": lpr_ready,
        "reached_stop_zone": reached_stop_zone,
        "reached_ptz_stop": reached_ptz_stop,
        "plate_bbox": plate_bbox.as_list(),
        "zoom_bbox": zoom_bbox.as_list(),
        "plate_size": {"width": plate_bbox.width, "height": plate_bbox.height},
        "plate_center": {"x": center_x, "y": center_y},
        "ptz_position": ptz_position.as_dict() if ptz_position else None,
        "ptz_stop_window": ptz_stop_window.as_dict() if ptz_stop_window else None,
    }
