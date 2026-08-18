from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import requests


def _as_text(value: Any) -> str:
    return str(value or "").strip()


# ── VideoCapture 開啟閘門 ────────────────────────────────────────────────
# 🛑 為什麼需要:OpenCV 的 FFmpeg backend 是在「open 的當下」才去
#    getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS") 讀參數,而我們全部是用
#    os.environ[...] = ... 把參數塞進去的。os.environ 的寫入等同 setenv(),
#    glibc 的 setenv 可能 realloc 整個 environ 陣列 —— 跟另一條 thread 正在
#    進行的 getenv() 併發就是 use-after-free,直接原生 SEGV,Python 層攔不到。
#
#    2026-08-18 08:33 在 87 實際炸掉:cam_2/3/4/5 四條串流在同一秒 30 秒逾時
#    → 四條 reader 同時走重連路徑 → 同時 setenv + VideoCapture()
#    → Fatal Python error: Segmentation fault,整個 traffic-api 被 systemd 重啟。
#    faulthandler 印出來的兩條 reader 剛好一條停在 cap.grab()、一條停在
#    cv2.VideoCapture(...) —— 正是這個併發組合。
#
#    順帶治好第二件事:這個環境變數是「整個 process 共用」,而
#    stream / congestion / lpr_stream 三個模組各自要不同的參數
#    (congestion 要 threads;1、LPR 要 stimeout;5000000)。沒有鎖的話
#    誰後寫誰贏,開出來的 cap 可能吃到別的模組的參數。鎖讓「設參數 → 開」
#    變成一個原子動作,各自要的參數才真的生效。
#
#    代價:同時間只能有一條 thread 在開 cap。開 RTSP 最久會卡到 stimeout(10 秒),
#    所以重連風暴時會變成排隊而不是並行 —— 這是刻意的取捨,重連本來就罕見,
#    而且比整個服務掛掉好太多。穩態(沒有人在重連)完全不受影響。
_CAP_OPEN_LOCK = threading.RLock()


@contextmanager
def capture_open_guard(options: str | None = None):
    """把「設 FFmpeg 參數 → cv2.VideoCapture(...)」包成序列化的原子區段。

    用法:
        with capture_open_guard(OPTS if is_rtsp else None):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    options 給 None 就只上鎖、不動環境變數(非 RTSP 來源不需要那些參數)。
    RLock:同一條 thread 巢狀進入不會自鎖。
    """
    with _CAP_OPEN_LOCK:
        if options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
        yield


def open_capture(source: Any, backend: Any = None, options: str | None = None):
    """cv2.VideoCapture 的替身 —— 開啟動作全程序序列化。

    等價於 `with capture_open_guard(options): cv2.VideoCapture(source[, backend])`,
    做成函式是為了讓呼叫端只要換函式名、不必改縮排。

    🛑 為什麼「沒有設參數的開啟」也要進閘門:race 的兩邊是 setenv 與 getenv,
       OpenCV 在每一次 open 都會 getenv。只鎖住寫的那一側沒有用 —— 讀的那一側
       同樣要排隊,否則照樣 use-after-free。
    """
    import cv2  # 延遲載入:camera_stream 也被沒有 cv2 的路徑 import
    with capture_open_guard(options):
        if backend is None:
            return cv2.VideoCapture(source)
        return cv2.VideoCapture(source, backend)


def _encode_rtsp_credential(value: Any) -> str:
    return quote(str(value or ""), safe="")


def _has_http_scheme(text: str) -> bool:
    value = _as_text(text).lower()
    return value.startswith("http://") or value.startswith("https://")


def _has_rtsp_scheme(text: str) -> bool:
    value = _as_text(text).lower()
    return value.startswith("rtsp://") or value.startswith("rtsps://")


def build_rtsp_source(ip: Any, username: Any, password: Any, port: Any, stream_path: Any) -> str:
    host = _as_text(ip)
    if not host:
        return ""
    if _has_rtsp_scheme(host):
        return host
    url = "rtsp://"
    user = _as_text(username)
    pwd = str(password or "")
    if user:
        url += _encode_rtsp_credential(user)
        if pwd:
            url += f":{_encode_rtsp_credential(pwd)}"
        url += "@"
    url += host
    port_text = _as_text(port)
    if port_text:
        url += f":{port_text}"
    path = _as_text(stream_path).lstrip("/")
    if path:
        url += f"/{path}"
    return url


def _internal_api_base_url() -> str:
    return _as_text(os.getenv("API_INTERNAL_BASE_URL")) or "http://127.0.0.1:8000"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_local_files_source(text: str) -> str:
    value = _as_text(text)
    if not value.startswith("/files/"):
        return ""
    output_root = (_project_root() / "output").resolve()
    rel = value[len("/files/"):].lstrip("/")
    candidate = (output_root / rel).resolve()
    try:
        candidate.relative_to(output_root)
    except Exception:
        return ""
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return ""


def _nx_settings_candidates() -> list[Path]:
    filename = "nx_settings.json"
    candidates: list[Path] = []
    env_dir = _as_text(os.getenv("SYSTEM_CONFIG_DIR"))
    if env_dir:
        candidates.append(Path(env_dir) / filename)
    candidates.append(Path("/workspace/config/system") / filename)
    candidates.append(_project_root() / "config" / "system" / filename)
    uniq: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(path)
    return uniq


def _looks_like_nx_proxy_source(text: str) -> bool:
    if not text:
        return False
    try:
        parsed = urlsplit(text)
        path = parsed.path or text
    except Exception:
        path = text
    return path.startswith("/api/nx/stream/")


def is_nx_proxy_source(source: Any) -> bool:
    return _looks_like_nx_proxy_source(_as_text(source))


def _nx_proxy_parts(text: str) -> tuple[str, dict[str, str]] | None:
    try:
        parsed = urlsplit(text)
        path = parsed.path or text
        params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    except Exception:
        path = text
        params = {}
    if not path.startswith("/api/nx/stream/"):
        return None
    device_id = unquote(path.rsplit("/", 1)[-1]).strip()
    if not device_id:
        return None
    return device_id, params


def _nx_settings() -> dict[str, Any]:
    for path in _nx_settings_candidates():
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    return {}


def _nx_server_base(settings: dict[str, Any]) -> str:
    return _as_text(settings.get("server_base_url")).rstrip("/")


def _nx_timeout(settings: dict[str, Any]) -> float:
    try:
        return max(3.0, float(settings.get("timeout_sec", 12)))
    except Exception:
        return 12.0


def _nx_verify_ssl(settings: dict[str, Any]) -> bool:
    return bool(settings.get("verify_ssl", False))


def _nx_nonce_path(settings: dict[str, Any]) -> str:
    return _as_text(settings.get("nonce_path")) or "/api/getNonce"


def _nx_media_path(settings: dict[str, Any], device_id: str, fmt: str) -> str:
    template = _as_text(settings.get("media_path_template")) or "/media/{device_id}.{format}"
    return template.format(device_id=quote(device_id, safe=""), format=fmt)


def _nx_rtsp_source(device_id: str, stream_index: int = 0) -> str:
    settings = _nx_settings()
    base = _nx_server_base(settings)
    if not base:
        return ""
    parsed = urlsplit(base)
    host = _as_text(parsed.hostname)
    if not host:
        return ""
    port = parsed.port or 7001
    username = _as_text(settings.get("username"))
    password = str(settings.get("password") or "")
    auth = ""
    if username:
        auth = _encode_rtsp_credential(username)
        if password:
            auth += f":{_encode_rtsp_credential(password)}"
        auth += "@"
    device_path = quote(device_id, safe="{}")
    query = urlencode({"stream": 1 if int(stream_index or 0) == 1 else 0})
    return f"rtsp://{auth}{host}:{port}/{device_path}?{query}"


def _nx_auth_query_token(settings: dict[str, Any], base: str) -> str:
    username = _as_text(settings.get("username"))
    password = str(settings.get("password") or "")
    if not username:
        return ""

    realm = "VMS"
    nonce_url = urljoin(f"{base}/", _nx_nonce_path(settings).lstrip("/"))
    timeout = _nx_timeout(settings)
    verify = _nx_verify_ssl(settings)
    try:
        with requests.get(nonce_url, timeout=timeout, verify=verify, headers={"Accept": "application/json"}) as resp:
            payload = resp.json() if resp.ok else {}
            reply = payload.get("reply") if isinstance(payload, dict) else None
            if isinstance(reply, dict):
                realm = _as_text(reply.get("realm")) or realm
            elif isinstance(payload, dict):
                realm = _as_text(payload.get("realm")) or realm
    except Exception:
        pass

    ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode("utf-8")).hexdigest()
    return base64.b64encode(f"{username}:{ha1}".encode("utf-8")).decode("ascii")


def _resolve_nx_capture_source(text: str) -> str:
    parsed = _nx_proxy_parts(text)
    if not parsed:
        return text
    device_id, params = parsed
    settings = _nx_settings()
    base = _nx_server_base(settings)
    if not base:
        return f"{_internal_api_base_url().rstrip('/')}{text}" if text.startswith("/api/") else text

    fmt = _as_text(params.get("format")) or "mpjpeg"
    auth = _nx_auth_query_token(settings, base)
    upstream_params: dict[str, str] = {}
    if auth:
        upstream_params["auth"] = auth
    for key in ("stream", "resolution", "pos", "endPos", "rotation", "sfd", "rt", "audio_only", "accurate_seek", "duration", "signature", "utc", "download"):
        value = _as_text(params.get(key))
        if value:
            upstream_params[key] = value
    path = _nx_media_path(settings, device_id, fmt).lstrip("/")
    url = urljoin(f"{base}/", path)
    if upstream_params:
        url = f"{url}?{urlencode(upstream_params)}"
    return url


def resolve_capture_source(source: Any) -> str:
    text = _as_text(source)
    if not text:
        return ""
    if _has_http_scheme(text):
        return text
    if text.startswith("/files/"):
        local_path = _resolve_local_files_source(text)
        if local_path:
            return local_path
        return f"{_internal_api_base_url().rstrip('/')}{text}"
    if _looks_like_nx_proxy_source(text):
        return f"{_internal_api_base_url().rstrip('/')}{text}"
    if text.startswith("/api/"):
        return f"{_internal_api_base_url().rstrip('/')}{text}"
    return text


def resolve_local_api_source(source: Any) -> str:
    text = _as_text(source)
    if not text:
        return ""
    if _has_http_scheme(text):
        return text
    if text.startswith("/files/"):
        local_path = _resolve_local_files_source(text)
        if local_path:
            return local_path
        return f"{_internal_api_base_url().rstrip('/')}{text}"
    if text.startswith("/api/"):
        return f"{_internal_api_base_url().rstrip('/')}{text}"
    return text


def _set_query_value(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params[key] = value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def _resolve_analysis_source_raw(camera: Any) -> str:
    source = _as_text(getattr(camera, "source", ""))
    cfg = getattr(camera, "detection_config", None)
    nx_relay_format = "mpegts"
    if not isinstance(cfg, dict):
        if source.startswith("/api/nx/stream/"):
            source = _set_query_value(source, "format", nx_relay_format)
        return resolve_capture_source(source)

    profile = _as_text(cfg.get("analysis_stream_profile")).lower() or "high"
    nx_analysis_source = _as_text(cfg.get("analysis_nx_source")).lower() or "relay"
    nx_relay_format = _as_text(cfg.get("analysis_nx_relay_format")).lower() or "mpegts"
    if nx_relay_format not in {"mpegts", "mp4"}:
        nx_relay_format = "mpegts"

    if source.startswith("/api/nx/stream/"):
        parsed = _nx_proxy_parts(source)
        device_id = parsed[0] if parsed else ""
        params = parsed[1] if parsed else {}
        stream_index = 1 if profile == "low" else (1 if _as_text(params.get("stream")) == "1" else 0)
        if nx_analysis_source == "rtsp" and device_id:
            rtsp_url = _nx_rtsp_source(device_id, stream_index)
            if rtsp_url:
                return rtsp_url
        source = _set_query_value(source, "format", nx_relay_format)

    if profile != "low":
        return resolve_capture_source(source)

    if source.startswith("http://") or source.startswith("https://"):
        low_url = _as_text(cfg.get("analysis_low_source"))
        return resolve_capture_source(low_url or source)

    if source.startswith("/api/nx/stream/"):
        return resolve_capture_source(_set_query_value(source, "stream", "1"))

    low_path = _as_text(cfg.get("analysis_low_stream_path"))
    if not low_path:
        return resolve_capture_source(source)

    built = build_rtsp_source(
        getattr(camera, "ip", ""),
        getattr(camera, "username", ""),
        getattr(camera, "password", ""),
        getattr(camera, "port", ""),
        low_path,
    )
    return resolve_capture_source(built or source)


# ---------------------------------------------------------------------------
# go2rtc restream 共用
# ---------------------------------------------------------------------------
# 2026-08-16 在 104 實測:每支相機同時被拉 3 條 RTSP —— go2rtc 1 條、
# traffic-api 自己 2 條(cv2.VideoCapture 直連相機,完全繞過 go2rtc)。
# frigate 反而是對的,它走 rtsp://127.0.0.1:8554/cam_N。
# 相機端要同時服務 3 個連線,對 111.70.34.184 那種「兩支共用一個 IP、
# 本來就會間歇斷」的機器是額外負擔。讓分析也走 go2rtc → 對外連線 3 降到 1。
#
# 🛑 安全前提:只有在 go2rtc 那條 stream 的 producer URL 跟我們要開的 URL
#    「完全一致」時才切換。URL 一樣 = 同一條流 = 同解析度 = ROI 座標零風險。
#    若分析走 profile1 低解析、而 go2rtc 走 profile2,兩者不一致就維持原樣,
#    絕不會因為換來源讓 ROI 比例跑掉。
#
# 解碼成本不會因此下降(traffic-api 仍要自己解碼跑 YOLO),省的是對外網路
# 連線數與相機端負載。分析率不受影響。
#
# 關掉:環境變數 TRAFFIC_SHARE_GO2RTC=0

_GO2RTC_API = os.getenv("GO2RTC_API", "http://127.0.0.1:1984")
_GO2RTC_RTSP = os.getenv("GO2RTC_RTSP", "rtsp://127.0.0.1:8554").rstrip("/")
_GO2RTC_CACHE_TTL = 30.0

# {producer_url: stream_name},30 秒快取;查不到 go2rtc 就是空 dict → 不切換
_go2rtc_cache: dict[str, Any] = {"ts": 0.0, "map": {}}
# {restream_url: 原始直連 url},給開流失敗時退回直連用
_go2rtc_reverse: dict[str, str] = {}


def _normalize_rtsp_url(url: str) -> str:
    """比對用的正規化:把 path 裡重複的斜線收成一個。

    現場 87 的 DB 存的是 `:554//axis-media/media.amp`(雙斜線),而 go2rtc
    自己記的是 `:554/axis-media/media.amp`(單斜線) —— 同一支相機、同一條流,
    但嚴格字串比對不相符,不正規化的話這個共用機制在 production 會整個空轉。
    只收斜線,不動 host/query,兩條真的不同的流不可能因此被誤判成同一條。
    """
    parts = urlsplit(_as_text(url))
    path = re.sub(r"/{2,}", "/", parts.path)
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _go2rtc_producer_map() -> dict[str, str]:
    """向 go2rtc 問各 stream 的 producer URL。失敗一律回空 dict(= 不切換)。"""
    now = time.time()
    if now - float(_go2rtc_cache["ts"]) < _GO2RTC_CACHE_TTL:
        return _go2rtc_cache["map"]
    mapping: dict[str, str] = {}
    try:
        resp = requests.get(f"{_GO2RTC_API}/api/streams", timeout=2.0)
        if resp.status_code == 200:
            for name, info in (resp.json() or {}).items():
                for producer in (info.get("producers") or []):
                    url = _as_text(producer.get("url"))
                    if url:
                        mapping[_normalize_rtsp_url(url)] = name
    except Exception:
        mapping = {}
    _go2rtc_cache["ts"] = now
    _go2rtc_cache["map"] = mapping
    return mapping


def resolve_shared_source(source: Any) -> str:
    """能共用 go2rtc restream 就回 restream 位址,否則原樣回傳。"""
    text = _as_text(source)
    if not text or not _has_rtsp_scheme(text):
        return text
    if os.getenv("TRAFFIC_SHARE_GO2RTC", "1") == "0":
        return text
    if text.startswith(_GO2RTC_RTSP):     # 已經是 restream,別再包一層
        return text
    name = _go2rtc_producer_map().get(_normalize_rtsp_url(text))
    if not name:
        return text
    shared = f"{_GO2RTC_RTSP}/{name}"
    _go2rtc_reverse[shared] = text
    return shared


def direct_source_for(source: Any) -> str:
    """restream 位址 → 原始直連位址。不是 restream 就原樣回傳。
    給開流連續失敗時退回直連用,確保 go2rtc 掛掉不會拖垮分析。"""
    text = _as_text(source)
    return _go2rtc_reverse.get(text, text)


def resolve_analysis_source(camera: Any) -> str:
    return resolve_shared_source(_resolve_analysis_source_raw(camera))
