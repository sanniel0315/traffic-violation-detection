from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

import requests


def _as_text(value: Any) -> str:
    return str(value or "").strip()


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


def resolve_analysis_source(camera: Any) -> str:
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
