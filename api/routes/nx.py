#!/usr/bin/env python3
"""Nx proxy routes."""
from __future__ import annotations

import base64
import hashlib
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.routes.system import load_nx_settings

router = APIRouter(prefix="/api/nx", tags=["nx"])
_NX_NONCE_CANDIDATES = ("/api/getNonce",)
_NX_LOGIN_CANDIDATES = ("/rest/v1/login/sessions", "/rest/v2/login/sessions", "/rest/v4/login/sessions", "/web/rest/v1/login/sessions")
_NX_BEARER_CACHE: Dict[str, Dict[str, Any]] = {}
_NX_BEARER_CACHE_LOCK = threading.Lock()
_NX_BEARER_TTL_SEC = 300.0


def _media_type_for_format(fmt: str) -> str:
    mapping = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "mpjpeg": "multipart/x-mixed-replace",
        "mpegts": "video/mp2t",
        "mkv": "video/x-matroska",
    }
    return mapping.get(str(fmt or "").lower(), "application/octet-stream")


def _nx_settings() -> Dict[str, Any]:
    return load_nx_settings()


def _nx_proxy_base(settings: Optional[Dict[str, Any]] = None) -> str:
    cfg = settings or _nx_settings()
    return str(cfg.get("proxy_base_url", "") or "").strip().rstrip("/")


def _nx_server_base(settings: Optional[Dict[str, Any]] = None) -> str:
    cfg = settings or _nx_settings()
    return str(cfg.get("server_base_url", "") or "").strip().rstrip("/")


def _nx_timeout(settings: Optional[Dict[str, Any]] = None) -> float:
    cfg = settings or _nx_settings()
    try:
        return max(3.0, float(cfg.get("timeout_sec", 12)))
    except Exception:
        return 12.0


def _nx_verify_ssl(settings: Optional[Dict[str, Any]] = None) -> bool:
    cfg = settings or _nx_settings()
    return bool(cfg.get("verify_ssl", True))


def _nx_auth(settings: Optional[Dict[str, Any]] = None) -> Optional[tuple[str, str]]:
    cfg = settings or _nx_settings()
    user = str(cfg.get("username", "") or "").strip()
    password = str(cfg.get("password", "") or "")
    if user:
        return (user, password)
    return None


def _nx_devices_path(settings: Optional[Dict[str, Any]] = None) -> str:
    cfg = settings or _nx_settings()
    return str(cfg.get("devices_path", "/rest/v2/devices") or "/rest/v2/devices").strip()


def _nx_media_path(device_id: str, fmt: str, settings: Optional[Dict[str, Any]] = None) -> str:
    cfg = settings or _nx_settings()
    template = str(cfg.get("media_path_template", "/media/{device_id}.{format}") or "/media/{device_id}.{format}")
    return template.format(device_id=device_id, format=fmt)


def _json_or_text(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def _normalize_devices(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("devices")
        if items is None:
            items = payload.get("items")
        if items is None:
            items = payload.get("results")
        if items is None and isinstance(payload.get("data"), list):
            items = payload.get("data")
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    out: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        device_id = (
            item.get("id")
            or item.get("deviceId")
            or item.get("device_id")
            or item.get("uuid")
            or item.get("resourceId")
        )
        if not device_id:
            continue
        name = (
            item.get("name")
            or item.get("displayName")
            or item.get("label")
            or f"Device {device_id}"
        )
        status = (
            item.get("status")
            or item.get("state")
            or item.get("connectionState")
            or item.get("online")
            or "Unknown"
        )
        if isinstance(status, bool):
            status = "Online" if status else "Offline"
        out.append(
            {
                "id": str(device_id),
                "name": str(name),
                "status": str(status),
            }
        )
    return out


def _upstream_stream(resp: requests.Response) -> Iterable[bytes]:
    try:
        try:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if chunk:
                    yield chunk
        except requests.RequestException:
            return
    finally:
        resp.close()


def _is_ssl_cert_error(error: BaseException) -> bool:
    text = str(error or "")
    return isinstance(error, requests.exceptions.SSLError) or "CERTIFICATE_VERIFY_FAILED" in text or "self-signed certificate" in text


def _request_with_ssl_fallback(
    session: requests.Session,
    method: str,
    url: str,
    *,
    verify: bool,
    **kwargs: Any,
) -> requests.Response:
    try:
        return session.request(method, url, verify=verify, **kwargs)
    except requests.RequestException as e:
        if verify and _is_ssl_cert_error(e):
            return session.request(method, url, verify=False, **kwargs)
        raise


def _nx_auth_query_token(
    session: requests.Session,
    base: str,
    username: str,
    password: str,
    timeout: float,
    verify: bool,
) -> str:
    last_error: requests.RequestException | None = None
    for path in _NX_NONCE_CANDIDATES:
        url = urljoin(f"{base}/", path.lstrip("/"))
        try:
            resp = _request_with_ssl_fallback(session, "GET", url, timeout=timeout, verify=verify)
        except requests.RequestException as e:
            last_error = e
            continue
        if not resp.ok:
            text = resp.text[:200]
            resp.close()
            raise HTTPException(status_code=resp.status_code, detail=f"NX getNonce failed: {text}")
        data = _json_or_text(resp)
        resp.close()
        reply = data.get("reply") if isinstance(data, dict) else None
        realm = "VMS"
        nonce = ""
        if isinstance(reply, dict):
            realm = str(reply.get("realm") or realm).strip() or "VMS"
            nonce = str(reply.get("nonce") or "").strip()
        elif isinstance(data, dict):
            realm = str(data.get("realm") or realm).strip() or "VMS"
            nonce = str(data.get("nonce") or "").strip()
        if not nonce:
            raise HTTPException(status_code=502, detail="NX getNonce did not return a nonce")
        login = username.lower()
        ha1 = hashlib.md5(f"{login}:{realm}:{password}".encode("utf-8")).hexdigest()
        ha2 = hashlib.md5("GET:".encode("utf-8")).hexdigest()
        digest = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode("utf-8")).hexdigest()
        token = base64.b64encode(f"{login}:{nonce}:{digest}".encode("utf-8")).decode("ascii")
        if token:
            return token
    if last_error:
        raise HTTPException(status_code=502, detail=f"NX getNonce request error: {last_error}") from last_error
    raise HTTPException(status_code=502, detail="NX getNonce endpoint not available")


def _nx_login_headers(
    session: requests.Session,
    base: str,
    username: str,
    password: str,
    timeout: float,
    verify: bool,
) -> Dict[str, str]:
    cache_key = f"{base}|{username}"
    now = time.time()
    with _NX_BEARER_CACHE_LOCK:
        cached = _NX_BEARER_CACHE.get(cache_key)
        if cached and str(cached.get("token") or "").strip() and (now - float(cached.get("ts") or 0.0) < _NX_BEARER_TTL_SEC):
            return {"Authorization": f"Bearer {cached['token']}"}

    payload = {"username": username, "password": password, "setCookie": False}
    last_error: requests.RequestException | None = None
    for path in _NX_LOGIN_CANDIDATES:
        url = urljoin(f"{base}/", path.lstrip("/"))
        try:
            resp = _request_with_ssl_fallback(
                session,
                "POST",
                url,
                timeout=timeout,
                verify=verify,
                json=payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
        except requests.RequestException as e:
            last_error = e
            continue
        if not resp.ok:
            text = resp.text[:200]
            resp.close()
            if resp.status_code in {401, 403, 404, 405}:
                continue
            raise HTTPException(status_code=resp.status_code, detail=f"NX login failed: {text}")
        data = _json_or_text(resp)
        resp.close()
        if not isinstance(data, dict):
            raise HTTPException(status_code=502, detail="NX login returned an unexpected payload")
        token = str(data.get("token") or data.get("id") or "").strip()
        if not token:
            raise HTTPException(status_code=502, detail="NX login did not return a bearer token")
        with _NX_BEARER_CACHE_LOCK:
            _NX_BEARER_CACHE[cache_key] = {"token": token, "ts": now}
        return {"Authorization": f"Bearer {token}"}
    if last_error:
        raise HTTPException(status_code=502, detail=f"NX login request error: {last_error}") from last_error
    raise HTTPException(status_code=502, detail="NX login endpoint not available")


def _nx_request_headers(
    session: requests.Session,
    cfg: Dict[str, Any],
    base: str,
    timeout: float,
    verify: bool,
    accept: str,
) -> Dict[str, str]:
    headers = {"Accept": accept}
    username = str(cfg.get("username", "") or "").strip()
    password = str(cfg.get("password", "") or "")
    if username:
        headers.update(_nx_login_headers(session, base, username, password, timeout, verify))
    return headers


def _nx_request_auth(
    session: requests.Session,
    cfg: Dict[str, Any],
    base: str,
    timeout: float,
    verify: bool,
    accept: str,
) -> Dict[str, Any]:
    headers = {"Accept": accept}
    params: Dict[str, str] = {}
    username = str(cfg.get("username", "") or "").strip()
    password = str(cfg.get("password", "") or "")
    if not username:
        return {"headers": headers, "params": params}
    try:
        headers.update(_nx_login_headers(session, base, username, password, timeout, verify))
        return {"headers": headers, "params": params}
    except HTTPException:
        params["auth"] = _nx_auth_query_token(session, base, username, password, timeout, verify)
        return {"headers": headers, "params": params}


def _prime_nx_session(
    session: requests.Session,
    base: str,
    auth: Optional[tuple[str, str]],
    timeout: float,
    verify: bool,
) -> None:
    for path in _NX_NONCE_CANDIDATES:
        try:
            resp = _request_with_ssl_fallback(
                session,
                "GET",
                urljoin(f"{base}/", path.lstrip("/")),
                auth=auth,
                timeout=timeout,
                verify=verify,
            )
            resp.close()
            if resp.ok:
                return
        except Exception:
            continue


def _call_direct_nx_devices(settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfg = settings or _nx_settings()
    base = _nx_server_base(cfg)
    if not base:
        raise HTTPException(status_code=503, detail="NX_SERVER_BASE_URL not configured")

    session = requests.Session()
    timeout = _nx_timeout(cfg)
    verify = _nx_verify_ssl(cfg)

    try:
        auth_request = _nx_request_auth(session, cfg, base, timeout, verify, "application/json")
        resp = _request_with_ssl_fallback(
            session,
            "GET",
            urljoin(f"{base}/", _nx_devices_path(cfg).lstrip("/")),
            timeout=timeout,
            verify=verify,
            headers=auth_request["headers"],
            params=auth_request["params"],
        )
        if not resp.ok:
            raise HTTPException(status_code=resp.status_code, detail=f"NX devices request failed: {resp.text[:200]}")
        return _normalize_devices(_json_or_text(resp))
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"NX devices request error: {e}") from e
    finally:
        session.close()


@router.get("/devices")
async def nx_devices():
    settings = _nx_settings()
    proxy_base = _nx_proxy_base(settings)
    if proxy_base:
        try:
            resp = requests.get(
                f"{proxy_base}/devices",
                timeout=_nx_timeout(settings),
                verify=_nx_verify_ssl(settings),
                headers={"Accept": "application/json"},
            )
            if not resp.ok:
                raise HTTPException(status_code=resp.status_code, detail=f"NX proxy devices failed: {resp.text[:200]}")
            payload = _json_or_text(resp)
            devices = payload.get("devices") if isinstance(payload, dict) else None
            return {"ok": True, "devices": devices if isinstance(devices, list) else _normalize_devices(payload)}
        except HTTPException:
            raise
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"NX proxy request error: {e}") from e

    devices = _call_direct_nx_devices(settings)
    return {"ok": True, "devices": devices}


@router.get("/stream/{device_id}")
async def nx_stream(
    device_id: str,
    format: str = Query("mp4"),
    stream: Optional[int] = Query(None),
    resolution: Optional[str] = Query(None),
):
    fmt = str(format or "mp4").strip().lower()
    settings = _nx_settings()
    proxy_base = _nx_proxy_base(settings)
    timeout = _nx_timeout(settings)
    verify = _nx_verify_ssl(settings)
    upstream_params: Dict[str, Any] = {}
    if stream in (0, 1):
        upstream_params["stream"] = int(stream)
    if str(resolution or "").strip():
        upstream_params["resolution"] = str(resolution).strip()

    if proxy_base:
        try:
            resp = requests.get(
                f"{proxy_base}/stream/{device_id}",
                params={"format": fmt, **upstream_params},
                stream=True,
                timeout=(timeout, timeout),
                verify=verify,
            )
            if not resp.ok:
                detail = resp.text[:200] if not resp.headers.get("content-type", "").startswith("video/") else "NX proxy stream failed"
                raise HTTPException(status_code=resp.status_code, detail=detail)
            return StreamingResponse(
                _upstream_stream(resp),
                media_type=resp.headers.get("content-type") or _media_type_for_format(fmt),
            )
        except HTTPException:
            raise
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"NX proxy stream error: {e}") from e

    base = _nx_server_base(settings)
    if not base:
        raise HTTPException(status_code=503, detail="NX proxy is not configured")

    session = requests.Session()
    try:
        auth_request = _nx_request_auth(session, settings, base, timeout, verify, "*/*")
        resp = _request_with_ssl_fallback(
            session,
            "GET",
            urljoin(f"{base}/", _nx_media_path(device_id, fmt, settings).lstrip("/")),
            timeout=(timeout, timeout),
            verify=verify,
            stream=True,
            headers=auth_request["headers"],
            params={**auth_request["params"], **upstream_params},
        )
        if not resp.ok:
            text = resp.text[:200] if "text" in (resp.headers.get("content-type") or "") else "NX stream failed"
            session.close()
            raise HTTPException(status_code=resp.status_code, detail=text)

        def _stream() -> Iterable[bytes]:
            try:
                try:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            yield chunk
                except requests.RequestException:
                    return
            finally:
                resp.close()
                session.close()

        return StreamingResponse(_stream(), media_type=resp.headers.get("content-type") or _media_type_for_format(fmt))
    except HTTPException:
        raise
    except requests.RequestException as e:
        session.close()
        raise HTTPException(status_code=502, detail=f"NX direct stream error: {e}") from e
