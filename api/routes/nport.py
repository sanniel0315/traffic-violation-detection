"""Moxa NPort 管理面板 —— 探測、登入、網頁介面代理。

為什麼需要代理:
NPort 掛在 192.168.1.x(87 的 enP5p4s0 那段),使用者遠端經 Tailscale 連進來時
瀏覽器到不了那個網段。分析器同時在兩段上,所以由後端把 NPort 的網頁轉出來。

路徑保留的理由跟 frigate VOD / go2rtc 代理一樣:老式設備網頁大量使用相對路徑,
代理必須維持同樣的路徑結構,相對路徑才解析得回這支端點。絕對路徑(NPort 會
307 轉到 /moxa/Login.htm)另外做 HTML 改寫。

🛑 這支只轉送,不儲存任何帳號密碼。憑證由瀏覽器每次帶上,後端不寫檔不入庫。
"""
from __future__ import annotations

import re
import socket
import time
from typing import Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from api.routes.auth import get_current_user

router = APIRouter(prefix="/api/nport", tags=["nport"])

# Moxa 的 OUI(MAC 前三碼),用來確認真的是 Moxa 設備而不是別的機器
_MOXA_OUI = {"00:90:e8", "00:90:E8"}
# NPort 常見的資料埠/命令埠:第 N 個序列埠 = 4000+N / 4900+N
_DATA_PORT_BASE = 4000
_CMD_PORT_BASE = 4900
_PROBE_TIMEOUT = 3.0
_HTTP_TIMEOUT = 8.0


class ProbeReq(BaseModel):
    host: str
    port: int = 4001          # 要探測的資料埠
    web_port: int = 80        # 網頁管理埠


def _tcp_open(host: str, port: int, timeout: float = _PROBE_TIMEOUT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _read_arp_mac(host: str) -> str:
    """從系統 ARP 表讀 MAC。讀不到不算錯 —— 跨網段本來就沒有 ARP。"""
    try:
        with open("/proc/net/arp", "r", encoding="utf-8") as fh:
            for line in fh.readlines()[1:]:
                cols = line.split()
                if len(cols) >= 4 and cols[0] == host and cols[3] != "00:00:00:00:00:00":
                    return cols[3]
    except OSError:
        pass
    return ""


def _probe_sync(req: ProbeReq) -> dict:
    t0 = time.time()
    out: dict = {
        "host": req.host,
        "data_port": req.port,
        "web_port": req.web_port,
        "reachable": False,
        "is_moxa": False,
        "mac": "",
        "data_port_open": False,
        "cmd_port_open": False,
        "web": {"status": None, "title": "", "redirect": ""},
        "serial_ports": [],
        "hint": "",
    }

    out["mac"] = _read_arp_mac(req.host)
    if out["mac"]:
        out["is_moxa"] = out["mac"][:8].lower() in {o.lower() for o in _MOXA_OUI}

    out["data_port_open"] = _tcp_open(req.host, req.port)
    # 命令埠與資料埠成對:4001 <-> 4900+(port-4000)
    idx = max(1, req.port - _DATA_PORT_BASE)
    out["cmd_port_open"] = _tcp_open(req.host, _CMD_PORT_BASE + idx)

    # 網頁管理介面
    try:
        r = requests.get(f"http://{req.host}:{req.web_port}/",
                         timeout=_HTTP_TIMEOUT, allow_redirects=True)
        out["web"]["status"] = r.status_code
        out["web"]["redirect"] = r.url
        m = re.search(r"<title>(.*?)</title>", r.text or "", re.I | re.S)
        if m:
            out["web"]["title"] = m.group(1).strip()[:80]
        if "moxa" in (r.text or "").lower() or "moxa" in (r.url or "").lower():
            out["is_moxa"] = True
    except Exception as exc:
        out["web"]["status"] = 0
        out["web"]["title"] = f"{type(exc).__name__}"

    # 掃相鄰資料埠,推斷是幾埠機種
    for n in range(1, 9):
        p = _DATA_PORT_BASE + n
        if _tcp_open(req.host, p, timeout=1.0):
            out["serial_ports"].append(p)

    out["reachable"] = bool(out["data_port_open"] or out["web"]["status"])
    out["elapsed_ms"] = int((time.time() - t0) * 1000)

    if not out["reachable"]:
        out["hint"] = ("設備沒有回應。ARP 學不到 MAC 通常代表斷電或網線脫落,"
                       "不是拒絕連線或連線數已滿。")
    elif out["data_port_open"] and not out["serial_ports"]:
        out["hint"] = "資料埠開著但相鄰埠都關 —— 單埠機種,或只啟用第 1 埠。"
    return out


@router.post("/probe", summary="探測指定 IP/Port 是否為 Moxa NPort")
async def probe(req: ProbeReq, _user=Depends(get_current_user)):
    return await run_in_threadpool(_probe_sync, req)


class TapReq(BaseModel):
    host: str
    port: int = 4001
    seconds: float = 10.0


@router.post("/tap", summary="唯讀擷取資料埠內容(不送出任何位元組)")
async def tap(req: TapReq, _user=Depends(get_current_user)):
    """連上資料埠純接收,用來確認號誌控制器有沒有在上傳。

    🛑 只讀不寫。都市交通控制通訊協定的號誌控制器是主動週期上傳,
       不需要輪詢,所以不必也不應該對它發送任何東西。
    """
    secs = max(1.0, min(30.0, float(req.seconds or 10)))

    def _run() -> dict:
        try:
            s = socket.create_connection((req.host, req.port), timeout=_PROBE_TIMEOUT)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}", "frames": []}
        s.settimeout(2.0)
        buf = b""
        frames: list[dict] = []
        t0 = time.time()
        try:
            while time.time() - t0 < secs:
                try:
                    d = s.recv(4096)
                except socket.timeout:
                    continue
                if not d:
                    break
                buf += d
                while True:
                    i = buf.find(b"\xaa\xbb")
                    if i < 0:
                        break
                    j = buf.find(b"\xaa\xcc", i + 2)
                    if j < 0 or len(buf) < j + 3:
                        break
                    fr = buf[i:j + 3]
                    buf = buf[j + 3:]
                    cks = 0
                    for b in fr[:-1]:
                        cks ^= b
                    frames.append({
                        "t": round(time.time() - t0, 2),
                        "hex": fr.hex(" ").upper(),
                        "cks_ok": cks == fr[-1],
                    })
        finally:
            try:
                s.close()
            except Exception:
                pass
        return {"ok": True, "seconds": secs, "count": len(frames), "frames": frames[:60]}

    return await run_in_threadpool(_run)


# ---------------------------------------------------------------------------
# 網頁介面代理
# ---------------------------------------------------------------------------
_PROXY_PREFIX = "/api/nport/web"
# 只轉送設備需要的 header,不要把我們自己的 cookie/auth 洩到設備上
_FORWARD_REQ_HEADERS = ("authorization", "content-type", "cookie", "referer", "user-agent")
_FORWARD_RES_HEADERS = ("content-type", "set-cookie", "location", "www-authenticate")


def _rewrite_html(body: bytes, host: str, web_port: int) -> bytes:
    """把設備網頁裡的絕對路徑改寫成走我們的代理。

    NPort 的網頁會用 /moxa/... 這種絕對路徑(首頁就是 307 轉到 /moxa/Login.htm),
    不改寫的話瀏覽器會拿去打「我們自己」的網站根目錄而 404。
    相對路徑不用動 —— 代理維持了同樣的路徑結構。
    """
    base = f"{_PROXY_PREFIX}/{host}/{web_port}"
    try:
        text = body.decode("utf-8", errors="ignore")
    except Exception:
        return body
    # 🛑 先處理「絕對 URL 帶設備自己的 host」——MiiNePort 的 307 就是回
    #    http://10.42.38.35/moxa/home.htm 這種。只改 / 開頭的不夠,遠端瀏覽器
    #    連不到設備 IP,會直接失敗(在 87 本機用 curl 測不出來,它連得到)。
    for scheme in ("http", "https"):
        text = text.replace(f"{scheme}://{host}:{web_port}/", f"{base}/")
        text = text.replace(f"{scheme}://{host}/", f"{base}/")
    # href/src/action="/xxx" → "<prefix>/xxx";已經是 http(s):// 或 // 的不動
    text = re.sub(r'(?i)\b(href|src|action)\s*=\s*(["\'])/(?!/)',
                  lambda m: f'{m.group(1)}={m.group(2)}{base}/', text)
    # JS 裡常見的 location='/xxx' / window.open('/xxx')
    text = re.sub(r'''(?i)(location(?:\.href)?\s*=\s*|window\.open\(\s*)(["\'])/(?!/)''',
                  lambda m: f'{m.group(1)}{m.group(2)}{base}/', text)
    return text.encode("utf-8")


@router.api_route("/web/{host}/{web_port}/{path:path}",
                  methods=["GET", "POST", "HEAD"],
                  summary="代理 NPort 網頁管理介面(遠端瀏覽器到不了那個網段)")
async def proxy_web(host: str, web_port: int, path: str, request: Request,
                    _user=Depends(get_current_user)):
    if ".." in (path or ""):
        raise HTTPException(status_code=400, detail="不合法的路徑")
    if not re.fullmatch(r"[0-9A-Fa-f:.]+", host or ""):
        raise HTTPException(status_code=400, detail="不合法的 host")

    query = str(request.url.query or "")
    url = f"http://{host}:{web_port}/{path}" + (f"?{query}" if query else "")
    headers = {k: v for k, v in request.headers.items()
               if k.lower() in _FORWARD_REQ_HEADERS}
    body = await request.body() if request.method == "POST" else None

    def _fetch():
        return requests.request(
            request.method, url, headers=headers, data=body,
            timeout=_HTTP_TIMEOUT, allow_redirects=False, stream=True,
        )

    try:
        up = await run_in_threadpool(_fetch)
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"NPort {host}:{web_port} 無回應: {type(exc).__name__}")

    out_headers = {"Cache-Control": "no-store"}
    for k in _FORWARD_RES_HEADERS:
        v = up.headers.get(k)
        if not v:
            continue
        if k == "location":
            # 轉址一律導回代理。設備可能回 / 開頭,也可能回帶自己 host 的絕對 URL
            # (MiiNePort 實測回 http://10.42.38.35/moxa/home.htm),兩種都要處理,
            # 否則遠端瀏覽器會被導去它連不到的設備 IP。
            _b = f"{_PROXY_PREFIX}/{host}/{web_port}"
            for _sch in ("http", "https"):
                if v.startswith(f"{_sch}://{host}:{web_port}/"):
                    v = _b + v[len(f"{_sch}://{host}:{web_port}"):]
                    break
                if v.startswith(f"{_sch}://{host}/"):
                    v = _b + v[len(f"{_sch}://{host}"):]
                    break
            else:
                if v.startswith("/"):
                    v = _b + v
        out_headers[k] = v

    ctype = (up.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        raw = await run_in_threadpool(lambda: up.content)
        up.close()
        return Response(content=_rewrite_html(raw, host, web_port),
                        status_code=up.status_code, headers=out_headers,
                        media_type=up.headers.get("content-type"))

    def body_iter():
        try:
            for chunk in up.iter_content(chunk_size=65536):
                if chunk:
                    yield chunk
        finally:
            up.close()

    return StreamingResponse(body_iter(), status_code=up.status_code,
                             headers=out_headers,
                             media_type=up.headers.get("content-type"))

# ---------------------------------------------------------------------------
# Moxa UDP 廣播搜尋
# ---------------------------------------------------------------------------
# 🛑 這是唯一能找到「IP 設錯」設備的方法,實務上非常重要。
#    2026-08-17 現場實例:NPort 的 IP 被改成 10.42.38.35(遮罩 /20、閘道
#    10.42.32.254),但網線還留在 192.168.1.x。它收到 ping 之後要回覆,會查
#    自己的路由表 → 送去 10.42.32.254 → 那個閘道不在這個廣播域 → 回不來。
#    所以 ping/TCP 全部不通,看起來像設備死了;但廣播搜尋走 L2 不需要路由,
#    照樣問得到,而且回應裡帶著它的 MAC 與它自己認定的 IP。
#
# 回應格式(依實測 24 bytes 回應歸納,非官方文件):
#    offset 0      0x81      對 0x01 查詢的回應
#    offset 2-3    長度
#    offset 14-19  MAC (6 bytes)
#    offset 20-23  IP  (4 bytes)
_MOXA_DISCOVER_PORT = 4800
_MOXA_DISCOVER_PROBE = bytes([0x01, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00])


class DiscoverReq(BaseModel):
    broadcast: str = "255.255.255.255"
    seconds: float = 3.0


@router.post("/discover", summary="Moxa UDP 廣播搜尋(IP 設錯也找得到)")
async def discover(req: DiscoverReq, _user=Depends(get_current_user)):
    secs = max(1.0, min(10.0, float(req.seconds or 3)))
    targets = [req.broadcast] if req.broadcast else ["255.255.255.255"]

    def _run() -> dict:
        found: dict[str, dict] = {}
        errors = []
        for bcast in targets:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sk.settimeout(1.0)
            try:
                sk.sendto(_MOXA_DISCOVER_PROBE, (bcast, _MOXA_DISCOVER_PORT))
                t0 = time.time()
                while time.time() - t0 < secs:
                    try:
                        data, addr = sk.recvfrom(2048)
                    except socket.timeout:
                        continue
                    item = {
                        "from": addr[0],
                        "bytes": len(data),
                        "raw": data[:32].hex(" ").upper(),
                        "mac": "",
                        "ip": "",
                    }
                    if len(data) >= 24:
                        mac = data[14:20]
                        item["mac"] = ":".join(f"{b:02x}" for b in mac)
                        item["ip"] = ".".join(str(b) for b in data[20:24])
                        item["is_moxa"] = item["mac"][:8].lower() in {
                            o.lower() for o in _MOXA_OUI}
                    key = item["mac"] or item["from"]
                    found[key] = item
            except Exception as exc:
                errors.append(f"{bcast}: {type(exc).__name__}")
            finally:
                try:
                    sk.close()
                except Exception:
                    pass
        items = list(found.values())
        hint = ""
        if items:
            odd = [i for i in items if i.get("ip") and i["ip"] != i["from"]]
            if odd:
                hint = ("有設備回報的 IP 與封包來源位址不同 —— 通常代表它的 IP/遮罩/閘道"
                        "設在別的子網,單播封包回不來。要連上它必須先把 IP 改成本網段可用位址,"
                        "或把網線接到該 IP 所屬的網段。")
        return {"ok": True, "count": len(items), "devices": items,
                "errors": errors, "hint": hint}

    return await run_in_threadpool(_run)
