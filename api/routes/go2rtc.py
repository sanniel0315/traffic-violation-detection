"""go2rtc 反向代理 —— 讓即時影像可以走 H.264，而且對外只需要開 :8000 一個埠。

為什麼要這一層：
2026-08-17 實測，go2rtc 的 :1984 與 frigate 的 :5000 都只綁在 127.0.0.1，
瀏覽器根本連不到 —— live_test.html 的 A/B/C 三種播法全紅。那不是瀏覽器
不支援 H.264（Chrome 對 H.264 + MSE/WebRTC 是普遍支援的），純粹是連不上埠。
現場對外只有 router 轉出來的 :8000，把 go2rtc 收進 :8000 之後單一埠就夠，
router 不用多開埠，也跟現有網頁同源。

為什麼路徑要原樣保留：
HLS 的 master.m3u8 內容是 `index-v1.m3u8`、index 裡是 `seg-N.ts`，全部是
相對路徑，瀏覽器會拿它們去接「目前這支 m3u8 的所在目錄」。代理必須保留
同樣的路徑結構，相對路徑才解析得回這支端點（跟 frigate VOD 代理同一個
理由，見 frigate.py proxy_frigate_vod）。

用法（前端）：
    ws     : /api/go2rtc/api/ws?src=cam_7          ← MSE/WebRTC，延遲最低
    HLS    : /api/go2rtc/api/stream.m3u8?src=cam_7 ← 相容性最好
    fMP4   : /api/go2rtc/api/stream.mp4?src=cam_7
    清單   : /api/go2rtc/api/streams
"""
from __future__ import annotations

import asyncio
import os
from urllib.parse import urlencode

import requests
import websockets
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.utils.shutdown import shutdown_event

router = APIRouter(prefix="/api/go2rtc", tags=["go2rtc"])

GO2RTC_HTTP = os.getenv("GO2RTC_API", "http://127.0.0.1:1984").rstrip("/")
GO2RTC_WS = GO2RTC_HTTP.replace("https://", "wss://").replace("http://", "ws://")

# 串流是無限長度的（stream.mp4 / mpegts），一次搬 64KB
_CHUNK = 65536
# 逐段轉送不要設短 timeout：連線本身是長連線，只有「建立」需要 timeout
_CONNECT_TIMEOUT = 10


@router.websocket("/api/ws")
async def proxy_ws(ws: WebSocket):
    """把瀏覽器的 WebSocket 橋接到 go2rtc。

    這是 MSE / WebRTC 兩種播法共用的訊令通道，也是延遲最低的一條路
    （單一 TCP 連線，穿得過 NAT，不必額外開 UDP 埠）。
    """
    await ws.accept()
    query = urlencode(dict(ws.query_params))
    upstream_url = f"{GO2RTC_WS}/api/ws" + (f"?{query}" if query else "")

    try:
        upstream = await websockets.connect(
            upstream_url, max_size=None, ping_interval=None, open_timeout=_CONNECT_TIMEOUT
        )
    except Exception as exc:
        await ws.close(code=1011, reason=f"go2rtc 連線失敗: {exc}"[:120])
        return

    async def browser_to_go2rtc():
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            if msg.get("text") is not None:
                await upstream.send(msg["text"])
            elif msg.get("bytes") is not None:
                await upstream.send(msg["bytes"])

    async def go2rtc_to_browser():
        async for msg in upstream:
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_text(msg)

    tasks = [asyncio.create_task(browser_to_go2rtc()), asyncio.create_task(go2rtc_to_browser())]
    try:
        # 任一邊斷掉就整組收掉，不要留半條在那邊漏
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
        await upstream.close()
        try:
            await ws.close()
        except Exception:
            pass


@router.api_route("/{go2rtc_path:path}", methods=["GET", "POST", "HEAD"])
async def proxy_http(go2rtc_path: str, request: Request):
    """go2rtc 的 HTTP 端點原樣轉送（清單、快照、HLS、fMP4）。"""
    path = str(go2rtc_path or "").strip().lstrip("/")
    if ".." in path:
        raise HTTPException(status_code=400, detail="不合法的路徑")

    query = urlencode(dict(request.query_params))
    url = f"{GO2RTC_HTTP}/{path}" + (f"?{query}" if query else "")
    is_head = request.method == "HEAD"

    headers = {}
    if request.headers.get("range"):
        headers["Range"] = request.headers["range"]
    body_bytes = await request.body() if request.method == "POST" else None

    def _connect():
        if is_head:
            return requests.head(url, timeout=_CONNECT_TIMEOUT, headers=headers, allow_redirects=True)
        if request.method == "POST":
            # WebRTC 的 SDP 交換走 POST
            return requests.post(url, timeout=_CONNECT_TIMEOUT, headers=headers,
                                 data=body_bytes, stream=True)
        return requests.get(url, timeout=_CONNECT_TIMEOUT, headers=headers, stream=True)

    try:
        # 建立連線是同步阻塞的,丟到 threadpool,不要卡住 event loop
        upstream = await run_in_threadpool(_connect)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"go2rtc 無回應: {exc}")

    out_headers = {"Cache-Control": "no-store"}
    for key in ("Content-Length", "Content-Range", "ETag", "Accept-Ranges"):
        value = upstream.headers.get(key)
        if value:
            out_headers[key] = value
    content_type = upstream.headers.get("content-type") or "application/octet-stream"

    if is_head:
        upstream.close()
        return Response(status_code=upstream.status_code, headers=out_headers, media_type=content_type)

    def body():
        # 🛑 stream.mp4 是無限長度的串流,不檢查 shutdown_event 的話
        #    關機時這條 generator 會卡到 SIGKILL(見 api/utils/shutdown.py)
        try:
            for chunk in upstream.iter_content(chunk_size=_CHUNK):
                if shutdown_event.is_set():
                    break
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return StreamingResponse(
        body(), status_code=upstream.status_code, headers=out_headers, media_type=content_type
    )

