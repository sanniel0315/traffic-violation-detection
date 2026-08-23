"""跨行程共用的「串流需求」登記表(on-demand 編碼用)。

annotated_streamer 只在「有人在看」時才編碼該變體(cam_X_annotated/lite/lofi),
沒人看就停,省 CPU。需求來自兩種來源:
- token(持久):go2rtc WS proxy 連上 add()、斷線 remove()。(proxy 目前非主要路徑)
- 心跳 touch(TTL):瀏覽器「直連 go2rtc」時,前端每幾秒 POST /api/go2rtc/want →
  touch() 更新 mtime;wanted() 看 TTL 內有沒有被 touch。直連繞過 proxy,故需要它。

🛑 用 /dev/shm 檔案而非 module 變數:偵測/編碼可能跟 web 不同行程。
🛑 不查 go2rtc consumer:它對「還沒有 producer」的串流不列 consumer(雞生蛋)。
"""
import os
import time
import uuid

_BASE = os.getenv("STREAM_DEMAND_DIR", "/dev/shm/tvd_stream_want")
_TTL = float(os.getenv("STREAM_DEMAND_TTL", "25") or 25)


def _safe(src: str) -> str:
    return "".join(c for c in (src or "") if c.isalnum() or c in "_-")


def _dir(src: str) -> str:
    return os.path.join(_BASE, _safe(src))


def clear_all() -> None:
    try:
        import shutil
        shutil.rmtree(_BASE, ignore_errors=True)
    except Exception:
        pass


def add(src: str):
    """持久登記(WS proxy 用),回傳 token。"""
    if not _safe(src):
        return None
    try:
        d = _dir(src)
        os.makedirs(d, exist_ok=True)
        token = "tok_" + uuid.uuid4().hex
        open(os.path.join(d, token), "w").close()
        return token
    except Exception:
        return None


def remove(src: str, token) -> None:
    if not _safe(src) or not token:
        return
    try:
        os.unlink(os.path.join(_dir(src), token))
    except Exception:
        pass


def touch(src: str) -> None:
    """心跳登記(直連 go2rtc 的前端每幾秒呼叫一次)。"""
    if not _safe(src):
        return
    try:
        d = _dir(src)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "hb")
        open(p, "a").close()
        os.utime(p, None)   # 更新 mtime = 現在
    except Exception:
        pass


def wanted(src: str) -> bool:
    """有人在看嗎:有持久 token,或 TTL 內被心跳 touch 過。"""
    d = _dir(src)
    try:
        now = time.time()
        with os.scandir(d) as it:
            for e in it:
                if e.name.startswith("tok_"):
                    return True
                if e.name == "hb":
                    try:
                        if now - e.stat().st_mtime < _TTL:
                            return True
                    except Exception:
                        pass
    except Exception:
        return False
    return False
