"""號誌控制器抄錄器 —— 都市交通控制通訊協定 3.0 版。

來源:號誌控制器 RS-232 → Moxa MiiNePort E1 → TCP。現場實測參數:
    IP 10.42.38.35   TCPDataPort 1001   Mode TCP / Role TCP Server   MaxConnect 1

🛑 只讀不寫。協定的號誌控制器是主動週期上傳(5F+03 時相資料主動回報),
   不需要輪詢,所以絕不對它送出任何位元組 —— 這是運作中的號誌通道。

🛑 MaxConnect=1:交控中心一旦接上,我們會被拒絕或踢掉。因此重連要退避、
   要能長期無資料而不噴錯,讓中心優先。

碼框(協定 2-6):
    DLE STX SEQ ADDR LEN INFO DLE ETX CKS
    AA  BB  1B  2B   2B  N    AA  CC  1B
    LEN = 10 + N        CKS = XOR(DLE..ETX)
    INFO 內的 0xAA 會被重複成 0xAA 0xAA(byte stuffing),解析前要還原(協定 2-8)
"""
from __future__ import annotations

import os
import socket
import threading
import time
from collections import Counter, deque
from typing import Optional

from fastapi import APIRouter, Depends

from api.routes.auth import get_current_user
from api.utils.shutdown import shutdown_event

router = APIRouter(prefix="/api/signal", tags=["signal"])

SIGNAL_HOST = os.getenv("SIGNAL_TC3_HOST", "10.42.38.35")
SIGNAL_PORT = int(os.getenv("SIGNAL_TC3_PORT", "1001") or 1001)
SIGNAL_ENABLED = os.getenv("SIGNAL_TC3_ENABLED", "1") != "0"

# 燈態方向(協定 P5-22 SignalMap bit map)
DIRECTIONS = ["北", "東北", "東", "東南", "南", "西南", "西", "西北"]
# 燈號狀態 bit map(協定 P5-22 SignalStatus)
LIGHT_BITS = [(0, "紅"), (1, "黃"), (2, "圓頭綠"), (3, "左綠"),
              (4, "直綠"), (5, "右綠"), (6, "行綠"), (7, "行紅")]
# StepID 特殊值(協定 P5-26)
STEP_SPECIAL = {
    0x9F: "啟動全紅3秒", 0xAF: "故障全紅", 0xBF: "固定時制閃光",
    0xCF: "綠綠衝突閃光", 0xDF: "現場操作閃光", 0xEF: "電源異常閃光",
    0xFF: "時制異常閃光",
}
DEVICE_NAMES = {0x0F: "設備共用", 0x5F: "號誌控制器",
                0x6F: "車輛偵測器", 0x8F: "資訊可變標誌"}

_state: dict = {
    "enabled": SIGNAL_ENABLED,
    "host": SIGNAL_HOST,
    "port": SIGNAL_PORT,
    "connected": False,
    "last_frame_at": 0.0,
    "last_error": "",
    "frames_total": 0,
    "cks_bad": 0,
    "reconnects": 0,
    "latest": None,          # 最近一筆解出來的燈態
}
_frames: deque = deque(maxlen=300)      # 最近訊框(raw + 解碼)
_coverage: Counter = Counter()          # 每個 device+cmd 看過幾次
_lock = threading.Lock()


def _unstuff(info: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(info):
        out.append(info[i])
        if info[i] == 0xAA and i + 1 < len(info) and info[i + 1] == 0xAA:
            i += 2
        else:
            i += 1
    return bytes(out)


def _light_text(b: int) -> str:
    on = [name for bit, name in LIGHT_BITS if b & (1 << bit)]
    if "行綠" in on and "行紅" in on:      # 兩者皆 1 = 行人綠閃(協定 P5-22)
        on = [x for x in on if x not in ("行綠", "行紅")] + ["行人綠閃"]
    return "+".join(on) if on else "(全暗)"


def decode_frame(frame: bytes) -> Optional[dict]:
    """解一個完整碼框。格式或 CKS 不對回 None(呼叫端自行記數)。"""
    if len(frame) < 11 or frame[0] != 0xAA or frame[1] != 0xBB:
        return None
    if frame[-3] != 0xAA or frame[-2] != 0xCC:
        return None
    cks = 0
    for b in frame[:-1]:
        cks ^= b
    ok = cks == frame[-1]
    info = _unstuff(frame[7:-3])
    out = {
        "ts": time.time(),
        "seq": frame[2],
        "addr": int.from_bytes(frame[3:5], "big"),
        "len": int.from_bytes(frame[5:7], "big"),
        "cks_ok": ok,
        "raw": frame.hex(" ").upper(),
    }
    if len(info) < 2:
        return out
    dev, cmd = info[0], info[1]
    out["device"] = f"{dev:02X}"
    out["cmd"] = f"{cmd:02X}"
    out["code"] = f"{dev:02X}{cmd:02X}"
    out["device_name"] = DEVICE_NAMES.get(dev, f"未知({dev:02X}H)")

    # 5F+03 時相資料主動回報(協定 P5-26)
    # 5F 03 + PhaseOrder + SignalMap + SignalCount + SubPhaseID + StepID + StepSec + SignalStatus(SignalCount)
    if dev == 0x5F and cmd == 0x03 and len(info) >= 9:
        p = info[2:]
        cnt = p[2]
        smap = p[1]
        used = [DIRECTIONS[b] for b in range(8) if smap & (1 << b)]
        st = p[7:7 + cnt]
        step = p[4]
        out["phase"] = {
            "phase_order": p[0],
            "signal_map": smap,
            "signal_count": cnt,
            "sub_phase_id": p[3],
            "step_id": step,
            "step_desc": STEP_SPECIAL.get(step, f"步階 {step}"),
            "step_sec": int.from_bytes(p[5:7], "big"),
            "lights": [
                {"dir": used[i] if i < len(used) else f"?{i}",
                 "value": st[i], "text": _light_text(st[i])}
                for i in range(len(st))
            ],
        }
    return out


def _recorder_loop() -> None:
    """常駐抄錄。斷線退避重連;被中心搶走(MaxConnect=1)時安靜等待。"""
    backoff = 2.0
    print(f"📶 [signal-tc3] 抄錄器啟動 {SIGNAL_HOST}:{SIGNAL_PORT}(只讀)", flush=True)
    while not shutdown_event.is_set():
        sock = None
        try:
            sock = socket.create_connection((SIGNAL_HOST, SIGNAL_PORT), timeout=8)
            sock.settimeout(3.0)
            with _lock:
                _state["connected"] = True
                _state["last_error"] = ""
            backoff = 2.0
            buf = b""
            while not shutdown_event.is_set():
                try:
                    d = sock.recv(4096)
                except socket.timeout:
                    continue
                if not d:
                    raise ConnectionError("對方關閉連線")
                buf += d
                # 依 AA BB … AA CC 切訊框
                while True:
                    i = buf.find(b"\xaa\xbb")
                    if i < 0:
                        if len(buf) > 65536:
                            buf = b""       # 垃圾資料不要無限長大
                        break
                    j = buf.find(b"\xaa\xcc", i + 2)
                    if j < 0 or len(buf) < j + 3:
                        if i > 0:
                            buf = buf[i:]
                        break
                    frame = buf[i:j + 3]
                    buf = buf[j + 3:]
                    rec = decode_frame(frame)
                    if rec is None:
                        continue
                    with _lock:
                        _state["frames_total"] += 1
                        _state["last_frame_at"] = rec["ts"]
                        if not rec["cks_ok"]:
                            _state["cks_bad"] += 1
                        if rec.get("code"):
                            _coverage[rec["code"]] += 1
                        if rec.get("phase"):
                            _state["latest"] = rec
                        _frames.append(rec)
        except Exception as exc:
            with _lock:
                _state["connected"] = False
                _state["last_error"] = f"{type(exc).__name__}: {exc}"[:160]
                _state["reconnects"] += 1
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        # MaxConnect=1:被中心佔住時要讓它,退避到最多 30 秒
        if shutdown_event.wait(backoff):
            break
        backoff = min(30.0, backoff * 1.6)
    print("📶 [signal-tc3] 抄錄器結束", flush=True)


_thread: Optional[threading.Thread] = None


def start_recorder() -> None:
    global _thread
    if not SIGNAL_ENABLED or (_thread is not None and _thread.is_alive()):
        return
    _thread = threading.Thread(target=_recorder_loop, daemon=True, name="signal-tc3")
    _thread.start()


@router.get("/status", summary="號誌即時燈態與抄錄器狀態")
async def status(_user=Depends(get_current_user)):
    with _lock:
        s = dict(_state)
        latest = s.get("latest")
    now = time.time()
    age = (now - s["last_frame_at"]) if s["last_frame_at"] else None
    return {
        **{k: s[k] for k in ("enabled", "host", "port", "connected",
                             "frames_total", "cks_bad", "reconnects", "last_error")},
        "age_sec": round(age, 2) if age is not None else None,
        # 超過 30 秒沒新訊框就標 stale —— 來源掉了要看得出來,不要顯示假的舊狀態
        "stale": (age is None or age > 30.0),
        "latest": latest,
    }


@router.get("/frames", summary="最近抄錄到的訊框(含解碼)")
async def frames(limit: int = 50, _user=Depends(get_current_user)):
    n = max(1, min(300, int(limit or 50)))
    with _lock:
        items = list(_frames)[-n:]
    return {"count": len(items), "frames": list(reversed(items))}


@router.get("/coverage", summary="TC3 命令覆蓋矩陣(抄到哪些訊息碼)")
async def coverage(_user=Depends(get_current_user)):
    with _lock:
        cov = dict(_coverage)
        total = _state["frames_total"]
    rows = [{"code": k, "device": k[:2], "cmd": k[2:],
             "device_name": DEVICE_NAMES.get(int(k[:2], 16), ""),
             "count": v} for k, v in sorted(cov.items(), key=lambda x: -x[1])]
    return {"frames_total": total, "distinct_codes": len(rows), "codes": rows}
