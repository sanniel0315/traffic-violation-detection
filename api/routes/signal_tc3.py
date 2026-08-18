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

import csv
import os
import pathlib
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
# 連上之後多久沒看到任何 TC3 訊框就判定「打到錯的機器」並主動斷線重試。
# 🛑 為什麼需要:10.42.38.35 這個 IP 在現場兩個網段各有一台設備(號誌模組
#    00:90:e8:89:11:42 與另一台只開 80 的機器)。ARP 被攪動時封包會打到錯的
#    那台,TCP 連得上卻永遠沒有資料 —— 沒有這道檢查就會傻等到天亮。
#    號誌是每 2 秒主動上傳,45 秒沒有任何訊框已經遠超正常間隔。
SIGNAL_PEER_TIMEOUT = float(os.getenv("SIGNAL_TC3_PEER_TIMEOUT", "45") or 45)
# 已經抄到過訊框、之後卻長時間靜默 → 一樣要斷線重連。
# 🛑 為什麼需要:原本的 peer 檢查只在「從頭到尾沒收到訊框」時才觸發。一旦
#    got_tc3=True,recv 逾時就無條件 continue —— socket 還開著、connected 仍是
#    True,但資料可以停在 45 分鐘前。2026-08-18 在 87 實際踩到:抄到 29 框後
#    靜默,狀態頁顯示「已連線」卻是 45 分鐘前的燈態。
#    號誌是每 2 秒主動上傳,60 秒沒有任何訊框就是不正常。
SIGNAL_STALL_TIMEOUT = float(os.getenv("SIGNAL_TC3_STALL_TIMEOUT", "60") or 60)

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
# 設備碼(協定 3-2/3-3 實查):0F 共用、5F 號誌、6F 車輛偵測器、AF 資訊可變標誌
DEVICE_NAMES = {0x0F: "設備共用", 0x5F: "號誌控制器",
                0x6F: "車輛偵測器", 0xAF: "資訊可變標誌"}

# 協定 3.0 全部 105 條訊息(33 條設備共用 + 72 條號誌控制器),由規範逐條抄錄。
# 用途:覆蓋矩陣要能回答「規範這 105 條,現場實際跑到哪幾條」—— 只列抄到的
# 沒有意義,驗收要看的是分母。CSV 放版控,改規範版本時只要換這個檔。
CATALOG_PATH = os.getenv(
    "SIGNAL_TC3_CATALOG",
    str(pathlib.Path(__file__).resolve().parents[2] / "config" / "tc3" / "command_catalog.csv"),
)
_catalog: Optional[list] = None


def load_catalog() -> list:
    """讀命令目錄(快取)。檔案不在就回空表,覆蓋矩陣退回「只列抄到的」。"""
    global _catalog
    if _catalog is not None:
        return _catalog
    rows: list = []
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                code = (r.get("code") or "").strip().upper()
                if len(code) != 4:
                    continue
                rows.append({
                    "code": code,
                    "device": code[:2],
                    "cmd": code[2:],
                    "device_name": DEVICE_NAMES.get(int(code[:2], 16), ""),
                    "scope": (r.get("scope") or "").strip(),
                    "category": (r.get("category") or "").strip(),
                    "message_type": (r.get("message_type") or "").strip(),
                    # 等級沿用規範原表的 A / B / O,不自行解釋
                    "level": (r.get("level") or "").strip(),
                    "spec_page": (r.get("spec_page") or "").strip(),
                    "center_command": (r.get("center_command") or "").strip().lower() == "true",
                })
    except FileNotFoundError:
        rows = []
    except Exception as exc:                       # 目錄壞掉不該讓抄錄器整個掛掉
        print(f"[signal_tc3] 命令目錄讀取失敗 {CATALOG_PATH}: {exc}")
        rows = []
    _catalog = rows
    return rows

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
    "bad_peer": 0,           # 連上但不是 TC3 來源(打到別台機器)的次數
    "stalls": 0,             # 連線還在、但來源靜默而主動重連的次數
    "peer_note": "",
}
_frames: deque = deque(maxlen=300)      # 最近訊框(raw + 解碼)
_coverage: Counter = Counter()          # 每個 device+cmd 看過幾次
_lock = threading.Lock()


def _find_etx(buf: bytes, start: int) -> int:
    """從 start 逐位元組找 DLE ETX(AA CC),正確跳過 byte stuffing 的成對 AA。

    🛑 不能直接 find(AA CC)。INFO 裡若本來就有 0xAA,傳送時會被重複成 AA AA
       (協定 2-8);當它後面剛好接著 0xCC,線上就是 ... AA AA CC ...,單純 find 會在
       「第二個 AA」命中而把訊框砍短。CKS 會擋下錯資料,但那一整框會靜默消失。
       StepSec 是 2 bytes,0xAACC 在值域內,真的會遇到。
    """
    i, n = start, len(buf)
    while i < n - 1:
        if buf[i] != 0xAA:
            i += 1
            continue
        nxt = buf[i + 1]
        if nxt == 0xAA:      # 成對 AA = 資料裡的 0xAA,整組跳過
            i += 2
            continue
        if nxt == 0xCC:      # 真正的 DLE ETX
            return i
        i += 1               # AA 接別的(BB/DD/EE...),不是結尾
    return -1


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
            connected_at = time.time()
            last_rx = connected_at      # 最近一次切出合法碼框的時間(停擺看門狗用)
            got_tc3 = False
            while not shutdown_event.is_set():
                try:
                    d = sock.recv(4096)
                except socket.timeout:
                    now = time.time()
                    # 連上但一直沒有 TC3 訊框 → 對方不是號誌來源,主動換一次
                    if not got_tc3:
                        if (now - connected_at) > SIGNAL_PEER_TIMEOUT:
                            with _lock:
                                _state["bad_peer"] += 1
                                _state["peer_note"] = (
                                    f"連上 {SIGNAL_HOST}:{SIGNAL_PORT} 但 "
                                    f"{int(SIGNAL_PEER_TIMEOUT)} 秒內沒有任何 TC3 訊框 —— "
                                    "很可能打到同 IP 的另一台設備(檢查 ARP/路由)")
                            raise ConnectionError("peer 不是 TC3 來源")
                    # 抄到過但停了 → socket 沒斷不代表資料還在,重連一次比較誠實
                    elif (now - last_rx) > SIGNAL_STALL_TIMEOUT:
                        with _lock:
                            _state["stalls"] += 1
                            _state["peer_note"] = (
                                f"已連線但 {int(now - last_rx)} 秒沒有新訊框 —— "
                                "來源停止上傳(可能被交控中心佔用 MaxConnect,或序列線中斷),"
                                "已主動重連")
                        raise ConnectionError("來源靜默")
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
                    j = _find_etx(buf, i + 2)
                    if j < 0 or len(buf) < j + 3:
                        if i > 0:
                            buf = buf[i:]
                        break
                    frame = buf[i:j + 3]
                    buf = buf[j + 3:]
                    rec = decode_frame(frame)
                    if rec is None:
                        continue
                    got_tc3 = True      # 切出合法碼框 = 對方確實是 TC3 來源
                    last_rx = rec["ts"]
                    with _lock:
                        _state["frames_total"] += 1
                        _state["last_frame_at"] = rec["ts"]
                        _frames.append(rec)      # 壞框也留著,方便查線路品質
                        if not rec["cks_ok"]:
                            # 🛑 CKS 不對就到此為止。不可以拿它更新「目前燈態」或覆蓋矩陣 ——
                            #    壞掉的框會變成前端顯示的當前燈態,也會讓驗收矩陣記到
                            #    根本沒發生過的訊息碼。
                            _state["cks_bad"] += 1
                            continue
                        if rec.get("code"):
                            _coverage[rec["code"]] += 1
                            _state["peer_note"] = ""
                        if rec.get("phase"):
                            _state["latest"] = rec
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
                             "frames_total", "cks_bad", "reconnects", "last_error",
                             "bad_peer", "stalls", "peer_note")},
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


@router.get("/coverage", summary="TC3 命令覆蓋矩陣(規範 105 條 vs 實際抄到)")
async def coverage(_user=Depends(get_current_user)):
    """規範全表 join 實際抄到的次數。

    驗收要的是「規範 105 條裡跑到哪幾條」,所以分母一律是目錄,不是抄到的那幾條。
    抄到但不在目錄裡的訊息碼也會列出(extra=True)—— 那代表現場用了規範外的東西,
    是要被看見的事,不能被 join 吃掉。
    """
    with _lock:
        cov = dict(_coverage)
        total = _state["frames_total"]
    catalog = load_catalog()
    known = {r["code"] for r in catalog}

    rows = []
    for r in catalog:
        n = cov.get(r["code"], 0)
        rows.append({**r, "count": n, "seen": n > 0, "extra": False})
    for code, n in cov.items():
        if code in known:
            continue
        rows.append({"code": code, "device": code[:2], "cmd": code[2:],
                     "device_name": DEVICE_NAMES.get(int(code[:2], 16), ""),
                     "scope": "", "category": "(不在規範目錄)", "message_type": "",
                     "level": "", "spec_page": "", "center_command": False,
                     "count": n, "seen": True, "extra": True})
    # 抄到的排前面(次數多到少),沒抄到的照規範順序排在後
    rows.sort(key=lambda r: (not r["seen"], -r["count"]))

    # seen_total 只算規範內的,extra 另計 —— 分母是 105,不要被規範外的訊息碼灌水
    seen_rows = [r for r in rows if r["seen"] and not r["extra"]]
    by_level: dict = {}
    for r in rows:
        if r["extra"]:
            continue
        lv = r["level"] or "-"
        b = by_level.setdefault(lv, {"level": lv, "total": 0, "seen": 0})
        b["total"] += 1
        b["seen"] += 1 if r["seen"] else 0
    by_scope: dict = {}
    for r in rows:
        if r["extra"]:
            continue
        sc = r["scope"] or "-"
        b = by_scope.setdefault(sc, {"scope": sc, "total": 0, "seen": 0})
        b["total"] += 1
        b["seen"] += 1 if r["seen"] else 0

    return {
        "frames_total": total,
        "catalog_total": len(catalog),
        "seen_total": len(seen_rows),
        "extra_total": sum(1 for r in rows if r["extra"]),
        "distinct_codes": len(cov),
        "by_level": sorted(by_level.values(), key=lambda x: x["level"]),
        "by_scope": sorted(by_scope.values(), key=lambda x: x["scope"]),
        "codes": rows,
    }
