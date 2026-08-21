"""號誌控制器抄錄器 —— 都市交通控制通訊協定 3.0 版。

來源:號誌控制器 RS-232 → Moxa MiiNePort E1 → TCP。現場實測參數:
    IP 10.42.38.35   TCPDataPort 1001   Mode TCP / Role TCP Server   MaxConnect 1

抄錄(接收)是無條件開著的。協定的號誌控制器是主動週期上傳
(5F+03 時相資料主動回報),不需要輪詢就看得到燈態。

🛑 號控(下傳)是另一回事,預設關閉。這是「對運轉中的號誌控制器送出位元組」——
   送錯一則 5F15(時制計畫)或 5F10(控制策略)會直接改變路口的實際號誌運轉。
   要啟用必須明確設 SIGNAL_TC3_CONTROL=1;而且預設再加一層
   SIGNAL_TC3_CONTROL_QUERY_ONLY=1,只准查詢類(不改變運轉)。
   見檔案末段 control_prepare / control_send。

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
import secrets
import socket
import threading
import time
from collections import Counter, deque
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from api.routes.auth import get_current_user
from api.routes.logs import add_log
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
# 門檻怎麼定的:2026-08-18 在 87 實測 3.5 分鐘 36 框 —— 平均間隔 6.0 秒、
# 最大 33.7 秒。上傳不是固定 2 秒一次(還混著中心查詢的回報),所以取實測最大值
# 的數倍當門檻,寧可晚一點重連也不要在正常間隔就誤斷。
SIGNAL_STALL_TIMEOUT = float(os.getenv("SIGNAL_TC3_STALL_TIMEOUT", "180") or 180)
# 前端「資料過期」的判定。同樣照實測:33.7 秒是正常會出現的間隔,
# 原本寫 30 秒會在正常運作時一直閃紅字。
SIGNAL_STALE_SEC = float(os.getenv("SIGNAL_TC3_STALE_SEC", "90") or 90)

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
# ── 號控(下傳)──────────────────────────────────────────────────────────
# 🛑 這是「對運轉中的號誌控制器送出位元組」的能力。送錯一則 5F15(時制計畫)
#    或 5F10(控制策略)會直接改變路口的實際號誌運轉,不是「沒反應」而已。
#    所以三道關卡:
#      ① 總開關預設關閉。沒有明確設 SIGNAL_TC3_CONTROL=1 一律拒絕。
#      ② 預設只准查詢類(指令碼高位元組 4~6)。查詢不改變運轉,
#         而且是驗證這條線 TX 到底通不通的最小風險方式。
#      ③ 兩段式送出:先 prepare 拿到解碼預覽與 token,再帶 token 送出。
#         「按錯一個按鈕就送出去」不會發生。
#    每一次送出都留完整紀錄(誰、何時、什麼碼、原始位元組、結果)。
CONTROL_ENABLED = os.getenv("SIGNAL_TC3_CONTROL", "0") == "1"
CONTROL_QUERY_ONLY = os.getenv("SIGNAL_TC3_CONTROL_QUERY_ONLY", "1") == "1"
CONTROL_ADDR = int(os.getenv("SIGNAL_TC3_ADDR", "0") or 0)   # 0 = 用抄到的位址
_PREPARE_TTL = 60.0                     # token 有效期,過了要重新確認

_send_lock = threading.Lock()           # 同時只有一個送出
_sock_ref: dict = {"sock": None}        # 抄錄器把目前的 socket 放這裡給送出用
_pending: dict = {}                     # token -> 準備好的送出內容
_sent_log: deque = deque(maxlen=200)    # 送出紀錄(稽核用)
_seq_next: dict = {"n": 0}              # 送出用的序號,逐次遞增

_frames: deque = deque(maxlen=300)      # 最近訊框(raw + 解碼)
_coverage: Counter = Counter()          # 每個 device+cmd 看過幾次
_by_addr: dict = {}                     # 設備位址 -> 最新一筆燈態(一台=一個路口)
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


# 碼框的四個控制位元組。用具名常數而不是字面值,一來讀得懂,
# 二來避免在多層字串裡被跳脫吃掉(2026-08-21 實際踩過)。
DLE = 0xAA
STX = 0xBB
ETX = 0xCC


def _stuff(info: bytes) -> bytes:
    """_unstuff 的反向:INFO 裡的 0xAA 要重複成 0xAA 0xAA(協定 2-8)。

    不做這件事的話,INFO 裡只要出現一個 0xAA,接收端就會把它當成 DLE,
    整個碼框從那裡被切斷。
    """
    out = bytearray()
    for b in info:
        out.append(b)
        if b == 0xAA:
            out.append(0xAA)
    return bytes(out)


def build_frame(addr: int, seq: int, info: bytes) -> bytes:
    """組一個 TC3 碼框。decode_frame 的反向。

        DLE STX SEQ ADDR LEN INFO DLE ETX CKS
        AA  BB  1B  2B   2B  N    AA  CC  1B

    🛑 LEN 是「整個碼框的長度」(10 + INFO 長度),不是 INFO 長度 ——
       這點是拿現場實際抄到的訊框逐一比對過的(21/21 相符)。
       而且要用 **stuffing 之後** 的 INFO 長度,LEN 才會等於真正送出去的位元組數。
    🛑 CKS = XOR(DLE .. ETX),含頭尾但不含自己。
    """
    # 🛑 這支會把位元組送進運轉中的號誌通道,寧可在這裡擋下來也不要送出去。
    #    INFO 至少要有「設備碼 + 指令碼」兩個位元組 —— 少於這個就不是有意義的
    #    命令,decode_frame 也會拒收(它要求整框 >= 11)。
    info = bytes(info)
    if len(info) < 2:
        raise ValueError("INFO 至少要有設備碼與指令碼兩個位元組")
    if not (0 <= int(addr) <= 0xFFFF):
        raise ValueError(f"位址超出範圍: {addr}")
    if not (0 <= int(seq) <= 0xFF):
        raise ValueError(f"序號超出範圍: {seq}")
    body = _stuff(info)
    total = 10 + len(body)
    frame = bytearray()
    frame += bytes((DLE, STX))
    frame.append(seq & 0xFF)
    frame += int(addr).to_bytes(2, "big")
    frame += int(total).to_bytes(2, "big")
    frame += body
    frame += bytes((DLE, ETX))
    cks = 0
    for b in frame:
        cks ^= b
    frame.append(cks)
    return bytes(frame)


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
            # 把目前這條 socket 交出去給號控用。TCP 的收送方向是獨立的,
            # 從別的執行緒 send 不會干擾這裡的 recv。
            _sock_ref["sock"] = sock
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
                            # 依設備位址各存一筆最新燈態 —— 一台控制器 = 一個路口。
                            # 現場現在只有 0xFFFF 一個,未來抄到多位址就自動多路口。
                            a = rec.get("addr")
                            if isinstance(a, int):
                                _by_addr[a] = rec
        except Exception as exc:
            with _lock:
                _state["connected"] = False
                _state["last_error"] = f"{type(exc).__name__}: {exc}"[:160]
                _state["reconnects"] += 1
        finally:
            # 🛑 先把交出去的參照清掉再關 socket,否則號控可能拿到已關閉的 fd。
            _sock_ref["sock"] = None
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
        by_addr = dict(_by_addr)
    now = time.time()
    age = (now - s["last_frame_at"]) if s["last_frame_at"] else None
    # 一台控制器 = 一個路口。每個位址一張,附各自的資料齡與是否過期。
    intersections = []
    for a, rec in sorted(by_addr.items()):
        r_age = (now - rec.get("ts", now)) if rec.get("ts") else None
        intersections.append({
            "addr": a,
            "addr_hex": f"0x{a:04X}",
            "phase": rec.get("phase"),
            "age_sec": round(r_age, 2) if r_age is not None else None,
            "stale": (r_age is None or r_age > SIGNAL_STALE_SEC),
        })
    return {
        **{k: s[k] for k in ("enabled", "host", "port", "connected",
                             "frames_total", "cks_bad", "reconnects", "last_error",
                             "bad_peer", "stalls", "peer_note")},
        "age_sec": round(age, 2) if age is not None else None,
        # 沒新訊框就標 stale —— 來源掉了要看得出來,不要顯示假的舊狀態。
        # 門檻見 SIGNAL_STALE_SEC(照現場實測間隔定,不是拍腦袋)
        "stale": (age is None or age > SIGNAL_STALE_SEC),
        "latest": latest,
        "intersections": intersections,     # 一台控制器一個路口
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


# ── 號控:下傳命令 ──────────────────────────────────────────────────────
# 兩段式。prepare 只組碼框並回傳解碼預覽,不碰 socket;send 才真的送出。


def _kind_by_nibble(cmd: int) -> str:
    """只看指令碼高位元組的粗略判斷(協定 3-3)。

    🛑 這個規則有例外,不要單獨拿它當安全閘門的依據 —— 見 _kind_of。
       用 105 條目錄實際比對:高位元組 8 那格是混的
       (0F8E 密碼代碼「設定」和 0F8F「設定回報」都落在 8)。
    """
    hi = (int(cmd) >> 4) & 0xF
    if hi == 0:
        return "主動回報"
    if 1 <= hi <= 3:
        return "設定"
    if 4 <= hi <= 6:
        return "查詢"
    if 8 <= hi <= 0xB:
        return "設定回報"
    if 0xC <= hi <= 0xE:
        return "查詢回報"
    return f"未知({hi:X})"


def _kind_of(cmd: int, device: int = 0x5F) -> str:
    """這則訊息是什麼型態。**以規範目錄為準**,目錄查不到才退回位元組規則。

    🛑 為什麼不直接用高位元組:它有例外。安全閘門是靠這個判斷「能不能送」的,
       用一個有例外的規則去擋,例外就是漏洞。目錄是逐條抄規範來的,它說了算。
    """
    code = f"{int(device):02X}{int(cmd):02X}"
    for row in load_catalog():
        if row.get("code") == code:
            mt = (row.get("message_type") or "").strip()
            if mt:
                # 「設定/解除」「設定、查詢」這種複合的,取第一段當主型態
                for sep in ("/", "、"):
                    if sep in mt:
                        return mt.split(sep)[0].strip()
                return mt
            break
    return _kind_by_nibble(cmd)


def _control_guard(cmd: int, device: int = 0x5F) -> Optional[str]:
    """能不能送。回錯誤訊息表示不能,回 None 表示可以。"""
    if not CONTROL_ENABLED:
        return ("號控未啟用。這台預設不對號誌通道送出任何位元組;"
                "要啟用請設 SIGNAL_TC3_CONTROL=1 並重啟服務。")
    kind = _kind_of(cmd, device)
    if kind in ("主動回報", "設定回報", "查詢回報"):
        return f"{kind} 是控制器回給中心的,中心不送這類訊息。"
    if CONTROL_QUERY_ONLY and kind != "查詢":
        return (f"目前限制為「只准查詢」,{kind} 類被擋下。"
                "查詢不改變控制器運轉;要開放設定類請設 "
                "SIGNAL_TC3_CONTROL_QUERY_ONLY=0。")
    return None


def _target_addr() -> Optional[int]:
    """送給誰。優先用設定值,否則用最近抄到的位址 —— 猜錯位址等於送給別的路口。"""
    if CONTROL_ADDR:
        return CONTROL_ADDR
    with _lock:
        items = list(_frames)
    for it in reversed(items):
        a = (it or {}).get("addr")
        if isinstance(a, int):
            return a
    return None


@router.get("/control/status", summary="號控狀態(是否啟用、限制、最近送出紀錄)")
async def control_status(_user=Depends(get_current_user)):
    addr = _target_addr()
    return {
        "enabled": CONTROL_ENABLED,
        "query_only": CONTROL_QUERY_ONLY,
        "link_connected": bool(_sock_ref.get("sock")),
        "target_addr": addr,
        "target_addr_hex": f"0x{addr:04X}" if addr is not None else None,
        "addr_source": "設定" if CONTROL_ADDR else "由抄錄到的訊框推得",
        "prepare_ttl_sec": _PREPARE_TTL,
        "recent": list(reversed(list(_sent_log)))[:30],
    }


@router.post("/control/prepare", summary="準備下傳(只組碼框與預覽,不送出)")
async def control_prepare(body: dict, _user=Depends(get_current_user)):
    """把要送的內容組成碼框並解回來給人看,確認無誤再用 token 送出。

    body: {"code": "5F45", "info_hex": "05"}   info_hex 是設備碼/指令碼之後的參數
    """
    code = str((body or {}).get("code") or "").strip().upper()
    if len(code) != 4:
        raise HTTPException(status_code=400, detail="code 要是 4 個十六進位字元,例如 5F45")
    try:
        dev = int(code[:2], 16)
        cmd = int(code[2:], 16)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"code 不是十六進位: {code}")

    why = _control_guard(cmd, dev)
    if why:
        raise HTTPException(status_code=403, detail=why)

    # 兩種給參數的方式:
    #   values = {...}  結構化 → 用 utc-tc3 的 encoder 照 schema 組(前端表單走這條)
    #   info_hex        原始十六進位 → 手動指定(進階/schema 不可用時的後路)
    # 🛑 兩者只能擇一,同時給會混淆「到底送了什麼」,直接擋。
    values = (body or {}).get("values")
    raw_hex = str((body or {}).get("info_hex") or "").replace(" ", "")
    if values is not None and raw_hex:
        raise HTTPException(status_code=400, detail="values 與 info_hex 只能擇一")

    if values is not None:
        if not isinstance(values, dict):
            raise HTTPException(status_code=400, detail="values 要是物件")
        schemas = load_command_schemas()
        try:
            import sys
            if UTC_TC3_PATH and UTC_TC3_PATH not in sys.path:
                sys.path.insert(0, UTC_TC3_PATH)
            from utc import messages as _M       # type: ignore
            msg = next((m for m in _M.ALL if m.code == code), None)
            if msg is None:
                raise HTTPException(status_code=400, detail=f"schema 裡沒有 {code}")
            # 🛑 bytes/var/tail 這幾種欄位,encoder 要的是 bytes 物件,但前端走 JSON
            #    只能送字串。這裡照 schema 把那些欄位的十六進位字串轉成 bytes,
            #    否則含密碼/協定/點陣的命令永遠編不出來(前端 UI 有欄位也沒用)。
            _sc = load_command_schemas().get(code, {})
            _hex_fields = {f.get("name") for f in _sc.get("fields", [])
                           if f.get("kind") in ("bytes", "var", "tail")}
            for _fn in _hex_fields:
                v = values.get(_fn)
                if isinstance(v, str) and v.strip():
                    try:
                        values[_fn] = bytes.fromhex(v.replace(" ", ""))
                    except ValueError:
                        raise HTTPException(status_code=400,
                                            detail=f"{_fn} 不是合法的十六進位")
            info = msg.encode(values)             # 回傳含設備碼+指令碼的完整 INFO
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400,
                                detail=f"依 schema 組不出來:{type(exc).__name__}: {exc}")
        # encode 已含 dev+cmd,frame 直接用;下面 build_frame 的 [dev,cmd]+params 不走
        addr = _target_addr()
        if addr is None:
            raise HTTPException(status_code=409,
                                detail="不知道要送給哪個位址:還沒抄到訊框,也沒設 SIGNAL_TC3_ADDR。")
        seq = (_seq_next["n"] + 1) & 0xFF
        try:
            frame = build_frame(addr, seq, info)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _finish_prepare(frame, code, cmd, dev, addr, seq)

    try:
        params = bytes.fromhex(raw_hex) if raw_hex else b""
    except ValueError:
        raise HTTPException(status_code=400, detail="info_hex 不是合法的十六進位字串")

    addr = _target_addr()
    if addr is None:
        raise HTTPException(status_code=409,
                            detail="不知道要送給哪個位址:還沒抄到任何訊框,也沒設 SIGNAL_TC3_ADDR。")

    seq = (_seq_next["n"] + 1) & 0xFF
    try:
        frame = build_frame(addr, seq, bytes([dev, cmd]) + params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _finish_prepare(frame, code, cmd, dev, addr, seq)


def _finish_prepare(frame, code, cmd, dev, addr, seq):
    """產生 token 並回傳預覽。兩條參數路徑(結構化 / info_hex)共用。"""
    token = secrets.token_urlsafe(16)
    now = time.time()
    for k in [k for k, v in _pending.items() if now - v["ts"] > _PREPARE_TTL]:
        _pending.pop(k, None)
    _pending[token] = {"ts": now, "frame": frame, "code": code, "seq": seq, "addr": addr}
    return {
        "token": token,
        "expires_in": _PREPARE_TTL,
        "code": code,
        "kind": _kind_of(cmd, dev),
        "addr": addr,
        "addr_hex": f"0x{addr:04X}",
        "seq": seq,
        "bytes": len(frame),
        "raw": frame.hex(" ").upper(),
        # 用自己的解析器解回來 —— 送出去的東西長什麼樣,人要看得到
        "decoded": decode_frame(frame),
    }


@router.post("/control/send", summary="真的送出(要帶 prepare 拿到的 token)")
async def control_send(body: dict, _user=Depends(get_current_user)):
    token = str((body or {}).get("token") or "")
    item = _pending.pop(token, None)
    if not item:
        raise HTTPException(status_code=400, detail="token 無效或已過期,請重新準備。")
    if time.time() - item["ts"] > _PREPARE_TTL:
        raise HTTPException(status_code=400, detail="token 已過期,請重新準備。")

    dev_cmd = item["code"]
    why = _control_guard(int(dev_cmd[2:], 16), int(dev_cmd[:2], 16))
    if why:                                  # 準備到送出之間設定可能變了
        raise HTTPException(status_code=403, detail=why)

    sock = _sock_ref.get("sock")
    if sock is None:
        raise HTTPException(status_code=409, detail="號誌通道目前沒有連線,無法送出。")

    user = getattr(_user, "username", None) or str(_user)
    rec = {
        "ts": time.time(),
        "user": user,
        "code": item["code"],
        "kind": _kind_of(int(item["code"][2:], 16), int(item["code"][:2], 16)),
        "addr": item["addr"],
        "seq": item["seq"],
        "raw": item["frame"].hex(" ").upper(),
        "ok": False,
        "error": "",
    }
    with _send_lock:
        try:
            sock.sendall(item["frame"])
            _seq_next["n"] = item["seq"]
            rec["ok"] = True
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:160]

    _sent_log.append(rec)
    # 🛑 送出一定要留在系統日誌,不能只存在記憶體裡 —— 服務重啟就沒了,
    #    而「誰在什麼時候對號誌送了什麼」是要能事後查的。
    print(f"[signal-tc3][控制] user={user} code={rec['code']} kind={rec['kind']} "
          f"addr=0x{rec['addr']:04X} seq={rec['seq']} ok={rec['ok']} "
          f"raw={rec['raw']}{(' err=' + rec['error']) if rec['error'] else ''}",
          flush=True)
    add_log("warning" if not rec["ok"] else "info",
            f"號控下傳 {rec['code']}({rec['kind']}) → 0x{rec['addr']:04X} "
            f"{'成功' if rec['ok'] else '失敗: ' + rec['error']}", "signal")

    if not rec["ok"]:
        raise HTTPException(status_code=502, detail=f"送出失敗: {rec['error']}")
    return {"ok": True, "sent": rec,
            "note": "已送出。控制器若有回應會出現在訊框清單(查詢類會回查詢回報)。"}


# ── 命令 schema(欄位定義)──────────────────────────────────────────────
# 前端要照每則命令的參數欄位產生表單,那份定義在另一個專案 utc-tc3
# (協定模擬器)的 utc.messages.ALL 裡,每則 .schema() 就是前端要的 JSON。
# 🛑 不把那 700 行 codec DSL 抄一份進來 —— 抄了兩邊會漂移,而號控送錯欄位
#    是會改到路口號誌的。改成執行時去 import,以那個專案為單一真相來源。
#    找不到就回空表,前端退回「手輸 info_hex」,不會整頁壞掉。
_SCHEMA_CACHE: Optional[dict] = None
UTC_TC3_PATH = os.getenv("UTC_TC3_PATH", "/home/ubuntu/utc-tc3")


def load_command_schemas() -> dict:
    """{code: schema dict}。匯入 utc-tc3;不可用就回空表。"""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    out: dict = {}
    try:
        import sys
        if UTC_TC3_PATH and UTC_TC3_PATH not in sys.path:
            sys.path.insert(0, UTC_TC3_PATH)
        from utc import messages as _M       # type: ignore
        for m in _M.ALL:
            sc = m.schema() if callable(getattr(m, "schema", None)) else None
            if sc and sc.get("code"):
                out[sc["code"]] = sc
    except Exception as exc:
        print(f"[signal_tc3] 命令 schema 匯入失敗({UTC_TC3_PATH}): {exc};"
              f"前端將退回手輸 info_hex", flush=True)
        out = {}
    _SCHEMA_CACHE = out
    return out


@router.get("/control/schemas", summary="每則命令的參數欄位定義(給前端產表單)")
async def control_schemas(_user=Depends(get_current_user)):
    """回可下傳的命令清單與其欄位。已套安全過濾:回不了的類別不列。"""
    schemas = load_command_schemas()
    items = []
    for code, sc in schemas.items():
        try:
            dev = int(code[:2], 16)
            cmd = int(code[2:], 16)
        except (ValueError, IndexError):
            continue
        # 只列「中心送得出去」的:總開關關著就全空;只准查詢就只列查詢
        if _control_guard(cmd, dev) is not None:
            continue
        items.append({
            "code": code,
            "name": sc.get("name"),
            "kind": _kind_of(cmd, dev),
            "level": sc.get("level"),
            "group": sc.get("group"),
            "page": sc.get("page"),
            "fields": sc.get("fields", []),
        })
    items.sort(key=lambda x: x["code"])
    return {
        "available": bool(schemas),
        "control_enabled": CONTROL_ENABLED,
        "query_only": CONTROL_QUERY_ONLY,
        "count": len(items),
        "commands": items,
    }


# ── 訊框深度解碼(欄位對照)──────────────────────────────────────────────
# decode_frame 在抄錄熱迴圈裡跑,只硬解了最常用的 5F03 燈態。其餘訊息
# (時制查詢回報、時相資料…)要看具名欄位,靠 utc-tc3 的 decoder 在「讀取時」解 ——
# 那是 API 執行緒,慢一點沒關係,不拖累抄錄。


def _decode_fields(code: str, raw_hex: str) -> Optional[list]:
    """把一個訊框解成 [{name, value, desc}] 的欄位表。解不出來回 None。"""
    try:
        import sys
        if UTC_TC3_PATH and UTC_TC3_PATH not in sys.path:
            sys.path.insert(0, UTC_TC3_PATH)
        from utc import messages as _M       # type: ignore
        msg = next((m for m in _M.ALL if m.code == code), None)
        if msg is None:
            return None
        raw = bytes.fromhex(str(raw_hex).replace(" ", ""))
        info = _unstuff(raw[7:-3])           # 去框頭尾 + 還原 stuffing
        vals = msg.decode(info)
        if not isinstance(vals, dict):
            return None
        # 欄位順序照 schema,值照解出來的;附上 desc 讓表格看得懂
        desc_map = {f.get("name"): f.get("desc") for f in msg.schema().get("fields", [])}
        out = []
        for k, v in vals.items():
            out.append({"name": k, "value": v, "desc": desc_map.get(k, "")})
        return out
    except Exception:
        return None


@router.get("/frames/decoded", summary="最近訊框(含 utc-tc3 深度解碼的欄位表)")
async def frames_decoded(limit: int = 30, _user=Depends(get_current_user)):
    """跟 /frames 一樣,但每一框多帶 fields(欄位名→值),給前端做對照表。"""
    n = max(1, min(200, int(limit or 30)))
    with _lock:
        items = list(_frames)[-n:]
    out = []
    for it in reversed(items):
        row = dict(it)
        code = it.get("code")
        if code:
            row["fields"] = _decode_fields(code, it.get("raw", ""))
            # 訊息名稱也帶上,前端表頭好標
            for m in load_command_schemas().values():
                if m.get("code") == code:
                    row["name"] = m.get("name")
                    break
        out.append(row)
    return {"count": len(out), "frames": out}


# ── 查詢紀錄:顯示 + 儲存 ──────────────────────────────────────────────────
# 前端的 sigQryLog 只在記憶體,關頁就沒了。查詢回報要能事後回查,存進 DB。
# 用獨立表,不動 SQLAlchemy models —— 這是純附加,跟既有 schema 無關。
import sqlite3 as _sqlite3
import json as _json_ctl

_QDB_PATH = os.getenv("SIGNAL_QUERY_DB",
                      str(pathlib.Path(__file__).resolve().parents[2] / "data" / "violations.db"))
_qdb_ready = False


def _query_db():
    global _qdb_ready
    conn = _sqlite3.connect(_QDB_PATH, timeout=20)
    conn.execute("PRAGMA busy_timeout=20000")
    if not _qdb_ready:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, user TEXT, query_code TEXT, reply_code TEXT,
                name TEXT, addr INTEGER, fields_json TEXT, raw TEXT
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sql_ts ON signal_query_log(ts)")
        conn.commit()
        _qdb_ready = True
    return conn


@router.post("/control/query-log", summary="儲存一筆查詢結果")
async def query_log_save(body: dict, _user=Depends(get_current_user)):
    """前端配對到查詢回報後呼叫,把結果存進 DB。"""
    user = getattr(_user, "username", None) or str(_user)
    fields = (body or {}).get("fields")
    row = (
        float((body or {}).get("ts") or time.time()),
        user,
        str((body or {}).get("query_code") or ""),
        str((body or {}).get("reply_code") or ""),
        str((body or {}).get("name") or ""),
        int((body or {}).get("addr") or 0),
        _json_ctl.dumps(fields, ensure_ascii=False) if fields is not None else None,
        str((body or {}).get("raw") or ""),
    )
    try:
        conn = _query_db()
        conn.execute(
            "INSERT INTO signal_query_log "
            "(ts,user,query_code,reply_code,name,addr,fields_json,raw) "
            "VALUES (?,?,?,?,?,?,?,?)", row)
        conn.commit()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"儲存失敗: {exc}")
    return {"ok": True}


@router.get("/control/query-log", summary="查詢紀錄(可依碼篩選)")
async def query_log_list(limit: int = 100, code: str = "", _user=Depends(get_current_user)):
    n = max(1, min(500, int(limit or 100)))
    try:
        conn = _query_db()
        sql = ("SELECT ts,user,query_code,reply_code,name,addr,fields_json,raw "
               "FROM signal_query_log")
        params: list = []
        if code:
            sql += " WHERE query_code=? OR reply_code=?"
            params += [code.upper(), code.upper()]
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(n)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"讀取失敗: {exc}")
    out = []
    for ts, user, qc, rc, name, addr, fj, raw in rows:
        out.append({
            "ts": ts, "user": user, "query_code": qc, "reply_code": rc,
            "name": name, "addr": addr,
            "fields": _json_ctl.loads(fj) if fj else None, "raw": raw,
        })
    return {"count": len(out), "items": out}
