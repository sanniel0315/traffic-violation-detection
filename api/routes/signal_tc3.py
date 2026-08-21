"""號誌控制器抄錄器 —— 都市交通控制通訊協定 3.0 版。

來源:號誌控制器 RS-232 → Moxa MiiNePort E1 → TCP。現場實測參數:
    IP 10.42.40.222  TCPDataPort 1001   Mode TCP / Role TCP Server   MaxConnect 1
    (2026-08-21 MiiNePort 由 10.42.38.35 移到 10.42.40.222;10.42.38.35 已改給分析器主機)

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
import queue as _queue
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

SIGNAL_HOST = os.getenv("SIGNAL_TC3_HOST", "10.42.40.222")
SIGNAL_PORT = int(os.getenv("SIGNAL_TC3_PORT", "1001") or 1001)
SIGNAL_ENABLED = os.getenv("SIGNAL_TC3_ENABLED", "1") != "0"
# 連上之後多久沒看到任何 TC3 訊框就判定「打到錯的機器」並主動斷線重試。
# 🛑 為什麼需要:號誌來源是 MiiNePort(現場 10.42.40.222)。MaxConnect=1、又只
#    此一條線,對端靜默或被交控中心搶走時,TCP 連得上卻永遠沒有資料 —— 沒有這
#    道檢查就會傻等到天亮。(2026-08-21 前 NPort 曾在 10.42.38.35 且網線落在
#    192.168.1.x,ARP 被攪動時封包會打到錯的那台,同樣靠這道檢查救回。)
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

# 路口名稱。真實控制器沒有內建路名,而 0xFFFF 是廣播位址,不能拿來當路口名顯示。
#   SIGNAL_TC3_SITE_NAME   單一站點名(這台現場只有一個路口時用)
#   SIGNAL_TC3_SITE_NAMES  多路口用,格式 "0x1230=興隆路三段,0x1231=..."
SITE_NAME = os.getenv("SIGNAL_TC3_SITE_NAME", "").strip()
_SITE_NAMES: dict = {}
for _pair in os.getenv("SIGNAL_TC3_SITE_NAMES", "").split(","):
    if "=" in _pair:
        _k, _v = _pair.split("=", 1)
        try:
            _SITE_NAMES[int(_k.strip(), 16)] = _v.strip()
        except ValueError:
            pass


def _addr_name(addr: int) -> str:
    """路口顯示名。優先查設定的路名;沒有的話廣播位址回站名或通用名,
    不把 0xFFFF 這種位址當名字秀出來。"""
    if addr in _SITE_NAMES:
        return _SITE_NAMES[addr]
    if addr == 0xFFFF:                       # 廣播 = 這條線唯一的路口
        return SITE_NAME or "號誌路口"
    return SITE_NAME or f"路口 {addr}"       # 具體編號的路口,用十進位比較像編號

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

# ── 中央電腦中繼(都三) ────────────────────────────────────────────────
# 架構:我們這台=中央眼中的號誌控制器。中央連我們 <本機>:1001(跟我們查 NPort
# 同一個 port),我們把 frame 轉給真控制器(佔據 NPort MaxConnect=1 那條),
# 控制器的主動回報/查詢回報再回傳給中央。雙向都側錄進 _frames(標 src)。
# 🛑 不是寫死的純通透管道 —— 保留雙向注入:
#    · 往控制器:_controller_send() 讓我們自己下號控(與中央下傳共用一條、加鎖序列化)
#    · 往中央:  _send_to_center() 讓我們「自己控制號誌後主動上報中央」,不必等控制器回報
#    預設行為是轉發;上面兩個鉤子讓我方邏輯可在任一方向插入自己的 frame。
# 🛑 opt-in:SIGNAL_TC3_CENTER_RELAY=1 才啟用;預設關,完全不動現有抄錄行為。
CENTER_RELAY_ENABLED = os.getenv("SIGNAL_TC3_CENTER_RELAY", "0") == "1"
CENTER_LISTEN_HOST = os.getenv("SIGNAL_TC3_CENTER_LISTEN_HOST", "0.0.0.0")
CENTER_LISTEN_PORT = int(os.getenv("SIGNAL_TC3_CENTER_LISTEN_PORT", "1001") or 1001)
# 🛑 廠商 HardwareStatus 說明「寫反」(極性顛倒):bit14 實際 1=信號驅動單元正常。
#    中央照寫反的說明會把 0x4000(正常)誤顯示「故障」。我們在轉給中央前翻 bit14,
#    補償中央的反向解讀,讓中央顯示正確。預設開,可用 env 關(=純通透)。
HW_STATUS_FIX = os.getenv("SIGNAL_TC3_FIX_HWSTATUS", "1") != "0"
HW_STATUS_FIX_CODES = ("0F04", "0FC1")   # 帶 HardwareStatus 的訊息
HW_STATUS_FIX_MASK = 0x4000              # 要翻的位元(bit14 信號驅動單元)
# 自我查詢比對:我方主動查控制器(5F40/5F48/5F44/0F41),回報預設「不轉發中央」,
# 避免中央看到它沒問的回報。用「有界計數 + 短窗」抑制:只擋掉我們預期筆數的回報,
# 中央若同碼查詢,其回報仍會有一筆通過(資料相同),不會被餓死。
SELF_PROBE_SUPPRESS = os.getenv("SIGNAL_TC3_SELFPROBE_NO_TEE", "1") != "0"
_self_probe_expect: dict = {}            # reply_code -> [remaining, expiry_ts]
_center_sock_ref: dict = {"sock": None}  # 目前中央的連線(單一,MaxConnect=1)
_ctrl_tx_lock = threading.Lock()         # 序列化「寫控制器」(號控下傳+中央轉發共用一條)
_center_tx_lock = threading.Lock()       # 序列化「寫中央」(轉發控制器回報+我方自報共用一條)
_center_state: dict = {
    "enabled": CENTER_RELAY_ENABLED,
    "listen": f"{CENTER_LISTEN_HOST}:{CENTER_LISTEN_PORT}",
    "connected": False, "peer": "", "since": 0.0,
    "from_center_bytes": 0, "to_center_bytes": 0, "center_frames": 0,
}

# 訊框持久化:抄到/中繼的每個 frame 都丟進背景 writer 寫 DB(監看要能存、篩選、重啟後還在)。
# 用 queue 解耦 —— 收框/中繼迴圈只 put_nowait,實際寫 DB 在單一 writer 執行緒,不卡熱路徑。
FRAME_PERSIST = os.getenv("SIGNAL_TC3_PERSIST_FRAMES", "1") != "0"
FRAME_RETAIN_DAYS = float(os.getenv("SIGNAL_TC3_FRAME_RETAIN_DAYS", "180") or 180)  # 保存 6 個月
_frame_q: "_queue.Queue" = _queue.Queue(maxsize=8000)


def _enqueue_frame(rec: dict) -> None:
    """把一個解好的 frame 排進持久化佇列(滿了就丟,監看不是關鍵資料,別回壓熱路徑)。"""
    if not FRAME_PERSIST:
        return
    try:
        _frame_q.put_nowait({
            "ts": rec.get("ts"), "src": rec.get("src", ""), "code": rec.get("code"),
            "seq": rec.get("seq"), "addr": rec.get("addr"), "len": rec.get("len"),
            "cks_ok": 1 if rec.get("cks_ok") else 0, "raw": rec.get("raw", ""),
            "user": rec.get("user", ""),
        })
    except _queue.Full:
        pass

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
                # 控制器→中央的轉發改「逐框」做(見下方切框處),才能對 0F04/0FC1 的
                # HardwareStatus 翻 bit14 補償廠商寫反。這裡不再原封 tee 整段 recv。
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
                        # 完整框但解不出:仍原封轉給中央,保持透明。
                        if CENTER_RELAY_ENABLED:
                            _tee_to_center(frame)
                        continue
                    rec["src"] = "controller"   # 來源:號誌控制器(上行)
                    # 逐框轉給中央(0F04/0FC1 會翻 HardwareStatus bit14 校正)。
                    if CENTER_RELAY_ENABLED:
                        _forward_controller_frame_to_center(frame, rec)
                    got_tc3 = True      # 切出合法碼框 = 對方確實是 TC3 來源
                    last_rx = rec["ts"]
                    with _lock:
                        _state["frames_total"] += 1
                        _state["last_frame_at"] = rec["ts"]
                        _frames.append(rec)      # 壞框也留著,方便查線路品質
                        _enqueue_frame(rec)      # 持久化(含壞框,查線路品質用)
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


# ── 中央中繼:寫控制器 / 轉發中央 / server 迴圈 ────────────────────────────
def _controller_send(data: bytes) -> bool:
    """把 bytes 寫進控制器 socket。號控下傳與中央轉發共用這條,靠 _ctrl_tx_lock
    序列化,避免兩邊同時 send 把 frame 交錯寫壞。控制器未連線回 False。"""
    ctrl = _sock_ref.get("sock")
    if ctrl is None:
        return False
    try:
        with _ctrl_tx_lock:
            ctrl.sendall(data)
        return True
    except Exception:
        return False


def _close_center() -> None:
    cs = _center_sock_ref.get("sock")
    _center_sock_ref["sock"] = None
    _center_state["connected"] = False
    if cs is not None:
        try:
            cs.close()
        except Exception:
            pass


def _tee_to_center(data: bytes) -> None:
    """控制器來的原始 bytes 原封轉給中央(透明轉發)。中央斷了就清掉。
    與 _send_to_center 共用 _center_tx_lock,避免轉發位元組跟我方自報 frame 交錯。"""
    cs = _center_sock_ref.get("sock")
    if cs is None:
        return
    try:
        with _center_tx_lock:
            cs.sendall(data)
        _center_state["to_center_bytes"] += len(data)
    except Exception:
        _close_center()


def _note_expected_reply(reply_code: str, n: int = 1) -> None:
    """記下「我方查詢預期收到的回報碼」n 筆,短窗內抑制轉發給中央。"""
    now = time.time()
    exp = _self_probe_expect.get(reply_code)
    if exp and now < exp[1]:
        exp[0] += n
        exp[1] = now + 10.0
    else:
        _self_probe_expect[reply_code] = [n, now + 10.0]


def _should_suppress_to_center(code: Optional[str]) -> bool:
    """此回報是否為我方查詢的結果(該筆不轉中央)。有界計數,過窗自動失效。"""
    if not code:
        return False
    exp = _self_probe_expect.get(code)
    if not exp:
        return False
    if time.time() >= exp[1] or exp[0] <= 0:
        _self_probe_expect.pop(code, None)
        return False
    exp[0] -= 1
    if exp[0] <= 0:
        _self_probe_expect.pop(code, None)
    return True


def _forward_controller_frame_to_center(frame: bytes, rec: dict) -> None:
    """把控制器的一個完整框轉給中央(透明中繼的上行)。
    🛑 例外1:我方自我查詢的回報,不轉中央(SELF_PROBE_SUPPRESS)。
    🛑 例外2:0F04/0FC1 的 HardwareStatus 翻 bit14(補償廠商『寫反』的說明,讓中央
       的反向解讀顯示正確);翻完重組碼框(build_frame 會重算 CKS + byte stuffing)。
       其餘一律原封轉發。出錯就原封轉,絕不擋線路。"""
    if SELF_PROBE_SUPPRESS and _should_suppress_to_center(rec.get("code")):
        return
    out = frame
    if (HW_STATUS_FIX and rec.get("cks_ok")
            and rec.get("code") in HW_STATUS_FIX_CODES
            and isinstance(rec.get("addr"), int) and isinstance(rec.get("seq"), int)):
        try:
            info = _unstuff(frame[7:-3])
            if len(info) >= 4:
                hs = ((info[2] << 8) | info[3]) ^ HW_STATUS_FIX_MASK
                info = info[:2] + bytes(((hs >> 8) & 0xFF, hs & 0xFF)) + info[4:]
                out = build_frame(rec["addr"], rec["seq"], info)
        except Exception:
            out = frame
    _tee_to_center(out)


def _send_to_center(frame: bytes) -> bool:
    """把「我方自己產生的」完整 TC3 碼框上報中央(不是轉發控制器的)。
    🛑 這是「不要寫死純通透」的鉤子 —— 我們自己控制號誌後,可用這個把結果/狀態
    主動回報中央,而不必等控制器的回報。與 _tee_to_center 共用 _center_tx_lock。
    中央未連線回 False。frame 必須是已組好(build_frame)的完整碼框。"""
    cs = _center_sock_ref.get("sock")
    if cs is None:
        return False
    try:
        with _center_tx_lock:
            cs.sendall(frame)
        _center_state["to_center_bytes"] += len(frame)
        return True
    except Exception:
        _close_center()
        return False


def _center_relay_loop() -> None:
    """中央電腦透明中繼 server。中央連進來 → 讀它的下傳原封轉給控制器,同時側錄。
    控制器→中央方向由 _recorder_loop 的 _tee_to_center 負責。"""
    print(f"📶 [signal-tc3] 中央中繼啟動,聽 {CENTER_LISTEN_HOST}:{CENTER_LISTEN_PORT}",
          flush=True)
    srv = None
    while not shutdown_event.is_set():
        try:
            if srv is None:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((CENTER_LISTEN_HOST, CENTER_LISTEN_PORT))
                srv.listen(1)
                srv.settimeout(1.0)
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            _close_center()                 # 新中央連入 → 取代舊的(MaxConnect=1)
            conn.settimeout(3.0)
            _center_sock_ref["sock"] = conn
            _center_state.update({"connected": True, "peer": f"{addr[0]}:{addr[1]}",
                                  "since": time.time()})
            print(f"📶 [signal-tc3] 中央已連入 {addr[0]}:{addr[1]}", flush=True)
            try:
                add_log("info", f"中央電腦連入中繼 {addr[0]}:{addr[1]}", "signal")
            except Exception:
                pass
            buf = b""
            while not shutdown_event.is_set():
                try:
                    d = conn.recv(4096)
                except socket.timeout:
                    continue
                if not d:
                    break
                # 中央→控制器:原封轉發(佔據那條)。控制器沒連上就丟棄(透明:等同源斷)。
                _controller_send(d)
                _center_state["from_center_bytes"] += len(d)
                # 側錄中央下傳的 frame(設定/查詢),標 src=center
                buf += d
                while True:
                    i = buf.find(b"\xaa\xbb")
                    if i < 0:
                        if len(buf) > 65536:
                            buf = b""
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
                    rec["src"] = "center"       # 來源:中央下傳(下行)
                    with _lock:
                        _center_state["center_frames"] += 1
                        _frames.append(rec)
                        _enqueue_frame(rec)     # 持久化中央下傳的指令
                        if rec.get("cks_ok") and rec.get("code"):
                            _coverage[rec["code"]] += 1
            _close_center()
            print("📶 [signal-tc3] 中央連線結束", flush=True)
        except Exception as exc:
            print(f"📶 [signal-tc3] 中央中繼錯誤: {exc}", flush=True)
            _close_center()
            if srv is not None:
                try:
                    srv.close()
                except Exception:
                    pass
            srv = None
            if shutdown_event.wait(2.0):
                break
    _close_center()
    if srv is not None:
        try:
            srv.close()
        except Exception:
            pass
    print("📶 [signal-tc3] 中央中繼結束", flush=True)


_center_thread: Optional[threading.Thread] = None


def start_center_relay() -> None:
    """啟動中央中繼 server(opt-in)。SIGNAL_TC3_CENTER_RELAY=1 才會真的起。"""
    global _center_thread
    if not CENTER_RELAY_ENABLED:
        return
    if _center_thread is not None and _center_thread.is_alive():
        return
    _center_thread = threading.Thread(target=_center_relay_loop, daemon=True,
                                      name="signal-tc3-center")
    _center_thread.start()


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
        ph = rec.get("phase") or {}
        # 🛑 StepSec 是「這一步的總長」,不是剩餘。控制器每次換步階才送一框,
        #    所以要用「總長 − 資料齡」現算剩餘,否則畫面會凍在總長不倒數。
        step_total = ph.get("step_sec")
        remain = None
        if isinstance(step_total, (int, float)) and r_age is not None:
            remain = max(0, int(round(step_total - r_age)))
        intersections.append({
            "addr": a,
            "addr_hex": f"0x{a:04X}",
            "name": _addr_name(a),
            "phase": rec.get("phase"),
            "age_sec": round(r_age, 2) if r_age is not None else None,
            "step_total_sec": step_total,       # 這一步的總長
            "remain_sec": remain,               # 現算的剩餘(倒數)
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
        "center": dict(_center_state),      # 中央中繼:是否啟用/中央連入/雙向流量
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
            with _ctrl_tx_lock:          # 與中央中繼轉發共用,避免交錯寫壞 frame
                sock.sendall(item["frame"])
            _seq_next["n"] = item["seq"]
            rec["ok"] = True
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:160]

    _sent_log.append(rec)
    # 統一表 B:我方送出的框也持久化(src=self + 操作者),進同一條 TC3 通訊紀錄。
    if rec["ok"]:
        _enqueue_frame({
            "ts": rec["ts"], "src": "self", "code": rec["code"],
            "seq": rec["seq"], "addr": rec["addr"], "len": len(item["frame"]),
            "cks_ok": True, "raw": rec["raw"], "user": rec["user"],
        })
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


# ── 時制計畫比對:5FC5(資料庫回報)+5FC4(基本參數回報) 合併成分組表 ──────────
# 參考號控中心「時制計畫下載」介面:依時制編號分組、每分相一列。
# 現場端 = 向控制器查到的即時值(從 query-log 取最新);
# 中心端 = 使用者按「設為基準」存下的參考版(signal_plan_baseline)。
# 🛑 唯讀 + 只寫我們自己的基準表,不對號誌控制器下傳任何位元組。
_plan_baseline_ready = False


def _plan_baseline_db():
    global _plan_baseline_ready
    conn = _sqlite3.connect(_QDB_PATH, timeout=20)
    conn.execute("PRAGMA busy_timeout=20000")
    if not _plan_baseline_ready:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_plan_baseline (
                plan_id INTEGER PRIMARY KEY, json TEXT, ts REAL, user TEXT
            )""")
        conn.commit()
        _plan_baseline_ready = True
    return conn


def _fmap(fields) -> dict:
    """[{name,value,desc}] → {name: value}。"""
    return {f.get("name"): f.get("value") for f in (fields or [])}


def _latest_reply_by_plan(reply_code: str) -> dict:
    """撈某回報碼、每個 PlanID 的最新一筆,重新解碼 raw。
    來源 = query-log(前端配對存的) + signal_frames(側錄/中繼/自我查詢抄到的),兩者合併取最新。
    回 {plan_id: {"vals": {...}, "ts": ts}}。"""
    out: dict = {}
    rows: list = []
    rc = reply_code.upper()
    try:
        conn = _query_db()
        rows += conn.execute(
            "SELECT ts, raw FROM signal_query_log WHERE reply_code=?", (rc,)).fetchall()
        conn.close()
    except Exception:
        pass
    try:
        conn = _frames_db()
        rows += conn.execute(
            "SELECT ts, raw FROM signal_frames WHERE code=? AND cks_ok=1", (rc,)).fetchall()
        conn.close()
    except Exception:
        pass
    rows.sort(key=lambda r: r[0] or 0, reverse=True)   # ts DESC,第一筆即最新
    for ts, raw in rows:
        if not raw:
            continue
        fields = _decode_fields(reply_code.upper(), raw)
        if not fields:
            continue
        vals = _fmap(fields)
        pid = vals.get("PlanID")
        if pid is None:
            continue
        pid = int(pid)
        if pid not in out:          # rows 已按 ts DESC,第一筆即最新
            out[pid] = {"vals": vals, "ts": ts}
    return out


def _merge_timing_plans() -> list:
    """合併 5FC5(綠燈/週期/時差) + 5FC4(最短綠/最長綠/黃/全紅/行人) 成分組表。"""
    db5 = _latest_reply_by_plan("5FC5")     # 資料庫回報
    db4 = _latest_reply_by_plan("5FC4")     # 基本參數回報
    plans: list = []
    for pid in sorted(set(db5) | set(db4)):
        g = db5.get(pid, {}).get("vals", {})
        b = db4.get(pid, {}).get("vals", {})
        greens = g.get("Green") or []
        subs = b.get("SubPhase") or []
        n = max(len(greens), len(subs),
                int(g.get("SubPhaseCount") or 0), int(b.get("SubPhaseCount") or 0))
        phases = []
        for i in range(n):
            sp = subs[i] if i < len(subs) and isinstance(subs[i], dict) else {}
            phases.append({
                "idx": i + 1,
                "green": greens[i] if i < len(greens) else None,
                "min_green": sp.get("MinGreen"),
                "max_green": sp.get("MaxGreen"),
                "yellow": sp.get("Yellow"),
                "all_red": sp.get("AllRed"),
                "ped_flash": sp.get("PedGreenFlash"),
                "ped_red": sp.get("PedRed"),
            })
        plans.append({
            "plan_id": pid,
            "direct": g.get("Direct"),
            "phase_order": g.get("PhaseOrder"),
            "cycle": g.get("CycleTime"),
            "offset": g.get("Offset"),
            "sub_phase_count": n,
            "phases": phases,
            "ts_green": db5.get(pid, {}).get("ts"),
            "ts_base": db4.get(pid, {}).get("ts"),
        })
    return plans


@router.get("/timing-plans", summary="時制計畫比對(現場查到的 vs 中心基準)")
async def timing_plans(_user=Depends(get_current_user)):
    field_plans = _merge_timing_plans()
    baseline: list = []
    try:
        conn = _plan_baseline_db()
        rows = conn.execute(
            "SELECT json FROM signal_plan_baseline ORDER BY plan_id").fetchall()
        conn.close()
        for (js,) in rows:
            try:
                baseline.append(_json_ctl.loads(js))
            except Exception:
                pass
    except Exception:
        pass
    return {"field": field_plans, "baseline": baseline}


@router.post("/timing-plans/baseline", summary="把目前現場查到的時制計畫存為中心基準")
async def timing_plans_set_baseline(_user=Depends(get_current_user)):
    """快照當下現場端 → 中心基準(只寫我們自己的 DB,不下傳控制器)。"""
    user = getattr(_user, "username", None) or str(_user)
    plans = _merge_timing_plans()
    if not plans:
        raise HTTPException(status_code=400, detail="目前沒有現場查詢結果可存為基準,請先查詢時制計畫")
    try:
        conn = _plan_baseline_db()
        now = time.time()
        for p in plans:
            conn.execute(
                "INSERT OR REPLACE INTO signal_plan_baseline (plan_id,json,ts,user) "
                "VALUES (?,?,?,?)",
                (int(p["plan_id"]), _json_ctl.dumps(p, ensure_ascii=False), now, user))
        conn.commit()
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"存基準失敗: {exc}")
    return {"ok": True, "count": len(plans)}


# ── 訊框持久化:背景 writer + 篩選查詢(監看要能存起來、篩選、重啟後還在) ──────
_frame_db_ready = False


def _frames_db():
    global _frame_db_ready
    conn = _sqlite3.connect(_QDB_PATH, timeout=20)
    conn.execute("PRAGMA busy_timeout=20000")
    if not _frame_db_ready:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, src TEXT, code TEXT, seq INTEGER, addr INTEGER,
                len INTEGER, cks_ok INTEGER, raw TEXT, user TEXT
            )""")
        # 舊表補 user 欄(統一表 B:我方送出也存,帶操作者)
        try:
            conn.execute("ALTER TABLE signal_frames ADD COLUMN user TEXT")
        except Exception:
            pass
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sf_ts ON signal_frames(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sf_src ON signal_frames(src)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sf_code ON signal_frames(code)")
        conn.commit()
        _frame_db_ready = True
    return conn


def _frame_writer_loop() -> None:
    """單一 writer:把 _frame_q 的 frame 批次寫進 signal_frames,順便定期清過期資料。
    批次+單連線,避免每框一次 commit 的 IO;佇列空就 block 等,不忙迴圈。"""
    print("📶 [signal-tc3] 訊框持久化 writer 啟動", flush=True)
    conn = None
    last_retain = 0.0
    while not shutdown_event.is_set():
        try:
            if conn is None:
                conn = _frames_db()
            # 先 block 等第一筆(最多 1s),再把佇列現有的一次抽乾(上限 500 筆/批)
            batch = []
            try:
                batch.append(_frame_q.get(timeout=1.0))
            except _queue.Empty:
                pass
            while len(batch) < 500:
                try:
                    batch.append(_frame_q.get_nowait())
                except _queue.Empty:
                    break
            if batch:
                conn.executemany(
                    "INSERT INTO signal_frames (ts,src,code,seq,addr,len,cks_ok,raw,user) "
                    "VALUES (:ts,:src,:code,:seq,:addr,:len,:cks_ok,:raw,:user)", batch)
                conn.commit()
            # 保存政策:每 ~5 分鐘清一次過期(預設 3 天)
            now = time.time()
            if now - last_retain > 300:
                last_retain = now
                cutoff = now - FRAME_RETAIN_DAYS * 86400
                conn.execute("DELETE FROM signal_frames WHERE ts < ?", (cutoff,))
                conn.commit()
        except Exception as exc:
            print(f"📶 [signal-tc3] 訊框 writer 錯誤: {exc}", flush=True)
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            conn = None
            if shutdown_event.wait(2.0):
                break
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    print("📶 [signal-tc3] 訊框持久化 writer 結束", flush=True)


_frame_writer_thread: Optional[threading.Thread] = None


def start_frame_writer() -> None:
    global _frame_writer_thread
    if not FRAME_PERSIST:
        return
    if _frame_writer_thread is not None and _frame_writer_thread.is_alive():
        return
    _frame_writer_thread = threading.Thread(target=_frame_writer_loop, daemon=True,
                                            name="signal-tc3-framewriter")
    _frame_writer_thread.start()


@router.get("/frames/log", summary="持久化訊框(可依方向/碼/CKS 篩選,重啟後還在)")
async def frames_log(limit: int = 200, src: str = "", code: str = "",
                     cks: str = "", _user=Depends(get_current_user)):
    """src: controller/center/空;code: 訊息碼(如 5FC5);cks: ok/bad/空。最新在前。"""
    n = max(1, min(2000, int(limit or 200)))
    where = []
    params: list = []
    if src in ("controller", "center", "self"):
        where.append("src=?")
        params.append(src)
    if code:
        where.append("code=?")
        params.append(code.upper())
    if cks == "ok":
        where.append("cks_ok=1")
    elif cks == "bad":
        where.append("cks_ok=0")
    sql = "SELECT ts,src,code,seq,addr,len,cks_ok,raw,user FROM signal_frames"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(n)
    try:
        conn = _frames_db()
        rows = conn.execute(sql, params).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM signal_frames").fetchone()[0]
        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"讀取失敗: {exc}")
    out = [{
        "ts": ts, "src": src_, "code": code_, "seq": seq, "addr": addr,
        "len": ln, "cks_ok": bool(cks_ok), "raw": raw, "user": user or "",
    } for ts, src_, code_, seq, addr, ln, cks_ok, raw, user in rows]
    return {"count": len(out), "total": total, "frames": out}


@router.get("/frames/decode", summary="解一個訊框的欄位(給 log 展開讀,可讀性)")
async def frames_decode_one(code: str, raw: str, _user=Depends(get_current_user)):
    """給前端 log 展開時呼叫:把該框 raw 解成 [{name,value,desc}]。解不出回 None。"""
    fields = _decode_fields((code or "").upper(), raw or "")
    # 附上訊息名稱,表頭好標
    name = ""
    for m in load_command_schemas().values():
        if m.get("code") == (code or "").upper():
            name = m.get("name", "")
            break
    return {"code": (code or "").upper(), "name": name, "fields": fields}


def _send_query_to_controller(code: str, params: bytes, user: str) -> bool:
    """我方主動送一則查詢給控制器(走 _controller_send,與中央/號控共用 tx 鎖)。
    記下預期回報碼(供抑制轉發中央),並把送出框存進通訊紀錄(src=self)。"""
    try:
        dev = int(code[:2], 16)
        cmd = int(code[2:], 16)
    except Exception:
        return False
    info = bytes((dev, cmd)) + (params or b"")
    addr = _target_addr()
    if addr is None:
        addr = 0xFFFF
    seq = (int(_seq_next.get("n", 0)) + 1) & 0xFF
    try:
        frame = build_frame(addr, seq, info)
    except Exception:
        return False
    if not _controller_send(frame):
        return False
    _seq_next["n"] = seq
    reply_code = f"{dev:02X}{(cmd + 0x80):02X}"   # 查詢 4x/5x/6x → 回報 Cx/Dx/Ex
    _note_expected_reply(reply_code, 1)
    ts = time.time()
    raw_hex = frame.hex(" ").upper()
    _sent_log.append({"ts": ts, "user": user, "code": code,
                      "kind": _kind_of(cmd, dev), "addr": addr, "seq": seq,
                      "raw": raw_hex, "ok": True, "error": ""})
    _enqueue_frame({"ts": ts, "src": "self", "code": code, "seq": seq,
                    "addr": addr, "len": len(frame), "cks_ok": True,
                    "raw": raw_hex, "user": user})
    return True


_self_probe_busy = {"on": False}


def _run_self_probe(user: str, plan_lo: int, plan_hi: int) -> None:
    """背景抄錄:控制策略/目前時制/硬體狀態 + 逐計畫(plan_lo~plan_hi) 5F45 資料庫 +
    5F44 基本參數。查詢多,間隔送,回報進 signal_frames(抑制轉中央)。"""
    try:
        for code in ("5F40", "5F48", "0F41"):
            if shutdown_event.is_set():
                return
            _send_query_to_controller(code, b"", user)
            time.sleep(0.12)
        for pid in range(plan_lo, plan_hi + 1):
            if shutdown_event.is_set():
                return
            p = bytes((pid & 0xFF,))
            _send_query_to_controller("5F45", p, user)   # 時制計畫資料庫(綠燈/週期/時差)
            time.sleep(0.12)
            _send_query_to_controller("5F44", p, user)   # 基本參數(最短綠/最長綠/黃/全紅/行人)
            time.sleep(0.12)
        _log("info", f"自我抄錄完成:控制策略/目前時制/硬體 + 計畫 {plan_lo}~{plan_hi}")
    finally:
        _self_probe_busy["on"] = False


@router.post("/control/self-probe", summary="自我查詢比對:主動抄錄控制器 控制策略/時制計畫(全)/基本參數/硬體狀態")
def control_self_probe(_user=Depends(get_current_user), plan_lo: int = 1, plan_hi: int = 40):
    """一鍵抄錄控制器:5F40(控制策略)/5F48(目前時制計畫)/0F41(硬體狀態) +
    逐計畫 5F45(資料庫)+5F44(基本參數),plan_lo~plan_hi 預設 1~40(全部)。
    背景執行緒送(查詢多),即時返回;回報進通訊紀錄+補滿時制計畫表,預設不轉發中央。"""
    if not CONTROL_ENABLED:
        raise HTTPException(status_code=403, detail="號控未啟用(SIGNAL_TC3_CONTROL)")
    if _sock_ref.get("sock") is None:
        raise HTTPException(status_code=409, detail="號誌通道未連線,無法查詢")
    if _self_probe_busy["on"]:
        return {"ok": False, "busy": True, "msg": "上一輪自我抄錄還在跑,請稍候"}
    lo = max(0, min(48, int(plan_lo)))
    hi = max(lo, min(48, int(plan_hi)))
    user = getattr(_user, "username", None) or str(_user)
    _self_probe_busy["on"] = True
    threading.Thread(target=_run_self_probe, args=(user, lo, hi), daemon=True,
                     name="signal-selfprobe").start()
    n = 3 + (hi - lo + 1) * 2
    return {"ok": True, "started": True, "plans": [lo, hi], "queries": n,
            "msg": f"開始抄錄:控制策略/目前時制/硬體 + 計畫 {lo}~{hi}（約 {n} 則,背景送）"}
