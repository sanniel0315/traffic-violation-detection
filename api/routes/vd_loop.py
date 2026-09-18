"""VD(車輛偵測器,線圈)逐分鐘資料收集 —— 我方攝影機計數的地面實況。

用途:線圈計數是獨立於攝影機的量測,拿來驗證我方車流計數與校準用的到達率。
    (2026-09-18 使用者同意接入 WN 下匝道線圈 VD-N8-E-9-O-WN-21-Loop。)

協定:《高快速公路交通控制系統中央電腦軟體雲端化通訊協定》
    3.2 碼框(文件頁 6~8):DLE(10) SOH(01) SEQ ADDR(2) LEN(2) 表頭LRC TEXT… LRC
        表頭 LRC = XOR(ADDR, LEN);整框 LRC = 除自己以外全部 byte XOR。
        🛑 **沒有 DLE 跳脫、沒有 DLE ETX 框尾**,框長完全靠 LEN。資料裡出現 0x10
           是一般資料(例如表頭 LRC=16 後面接指令碼 10),不可以當成框邊界。
    ACK:DLE ACK(06) SEQ ADDR(2) LRC —— 帶回同一個 SEQ。
        設備收不到 ACK 會重送最多 3 次,之後判通訊故障,所以**一定要回**。
    10H 週期性資料(表 4.2-2,文件頁 95~99):長度 = 10 + 車道數 × 12
        response_type(1) hardware_status(4) day hour minute lane_count
        每車道:小型/大型/聯結 各 (流量, 速度 km/h, 車長 0.1m) + 車間距(2, 0.1s) + 佔有率(%)

🛑 只收資料、只回 ACK,**不送任何查詢或設定命令** —— 這台設備的主人是中央,
   我方只是旁聽者;改它的設定(例如回報週期 03H)會影響中央拿到的資料。
🛑 設備時鐘會偏:2026-09-18 實測 13:16:00 收到的框時間戳是 12:46(慢 30 分鐘)。
   對照攝影機一律用**我方收到的時刻**,設備時間戳只記錄、並回報偏差。

設定:SIGNAL_VD_DEVICES="名稱@host:port,名稱@host:port"(空 = 不啟動)
"""
from __future__ import annotations

import os
import socket
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.routes.auth import get_current_user

router = APIRouter(prefix="/api/vd", tags=["vd"])

_DB = os.getenv("SIGNAL_VD_DB", "data/signal_shadow.db")
_DEVICES_RAW = os.getenv("SIGNAL_VD_DEVICES", "") or ""

_state: dict = {}          # name -> {connected, last_rx, frames, bad_lrc, acks, errors, clock_offset_sec}
_threads: dict = {}
_stop = threading.Event()


def parse_devices(raw: str) -> list:
    """'名稱@host:port,…' → [(名稱, host, port)]。格式錯的項目略過。"""
    out = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item or "@" not in item or ":" not in item.split("@", 1)[1]:
            continue
        name, hp = item.split("@", 1)
        host, port = hp.rsplit(":", 1)
        try:
            out.append((name.strip(), host.strip(), int(port)))
        except ValueError:
            continue
    return out


def lrc(bs: bytes) -> int:
    x = 0
    for b in bs:
        x ^= b
    return x


def split_frames(buf: bytes) -> tuple:
    """從緩衝區切出完整資料框。回 (frames, 剩餘緩衝, 丟棄的雜訊 byte 數)。

    只認 DLE SOH 開頭的資料框;長度由表頭 LEN 決定(本協定沒有框尾)。
    表頭 LRC 不對就只丟掉這個 DLE,往後找下一個起點 —— 不可整段丟,
    否則一個壞 byte 會吃掉後面好幾筆正常資料。
    """
    frames, junk = [], 0
    while True:
        i = buf.find(b"\x10\x01")
        if i < 0:
            junk += max(0, len(buf) - 1)
            return frames, buf[-1:] if buf.endswith(b"\x10") else b"", junk
        if i > 0:
            junk += i
            buf = buf[i:]
        if len(buf) < 8:
            return frames, buf, junk
        if lrc(buf[3:7]) != buf[7]:
            buf = buf[1:]
            junk += 1
            continue
        need = 8 + int.from_bytes(buf[5:7], "big") + 1
        if len(buf) < need:
            return frames, buf, junk
        frames.append(buf[:need])
        buf = buf[need:]


def ack_for(frame: bytes) -> bytes:
    """對資料框回的 ACK:DLE ACK SEQ ADDR(2) LRC。"""
    a = bytes([0x10, 0x06, frame[2]]) + frame[3:5]
    return a + bytes([lrc(a)])


def seq_step(prev: int, cur: int) -> int:
    """兩框序號的前進量。1 = 連續;0 = 重送同一框;>1 = 中間漏了 (step−1) 框。

    協定 3.2:終端設備自行指定的 SEQ 落在 128~255,所以序號在 0x80~0xFF 這 128 格裡循環
    (FF 的下一個是 80,不是 00)。不在這個範圍的序號就用一般的 256 循環算。
    """
    if 0x80 <= prev <= 0xFF and 0x80 <= cur <= 0xFF:
        return (cur - prev) % 128
    return (cur - prev) % 256


def decode_10h(text: bytes) -> Optional[dict]:
    """10H 週期性資料。長度不符文件規定就回 None(不猜)。"""
    if len(text) < 10 or text[0] != 0x10:
        return None
    n = text[9]
    if len(text) != 10 + n * 12:
        return None
    lanes = []
    for i in range(n):
        L = text[10 + i * 12: 22 + i * 12]
        fault = all(b == 0xFF for b in L)          # 文件:線圈/偵測器故障時整車道填 FF
        lanes.append({
            "lane": i + 1, "fault": fault,
            "small_n": None if fault else L[0], "small_kmh": None if fault else L[1],
            "small_len_m": None if fault else L[2] / 10.0,
            "large_n": None if fault else L[3], "large_kmh": None if fault else L[4],
            "large_len_m": None if fault else L[5] / 10.0,
            "trailer_n": None if fault else L[6], "trailer_kmh": None if fault else L[7],
            "headway_s": None if fault else int.from_bytes(L[9:11], "big") / 10.0,
            "occ_pct": None if fault else L[11],
        })
    return {"response_type": text[1], "hw_status": text[2:6].hex(),
            "day": text[6], "hour": text[7], "minute": text[8], "lanes": lanes}


def device_clock_offset(recv_ts: float, day: int, hour: int, minute: int) -> Optional[float]:
    """設備時間戳(只有日/時/分)與我方收到時刻的差(秒,正值 = 設備慢)。

    跨月邊界時以「離收到時刻最近的那一天」為準。
    """
    r = datetime.fromtimestamp(recv_ts)
    best = None
    for dm in (-1, 0, 1):
        base = (r.replace(day=1) + timedelta(days=32 * dm)).replace(day=1) if dm else r
        try:
            cand = base.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            continue
        d = (r - cand).total_seconds()
        if best is None or abs(d) < abs(best):
            best = d
    return best


def _conn():
    c = sqlite3.connect(_DB, timeout=10)
    c.execute("""CREATE TABLE IF NOT EXISTS vd_minute (
        id INTEGER PRIMARY KEY AUTOINCREMENT, device TEXT, recv_ts REAL, recv_iso TEXT,
        dev_time TEXT, clock_offset_sec REAL, hw_status TEXT, lane INTEGER, fault INTEGER,
        small_n INTEGER, small_kmh INTEGER, small_len_m REAL,
        large_n INTEGER, large_kmh INTEGER, large_len_m REAL,
        trailer_n INTEGER, trailer_kmh INTEGER, headway_s REAL, occ_pct INTEGER, raw TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_vd_minute_ts ON vd_minute(device, recv_ts)")
    return c


def _store(name: str, recv_ts: float, frame: bytes, d: dict, offset: Optional[float]) -> None:
    c = _conn()
    try:
        for L in d["lanes"]:
            c.execute(
                "INSERT INTO vd_minute(device,recv_ts,recv_iso,dev_time,clock_offset_sec,hw_status,"
                "lane,fault,small_n,small_kmh,small_len_m,large_n,large_kmh,large_len_m,"
                "trailer_n,trailer_kmh,headway_s,occ_pct,raw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (name, recv_ts, datetime.fromtimestamp(recv_ts).isoformat(timespec="seconds"),
                 "%02d日 %02d:%02d" % (d["day"], d["hour"], d["minute"]), offset, d["hw_status"],
                 L["lane"], 1 if L["fault"] else 0,
                 L["small_n"], L["small_kmh"], L["small_len_m"],
                 L["large_n"], L["large_kmh"], L["large_len_m"],
                 L["trailer_n"], L["trailer_kmh"], L["headway_s"], L["occ_pct"],
                 frame.hex(" ").upper()))
        c.commit()
    finally:
        c.close()


_SPOOL = os.getenv("SIGNAL_VD_SPOOL", "data/vd_spool.jsonl")


def _log(name: str, msg: str) -> None:
    """異常一律進系統日誌 —— 2026-09-18 14:06 漏收一框,當時沒有任何紀錄可查。"""
    print("[vd][%s] %s" % (name, msg), flush=True)


def _spool(name: str, recv_ts: float, frame: bytes) -> None:
    """寫庫失敗時先存原始框,之後補寫。🛑 寫入失敗不可以丟資料,也不可以因此斷線。"""
    import json as _j
    with open(_SPOOL, "a", encoding="utf-8") as f:
        f.write(_j.dumps({"device": name, "recv_ts": recv_ts, "raw": frame.hex()}) + "\n")


def _flush_spool(name: str) -> int:
    """把暫存檔裡屬於這台設備的框補寫進資料庫。回補寫筆數。"""
    import json as _j
    if not os.path.exists(_SPOOL):
        return 0
    keep, done = [], 0
    with open(_SPOOL, encoding="utf-8") as f:
        lines = f.readlines()
    for ln in lines:
        try:
            r = _j.loads(ln)
        except Exception:
            continue
        if r.get("device") != name:
            keep.append(ln)
            continue
        fr = bytes.fromhex(r["raw"])
        d = decode_10h(fr[8:-1])
        if d is None:
            continue
        try:
            _store(name, r["recv_ts"], fr, d,
                   device_clock_offset(r["recv_ts"], d["day"], d["hour"], d["minute"]))
            done += 1
        except Exception:
            keep.append(ln)
    with open(_SPOOL, "w", encoding="utf-8") as f:
        f.writelines(keep)
    return done


def _run(name: str, host: str, port: int) -> None:
    st = _state.setdefault(name, {"host": host, "port": port, "connected": False, "last_rx": None,
                                  "frames": 0, "bad_lrc": 0, "junk_bytes": 0, "acks": 0,
                                  "errors": 0, "last_error": "", "clock_offset_sec": None,
                                  "other_codes": {}, "store_fail": 0, "spooled": 0,
                                  "seq_gaps": 0, "seq_gap_last": None, "last_seq": None})
    backoff = 5.0
    while not _stop.is_set():
        s = None
        try:
            s = socket.create_connection((host, port), timeout=10)
            s.settimeout(5)
            st["connected"] = True
            backoff = 5.0
            _log(name, "已連線 %s:%d" % (host, port))
            try:
                n = _flush_spool(name)
                if n:
                    _log(name, "補寫暫存的 %d 框" % n)
            except Exception as exc:
                _log(name, "補寫暫存失敗:%s" % exc)
            buf = b""
            while not _stop.is_set():
                try:
                    data = s.recv(4096)
                except socket.timeout:
                    # 每分鐘一筆;超過 3 分鐘沒資料就重連(連線可能半開)
                    if st["last_rx"] and time.time() - st["last_rx"] > 180:
                        raise ConnectionError("超過 3 分鐘沒有收到資料")
                    continue
                if not data:
                    raise ConnectionError("對方關閉連線")
                buf += data
                frames, buf, junk = split_frames(buf)
                st["junk_bytes"] += junk
                for fr in frames:
                    now = time.time()
                    st["last_rx"] = now
                    if lrc(fr[:-1]) != fr[-1]:
                        st["bad_lrc"] += 1          # 不回 ACK,讓設備依協定重送
                        _log(name, "整框 LRC 錯,不回 ACK 等重送:%s" % fr.hex(" "))
                        continue
                    st["frames"] += 1
                    s.sendall(ack_for(fr))
                    st["acks"] += 1
                    # 序號連續性:設備每送一框 +1。
                    # 🛑 跳號 = 設備有送、我們沒存到(14:06、14:10 就是這樣被發現的)
                    seq = fr[2]
                    if st["last_seq"] is not None:
                        step = seq_step(st["last_seq"], seq)
                        if step == 0:
                            _log(name, "序號 %02X 重複(設備重送,已回 ACK,不重複存)" % seq)
                            continue
                        if step > 1:
                            st["seq_gaps"] += step - 1
                            st["seq_gap_last"] = time.time()
                            _log(name, "序號由 %02X 跳到 %02X,漏收 %d 框" % (st["last_seq"], seq, step - 1))
                    st["last_seq"] = seq
                    text = fr[8:-1]
                    d = decode_10h(text)
                    if d is None:
                        code = "%02X" % text[0] if text else "??"
                        st["other_codes"][code] = st["other_codes"].get(code, 0) + 1
                        _log(name, "非 10H 或長度不符,只記錄:%s" % fr.hex(" "))
                        continue
                    off = device_clock_offset(now, d["day"], d["hour"], d["minute"])
                    st["clock_offset_sec"] = off
                    # 🛑 寫庫失敗不可以讓整條連線斷掉 —— 那一框會就此遺失。先暫存,之後補寫。
                    try:
                        _store(name, now, fr, d, off)
                    except Exception as exc:
                        st["store_fail"] += 1
                        _log(name, "寫庫失敗(%s),先暫存原始框" % exc)
                        try:
                            _spool(name, now, fr)
                            st["spooled"] += 1
                        except Exception as exc2:
                            _log(name, "暫存也失敗,這一框遺失:%s | %s" % (exc2, fr.hex(" ")))
        except Exception as exc:
            st["errors"] += 1
            st["last_error"] = "%s: %s" % (type(exc).__name__, exc)
            _log(name, "連線中斷:%s(%.0f 秒後重連)" % (st["last_error"], backoff))
        finally:
            st["connected"] = False
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
        _stop.wait(backoff)
        backoff = min(backoff * 2, 120.0)


def start_vd() -> list:
    """依 SIGNAL_VD_DEVICES 啟動收集執行緒(冪等)。回啟動的設備名稱。"""
    started = []
    for name, host, port in parse_devices(_DEVICES_RAW):
        t = _threads.get(name)
        if t is not None and t.is_alive():
            continue
        t = threading.Thread(target=_run, args=(name, host, port), daemon=True, name="vd-" + name)
        _threads[name] = t
        t.start()
        started.append(name)
    return started


def _camera_minutes(since_utc: str, until_utc: str) -> dict:
    """我方攝影機逐分鐘計數(下匝道那一相的基準測點與車道),key = 本地時間 'YYYY-MM-DD HH:MM'。

    🛑 鍵要含日期:只用 HH:MM 時,查「昨日全日 + 今天」會把兩天同一分鐘加在一起。

    🛑 與 signal_eval 同一口徑:排除進出線事件(IN/OUT/EXIT),那是 OPAC 的,不當通過量。
    🛑 traffic_events.created_at 是 UTC,這裡轉本地時間再對齊。
    """
    try:
        from detection.signal_eval import FLOW_EXCLUDE_DIRECTIONS
        from detection.signal_timing_lookup import phase_of_role, phase_role
        from api.routes.signal_shadow import _phase_lanes
        off = phase_of_role("off_ramp")
        cc = str((phase_role(off) or {}).get("constraint_camera") or "")
        cam = int(cc[2:]) if cc.startswith("ID") and cc[2:].isdigit() else None
        lanes = _phase_lanes(off).get(cam) or [] if cam else []
        if not cam or not lanes:
            return {"camera": None, "minutes": {}}
        c = sqlite3.connect("file:data/violations.db?mode=ro", uri=True, timeout=10)
        try:
            rows = c.execute(
                "SELECT strftime('%%Y-%%m-%%d %%H:%%M', created_at, 'localtime'), count(*), "
                "sum(CASE WHEN label IN ('truck','heavy_truck','bus','trailer') THEN 1 ELSE 0 END) "
                "FROM traffic_events WHERE camera_id=? AND lane_no IN (%s) AND direction NOT IN (%s) "
                "AND created_at>=? AND created_at<? GROUP BY 1"
                % (",".join(str(int(x)) for x in lanes),
                   ",".join("'%s'" % d for d in FLOW_EXCLUDE_DIRECTIONS)),
                (cam, since_utc, until_utc)).fetchall()
        finally:
            c.close()
        return {"camera": cam, "lanes": lanes,
                "minutes": {k: {"n": n, "large": lg or 0} for k, n, lg in rows}}
    except Exception as exc:
        return {"camera": None, "minutes": {}, "error": str(exc)[:160]}


def completeness(rows: list) -> dict:
    """依存下的框序號,算「設備有送、我們沒存到」幾框、是哪幾分鐘。rows = [(recv_ts, raw), ...]

    🛑 用資料庫裡的序號算,不用記憶體計數:服務一重啟記憶體歸零,
       但重啟正是最容易漏的時刻(14:10 就是)。
    """
    missed, prev = [], None
    for ts, raw in rows:
        try:
            seq = int(str(raw).split()[2], 16)
        except Exception:
            continue
        if prev is not None:
            step = seq_step(prev[1], seq)
            for k in range(1, step):
                missed.append((datetime.fromtimestamp(prev[0]) + timedelta(minutes=k)).strftime("%H:%M"))
        prev = (ts, seq)
    got = len(rows)
    return {"received": got, "missed": len(missed), "missed_minutes": missed[-20:],
            "rate_pct": (round(got / (got + len(missed)) * 100, 1) if got else None)}


@router.get("/status", summary="VD 線圈:連線狀態、時鐘偏差、紀錄與攝影機對照(可查詢區間、可分組)")
def vd_status(device: str = Query("", description="設備名稱;空 = 第一台"),
              minutes: int = Query(60, ge=5, le=10080),
              since: str = Query("", description="起(ISO);給了就蓋過 minutes"),
              until: str = Query("", description="訖(ISO);空 = 現在"),
              bucket: int = Query(1, description="分組分鐘數:1/5/15/60"),
              _user=Depends(get_current_user)):
    """線圈紀錄 + 同一時段我方攝影機的計數,大小車分開對照。

    對照用**我方收到的時刻**:設備在 hh:mm:00 送出的那一筆,算作前一分鐘
    (hh:mm-1)的車流 —— 設備時間戳會偏(實測慢 30 分鐘),不拿來對齊。
    🛑 線圈與攝影機看的是同一條下匝道、同一批車(使用者 2026-09-18),
       差異就是我方計數的誤差,不是位置不同。
    🛑 區間跟著運作統計頁的查詢列走(使用者:「也要有查詢功能」)。
    """
    # 直接呼叫(非經 HTTP)時,預設值會是 FastAPI 的 Query 物件,不是字串/數字
    device = device if isinstance(device, str) else ""
    since = since if isinstance(since, str) else ""
    until = until if isinstance(until, str) else ""
    minutes = minutes if isinstance(minutes, int) else 60
    names = [n for n, _, _ in parse_devices(_DEVICES_RAW)]
    name = device or (names[0] if names else "")
    st = dict(_state.get(name) or {})
    bucket = bucket if bucket in (1, 5, 15, 60) else 1

    def _ep(v):
        try:
            return datetime.fromisoformat(v).timestamp()
        except Exception:
            return None
    now = time.time()
    t_until = _ep(until) if until else now
    t_since = _ep(since) if since else t_until - minutes * 60
    if t_since is None or t_until is None or t_since >= t_until:
        return {"available": False, "reason": "查詢區間不正確", "devices": names}

    rows, comp = [], {}
    try:
        c = _conn()
        comp = completeness(c.execute(
            "SELECT recv_ts, raw FROM vd_minute WHERE device=? AND recv_ts>=? AND recv_ts<? "
            "AND lane=1 ORDER BY recv_ts", (name, t_since + 30, t_until + 30)).fetchall())
        rows = c.execute(
            "SELECT recv_ts,dev_time,clock_offset_sec,fault,small_n,small_kmh,large_n,large_kmh,"
            "trailer_n,headway_s,occ_pct FROM vd_minute WHERE device=? AND recv_ts>=? AND recv_ts<? "
            "ORDER BY recv_ts", (name, t_since + 30, t_until + 30)).fetchall()
        c.close()
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:160], "devices": names}

    utc = lambda t: datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    cam = _camera_minutes(utc(t_since - 120), utc(t_until + 60))
    cm_all = cam.get("minutes") or {}
    has_cam = bool(cam.get("camera"))

    agg: dict = {}
    for (rts, dtime, off, fault, sn, sv, ln, lv, tn, hwy, occ) in rows:
        m = datetime.fromtimestamp(rts) - timedelta(seconds=30)
        m = m.replace(second=0, microsecond=0)
        b0 = m - timedelta(minutes=(m.hour * 60 + m.minute) % bucket)
        a = agg.setdefault(b0, {"minutes": 0, "fault": 0, "vd_small": 0, "vd_large": 0,
                                "cam_small": 0, "cam_large": 0, "occ": [], "kmh": [],
                                "dev_time": dtime, "headway": [], "clock_offset_sec": off})
        a["minutes"] += 1
        if fault:
            a["fault"] += 1
            continue
        a["vd_small"] += sn or 0
        a["vd_large"] += (ln or 0) + (tn or 0)
        cmm = cm_all.get(m.strftime("%Y-%m-%d %H:%M")) or {}
        a["cam_large"] += cmm.get("large", 0) or 0
        a["cam_small"] += (cmm.get("n", 0) or 0) - (cmm.get("large", 0) or 0)
        if occ is not None:
            a["occ"].append(occ)
        if sv:
            a["kmh"].append(sv)
        if hwy:
            a["headway"].append(hwy)

    def pct(c_, v_):
        return round((c_ - v_) / v_ * 100, 1) if v_ else None

    items = []
    for b0 in sorted(agg, reverse=True)[:1500]:
        a = agg[b0]
        vt, ct = a["vd_small"] + a["vd_large"], a["cam_small"] + a["cam_large"]
        items.append({
            "t": b0.strftime("%Y-%m-%d %H:%M"), "minutes": a["minutes"], "fault_minutes": a["fault"],
            "vd_small": a["vd_small"], "vd_large": a["vd_large"], "vd_total": vt,
            "cam_small": a["cam_small"] if has_cam else None,
            "cam_large": a["cam_large"] if has_cam else None,
            "cam_total": ct if has_cam else None,
            "diff_pct": pct(ct, vt) if has_cam else None,
            "occ_pct": round(sum(a["occ"]) / len(a["occ"]), 1) if a["occ"] else None,
            "small_kmh": round(sum(a["kmh"]) / len(a["kmh"])) if a["kmh"] else None,
            "headway_s": round(sum(a["headway"]) / len(a["headway"]), 1) if a["headway"] else None,
            "device_time": a["dev_time"] if bucket == 1 else None,
        })
    tv_s = sum(a["vd_small"] for a in agg.values()); tv_l = sum(a["vd_large"] for a in agg.values())
    tc_s = sum(a["cam_small"] for a in agg.values()); tc_l = sum(a["cam_large"] for a in agg.values())
    summary = {
        "vd_small": tv_s, "vd_large": tv_l, "vd_total": tv_s + tv_l,
        "cam_small": tc_s if has_cam else None, "cam_large": tc_l if has_cam else None,
        "cam_total": (tc_s + tc_l) if has_cam else None,
        "small_pct": pct(tc_s, tv_s) if has_cam else None,
        "large_pct": pct(tc_l, tv_l) if has_cam else None,
        "total_pct": pct(tc_s + tc_l, tv_s + tv_l) if has_cam else None,
        "mae_per_bucket": (round(sum(abs(i["cam_total"] - i["vd_total"]) for i in items) / len(items), 1)
                           if items and has_cam else None),
        "buckets": len(items),
    }
    return {
        "available": True, "device": name, "devices": names, "bucket": bucket,
        "since": datetime.fromtimestamp(t_since).isoformat(timespec="seconds"),
        "until": datetime.fromtimestamp(t_until).isoformat(timespec="seconds"),
        "status": {k: st.get(k) for k in ("host", "port", "connected", "last_rx", "frames",
                                          "bad_lrc", "junk_bytes", "acks", "errors",
                                          "last_error", "clock_offset_sec", "other_codes",
                                          "store_fail", "spooled", "seq_gaps", "seq_gap_last")},
        "completeness": comp,
        "camera": {"camera_id": cam.get("camera"), "lanes": cam.get("lanes"),
                   "note": "下匝道那一相的基準測點,排除進出線事件(IN/OUT/EXIT)"},
        "summary": summary,
        "items": items,
        "note": ("線圈與攝影機看的是同一條下匝道、同一批車,差異就是攝影機計數的誤差。"
                 "對照以我方收到時刻為準;設備時間戳只記錄不對齊。"),
    }
