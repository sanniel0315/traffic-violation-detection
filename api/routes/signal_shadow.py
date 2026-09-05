# -*- coding: utf-8 -*-
"""動態號誌影子模式 —— 我方決策全速運轉，但**只記錄不下發**。

用途（bypass OPAC 的前置驗證）：
    OPAC 正在控制路口時，我方每 N 秒用同一份現場資料算出「換我會怎麼切」，
    與控制器**實際發生的動作**並排記錄。累積夠多之後用成效指標
    （總延滯／排隊／主線回堵次數）比較兩套控制的好壞，而不是比逐筆一致。

🛑 **絕對不下發。** 本模組只讀 congestion 的排隊與 signal_tc3 抄錄的燈態，
   算完寫進 DB 就結束。任何下發都要走 signal_tc3 的 control/prepare +
   control/send，那條路預設關閉（SIGNAL_TC3_CONTROL）。

🛑 為什麼要趁現在做：OPAC 停掉之後就沒有對照組了。將來要接管，
   必須拿得出「我方演算法在真實車流上表現如何」的依據。

實際動作怎麼判斷：我方 signal_tc3 抄錄的 5F03 帶 sub_phase_id，
分相編號一變就是發生了切換（SWITCH），沒變就是 KEEP。
這與 icagent 讀的是同一顆控制器的同一份訊框，所以是權威值。
"""
from __future__ import annotations

import os
import json
import sqlite3 as _sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query

from api.routes.auth import get_current_user
from api.routes.logs import add_log
from api.routes.push import push_alert

router = APIRouter(prefix="/api/signal/shadow", tags=["signal-shadow"])

# 取樣週期(秒)。OPAC 是 5 秒一次決策，對齊它才好比對。
SHADOW_INTERVAL_SEC = float(os.getenv("SIGNAL_SHADOW_INTERVAL_SEC", "5") or 5)
# 回堵判定比例(與決策引擎同一個值,不另外訂一套)
from detection.signal_decision_engine import (  # noqa: E402
    DEFAULT_SPILLBACK_RATIO as DEFAULT_SPILLBACK_RATIO_LOCAL,
)
# 影子模式開關。預設關 —— 要明確開啟才跑（雖然它不下發，仍是背景負載）。
SHADOW_ENABLED = os.getenv("SIGNAL_SHADOW_ENABLED", "0") != "0"
# 分相 → 提供該分相排隊量測的相機 id（對照 ramp_timing_baseline.json 的
# phases[].constraint_camera：分相1=ID3、分相2=ID4）
# 每個分相對應的**所有**相機(不是只有一台)。
# 🛑 2026-09-05 抓到的缺口:現場四台 NE-1 / NE-2 / WN-1 / WN-2(相機 id 2/3/4/5),
#    但先前只用 constraint_camera 各取一台 —— 分相1 只看 NE-2、分相2 只看
#    WN-1,另外兩台的排隊完全沒有進到決策。決策用的 queue_m 因此系統性低估
#    (少看一半的進場),而 switch_gain 直接由排隊車數算出來。
#    基準表 phases[1].cameras 也漏登記 NE-1,一併補上。
def _phase_cams(env_key: str, default: str) -> list:
    raw = os.getenv(env_key, default) or default
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out or [int(default.split(",")[0])]


# 基準表用 "ID2".."ID5" 當相機鍵,但現場設備牌與所有文件用的是
# CCTV-N8-E-9-L-**-SIG 的中段代號。介面上要顯示看得懂的那個,
# 不能丟 ID3 給操作的人自己心算是哪一台。
CAMERA_LABELS = {"ID2": "NE-1", "ID3": "NE-2", "ID4": "WN-1", "ID5": "WN-2",
                 2: "NE-1", 3: "NE-2", 4: "WN-1", 5: "WN-2"}


def camera_label(key) -> str:
    """把基準表的 IDn 或相機 id 轉成現場名稱;不認得就原樣回傳。"""
    return CAMERA_LABELS.get(key, str(key) if key is not None else "—")


PHASE_CAMERAS = {
    1: _phase_cams("SIGNAL_SHADOW_CAMS_PHASE1", "2,3"),   # NE-1, NE-2 上匝道
    2: _phase_cams("SIGNAL_SHADOW_CAMS_PHASE2", "4,5"),   # WN-1, WN-2 下匝道
}
# 🛑 PHASE_CAMERA 維持原意 = 官方時制表的 constraint_camera(該相的「基準測點」),
#    不可以改成「清單第一台」—— 那是語意漂移,會讓依賴它的地方悄悄換了意思。
#    (加聚合時差點就這樣改掉,既有測試 test_phase_camera_mapping_matches_baseline
#     擋下來了。)聚合請用 PHASE_CAMERAS。
PHASE_CAMERA = {
    1: int(os.getenv("SIGNAL_SHADOW_CAM_PHASE1", "3") or 3),
    2: int(os.getenv("SIGNAL_SHADOW_CAM_PHASE2", "4") or 4),
}

# 抄錄器所在的獨立服務(traffic-signal.service)。燈態只有它有。
SIGNAL_DAEMON_URL = os.getenv("SIGNAL_DAEMON_URL", "http://127.0.0.1:8012").rstrip("/")
# 自動回報週期(秒)。影子跑再久，沒人去撈 DB 就等於沒回報 —— 這是實際踩到的問題。
SHADOW_REPORT_SEC = float(os.getenv("SIGNAL_SHADOW_REPORT_SEC", "3600") or 3600)
# 一致率低於此值就推播(只在「有車」的樣本上算，夜間無車不觸發)。
SHADOW_ALERT_RATE = float(os.getenv("SIGNAL_SHADOW_ALERT_RATE", "0.75") or 0.75)
# 一小時內「有車樣本」少於這個數就不評分(車太少，比率沒有意義)。
SHADOW_MIN_ACTIVE = int(os.getenv("SIGNAL_SHADOW_MIN_ACTIVE", "60") or 60)
# congestion_samples / traffic_events 所在(成效報告讀它;存 UTC)
_VIOL_DB = os.getenv("TRAFFIC_VIOL_DB", "data/violations.db")
_DB_PATH = os.getenv("SIGNAL_SHADOW_DB",
                     os.getenv("SIGNAL_TC3_QDB", "data/signal_shadow.db"))
_lock = threading.Lock()
_thread: Optional[threading.Thread] = None
_stop = threading.Event()
_stats = {"started_at": None, "samples": 0, "last_error": "", "last_at": None}
_db_ready = False
_last_report = [0.0]   # 上次自動回報的時刻(list 才好在 _loop 內改;0=尚未從 DB 讀回)
# 影子迴圈追蹤中的綠燈起始時刻與分相。/plan 直接讀這個算 green_elapsed ——
# 先前是去撈「最後一筆樣本的 green_elapsed」,那個值最多差 5 秒,
# 而且分相剛換的瞬間拿到的是「上一個分相」的秒數,控制盤會顯示錯的已亮秒數。
_live_green = {"since": None, "phase": None}
# 現場量測的飽和流(輛/小時),每相一個。None = 還沒量到,用預設值。
# 🛑 為什麼要量:change_cost = 損失時間 × 飽和流 × 損失時間,用教科書的
#    1800 vph 算出來是 12.5;但 2026-09-04 現場實測只有 598~777 vph,
#    用實測值算約 4.75 —— 換相門檻差了 2.6 倍,直接影響「值不值得切」。
#    飽和流是物理量,本來就該量,不是套一個假設。
_measured_sat = {"vph": {}, "ts": None, "source": "default"}
# 🛑 量到的飽和流要落地。2026-09-05 教訓:它原本只存記憶體,每次部署重啟
#    就歸零;重啟後用「最近 6 小時」重量,半夜量出 116/137 vph 低於下限被
#    判無效 → 退回預設 1800 → 之後每小時重量都還是半夜資料,永遠回不來。
#    早尖峰整段跑假設值,跨日比對整組作廢。
_SAT_FILE = os.path.join(os.path.dirname(_DB_PATH) or ".", "signal_saturation.json")
# 只從「綠燈開始時至少排了兩台車」的段量飽和流(見 estimate_saturation)
SAT_MIN_START_QUEUE_M = float(os.getenv("SIGNAL_SAT_MIN_START_QUEUE_M", "14") or 14)
SAT_WINDOW_HOURS = float(os.getenv("SIGNAL_SAT_WINDOW_HOURS", "24") or 24)


def _save_saturation() -> None:
    """量到就寫檔(先寫暫存再 rename,半途斷電不會留下半個 JSON)。"""
    try:
        tmp = _SAT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_measured_sat, f, ensure_ascii=False)
        os.replace(tmp, _SAT_FILE)
    except Exception as e:
        # 🛑 不可以靜默吞掉。第一版就是 json 沒 import,NameError 被 pass 掉,
        #    測試才抓到 —— 正式環境裡這種失敗永遠不會有人看見。
        _stats["last_error"] = f"saturation save failed: {e}"


def _load_saturation() -> bool:
    """啟動時把上次量到的載回來。回傳是否載到有效值。"""
    try:
        with open(_SAT_FILE, encoding="utf-8") as f:
            d = json.load(f)
        vph = {int(k): float(v) for k, v in (d.get("vph") or {}).items()
               if v and SAT_MIN_VPH <= float(v) <= SAT_MAX_VPH}
    except FileNotFoundError:
        return False                        # 第一次啟動本來就沒有檔,不是錯
    except Exception as e:
        _stats["last_error"] = f"saturation load failed: {e}"
        return False
    if not vph:
        return False
    _measured_sat["vph"] = vph
    _measured_sat["ts"] = d.get("ts")
    _measured_sat["source"] = "measured(restored)"
    _measured_sat["window_hours"] = d.get("window_hours")
    _measured_sat["samples"] = d.get("samples")
    return True
# 量出來的值超出這個範圍就不採信(視為量測異常),退回預設值。
# 最大綠:使用者指示固定 100 秒(2026-09-05)。官方時制表寫 210,但那是控制器的
# 硬上限,不是我方要用的操作上限 —— 綠燈放到 100 秒還沒切,紅側已經等太久。
# 設 0 = 退回時制表的值。/plan 的 constants 會標明現在用哪一個。
MAX_GREEN_SEC = float(os.getenv("SIGNAL_MAX_GREEN_SEC", "100") or 0)


def _max_green(pp: dict) -> float:
    """該相的最大綠:有固定設定就用固定值,否則用時制表(再沒有就 210)。"""
    if MAX_GREEN_SEC > 0:
        return MAX_GREEN_SEC
    return float((pp or {}).get("max_green") or 210)


SAT_MIN_VPH = float(os.getenv("SIGNAL_SAT_MIN_VPH", "200") or 200)
SAT_MAX_VPH = float(os.getenv("SIGNAL_SAT_MAX_VPH", "2200") or 2200)
SAT_REFRESH_SEC = float(os.getenv("SIGNAL_SAT_REFRESH_SEC", "3600") or 3600)
_last_sat_refresh = [0.0]


def _db():
    global _db_ready
    conn = _sqlite3.connect(_DB_PATH, timeout=20)
    conn.execute("PRAGMA busy_timeout=20000")
    if not _db_ready:
        conn.execute("""CREATE TABLE IF NOT EXISTS signal_shadow_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, green_phase INTEGER, green_elapsed REAL,
            queue_m_1 REAL, queue_m_2 REAL,
            ours TEXT, actual TEXT, agree INTEGER,
            switch_gain REAL, keep_gain REAL, change_cost REAL,
            forced INTEGER, blocked INTEGER, reason TEXT,
            step_id INTEGER, clearance INTEGER, control_mode TEXT,
            flow_vpm_1 REAL, flow_vpm_2 REAL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_shadow_ts "
                     "ON signal_shadow_log(ts)")
        # 🛑 上次回報時刻必須落地。放行程內變數的話,每次部署重啟都把一小時的
        #    計時歸零 —— 2026-09-03 部署頻繁,14:04 與 14:56 兩次重啟就讓
        #    14 點那個小時整個沒有回報,使用者以為影子壞了。
        conn.execute("CREATE TABLE IF NOT EXISTS signal_shadow_meta ("
                     "k TEXT PRIMARY KEY, v TEXT)")
        # 既有 DB(9/2 起累積的那份)沒有這三欄,補上 —— 舊列留 NULL,
        # summarize 會把 control_mode IS NULL 的樣本當「前提不明」排除。
        have = {r[1] for r in conn.execute("PRAGMA table_info(signal_shadow_log)")}
        # 🛑 flow_vpm 一定要存。2026-09-04 做模擬驗證時卡在這裡:
        #    只有 queue_m 沒有流量計數,而 queue_m 不守恆 —— 從紅燈成長推的
        #    「到達」與從綠燈消退推的「離開」對不起來(分相2 推出的到達率
        #    0.106 > 有效容量 0.083,模型必然無限累積,但現實排隊是有界的)。
        #    原因是 ROI 只看得到部分路段、而且它是停等長度估計不是車輛計數。
        #    flow_vpm 是通過流量計數,守恆,才撐得起模擬。
        for col, typ in (("step_id", "INTEGER"), ("clearance", "INTEGER"),
                         ("control_mode", "TEXT"),
                         ("flow_vpm_1", "REAL"), ("flow_vpm_2", "REAL")):
            if col not in have:
                conn.execute(f"ALTER TABLE signal_shadow_log ADD COLUMN {col} {typ}")
        conn.commit()
        _db_ready = True
    return conn


def _last_report_at() -> float:
    """讀回上次回報時刻(跨重啟保留)。讀不到就當「從未回報」。"""
    try:
        conn = _db()
        row = conn.execute("SELECT v FROM signal_shadow_meta WHERE k='last_report'"
                           ).fetchone()
        conn.close()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0


def _mark_reported(ts: float) -> None:
    try:
        conn = _db()
        conn.execute("INSERT INTO signal_shadow_meta(k,v) VALUES('last_report',?) "
                     "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (str(ts),))
        conn.commit()
        conn.close()
    except Exception:
        pass


def _refresh_saturation(hours: float = None) -> None:
    """從最近 N 小時的紀錄量飽和流,寫進 _measured_sat。

    🛑 量出來不合理就不採信 —— **保留上一次有效值**,不要讓一次異常量測
       把控制邏輯帶偏。逐相合併:相 1 量到、相 2 沒量到,只更新相 1,
       相 2 維持原值(先前整個 dict 覆蓋,會把另一相打回預設)。
    視窗預設 24 小時且只取「綠燈開始時有隊伍」的段 —— 任何時刻重量都會
    涵蓋到一個尖峰,而且不會被半夜的零星到達拉低。
    """
    if hours is None:
        hours = SAT_WINDOW_HOURS
    from detection.signal_sim import estimate_arrivals, estimate_saturation
    since = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")
    until = datetime.now().isoformat(timespec="seconds")
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase,queue_m_1,queue_m_2 FROM signal_shadow_log "
            "WHERE ts>=? AND ts<=? AND control_mode='external_dynamic' ORDER BY ts",
            (since, until)).fetchall()
        conn.close()
    except Exception:
        return
    if len(rows) < 200:
        return
    arr = estimate_arrivals(rows)
    sat = estimate_saturation(rows, arr, min_start_queue_m=SAT_MIN_START_QUEUE_M)
    good = {}
    for ph in (1, 2):
        v = (sat.get(ph) or {}).get("vph")
        if v and SAT_MIN_VPH <= v <= SAT_MAX_VPH:
            good[ph] = float(v)
    if good:
        merged = dict(_measured_sat.get("vph") or {})
        merged.update(good)                 # 逐相合併,沒量到的那一相不動
        _measured_sat["vph"] = merged
        _measured_sat["ts"] = datetime.now().isoformat(timespec="seconds")
        _measured_sat["source"] = "measured"
        _measured_sat["window_hours"] = hours
        _measured_sat["samples"] = len(rows)
        _save_saturation()


# ── 其他決策參數也要實測落地(使用者:「都是要落地的準確值」)────────────
# 損失時間:控制器 5F03 每秒回報,綠燈結束到下一相開始的秒數 = 黃燈 + 全紅,
#          逐相取中位數(時制表寫 3+2=5,但要用量到的)。
# 每車佔用長度:停止線相機 5 秒取樣,排隊公尺 ÷ 停等車數(停等 ≥ 2 台才算),取中位數。
# 回堵判定比例 80% 是政策門檻,不是物理量,維持設定值並在 /plan 標明。
_PARAMS_FILE = os.path.join(os.path.dirname(_DB_PATH) or ".", "signal_params.json")
_measured_params = {"lost_time_sec": {}, "meters_per_vehicle": None, "ts": None, "source": "default",
                    "samples": {}}
LOST_TIME_MIN, LOST_TIME_MAX = 3.0, 15.0
MPV_MIN, MPV_MAX = 4.0, 12.0


def _save_params() -> None:
    try:
        tmp = _PARAMS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_measured_params, f, ensure_ascii=False)
        os.replace(tmp, _PARAMS_FILE)
    except Exception as e:
        _stats["last_error"] = f"params save failed: {e}"


def _load_params() -> bool:
    try:
        with open(_PARAMS_FILE, encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        return False
    except Exception as e:
        _stats["last_error"] = f"params load failed: {e}"
        return False
    lt = {int(k): float(v) for k, v in (d.get("lost_time_sec") or {}).items()
          if v and LOST_TIME_MIN <= float(v) <= LOST_TIME_MAX}
    mpv = d.get("meters_per_vehicle")
    mpv = float(mpv) if (mpv and MPV_MIN <= float(mpv) <= MPV_MAX) else None
    if not lt and mpv is None:
        return False
    _measured_params.update({"lost_time_sec": lt, "meters_per_vehicle": mpv, "ts": d.get("ts"),
                             "source": "measured(restored)", "samples": d.get("samples") or {}})
    return True


def _median(v: list):
    if not v:
        return None
    s_ = sorted(v)
    n = len(s_)
    return s_[n // 2] if n % 2 else (s_[n // 2 - 1] + s_[n // 2]) / 2.0


def _frames_spacing(since_iso: str, until_iso: str) -> Optional[float]:
    """該時段 5F03 框的中位間隔(秒)。1.0 = 每秒一框;>1.5 代表控制器回報變慢,
    用框算出來的秒數精度就只有一個框距,量損失時間會被灌水。"""
    try:
        from api.routes.signal_tc3 import _QDB_PATH
        import sqlite3 as _sq
        a = datetime.fromisoformat(since_iso).timestamp()
        b = datetime.fromisoformat(until_iso).timestamp()
        conn = _sq.connect(f"file:{_QDB_PATH}?mode=ro", uri=True, timeout=10)
        ts = [r[0] for r in conn.execute("SELECT ts FROM signal_frames WHERE code='5F03' AND cks_ok=1 "
                                         "AND ts>=? AND ts<? ORDER BY ts", (a, b)).fetchall()]
        conn.close()
    except Exception:
        return None
    if len(ts) < 30:
        return None
    gaps = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
    return round(gaps[len(gaps) // 2], 2)


def _refresh_params(hours: float = 24.0) -> None:
    """量損失時間與每車長度;量到合理值才更新(逐項合併),量不到保留上一次。"""
    since = (datetime.now() - timedelta(hours=hours)).isoformat(timespec="seconds")
    until = datetime.now().isoformat(timespec="seconds")
    changed = False
    # 損失時間:清道 = 綠燈結束 → 該相結束(下一相開始)。
    # 🛑 只有 5F03 每秒一框時才能這樣量。2026-09-05 實測:回報變成每 6~7 秒一框後,
    #    量出 8.5 秒(真值 3+2=5),差的就是一個框距 —— 那是取樣假象不是物理量,
    #    而且會把 change_cost 從 3.5 灌到 10.6。框距 >1.5 秒 → 改用控制器回報的
    #    時制設定(5FC4 的黃燈+全紅),那也是控制器自己說的數字,不是假設。
    spacing = _frames_spacing(since, until)
    _measured_params["samples"]["frame_interval_sec"] = spacing
    try:
        if spacing is not None and spacing <= 1.5:
            segs = _actual_runs_from_frames(since, until) or []
            for ph in (1, 2):
                vals = [sg["end"] - sg["green_end"] for sg in segs
                        if sg["phase"] == ph and sg["end"] > sg["green_end"]]
                vals = [v for v in vals if LOST_TIME_MIN <= v <= LOST_TIME_MAX]
                m = _median(vals)
                if m is not None and len(vals) >= 20:
                    _measured_params["lost_time_sec"][ph] = round(m, 1)
                    _measured_params["samples"]["lost_time_%d" % ph] = len(vals)
                    _measured_params["samples"]["lost_time_source"] = "frames_1hz"
                    changed = True
        else:
            from detection.signal_timing_lookup import plan_params, current_base_plan
            pp = plan_params(current_base_plan()) or {}
            y, r = pp.get("yellow"), pp.get("all_red")
            if y is not None and r is not None:
                lt = float(y) + float(r)
                if LOST_TIME_MIN <= lt <= LOST_TIME_MAX:
                    for ph in (1, 2):
                        _measured_params["lost_time_sec"][ph] = lt
                    _measured_params["samples"]["lost_time_source"] = "timing_plan_5FC4"
                    changed = True
    except Exception as e:
        _stats["last_error"] = f"lost_time measure failed: {e}"
    # 每車佔用長度:停止線相機,停等 ≥ 2 台
    try:
        from detection import signal_eval as E
        cams = sorted(set(PHASE_STOPLINE.values()))
        cong = E.load_congestion(_VIOL_DB, cams, since, until)
        ratios = []
        for cam in cams:
            for (_ts, stopped, _veh, qm) in cong.get(cam, []):
                if stopped >= 2 and qm > 0:
                    ratios.append(qm / stopped)
        ratios = [r for r in ratios if MPV_MIN <= r <= MPV_MAX]
        m = _median(ratios)
        if m is not None and len(ratios) >= 100:
            _measured_params["meters_per_vehicle"] = round(m, 2)
            _measured_params["samples"]["mpv"] = len(ratios)
            changed = True
    except Exception as e:
        _stats["last_error"] = f"mpv measure failed: {e}"
    if changed:
        _measured_params["ts"] = datetime.now().isoformat(timespec="seconds")
        _measured_params["source"] = "measured"
        _save_params()


def _lost_time_for(phase: int) -> float:
    from detection.signal_decision_engine import DEFAULT_LOST_TIME_SEC
    return float(_measured_params["lost_time_sec"].get(phase) or DEFAULT_LOST_TIME_SEC)


def _mpv() -> float:
    from detection.signal_decision_engine import DEFAULT_METERS_PER_VEHICLE
    return float(_measured_params["meters_per_vehicle"] or DEFAULT_METERS_PER_VEHICLE)


def _sat_for(phase: int) -> float:
    """該相要用的飽和流(輛/小時)。量到就用量到的,否則用引擎預設。"""
    from detection.signal_decision_engine import DEFAULT_SATURATION_VPH
    return float(_measured_sat["vph"].get(phase) or DEFAULT_SATURATION_VPH)


# ── 戰情頁用:每台相機的即時量測、每相今日放行統計 ──────────────────
def _camera_live() -> list:
    """四台相機各自的即時量測。

    決策用的是「同相取最大」的聚合值,但戰情牆要看的是**每一台各自**的狀況
    —— 上游那台有車而停等線那台沒有,代表隊伍還沒排到停等區,聚合之後
    這個訊息就沒了。所以這裡刻意不聚合。
    """
    from api.routes.congestion import congestion_results
    out = []
    for phase in sorted(PHASE_CAMERAS):
        for cam in PHASE_CAMERAS[phase]:
            r = congestion_results.get(cam) or {}
            out.append({
                "camera_id": cam,
                "name": camera_label(cam),
                "phase_no": phase,
                "online": bool(r),
                # 顯示用一律取平滑後的佔用率。原始瞬時值抖動大,
                # 掛在牆上會一直跳,而且跟等級判定用的不是同一個數。
                "occupancy": r.get("occupancy"),
                "raw_occupancy": r.get("raw_occupancy"),
                "vehicle_count": r.get("vehicle_count"),
                "stopped_vehicle_count": r.get("stopped_vehicle_count"),
                "flow_vpm": r.get("flow_vpm"),
                "queue_m": r.get("estimated_queue_length_m"),
                # 🛑 鍵是 level / level_name(見 congestion_detector 的 result),
                #    第一版寫成 congestion_level,四台全回 None,圖上全是「未量到」灰。
                "level": r.get("level"),
                "level_name": r.get("level_name"),
            })
    return out


_release_cache = {"ts": 0.0, "data": None}


def _release_stats() -> dict:
    """今日各分相的放行統計:累計綠燈秒數 / 放行次數 / 前一次多長。

    🛑 加 30 秒快取。/plan 每 5 秒被打一次,而這裡要掃當日整份 log
       (尖峰一天上萬筆),不快取等於把決策盤的輪詢變成資料庫壓力來源。
    """
    now = time.time()
    if _release_cache["data"] is not None and now - _release_cache["ts"] < 30:
        return _release_cache["data"]
    out: dict = {}
    try:
        day = datetime.now().strftime("%Y-%m-%d")
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase FROM signal_shadow_log "
            "WHERE ts>=? AND control_mode='external_dynamic' ORDER BY ts",
            (day + "T00:00:00",)).fetchall()
        conn.close()
    except Exception:
        rows = []
    # 逐筆掃出連續同相的區段。區段長度用「頭尾時間差」,不是筆數×取樣週期——
    # 抄錄過期被跳過的取樣會讓筆數變少,乘出來會低估。
    runs: dict = {}
    cur_ph = None
    t0 = t1 = None
    last_ph = None          # 最後一段屬於哪一相(它可能還在進行中)

    def close_run():
        if cur_ph is None or t0 is None:
            return
        runs.setdefault(cur_ph, []).append(max(0.0, t1 - t0))

    for ts_s, ph in rows:
        try:
            t = datetime.fromisoformat(ts_s).timestamp()
        except Exception:
            continue
        if ph != cur_ph:
            close_run()
            cur_ph, t0 = ph, t
        t1 = t
    close_run()
    last_ph = cur_ph
    # 🛑「前次」必須是上一段**已結束**的放行,不能拿還在跑的這一段。
    #    最後一段若尾巴貼近現在(取樣週期 5 秒,放寬到 15 秒),它就還在進行中 ——
    #    拿它當前次會跟畫面上的「已亮 N 秒」是同一個數,等於白佔一個欄位,
    #    而且會讓人以為上一輪只放行了 5 秒。
    running = (t1 is not None and time.time() - t1 <= 15)
    for ph, lens in runs.items():
        if ph is None:
            continue
        done = lens[:-1] if (running and ph == last_ph) else lens
        out[str(int(ph))] = {
            "count": len(lens),
            "total_sec": int(round(sum(lens))),
            "last_sec": int(round(done[-1])) if done else None,
            "avg_sec": round(sum(done) / len(done), 1) if done else None,
            "running": bool(running and ph == last_ph),
        }
    _release_cache.update({"ts": now, "data": out})
    return out


def _phase_measure(phase: int) -> dict:
    """把一個分相底下所有相機的量測聚合成一組。

    🛑 全部取**最大**,不取總和 —— 這點 2026-09-05 用實際座標更正過:
       原本以為同相的兩台是「相鄰車道」,所以車數該加總。但量了實際距離:
         分相1  NE-1 ↔ NE-2  相距 52.7 m
         分相2  WN-1 ↔ WN-2  相距 16.0 m
       相鄰車道只會差 3~4 公尺。數十公尺代表它們是**同一個進場的不同位置**
       (一台在停等區、一台在上游),看的是同一批車。
       車流區設定也證實:NE-2「上匝道前停等區」、NE-1「上高速公路前平面道路」
       —— 上下游關係,不是並排車道。
       這種情況下加總會把同一批車算兩次,switch_gain 會膨脹一倍,
       決策直接受影響。取最大才對:上游那台在隊伍長到超出停等區視野時
       才會給出更大的值,正好補上單台看不到的部分。
    """
    from api.routes.congestion import congestion_results
    qmax = None
    fmax = None
    veh = 0.0
    seen = 0
    for cam in PHASE_CAMERAS.get(phase, []):
        r = congestion_results.get(cam) or {}
        if not r:
            continue
        seen += 1
        q = r.get("estimated_queue_length_m")
        if q is not None:
            qv = float(q)
            qmax = qv if qmax is None else max(qmax, qv)
        n = r.get("stopped_vehicle_count")
        if n is None:
            n = r.get("vehicle_count")
        if n is not None:
            veh = max(veh, float(n))
        f2 = r.get("flow_vpm")
        if f2 is not None:
            fmax = f2 if fmax is None else max(fmax, float(f2))
    return {"queue_m": qmax, "flow_vpm": fmax,
            "vehicles": veh, "cameras": seen}


def _queue_m(camera_id: int) -> Optional[float]:
    """取該相機當下的排隊公尺（我方壅塞偵測的量測值）。"""
    try:
        from api.routes.congestion import congestion_results
        r = congestion_results.get(camera_id) or {}
        v = r.get("estimated_queue_length_m")
        return float(v) if v is not None else None
    except Exception:
        return None


def _flow_vpm(camera_id: int) -> Optional[float]:
    """取該相機當下的到達流量(輛/分)。綠側價值要靠它,不能只看靜態排隊。"""
    try:
        from api.routes.congestion import congestion_results
        r = congestion_results.get(camera_id) or {}
        v = r.get("flow_vpm")
        return float(v) if v is not None else None
    except Exception:
        return None


def _live_phase() -> Optional[dict]:
    """取控制器當下的分相/步階（5F03）。

    🛑 **必須跟 signal_daemon 要，不可以讀 traffic-api 行程內的 _by_addr。**
       抄錄器是獨立服務 `traffic-signal.service`
       (`uvicorn services.signal_daemon:app --port 8012`)，
       它才是真正連著 MiiNePort :1001 在抄錄的那一個。traffic-api 這個行程
       的 _by_addr 永遠是空的 —— 影子模式第一版就是讀它，結果空轉沒資料。
       而且 services/signal_daemon.py 明寫「traffic-api 那邊絕不可再
       start_recorder，否則搶 :1001 / 雙抄錄」，所以也不能自己開一份。
    """
    try:
        import urllib.request as _u
        with _u.urlopen(f"{SIGNAL_DAEMON_URL}/api/signal/status", timeout=3) as r:
            import json as _j
            data = _j.load(r)
        for xn in (data.get("intersections") or []):
            ph = xn.get("phase") or {}
            if ph.get("sub_phase_id") is None:
                continue
            lights = ph.get("lights") or []
            # 清道判定不寫死步階編號 —— 直接看有沒有任何方向亮綠
            # (協定 bit2 圓頭綠 / bit3 左 / bit4 直 / bit5 右)。
            # 黃燈與全紅期間控制器已經committed,那時的 KEEP/SWITCH 判斷沒有意義。
            any_green = any((int(l.get("value") or 0) & 0x3C) for l in lights)
            cm = xn.get("control_mode") or {}
            return {"sub_phase_id": int(ph["sub_phase_id"]),
                    "step_id": ph.get("step_id"),
                    "clearance": not any_green,
                    "stale": bool(xn.get("stale")),
                    "age_sec": xn.get("age_sec"),
                    # 🛑 已亮秒數以抄錄器逐框追蹤的值為準。它每秒都收到 5F03,
                    #    精確到訊框;影子自己每 5 秒輪詢推算最多差一個週期,
                    #    而 min/max green 的安全閘門就是拿這個數字去比。
                    "phase_elapsed_sec": xn.get("phase_elapsed_sec"),
                    "control_mode": cm.get("code")}
    except Exception:
        pass
    return None


def _loop():
    """影子迴圈：取樣 → 算我方決策 → 與實際動作對照 → 落 DB。不下發。"""
    from detection.signal_decision_engine import ApproachState, decide
    from detection.signal_timing_lookup import (
        current_base_plan, plan_params, phase_role,
    )

    prev_phase: Optional[int] = None
    green_since: float = time.time()
    while not _stop.is_set():
        try:
            live = _live_phase()
            if live is None:
                _stop.wait(SHADOW_INTERVAL_SEC)
                continue
            # 🛑 抄錄斷線時 latest 會凍結在最後一幀,sub_phase_id 不再變,
            #    green_elapsed 會無限累加,樣本卻照記 —— 一致率被污染而且看不出來。
            #    (9/3 實測有 73 筆 green_elapsed>210s,若不排除無從分辨真假長綠。)
            if live.get("stale"):
                with _lock:
                    _stats["skipped_stale"] = _stats.get("skipped_stale", 0) + 1
                prev_phase = None       # 重連後不要拿斷線前的分相當基準
                _stop.wait(SHADOW_INTERVAL_SEC)
                continue
            cur_phase = live["sub_phase_id"]
            now = time.time()
            # 分相變了 = 控制器實際發生了切換
            actual = "KEEP"
            if prev_phase is not None and cur_phase != prev_phase:
                actual = "SWITCH"
                green_since = now
            elif prev_phase is None:
                green_since = now
            prev_phase = cur_phase
            # 優先用抄錄器逐框追蹤的精確值;取不到才退回自己推算(誤差 ≤ 取樣週期)
            exact = live.get("phase_elapsed_sec")
            if isinstance(exact, (int, float)):
                green_elapsed = float(exact)
                green_since = now - green_elapsed
            else:
                green_elapsed = max(0.0, now - green_since)
            _live_green["since"] = green_since
            _live_green["phase"] = cur_phase

            m1 = _phase_measure(1)
            m2 = _phase_measure(2)
            q1, q2 = m1["queue_m"], m2["queue_m"]
            f1, f2 = m1["flow_vpm"], m2["flow_vpm"]
            # 綠燈側 = 當下分相；紅燈側 = 另一相
            g_no, r_no = (cur_phase, 2 if cur_phase == 1 else 1)
            q_map = {1: q1, 2: q2}
            f_map = {1: f1, 2: f2}
            g_role = phase_role(g_no) or {}
            r_role = phase_role(r_no) or {}
            pp = plan_params(current_base_plan()) or {}
            mins = pp.get("min_green") or [15, 15]
            min_green = float(mins[g_no - 1] if len(mins) >= g_no else 15)

            d = decide(
                green_phase=g_no, green_elapsed_sec=green_elapsed,
                green_side=ApproachState(
                    g_no, queue_m=q_map.get(g_no),
                    flow_vpm=f_map.get(g_no),
                    storage_m=g_role.get("storage_m"),
                    priority=bool(g_role.get("priority"))),
                red_side=ApproachState(
                    r_no, queue_m=q_map.get(r_no),
                    flow_vpm=f_map.get(r_no),
                    storage_m=r_role.get("storage_m"),
                    priority=bool(r_role.get("priority")),
                    waiting_sec=green_elapsed),
                min_green_sec=min_green,
                max_green_sec=_max_green(pp),
                # 飽和流 / 損失時間 / 每車長度全部用現場量到的(見 _measured_* 的說明)
                saturation_vph=_sat_for(g_no),
                meters_per_vehicle=_mpv(),
                lost_time_sec=_lost_time_for(g_no),
            )

            conn = _db()
            conn.execute(
                "INSERT INTO signal_shadow_log(ts,green_phase,green_elapsed,"
                "queue_m_1,queue_m_2,ours,actual,agree,switch_gain,keep_gain,"
                "change_cost,forced,blocked,reason,step_id,clearance,control_mode,"
                "flow_vpm_1,flow_vpm_2)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), g_no,
                 round(green_elapsed, 1), q1, q2, d.action, actual,
                 # 🛑 切換剛發生的那一筆不列入一致率(agree=NULL)。
                 #    偵測到分相變了才記 actual=SWITCH,但那一刻 green_elapsed
                 #    已重設為 0,我方引擎因 min-green 未滿必然回 KEEP ——
                 #    這是取樣時序造成的假不一致,不是真的決策分歧。
                 #    (實測:18 筆裡 2 筆不一致全都是 green_elapsed=0 那筆)
                 # 🛑 三種樣本不列入一致率(agree=NULL),因為前提根本不成立:
                 #  (a) 偵測到換相的那一筆:actual=SWITCH 代表「分相已經變了」,
                 #      那是**已經發生的過去事件**;我方引擎在這一刻評估的是
                 #      「新分相要不要再切」,兩者問的不是同一件事,無從比對。
                 #      🛑 舊條件寫 green_elapsed < 1.0 —— 那是自己推算秒數時
                 #      「切換瞬間必為 0」的權宜寫法。2026-09-04 改用抄錄器的
                 #      精確已亮秒數後,同一筆變成 1.8 秒,條件失效,每一次換相
                 #      都被算成岐異(6 小時約 530 次)。改成只看 actual。
                 #  (b) 清道期間(黃燈/全紅):控制器已經committed要換相,
                 #      這時候問「該不該切」沒有意義。
                 #  (c) 不是外部動態控制:定時/手動時 actual 不是 OPAC 的決策,
                 #      拿我方演算法去比一個根本沒在做決策的控制器毫無意義。
                 (None if (
                     actual == "SWITCH"
                     or live.get("clearance")
                     or live.get("control_mode") != "external_dynamic"
                 ) else (1 if d.action == actual else 0)),
                 d.switch_gain, d.keep_gain,
                 d.change_cost, 1 if d.forced_by_max_green else 0,
                 1 if d.blocked_by_priority else 0, d.reason,
                 live.get("step_id"), 1 if live.get("clearance") else 0,
                 live.get("control_mode"), f1, f2))
            conn.commit()
            conn.close()
            with _lock:
                _stats["samples"] += 1
                _stats["last_at"] = datetime.now().isoformat(timespec="seconds")
                _stats["last_error"] = ""
        except Exception as e:
            with _lock:
                _stats["last_error"] = str(e)
        # 逐時評估:整點過 2 分算前一小時(輕量,大多數輪次直接 return)
        try:
            _hourly_tick()
        except Exception:
            pass
        # 定期重量飽和流。它會隨車種組成與天候變動,不是固定不變的。
        if time.time() - _last_sat_refresh[0] >= SAT_REFRESH_SEC:
            _last_sat_refresh[0] = time.time()
            try:
                _refresh_saturation()
                _refresh_params()
            except Exception:
                pass
        # 到點就自己回報一次 —— 不必等人去撈 DB。
        # 上次時刻從 DB 讀回,重啟不會把計時歸零。
        if not _last_report[0]:
            _last_report[0] = _last_report_at() or time.time()
        if time.time() - _last_report[0] >= SHADOW_REPORT_SEC:
            _last_report[0] = time.time()
            _mark_reported(_last_report[0])
            try:
                _report()
            except Exception:
                pass
        _stop.wait(SHADOW_INTERVAL_SEC)


def summarize(minutes: int = 60, since: Optional[str] = None,
              until: Optional[str] = None) -> dict:
    """把最近 N 分鐘的影子結果壓成一份摘要。

    🛑 一致率一定要分「有車/無車」算。夜間兩側排隊都是 0，兩邊都 KEEP，
       一致率會漂到 98% —— 那個數字沒有資訊量，會蓋掉尖峰的真實表現。
       實測 13.5 小時:整體 87.4%，但只看有車樣本，08 時只有 54.7%。
    """
    # 🛑 「最近 N 分鐘」會隨查詢時間漂移 —— 要比對固定時段(例如尖峰
    #    06:00~12:00)就必須能指定起訖,否則早一分鐘晚一分鐘查到的不是同一段,
    #    兩次結果沒有可比性。
    if since:
        since_iso = since
        until_iso = until or datetime.now().isoformat(timespec="seconds")
    else:
        since_iso = datetime.fromtimestamp(
            time.time() - minutes * 60).isoformat(timespec="seconds")
        until_iso = datetime.now().isoformat(timespec="seconds")
    out = {"minutes": minutes, "since": since_iso, "until": until_iso,
           "samples": 0, "judged_samples": 0,
           "active_samples": 0,
           "agree_rate": None, "active_agree_rate": None,
           "excluded_clearance": 0, "excluded_not_opac": 0,
           "excluded_switch_instant": 0,
           "disagree_switch_early": 0, "disagree_switch_late": 0,
           "keep_gain_zero": 0, "forced": 0, "blocked": 0,
           "max_green_elapsed": None}
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT green_phase,green_elapsed,queue_m_1,queue_m_2,ours,actual,"
            "agree,keep_gain,forced,blocked,clearance,control_mode,ts "
            "FROM signal_shadow_log WHERE ts>=? AND ts<=?",
            (since_iso, until_iso)).fetchall()
        conn.close()
    except Exception as e:
        out["error"] = str(e)
        return out
    if not rows:
        return out
    out["samples"] = len(rows)
    # 🛑 一致率只能在「前提成立」的樣本上算。把排除掉的量攤開,
    #    否則看到一個裸數字沒人知道它是拿什麼比出來的。
    out["excluded_clearance"] = sum(1 for r in rows if r[10])
    out["excluded_not_opac"] = sum(
        1 for r in rows if r[11] is not None and r[11] != "external_dynamic")
    out["excluded_switch_instant"] = sum(
        1 for r in rows if r[6] is None and not r[10]
        and (r[11] == "external_dynamic" or r[11] is None))
    judged = [r for r in rows if r[6] is not None]
    out["judged_samples"] = len(judged)
    # 有車 = 任一側量到排隊。只有這些樣本的一致率才有意義。
    active = [r for r in judged if (r[2] or 0) > 0 or (r[3] or 0) > 0]
    out["active_samples"] = len(active)
    if judged:
        out["agree_rate"] = round(sum(r[6] for r in judged) / len(judged), 3)
    if active:
        out["active_agree_rate"] = round(sum(r[6] for r in active) / len(active), 3)
    out["disagree_switch_early"] = sum(
        1 for r in rows if r[4] == "SWITCH" and r[5] == "KEEP")
    out["disagree_switch_late"] = sum(
        1 for r in rows if r[4] == "KEEP" and r[5] == "SWITCH" and r[6] is not None)
    out["keep_gain_zero"] = sum(
        1 for r in rows if r[4] == "SWITCH" and r[5] == "KEEP" and not r[7])
    out["forced"] = sum(1 for r in rows if r[8])
    out["blocked"] = sum(1 for r in rows if r[9])
    out["max_green_elapsed"] = round(max((r[1] or 0) for r in rows), 1)

    # 逐時拆解。整段的平均會被無車時段稀釋 —— 尖峰哪一小時掉下來,
    # 只有拆開才看得見(2026-09-03 實測:整體 87.4% 但 08 時只有 54.7%)。
    buckets: dict = {}
    for r in rows:
        h = str(r[12])[11:13]
        b = buckets.setdefault(h, {"hour": h, "samples": 0, "judged": 0,
                                   "active": 0, "active_agree": 0})
        b["samples"] += 1
        if r[6] is None:
            continue
        b["judged"] += 1
        if (r[2] or 0) > 0 or (r[3] or 0) > 0:
            b["active"] += 1
            b["active_agree"] += r[6]
    for b in buckets.values():
        b["active_agree_rate"] = (round(b["active_agree"] / b["active"], 3)
                                  if b["active"] else None)
    out["by_hour"] = [buckets[k] for k in sorted(buckets)]
    return out


def _report() -> None:
    """把摘要寫進系統日誌;有車而一致率偏低就推播。"""
    s = summarize(int(SHADOW_REPORT_SEC // 60) or 60)
    if not s["samples"]:
        return
    ar = s["active_agree_rate"]
    pct = "—" if ar is None else f"{ar * 100:.1f}%"
    detail = (f"樣本 {s['samples']} → 可比對 {s['judged_samples']}"
              f"(排除:清道 {s['excluded_clearance']}、"
              f"非外部動態 {s['excluded_not_opac']}、"
              f"切換瞬間 {s['excluded_switch_instant']})、"
              f"其中有車 {s['active_samples']}、有車一致率 {pct}、"
              f"我方提早切 {s['disagree_switch_early']} 次"
              f"(其中綠側價值=0 佔 {s['keep_gain_zero']})、"
              f"最大綠強制 {s['forced']}、最長綠 {s['max_green_elapsed']}s")
    try:
        add_log("info", f"號誌影子決策回報 — {detail}", source="signal-shadow")
    except Exception:
        pass
    # 車太少不評分:比率會被少數樣本帶著跳。
    if ar is not None and s["active_samples"] >= SHADOW_MIN_ACTIVE and ar < SHADOW_ALERT_RATE:
        try:
            push_alert("號誌影子決策:與現行控制差異偏大",
                       f"有車時一致率僅 {pct} — {detail}",
                       {"active_agree_rate": ar,
                        "active_samples": s["active_samples"]},
                       category="signal")
        except Exception:
            pass


def start_shadow() -> bool:
    """啟動影子執行緒（冪等）。回傳是否真的啟動。"""
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _stop.clear()
        _stats["started_at"] = datetime.now().isoformat(timespec="seconds")
        # 先載回上次量到的,再嘗試重量;重量失敗也不會是空的
        _load_saturation()
        _load_params()
        try:
            _refresh_saturation()
            _refresh_params()
        except Exception:
            pass
        _thread = threading.Thread(target=_loop, name="signal-shadow", daemon=True)
        _thread.start()
        # 逐時表的缺漏(重啟期間、部署前的小時)丟背景回填,不擋啟動
        try:
            for d in (datetime.now() - timedelta(days=1), datetime.now()):
                hourly_rows(d.strftime("%Y-%m-%d"), compute_missing=True, max_sync=0)
        except Exception:
            pass
        return True


def stop_shadow() -> None:
    _stop.set()


@router.get("", summary="影子模式狀態與最近決策")
async def shadow_status(limit: int = Query(50, ge=1, le=500),
                        _user=Depends(get_current_user)):
    rows = []
    try:
        conn = _db()
        cur = conn.execute(
            "SELECT ts,green_phase,green_elapsed,queue_m_1,queue_m_2,ours,actual,"
            "agree,switch_gain,keep_gain,forced,blocked,reason "
            "FROM signal_shadow_log ORDER BY id DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        with _lock:
            _stats["last_error"] = str(e)
    with _lock:
        st = dict(_stats)
    running = _thread is not None and _thread.is_alive()
    agree = [r["agree"] for r in rows if r.get("agree") is not None]
    return {
        "enabled": SHADOW_ENABLED,
        "running": running,
        "interval_sec": SHADOW_INTERVAL_SEC,
        "note": "影子模式只記錄不下發，路口仍由現行控制方(OPAC)控制",
        **st,
        "agree_rate": round(sum(agree) / len(agree), 3) if agree else None,
        "recent": rows,
    }


def _outcome_window(since_iso: str, until_iso: str) -> dict:
    """算一個時段的實際成效指標(現場真的發生了什麼)。

    🛑 這是「現況基準」,量的是控制器實際運轉下的結果。
       它**不是**我方演算法的成效 —— 我方沒有真的控制過,那是反事實,量不到。
    """
    from detection.signal_decision_engine import evaluate_outcome
    from detection.signal_timing_lookup import phase_role
    samples = []
    try:
        conn = _db()
        # 🛑 一定要用 ISO(T 分隔)字串比,不能用 datetime('now',...) ——
        #    那個回空格分隔,而 ts 是 T 分隔;字串比較下 'T'(0x54) > ' '(0x20),
        #    條件永遠成立,時間過濾等於沒作用(舊版 minutes 參數完全被忽略,
        #    不論填多少都回傳當天全部樣本)。
        cur = conn.execute(
            "SELECT queue_m_1,queue_m_2,actual FROM signal_shadow_log "
            "WHERE ts>=? AND ts<=? AND control_mode='external_dynamic' ORDER BY id",
            (since_iso, until_iso))
        st2 = (phase_role(2) or {}).get("storage_m")
        for q1, q2, actual in cur.fetchall():
            samples.append({"queue_m_1": q1 or 0, "queue_m_2": q2 or 0,
                            "storage_2": st2, "interval_sec": SHADOW_INTERVAL_SEC,
                            "switched": (actual == "SWITCH")})
        conn.close()
    except Exception as e:
        return {"error": str(e), "since": since_iso, "until": until_iso}
    out = evaluate_outcome(samples) or {}
    out.update({"since": since_iso, "until": until_iso})
    if not samples:
        out["insufficient_data"] = True
    return out


@router.get("/outcome/compare", summary="兩時段成效對比(A/B)")
async def shadow_outcome_compare(
        a_since: str = Query(..., description="A 段起(ISO)"),
        a_until: str = Query(..., description="A 段訖(ISO)"),
        b_since: str = Query(..., description="B 段起(ISO)"),
        b_until: str = Query(..., description="B 段訖(ISO)"),
        _user=Depends(get_current_user)):
    """比較兩個時段的實際成效。

    🛑 這個端點是為了 L5 分階段接管後的 A/B 對照而做的:
       例如 A = 我方控制的時段、B = OPAC 控制的同性質時段。
       **在我方真的接管之前,兩段都是 OPAC 的成效**,只能拿來看不同時段的
       基準差異(例如尖峰 vs 離峰),不能拿來宣稱我方比較好。
       想證明「優於現今」只有三條路,見 docs/上線報告_骨架.md。
    """
    a = _outcome_window(a_since, a_until)
    b = _outcome_window(b_since, b_until)
    keys = ("total_delay_veh_sec", "avg_queue_m_1", "avg_queue_m_2",
            "max_queue_m_1", "max_queue_m_2", "spillback_events_2",
            "switch_per_min")
    delta = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta[k] = {"a": va, "b": vb, "diff": round(va - vb, 2),
                        "pct": (round((va - vb) / vb * 100, 1) if vb else None)}
    return {"a": a, "b": b, "delta": delta,
            "note": "越小越好(switch_per_min 除外,太頻繁代表浪費在換相損失)。"
                    "在我方真的接管控制權之前,兩段量到的都是現行控制方的成效。"}


@router.get("/outcome", summary="成效基準(總延滯/排隊/回堵次數)")
async def shadow_outcome(minutes: int = Query(60, ge=1, le=1440),
                         since: str = Query("", description="起(ISO),給了就用固定時段"),
                         until: str = Query("", description="訖(ISO)"),
                         _user=Depends(get_current_user)):
    """量這段時間**實際發生**的成效指標。

    🛑 名稱從「成效比較」改成「成效基準」—— 它量的是現行控制方的實際結果,
       不是我方演算法的成效。我方沒有真的控制過,那是反事實,量不到。
    """
    if since:
        since_iso = since
        until_iso = until or datetime.now().isoformat(timespec="seconds")
    else:
        since_iso = datetime.fromtimestamp(
            time.time() - int(minutes) * 60).isoformat(timespec="seconds")
        until_iso = datetime.now().isoformat(timespec="seconds")
    res = _outcome_window(since_iso, until_iso)
    res["minutes"] = minutes
    return res

    return {"minutes": minutes, "outcome": evaluate_outcome(samples)}


@router.post("/start", summary="啟動影子模式(只記錄不下發)")
async def shadow_start(_user=Depends(get_current_user)):
    started = start_shadow()
    return {"started": started, "running": True,
            "note": "影子模式不會對號誌控制器送出任何指令"}


@router.post("/stop", summary="停止影子模式")
async def shadow_stop(_user=Depends(get_current_user)):
    stop_shadow()
    return {"stopped": True}


@router.get("/summary", summary="影子結果摘要(有車/無車分開算一致率,可指定時段)")
async def shadow_summary(minutes: int = Query(60, ge=5, le=1440),
                         since: str = Query("", description="起(ISO,例 2026-09-04T06:00:00)"),
                         until: str = Query("", description="訖(ISO)"),
                         _user=Depends(get_current_user)):
    """給 since/until 就比對那個固定時段;不給就看最近 minutes 分鐘。"""
    return summarize(minutes, since or None, until or None)


@router.get("/plan", summary="即時決策盤(輸入/算式/安全閘門逐項攤開)")
async def shadow_plan(_user=Depends(get_current_user)):
    """回傳「這一刻我方演算法在想什麼」的完整快照。

    🛑 摘要頁只給一個一致率,看不出演算法憑什麼這樣判。要接管控制權之前,
       每一筆判斷都必須可稽核 —— 輸入是什麼、算式每一項多少、
       哪一道安全閘門先攔下來,都要攤在同一個畫面上。
    """
    from detection.signal_decision_engine import (
        ApproachState, decide,
        DEFAULT_SATURATION_VPH, DEFAULT_METERS_PER_VEHICLE,
        DEFAULT_SPILLBACK_RATIO, DEFAULT_LOST_TIME_SEC,
    )
    from detection.signal_timing_lookup import (
        phase_role, plan_params, current_base_plan,
    )

    live = _live_phase()
    if not live:
        return {"available": False, "reason": "抄錄器沒有燈態資料(traffic-signal 未連線?)"}
    if live.get("stale"):
        return {"available": False, "reason": "燈態資料已過期,不做決策(避免用凍結的分相判斷)"}

    g_no = live["sub_phase_id"]
    r_no = 2 if g_no == 1 else 1
    # green_elapsed 以影子迴圈追蹤中的綠燈起始時刻為準(同一個行程,即時且準)。
    # 只有在迴圈還沒起來、或它追的分相跟現在不同(剛換相的瞬間)時,
    # 才退回撈最後一筆樣本 —— 那個值最多差一個取樣週期。
    green_elapsed = 0.0
    exact = live.get("phase_elapsed_sec")
    if isinstance(exact, (int, float)):
        green_elapsed = float(exact)
    elif _live_green["since"] and _live_green["phase"] == g_no:
        green_elapsed = max(0.0, time.time() - _live_green["since"])
    else:
        try:
            conn = _db()
            row = conn.execute("SELECT green_elapsed FROM signal_shadow_log "
                               "ORDER BY id DESC LIMIT 1").fetchone()
            conn.close()
            if row:
                green_elapsed = float(row[0] or 0.0)
        except Exception:
            pass

    meas = {1: _phase_measure(1), 2: _phase_measure(2)}
    q = {p: meas[p]["queue_m"] for p in (1, 2)}
    f = {p: meas[p]["flow_vpm"] for p in (1, 2)}
    roles = {1: phase_role(1) or {}, 2: phase_role(2) or {}}
    plan_id = current_base_plan()
    pp = plan_params(plan_id) or {}
    mins = pp.get("min_green") or [15, 15]
    min_green = float(mins[g_no - 1] if len(mins) >= g_no else 15)
    max_green = _max_green(pp)

    green = ApproachState(g_no, queue_m=q.get(g_no), flow_vpm=f.get(g_no),
                          storage_m=roles[g_no].get("storage_m"),
                          priority=bool(roles[g_no].get("priority")))
    # 🛑 紅側的 priority 一定要帶。主線保護閘門目前只看綠側,所以漏傳不影響
    #    決策 —— 但控制盤把它顯示成「否」就是錯的,而這個畫面是要拿來稽核的。
    red = ApproachState(r_no, queue_m=q.get(r_no), flow_vpm=f.get(r_no),
                        storage_m=roles[r_no].get("storage_m"),
                        priority=bool(roles[r_no].get("priority")),
                        waiting_sec=green_elapsed)
    d = decide(green_phase=g_no, green_elapsed_sec=green_elapsed,
               green_side=green, red_side=red,
               min_green_sec=min_green, max_green_sec=max_green,
               saturation_vph=_sat_for(g_no),
               meters_per_vehicle=_mpv(), lost_time_sec=_lost_time_for(g_no))

    def side(a: ApproachState, role: dict) -> dict:
        sr = a.spillback_ratio()
        return {
            "phase_no": a.phase_no,
            "ramp": role.get("ramp"), "label": role.get("label"),
            "camera": camera_label(role.get("constraint_camera")),
            "camera_key": role.get("constraint_camera"),
            "cameras_used": meas[a.phase_no]["cameras"],
            "camera_ids": PHASE_CAMERAS.get(a.phase_no, []),
            "camera_names": [camera_label(c) for c in PHASE_CAMERAS.get(a.phase_no, [])],
            "vehicles_measured": round(meas[a.phase_no]["vehicles"], 1),
            "queue_m": a.queue_m,
            "queue_vehicles": round(a.queue_vehicles(_mpv()), 2),
            "flow_vpm": a.flow_vpm,
            "arrival_per_sec": round(a.arrival_rate_per_sec(), 3),
            "storage_m": a.storage_m,
            "spillback_ratio": None if sr is None else round(sr, 3),
            "priority": a.priority,
        }

    # 安全閘門依引擎內的判定順序列出,並標出是哪一道實際生效
    gates = [
        {"order": 1, "name": "最小綠", "rule": f"已亮 {green_elapsed:.0f}s < {min_green:.0f}s 則強制 KEEP",
         "hit": green_elapsed < min_green},
        {"order": 2, "name": "最大綠", "rule": f"已亮 {green_elapsed:.0f}s ≥ {max_green:.0f}s 則強制 SWITCH",
         "hit": bool(d.forced_by_max_green)},
        {"order": 3, "name": "主線保護",
         "rule": f"綠側為優先相且排隊達儲車上限 {DEFAULT_SPILLBACK_RATIO*100:.0f}% 則不可切走",
         "hit": bool(d.blocked_by_priority)},
        {"order": 4, "name": "延滯成本比較",
         "rule": "紅側延滯 > 綠側價值 + 換相成本 才切", "hit": not any(
             (green_elapsed < min_green, d.forced_by_max_green, d.blocked_by_priority))},
    ]

    cams = _camera_live()
    releases = _release_stats()
    return {
        "cameras": cams,
        "release_stats": releases,
        "available": True,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "control_mode": live.get("control_mode"),
        "clearance": bool(live.get("clearance")),
        "step_id": live.get("step_id"),
        "green_phase": g_no, "red_phase": r_no,
        "green_elapsed_sec": round(green_elapsed, 1),
        "plan": {"plan_id": plan_id, "min_green_sec": min_green,
                 "max_green_sec": max_green,
                 "cycle": pp.get("cycle"), "yellow": pp.get("yellow"),
                 "all_red": pp.get("all_red")},
        "green_side": side(green, roles[g_no]),
        "red_side": side(red, roles[r_no]),
        "terms": {
            "switch_gain": d.switch_gain, "keep_gain": d.keep_gain,
            "change_cost": d.change_cost,
            "threshold": round(d.keep_gain + d.change_cost, 2),
            "margin": round(d.switch_gain - d.keep_gain - d.change_cost, 2),
            **d.detail,
        },
        "constants": {
            # 🛑 標出這個值是量到的還是預設的 —— 它直接決定換相門檻,
            #    看報表的人必須知道自己在看哪一種。
            "saturation_vph": _sat_for(g_no),
            "max_green_sec": max_green,
            "max_green_source": ("固定 %g 秒(使用者設定)" % MAX_GREEN_SEC) if MAX_GREEN_SEC > 0 else "時制計畫",
            "saturation_source": _measured_sat.get("source", "default"),
            "saturation_measured_at": _measured_sat.get("ts"),
            "saturation_default_vph": DEFAULT_SATURATION_VPH,
            "meters_per_vehicle": _mpv(),
            "meters_per_vehicle_source": ("實測(停止線排隊÷停等車數,%s 筆)" % _measured_params["samples"].get("mpv"))
                                          if _measured_params.get("meters_per_vehicle") else "預設 %g m(未量到)" % DEFAULT_METERS_PER_VEHICLE,
            "spillback_ratio": DEFAULT_SPILLBACK_RATIO,
            "lost_time_sec": _lost_time_for(g_no),
            "lost_time_source": (
                ("實測(控制器 5F03 每秒回報的黃燈+全紅,%s 段)" % _measured_params["samples"].get("lost_time_%d" % g_no))
                if (_measured_params["samples"].get("lost_time_source") == "frames_1hz" and _measured_params["lost_time_sec"].get(g_no))
                else ("控制器時制設定 5FC4(黃燈+全紅;5F03 框距 %s 秒不足以逐秒量)" % _measured_params["samples"].get("frame_interval_sec"))
                if _measured_params["lost_time_sec"].get(g_no) else "預設 %g s(未量到)" % DEFAULT_LOST_TIME_SEC),
            "frame_interval_sec": _measured_params["samples"].get("frame_interval_sec"),
            "lost_time_by_phase": {str(k): v for k, v in _measured_params["lost_time_sec"].items()},
            "spillback_ratio_source": "設定值(政策門檻,非物理量)",
            "params_measured_at": _measured_params.get("ts"),
        },
        "gates": gates,
        "action": d.action, "reason": d.reason,
        "would_send": False,
        "note": "影子模式:本決策只記錄不下發,路口仍由現行控制方控制",
    }

def _ts_gap(prev_iso: Optional[str], cur_iso: str) -> Optional[float]:
    """兩筆取樣的間隔秒數。解不出來回 None。"""
    if not prev_iso:
        return None
    try:
        return (datetime.fromisoformat(cur_iso)
                - datetime.fromisoformat(prev_iso)).total_seconds()
    except Exception:
        return None


def _green_runs(rows: list) -> list:
    """把 5 秒一筆的取樣重建成「一次一次的綠燈」。

    rows 需為 (ts, green_phase, green_elapsed, forced, queue_m_1, queue_m_2,
    flow_1, flow_2) 且依時間排序。

    判定方式:同一分相內 green_elapsed 是遞增的,一旦變小就是換相了 ——
    這比去比對 sub_phase_id 更穩,因為分相會在 1/2 之間來回,單看編號
    分不出「同一個分相的第二輪」。

    🛑 每一段取「最後一筆的 green_elapsed」當這次綠燈長度,所以會低估
       最多一個取樣週期(5 秒),而且被 stale 跳過的取樣會讓該段直接斷開。
       回傳裡帶 truncated 標記,呈現時要說清楚,不要假裝是精確值。
    """
    runs = []
    cur = None
    prev_ts = None
    for (ts, ph, el, forced, q1, q2, f1, f2) in rows:
        el = float(el or 0.0)
        gap = _ts_gap(prev_ts, ts)
        prev_ts = ts
        if cur is None or ph != cur["phase"] or el < cur["last_elapsed"]:
            if cur is not None:
                runs.append(cur)
            cur = {"phase": ph, "start_ts": ts, "end_ts": ts,
                   "green_sec": el, "last_elapsed": el, "forced": bool(forced),
                   "samples": 1,
                   # 段「之前」有斷點 → 這段可能是被截斷後的後半截
                   "gap_before": gap}
        else:
            cur["end_ts"] = ts
            cur["green_sec"] = el
            cur["last_elapsed"] = el
            cur["samples"] += 1
            # 段「之內」有斷點 → 這段的長度不可信。
            # 🛑 要比「異常大的間隔」,不是「有沒有間隔」—— 正常取樣本來就每
            #    5 秒一筆,寫成 gap > 0 會把每一段都標成不可信(實測 417 段
            #    全被標記,等於這個欄位完全失去意義)。
            if gap and gap > SHADOW_INTERVAL_SEC * 1.6                     and gap > cur.get("max_inner_gap", 0):
                cur["max_inner_gap"] = gap
            if forced:
                cur["forced"] = True
    if cur is not None:
        runs.append(cur)
    return runs


def _run_after_gap(run: dict) -> bool:
    """這一段是不是接在一個取樣斷點之後(可能是被截斷的後半截)。"""
    g = run.get("gap_before")
    return bool(g and g > SHADOW_INTERVAL_SEC * 1.6)


def _stat(vals: list) -> dict:
    """樣本數/平均/變異數/標準差。空的回 None —— 不用 0 代表「沒量到」。"""
    n = len(vals)
    if not n:
        return {"n": 0, "avg": None, "variance": None, "stddev": None}
    avg = sum(vals) / n
    var = sum((v - avg) ** 2 for v in vals) / n
    return {"n": n, "avg": round(avg, 1), "variance": round(var, 1),
            "stddev": round(var ** 0.5, 1)}


@router.get("/stats", summary="運作統計(綠燈長度/切換次數/滯留,依方向)")
async def shadow_stats(minutes: int = Query(360, ge=5, le=10080),
                       since: str = Query("", description="起(ISO)"),
                       until: str = Query("", description="訖(ISO)"),
                       trend_limit: int = Query(120, ge=10, le=1000),
                       _user=Depends(get_current_user)):
    """所選區間的運作統計。

    🛑 查無樣本時各指標回 None 並附 insufficient_data —— **不以 0 代表統計值**。
       0 次切換和「沒有資料」是完全不同的兩件事,混在一起看會做出錯的判斷。
    """
    from detection.signal_timing_lookup import phase_role, plan_params, current_base_plan

    if since:
        since_iso, until_iso = since, (until or datetime.now().isoformat(timespec="seconds"))
    else:
        since_iso = datetime.fromtimestamp(
            time.time() - minutes * 60).isoformat(timespec="seconds")
        until_iso = datetime.now().isoformat(timespec="seconds")

    out = {"since": since_iso, "until": until_iso, "insufficient_data": True,
           "samples": 0, "runs": 0, "switch_count": None,
           "forced_count": None, "forced_ratio": None,
           "by_direction": [], "trend": [], "trend_total": 0,
           "exit_queue_m": None, "exit_queue_vehicles": None,
           "vehicles_per_green_sec": None,
           "dropped_unobserved": 0, "runs_used": 0,
           "note": "綠燈長度由 5 秒取樣重建,最多低估一個取樣週期;"
                   "抄錄過期(stale)被跳過的取樣會讓該段斷開。"
                   "綠燈長度是區間量測:真值落在 [green_sec, green_sec+取樣週期)。"
                   "below_min_green 以上界判定,且排除接在斷點之後或段內有斷點的段,"
                   "所以它只計入『確定低於最小綠』的次數;不確定的計入 uncertain_truncated"}
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase,green_elapsed,forced,queue_m_1,queue_m_2 "
            "FROM signal_shadow_log WHERE ts>=? AND ts<=? "
            "AND control_mode='external_dynamic' ORDER BY ts",
            (since_iso, until_iso)).fetchall()
        conn.close()
    except Exception as e:
        out["error"] = str(e)
        return out
    if not rows:
        return out

    out["samples"] = len(rows)
    out["insufficient_data"] = False
    runs = _green_runs([(r[0], r[1], r[2], r[3], r[4], r[5], None, None)
                        for r in rows])
    # 第一段的起點在區間之前就開始了,長度不完整,不列入統計
    if len(runs) > 1:
        runs = runs[1:]
    out["runs"] = len(runs)

    # 🛑 長度量到 0 秒的段不是量測結果,是取樣斷開造成的假段:抄錄 stale
    #    被跳過時 prev_phase 會清掉,下一筆重新起算 green_elapsed=0,
    #    若連兩筆都落在 0 就會拼出一個「0 秒的綠燈」——分相2 最小綠 20 秒,
    #    物理上不可能。用「長度<=0」判,不能用「取樣數<2」判(實測那段有 2 筆)。
    dropped = [r for r in runs if r["green_sec"] <= 0]
    runs = [r for r in runs if r["green_sec"] > 0]
    out["dropped_unobserved"] = len(dropped)
    out["runs_used"] = len(runs)
    # 切換次數 = 綠燈段數 - 1(段與段之間各一次換相)
    out["switch_count"] = max(0, len(runs) - 1)
    out["forced_count"] = sum(1 for r in runs if r["forced"])
    out["forced_ratio"] = (round(out["forced_count"] / len(runs), 3)
                           if runs else None)

    pp = plan_params(current_base_plan()) or {}
    mins = pp.get("min_green") or [15, 15]
    for ph in (1, 2):
        role = phase_role(ph) or {}
        vals = [r["green_sec"] for r in runs if r["phase"] == ph]
        st = _stat(vals)
        out["by_direction"].append({
            "phase_no": ph,
            "ramp": role.get("ramp"), "label": role.get("label"),
            "min_green_sec": float(mins[ph - 1]) if len(mins) >= ph else None,
            "max_green_sec": _max_green(pp),
            **st,
            "min_observed": round(min(vals), 1) if vals else None,
            "max_observed": round(max(vals), 1) if vals else None,
            # 平均的最佳估計:量到的值 + 半個取樣週期(真值均勻落在
            # [量到, 量到+週期) 之間)
            "avg_estimated": (round(st["avg"] + SHADOW_INTERVAL_SEC / 2, 1)
                              if st["avg"] is not None else None),
            # 🛑 判定「低於最小綠」必須用上界,不能用量到的值。
            #    綠燈長度是區間量測:量到 15 秒的段,真值在 [15, 20) ——
            #    一段真實 20 秒(= 最小綠)的綠燈,用 5 秒取樣量出來就是 15 秒。
            #    2026-09-03 就是這樣誤判:分相2 出現 11 次「低於最小綠 20 秒」,
            #    逐段查證後 5 段緊鄰取樣斷點(截斷假象)、6 段都恰好 15.x 秒且
            #    各 4 個取樣 —— 全部都是量測下限,不是控制器違規。
            "below_min_green": sum(
                1 for r in runs
                if r["phase"] == ph and len(mins) >= ph
                and r["green_sec"] + SHADOW_INTERVAL_SEC < float(mins[ph - 1])
                and not r.get("max_inner_gap") and not _run_after_gap(r)),
            "uncertain_truncated": sum(
                1 for r in runs
                if r["phase"] == ph
                and (r.get("max_inner_gap") or _run_after_gap(r))),
        })

    out["trend_total"] = len(runs)
    out["trend"] = [{"ts": r["start_ts"], "phase_no": r["phase"],
                     "green_sec": round(r["green_sec"], 1),
                     # 真值落在 [green_sec, green_sec + 取樣週期)
                     "green_sec_upper": round(r["green_sec"] + SHADOW_INTERVAL_SEC, 1),
                     "forced": r["forced"],
                     "truncated": bool(r.get("max_inner_gap") or _run_after_gap(r))}
                    for r in runs[-trend_limit:]]

    # 出口(下匝道 = 分相2)滯留:取區間內的平均與最大,這是主線回堵的前哨
    q2 = [float(r[5]) for r in rows if r[5] is not None]
    if q2:
        from detection.signal_decision_engine import DEFAULT_METERS_PER_VEHICLE as MPV
        out["exit_queue_m"] = {"avg": round(sum(q2) / len(q2), 1),
                               "max": round(max(q2), 1)}
        out["exit_queue_vehicles"] = {
            "avg": round(sum(q2) / len(q2) / MPV, 1),
            "max": round(max(q2) / MPV, 1)}
    return out

@router.get("/simulate", summary="模擬驗證(先校準,校準過才給比較結果)")
async def shadow_simulate(minutes: int = Query(360, ge=30, le=1440),
                          since: str = Query(""), until: str = Query(""),
                          _user=Depends(get_current_user)):
    """用同一份到達流量餵兩套演算法,比較成效。

    🛑 流程刻意是「先校準、再比較」,而且**校準沒過就不回傳比較結果**:
       把現場實際的換相序列餵進模型,看模擬排隊能不能重現實際量到的排隊。
       重現不了就代表模型無法在已知控制下描述現場,更不可能預測「換另一套
       控制會怎樣」—— 這時候給比較數字只會製造假結論。
    """
    from detection.signal_sim import (
        SimConfig, arrival_profile, calibrate, estimate_arrivals,
        estimate_saturation, profile_rate_fn, replay_actual, simulate,
    )
    from detection.signal_decision_engine import ApproachState, decide
    from detection.signal_timing_lookup import (
        current_base_plan, phase_role, plan_params,
    )

    if since:
        since_iso = since
        until_iso = until or datetime.now().isoformat(timespec="seconds")
    else:
        since_iso = datetime.fromtimestamp(
            time.time() - int(minutes) * 60).isoformat(timespec="seconds")
        until_iso = datetime.now().isoformat(timespec="seconds")
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase,queue_m_1,queue_m_2 FROM signal_shadow_log "
            "WHERE ts>=? AND ts<=? AND control_mode='external_dynamic' "
            "ORDER BY ts", (since_iso, until_iso)).fetchall()
        conn.close()
    except Exception as e:
        return {"error": str(e)}
    if len(rows) < 120:
        return {"available": False, "since": since_iso, "until": until_iso,
                "reason": f"樣本僅 {len(rows)} 筆,不足以校準(需 ≥120)"}

    pp = plan_params(current_base_plan()) or {}
    mins = pp.get("min_green") or [10, 20]
    cfg = SimConfig(dt_sec=SHADOW_INTERVAL_SEC,
                    min_green_sec={1: float(mins[0]), 2: float(mins[1])},
                    max_green_sec=_max_green(pp))

    # 🛑 用時變到達率,不用單一中位數。首次校準用固定率時相關係數只有
    #    -0.006 / -0.04(完全沒跟上動態)—— 固定率撐不起數小時的模擬。
    overall = estimate_arrivals(rows)
    # 飽和流現場量,不用教科書的 1800 vph —— 那是物理量,假設值差 13 倍
    sat = estimate_saturation(rows, overall)
    cfg.saturation_by_phase = {p: (sat[p]["veh_per_sec"] or None) for p in (1, 2)}
    profile = arrival_profile(rows)
    rate_fn = profile_rate_fn(profile)
    base = replay_actual(rows, rate_fn, cfg)
    cal = calibrate(rows, base)

    result = {
        "available": True, "since": since_iso, "until": until_iso,
        "samples": len(rows),
        "arrivals_overall": overall,
        "saturation_measured": sat,
        "arrival_windows": len(profile.get("windows") or []),
        "arrival_window_sec": profile.get("window_sec"),
        "calibration": cal,
        "baseline_sim": {k: v for k, v in base.items() if k != "trajectory"},
    }
    if not cal.get("usable"):
        result["comparison"] = None
        result["conclusion"] = ("校準未通過,不提供比較結果 —— "
                                "模型無法在已知控制下重現現場排隊。")
        return result

    # 校準過了才跑我方演算法
    roles = {p: (phase_role(p) or {}) for p in (1, 2)}
    mpv = cfg.meters_per_vehicle

    def ours(state):
        g = state["green_phase"]
        r = 2 if g == 1 else 1
        qv = state["queue_veh"]
        gs = ApproachState(g, queue_m=qv[g] * mpv,
                           storage_m=roles[g].get("storage_m"),
                           priority=bool(roles[g].get("priority")))
        rs = ApproachState(r, queue_m=qv[r] * mpv,
                           storage_m=roles[r].get("storage_m"),
                           priority=bool(roles[r].get("priority")),
                           waiting_sec=state["green_elapsed"])
        d = decide(green_phase=g, green_elapsed_sec=state["green_elapsed"],
                   green_side=gs, red_side=rs,
                   min_green_sec=cfg.min_green_sec.get(g, 10.0),
                   max_green_sec=cfg.max_green_sec)
        return d.action == "SWITCH"

    mine = simulate(rate_fn, ours, base["duration_sec"], cfg,
                    start_phase=rows[0][1] or 1)
    keys = ("total_delay_veh_sec", "avg_queue_m_1", "avg_queue_m_2",
            "max_queue_m_1", "max_queue_m_2", "switch_per_min")
    delta = {}
    for k in keys:
        a, b = mine.get(k), base.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta[k] = {"ours": a, "actual": b, "diff": round(a - b, 2),
                        "pct": round((a - b) / b * 100, 1) if b else None}
    d0 = delta.get("total_delay_veh_sec") or {}
    pct = d0.get("pct")
    result["comparison"] = {
        "ours_sim": {k: v for k, v in mine.items() if k != "trajectory"},
        "delta": delta,
    }
    if pct is None:
        result["conclusion"] = "無法計算延滯差異"
    elif pct < -5:
        result["conclusion"] = f"模擬中我方總延滯較低 {abs(pct):.1f}%"
    elif pct > 5:
        result["conclusion"] = f"模擬中我方總延滯較高 {pct:.1f}%"
    else:
        result["conclusion"] = f"模擬中兩者差異在 5% 以內({pct:+.1f}%),視為無顯著差異"
    result["caveat"] = ("這是模擬結論,不是現場實績。模型已通過校準"
                        "(能在已知控制下重現現場排隊),但仍是確定性排隊模型,"
                        "沒有納入車輛異質性、上游號誌連鎖與駕駛行為。"
                        "要主張現場實績仍須做 A/B 交替時段。")
    return result

# ── 逐次綠燈配對(精確比對)────────────────────────────────────────
# 🛑 取樣層級的「一致率」只回答「同一秒兩邊說的一不一樣」,答不了
#    「我方會早幾秒切」。控制上真正有意義的是:每一次實際綠燈,
#    我方第一次判 SWITCH 落在第幾秒,跟實際換相差多少。
#    這裡以「一次綠燈」為單位配對,每一段只算一次,不會被長綠燈灌樣本。
def _paired_runs(rows: list, interval: float = None) -> dict:
    """rows: (ts, green_phase, green_elapsed, ours, actual, forced, clearance,
              queue_m_1, queue_m_2) 依時間排序,且已限定 external_dynamic。

    每段回:
      actual_sec        實際綠燈長度(區間量測:真值在 [actual_sec, +取樣週期))
      ours_sec          我方第一次 SWITCH 時的已亮秒數;整段都 KEEP 則 None
      delta_sec         ours_sec − actual_sec;負 = 我方會比實際早切
      red_waiting       我方判 SWITCH 那一刻紅側有沒有排隊(有 = 早切是有意義的)
      waste_sec         紅側有排隊時,從我方判切到實際換相之間的秒數(有代價空放)
      truncated         段內或段前有斷點,長度不可信 → 不計入統計
    """
    if interval is None:
        interval = SHADOW_INTERVAL_SEC
    runs = []
    cur = None
    prev_ts = None
    for (ts, ph, el, ours, actual, forced, clr, q1, q2) in rows:
        el = float(el or 0.0)
        gap = _ts_gap(prev_ts, ts)
        prev_ts = ts
        if cur is None or ph != cur["phase"] or el < cur["last_elapsed"]:
            if cur is not None:
                runs.append(cur)
            cur = {"phase": ph, "start_ts": ts, "end_ts": ts, "actual_sec": el,
                   "last_elapsed": el, "ours_sec": None, "red_waiting": None,
                   "forced": bool(forced), "samples": 1,
                   "truncated": bool(gap and gap > interval * 1.6)}
        else:
            cur["end_ts"] = ts
            cur["actual_sec"] = el
            cur["last_elapsed"] = el
            cur["samples"] += 1
            if gap and gap > interval * 1.6:
                cur["truncated"] = True
            if forced:
                cur["forced"] = True
        # 清道期間不評估;第一次 SWITCH 才算(之後每筆都會一直說 SWITCH)
        if cur["ours_sec"] is None and ours == "SWITCH" and not clr:
            cur["ours_sec"] = el
            red_q = (q2 if ph == 1 else q1)
            cur["red_waiting"] = bool(red_q and float(red_q) > 0)
    if cur is not None:
        runs.append(cur)

    out_runs = []
    for r in runs:
        if r["samples"] < 2:
            continue                     # 只有一筆的「段」是切換瞬間的殘影
        d = None
        waste = 0.0
        if r["ours_sec"] is not None:
            d = round(r["ours_sec"] - r["actual_sec"], 1)
            # 🛑 代價只算「超出一個取樣週期」的早切。5 秒取樣分不出 3 秒的差,
            #    把容忍內的也算進去會把量測誤差當成代價(實測 313 vs 273 秒)。
            if r["red_waiting"] and d < -interval:
                waste = round(-d, 1)
        out_runs.append({
            "phase": r["phase"], "start_ts": r["start_ts"], "end_ts": r["end_ts"],
            "actual_sec": round(r["actual_sec"], 1),
            "ours_sec": None if r["ours_sec"] is None else round(r["ours_sec"], 1),
            "delta_sec": d, "red_waiting": r["red_waiting"], "waste_sec": waste,
            "forced": r["forced"], "truncated": r["truncated"], "samples": r["samples"],
        })

    usable = [r for r in out_runs if not r["truncated"]]
    # 分類。「同時」給一個取樣週期的容忍 —— 5 秒取樣本來就分不出 3 秒的差。
    earlier = [r for r in usable if r["delta_sec"] is not None and r["delta_sec"] < -interval]
    same = [r for r in usable if r["delta_sec"] is not None and -interval <= r["delta_sec"] <= interval]
    hold = [r for r in usable if r["delta_sec"] is None]       # 整段都同意續綠
    later = [r for r in usable if r["delta_sec"] is not None and r["delta_sec"] > interval]
    meaningful = [r for r in earlier if r["red_waiting"]]      # 早切且紅側真的有車在等
    idle = [r for r in earlier if not r["red_waiting"]]         # 早切但紅側沒車 —— 切了也沒意義

    def stat(vals):
        return _stat(vals) if vals else None

    by_phase = {}
    for ph in (1, 2):
        sub = [r for r in usable if r["phase"] == ph]
        by_phase[str(ph)] = {
            "runs": len(sub),
            "earlier": sum(1 for r in sub if r["delta_sec"] is not None and r["delta_sec"] < -interval),
            "hold": sum(1 for r in sub if r["delta_sec"] is None),
            "delta": stat([r["delta_sec"] for r in sub if r["delta_sec"] is not None]),
            "waste_sec": round(sum(r["waste_sec"] for r in sub), 1),
        }
    return {
        "interval_sec": interval,
        "runs_total": len(out_runs), "runs_usable": len(usable),
        "runs_truncated": len(out_runs) - len(usable),
        "earlier": len(earlier), "earlier_meaningful": len(meaningful), "earlier_idle": len(idle),
        "same": len(same), "hold": len(hold), "later": len(later),
        "delta_all": stat([r["delta_sec"] for r in usable if r["delta_sec"] is not None]),
        "delta_meaningful": stat([r["delta_sec"] for r in meaningful]),
        "waste_sec_total": round(sum(r["waste_sec"] for r in usable), 1),
        "by_phase": by_phase,
        "runs": out_runs,
        "note": "以一次綠燈為單位配對。delta_sec = 我方第一次判 SWITCH 的已亮秒數 − 實際綠燈長度;"
                "負值 = 我方會早切。earlier_meaningful 只算紅側當時真的有排隊的段。"
                "actual_sec 是區間量測(真值在 [值, 值+取樣週期)),同時的容忍 = 一個取樣週期。",
    }


def _actual_runs_from_frames(since_iso: str, until_iso: str) -> Optional[list]:
    """用控制器自己回報的 5F03 框(每秒一框)重建精確的綠燈段。

    🛑 使用者要求:影子比對要用「號誌控制器被操控的秒數」,不是影子 5 秒取樣
       重建的近似值。5F03 每秒一框帶 SubPhaseID/StepID/StepSec,分相何時開始、
       綠燈(StepID=1)何時結束,精確到 1 秒;取樣法最多差 5 秒。
    回傳 [{phase, start, green_end, end, green_sec}],時間為 epoch 秒;
    抄錄框 DB 不可用或該段沒框 → None(呼叫端退回取樣法並標示)。
    """
    try:
        from api.routes.signal_tc3 import decode_frame, _QDB_PATH
        import sqlite3 as _sq
        a = datetime.fromisoformat(since_iso).timestamp()
        b = datetime.fromisoformat(until_iso).timestamp()
        conn = _sq.connect(f"file:{_QDB_PATH}?mode=ro", uri=True, timeout=10)
        rows = conn.execute(
            "SELECT ts, raw FROM signal_frames WHERE code='5F03' AND cks_ok=1 "
            "AND ts>=? AND ts<? ORDER BY ts", (a, b)).fetchall()
        conn.close()
    except Exception:
        return None
    if len(rows) < 30:
        return None
    segs = []
    cur = None
    for ts, raw in rows:
        try:
            d = decode_frame(bytes.fromhex(str(raw).replace(" ", "")))
        except Exception:
            continue
        ph = (d or {}).get("phase") or {}
        sp, st = ph.get("sub_phase_id"), ph.get("step_id")
        if sp is None:
            continue
        if cur is None or sp != cur["phase"]:
            if cur is not None:
                segs.append(cur)
            cur = {"phase": sp, "start": float(ts), "green_end": None, "end": float(ts)}
        cur["end"] = float(ts)
        if st == 1:
            cur["green_end"] = float(ts)
    if cur is not None:
        segs.append(cur)
    out = []
    for sg in segs:
        if sg["green_end"] is None:
            continue
        out.append({"phase": sg["phase"], "start": sg["start"], "green_end": sg["green_end"],
                    "end": sg["end"], "green_sec": round(sg["green_end"] - sg["start"], 1)})
    return out


def _paired_precise(rows: list, actual: list, interval: float = None) -> dict:
    """精確配對:綠燈段來自控制器框(_actual_runs_from_frames),
    我方判斷來自影子 log。每段找落在 [start, green_end] 內第一筆 SWITCH。

    rows: (ts, green_phase, green_elapsed, ours, actual, forced, clearance,
           queue_m_1, queue_m_2, switch_gain, keep_gain, change_cost)
    續綠段(整段沒說 SWITCH)也給比對數據:OPAC 切相那一刻我方的裕度
    (門檻 − 紅側延滯)、紅側有沒有車、有沒有撞最大綠 —— 「我方同意續綠」
    到底同意得多堅定,這裡看得到。
    """
    if interval is None:
        interval = SHADOW_INTERVAL_SEC
    import bisect
    ts_list = []
    for r in rows:
        try:
            ts_list.append(datetime.fromisoformat(r[0]).timestamp())
        except Exception:
            ts_list.append(0.0)
    out_runs = []
    for seg in actual:
        i = bisect.bisect_left(ts_list, seg["start"] - 0.5)
        j = bisect.bisect_right(ts_list, seg["green_end"] + 0.5)
        samp = rows[i:j]
        if not samp:
            continue
        ours_sec = None
        red_waiting = None
        for r in samp:
            if r[3] == "SWITCH" and not r[6]:
                ours_sec = float(r[2] or 0.0)
                rq = r[8] if seg["phase"] == 1 else r[7]
                red_waiting = bool(rq and float(rq) > 0)
                break
        last = samp[-1]
        margin = None
        if last[9] is not None and last[10] is not None and last[11] is not None:
            margin = round(float(last[10]) + float(last[11]) - float(last[9]), 1)
        rq_last = last[8] if seg["phase"] == 1 else last[7]
        d = None
        waste = 0.0
        if ours_sec is not None:
            d = round(ours_sec - seg["green_sec"], 1)
            if red_waiting and d < -interval:
                waste = round(-d, 1)
        out_runs.append({
            "phase": seg["phase"], "start_ts": datetime.fromtimestamp(seg["start"]).isoformat(timespec="seconds"),
            "actual_sec": seg["green_sec"], "ours_sec": ours_sec, "delta_sec": d,
            "red_waiting": red_waiting, "waste_sec": waste, "samples": len(samp),
            "forced": any(bool(r[5]) for r in samp),
            # 續綠段的比對數據
            "margin_at_switch": margin,
            "red_queue_at_switch": round(float(rq_last), 1) if rq_last is not None else None,
        })
    usable = out_runs
    earlier = [r for r in usable if r["delta_sec"] is not None and r["delta_sec"] < -interval]
    meaningful = [r for r in earlier if r["red_waiting"]]
    same = [r for r in usable if r["delta_sec"] is not None and -interval <= r["delta_sec"] <= interval]
    hold = [r for r in usable if r["delta_sec"] is None]
    later = [r for r in usable if r["delta_sec"] is not None and r["delta_sec"] > interval]
    stat = lambda v: (_stat(v) if v else None)
    hold_margin = [r["margin_at_switch"] for r in hold if r["margin_at_switch"] is not None]
    hold_redq = [r for r in hold if r["red_queue_at_switch"] and r["red_queue_at_switch"] > 0]
    by_phase = {}
    for ph in (1, 2):
        sub = [r for r in usable if r["phase"] == ph]
        by_phase[str(ph)] = {
            "runs": len(sub),
            "earlier": sum(1 for r in sub if r["delta_sec"] is not None and r["delta_sec"] < -interval),
            "hold": sum(1 for r in sub if r["delta_sec"] is None),
            "actual_green": stat([r["actual_sec"] for r in sub]),
            "delta": stat([r["delta_sec"] for r in sub if r["delta_sec"] is not None]),
            "waste_sec": round(sum(r["waste_sec"] for r in sub), 1),
        }
    return {
        "source": "controller_5F03", "interval_sec": interval,
        "runs_usable": len(usable), "runs_truncated": 0,
        "earlier": len(earlier), "earlier_meaningful": len(meaningful),
        "earlier_idle": len(earlier) - len(meaningful),
        "same": len(same), "hold": len(hold), "later": len(later),
        "delta_all": stat([r["delta_sec"] for r in usable if r["delta_sec"] is not None]),
        "delta_meaningful": stat([r["delta_sec"] for r in meaningful]),
        "waste_sec_total": round(sum(r["waste_sec"] for r in usable), 1),
        "hold_compare": {
            "runs": len(hold),
            "margin_at_switch": stat(hold_margin),
            "red_waiting_at_switch": len(hold_redq),
            "red_waiting_ratio": round(len(hold_redq) / len(hold), 3) if hold else None,
            "forced_max_green": sum(1 for r in hold if r["forced"]),
            "note": "OPAC 切相那一刻,我方仍判續綠的裕度(門檻 − 紅側延滯,車·秒)。"
                    "裕度越大代表我方越堅定認為還不該切;紅側有車卻仍續綠的比例,是"
                    "「我方比 OPAC 更保守」的量。",
        },
        "by_phase": by_phase,
        "runs": out_runs,
        "note": "實際綠燈秒數取自控制器每秒回報的 5F03(SubPhaseID/StepID),精確到 1 秒;"
                "我方判斷取自影子 log(5 秒取樣)。delta_sec = 我方第一次判 SWITCH 的已亮秒數 −"
                " 控制器實際綠燈秒數;負 = 我方會早切。同時的容忍 = 一個取樣週期。",
    }


# ── 成效報告(工程 / 技術 / 完整)──────────────────────────────────────
# 停止線相機(每相一台,通過事件與排隊都以它為準)與進場道長度(上游台到停止線台)。
# 🛑 這兩個是站點幾何,不是量測值;換站點改 env。
PHASE_STOPLINE = {
    1: int(os.getenv("SIGNAL_EVAL_STOPLINE_PHASE1", "3") or 3),
    2: int(os.getenv("SIGNAL_EVAL_STOPLINE_PHASE2", "5") or 5),
}
APPROACH_LEN_M = {
    1: float(os.getenv("SIGNAL_EVAL_APPROACH_M_PHASE1", "52.7") or 52.7),
    2: float(os.getenv("SIGNAL_EVAL_APPROACH_M_PHASE2", "16.0") or 16.0),
}
PEAK_WINDOWS = ((9, 12), (17, 20))      # 使用者定義的尖峰:每天 09-12、17-20


def _eval_window(since_iso: str, until_iso: str) -> dict:
    """一個時段、兩相各自的逐週期指標。回 {phase: rows}。"""
    from detection import signal_eval as E
    from detection.signal_timing_lookup import phase_role
    cycles_all = _actual_runs_from_frames(since_iso, until_iso) or []
    cams = sorted(set(PHASE_STOPLINE.values()))
    cong = E.load_congestion(_VIOL_DB, cams, since_iso, until_iso)
    passes = E.load_passes(_VIOL_DB, cams, since_iso, until_iso)
    out = {}
    for ph in (1, 2):
        cyc = [c for c in cycles_all if c["phase"] == ph]
        cam = PHASE_STOPLINE[ph]
        role = phase_role(ph) or {}
        out[ph] = E.per_cycle_metrics(cyc, cong.get(cam, []), passes.get(cam, []),
                                      role.get("storage_m"), APPROACH_LEN_M.get(ph))
    return out


def _is_peak(ts: float) -> bool:
    h = datetime.fromtimestamp(ts).hour
    return any(a <= h < b for a, b in PEAK_WINDOWS)


@router.get("/benchmark", summary="演算法驗收:我方 vs 公認基準(固定時制/Webster/感應/MaxPressure)")
async def shadow_benchmark(minutes: int = Query(360, ge=30, le=1440),
                           since: str = Query(""), until: str = Query(""),
                           _user=Depends(get_current_user)):
    """把我方演算法跟交通工程的公認基準比,**現行控制(OPAC)不在對照組**。

    🛑 使用者 2026-09-05:「OPAC 是最差的控制,沒法驗收」。
       拿沒調好的系統當對照,贏了也只證明對方沒調好;而且它的參數不在
       我方手上,基準會漂移。OPAC 在這裡只剩校準用途 —— 用它實際發生的
       換相序列驗證模型能重現現場排隊,校準沒過一律不給比較結果。

    對照組(同一份實測到達流量、同一組安全約束):
      固定時制(現行時制表) / Webster 最佳固定時制 / 感應控制 / MaxPressure
    """
    from detection.signal_sim import (
        SimConfig, arrival_profile, calibrate, estimate_arrivals,
        estimate_saturation, profile_rate_fn, replay_actual,
    )
    from detection.signal_baselines import run_benchmark
    from detection.signal_decision_engine import ApproachState, decide
    from detection.signal_timing_lookup import (
        current_base_plan, phase_role, plan_params,
    )

    if since:
        since_iso = since
        until_iso = until or datetime.now().isoformat(timespec="seconds")
    else:
        since_iso = datetime.fromtimestamp(
            time.time() - int(minutes) * 60).isoformat(timespec="seconds")
        until_iso = datetime.now().isoformat(timespec="seconds")
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase,queue_m_1,queue_m_2 FROM signal_shadow_log "
            "WHERE ts>=? AND ts<=? AND control_mode='external_dynamic' "
            "ORDER BY ts", (since_iso, until_iso)).fetchall()
        conn.close()
    except Exception as e:
        return {"error": str(e)}
    if len(rows) < 120:
        return {"available": False, "since": since_iso, "until": until_iso,
                "reason": f"樣本僅 {len(rows)} 筆,不足以校準(需 >=120)"}

    pp = plan_params(current_base_plan()) or {}
    mins = pp.get("min_green") or [10, 20]
    cfg = SimConfig(dt_sec=SHADOW_INTERVAL_SEC,
                    min_green_sec={1: float(mins[0]), 2: float(mins[1])},
                    max_green_sec=_max_green(pp),
                    lost_time_sec=_lost_time_for(1),
                    meters_per_vehicle=_mpv())
    overall = estimate_arrivals(rows, cfg.meters_per_vehicle)
    sat = estimate_saturation(rows, overall,
                              min_start_queue_m=SAT_MIN_START_QUEUE_M)
    cfg.saturation_by_phase = {p: (sat[p]["veh_per_sec"] or None) for p in (1, 2)}
    profile = arrival_profile(rows, mpv=cfg.meters_per_vehicle)
    rate_fn = profile_rate_fn(profile)

    # ── 校準:OPAC 的實際換相序列能不能被模型重現 ──
    replay = replay_actual(rows, rate_fn, cfg)
    cal = calibrate(rows, replay, cfg.meters_per_vehicle)
    out = {
        "available": True, "since": since_iso, "until": until_iso,
        "samples": len(rows),
        "arrivals_measured": overall,
        "saturation_measured": sat,
        "calibration": cal,
        "constants": {"lost_time_sec": cfg.lost_time_sec,
                      "meters_per_vehicle": cfg.meters_per_vehicle,
                      "min_green_sec": cfg.min_green_sec,
                      "max_green_sec": cfg.max_green_sec},
    }
    if not cal.get("usable"):
        out["benchmark"] = None
        out["conclusion"] = ("校準未通過,不提供比較結果 —— "
                             "模型無法在已知控制下重現現場排隊。")
        return out

    roles = {p: (phase_role(p) or {}) for p in (1, 2)}
    mpv = cfg.meters_per_vehicle

    def ours(state):
        g = state["green_phase"]
        r = 2 if g == 1 else 1
        qv = state["queue_veh"]
        gs = ApproachState(g, queue_m=qv[g] * mpv,
                           storage_m=roles[g].get("storage_m"),
                           priority=bool(roles[g].get("priority")))
        rs = ApproachState(r, queue_m=qv[r] * mpv,
                           storage_m=roles[r].get("storage_m"),
                           priority=bool(roles[r].get("priority")),
                           waiting_sec=state["green_elapsed"])
        d = decide(green_phase=g, green_elapsed_sec=state["green_elapsed"],
                   green_side=gs, red_side=rs,
                   min_green_sec=cfg.min_green_sec.get(g, 10.0),
                   max_green_sec=cfg.max_green_sec,
                   saturation_vph=_sat_for(g),
                   meters_per_vehicle=mpv,
                   lost_time_sec=cfg.lost_time_sec)
        return d.action == "SWITCH"

    plan_green = None
    if pp.get("phase1_green") and pp.get("phase2_green"):
        plan_green = {1: float(pp["phase1_green"]), 2: float(pp["phase2_green"])}
    flows = {p: (overall[p]["veh_per_sec"] or 0.0) for p in (1, 2)}
    bench = run_benchmark(rate_fn, replay["duration_sec"], cfg, ours,
                          start_phase=rows[0][1] or 1,
                          plan_green=plan_green, flow_veh_per_sec=flows)
    out["benchmark"] = bench

    # 結論:對每個基準,我方每車延滯差幾 %(負 = 我方較低 = 較好)
    parts = []
    for row in (bench.get("vs_baseline") or {}).values():
        pct = (row.get("delay_per_veh_sec") or {}).get("pct")
        if pct is not None:
            parts.append(f"{row['label']} {pct:+.1f}%")
    out["conclusion"] = ("模擬中我方每車延滯 vs 各基準(負=我方較低):"
                         + "、".join(parts)) if parts else "無可比較的基準"
    out["caveat"] = ("這是模擬結論,不是現場實績。模型已通過校準(能在已知控制下"
                     "重現現場排隊),但仍是確定性排隊模型,沒有納入車輛異質性、"
                     "上游號誌連鎖與駕駛行為。現場實績仍須接管後做 A/B 交替時段。"
                     "現行控制(OPAC)不在對照組內,只用於校準。")
    return out


@router.get("/report", summary="成效報告:工程(min)/技術(standard)/完整(full),可 A/B 兩時段")
async def shadow_report(since: str = Query(...), until: str = Query(...),
                        b_since: str = Query(""), b_until: str = Query(""),
                        tier: str = Query("full"),
                        _user=Depends(get_current_user)):
    """指標定義見 detection/signal_eval.py。每個指標都帶 method(measured/approx),
    報告不可以把近似值寫成實測值。統計單位是控制器 5F03 重建的號誌週期。"""
    from detection import signal_eval as E
    tier = tier if tier in ("min", "standard", "full") else "full"
    from services import tdx_travel as T
    A = _eval_window(since, until)
    res = {"tier": tier, "unit": "cycle(controller_5F03)",
           "a": {"since": since, "until": until,
                 "by_phase": {str(ph): E.summarize_cycles(rows, tier) for ph, rows in A.items()},
                 "all": E.summarize_cycles(A[1] + A[2], tier),
                 # 🛑 TDX 量的是國道主線門架之間,現場量的是匝道端進場道,兩個路段不同,
                 #    分開放、分開標,不可以合併成一個「旅行時間」。
                 "travel_time_tdx": T.summarize(since, until) if tier != "min" else None}}
    if tier == "full":
        pk = [r for ph in A for r in A[ph] if _is_peak(r["start"])]
        off = [r for ph in A for r in A[ph] if not _is_peak(r["start"])]
        res["a"]["peak"] = E.summarize_cycles(pk, "standard")
        res["a"]["offpeak"] = E.summarize_cycles(off, "standard")
        res["a"]["peak_vs_offpeak"] = E.compare(off, pk)
    if b_since and b_until:
        B = _eval_window(b_since, b_until)
        res["b"] = {"since": b_since, "until": b_until,
                    "by_phase": {str(ph): E.summarize_cycles(rows, tier) for ph, rows in B.items()},
                    "all": E.summarize_cycles(B[1] + B[2], tier),
                    "travel_time_tdx": T.summarize(b_since, b_until) if tier != "min" else None}
        if tier == "full":
            res["ab_test"] = {"all": E.compare(A[1] + A[2], B[1] + B[2]),
                              "phase_1": E.compare(A[1], B[1]), "phase_2": E.compare(A[2], B[2]),
                              "note": "Welch t-test,樣本 = 號誌週期;b − a 為正代表 B 較大。"
                                      "Cohen's d:<0.2 negligible, <0.5 small, <0.8 medium, ≥0.8 large。"}
    return res


# ── TDX eTag 旅行時間(國道主線,實測)────────────────────────────────
@router.get("/tdx", summary="TDX eTag 站間旅行時間:抓取狀態與時段平均")
async def shadow_tdx(since: str = Query(""), until: str = Query(""),
                     _user=Depends(get_current_user)):
    from services import tdx_travel as T
    out = {"status": T.status()}
    if since:
        out["summary"] = T.summarize(since, until or datetime.now().isoformat(timespec="seconds"))
    return out


# 站點座標:使用者在地圖頁定的(國道 8 號新市交流道路口),跟前端 MAP_SITE_CENTER 同值。
SITE_LAT = float(os.getenv("SIGNAL_SITE_LAT", "23.063772") or 23.063772)
SITE_LON = float(os.getenv("SIGNAL_SITE_LON", "120.279169") or 120.279169)


@router.get("/tdx/discover", summary="用站點座標找最近的 eTag 配對(挑 TDX_ETAG_PAIRS 用)")
async def shadow_tdx_discover(lat: float = Query(None), lng: float = Query(None),
                              km: float = Query(None, description="(選用)交流道里程,只當交叉驗證"),
                              road_id: str = Query(""), radius_km: float = Query(8.0),
                              _user=Depends(get_current_user)):
    """🛑 以座標為準,不靠里程猜。里程(維基:新市交流道 9.7K)只拿來交叉驗證。"""
    from services import tdx_travel as T
    if not T.enabled():
        return {"enabled": False, "error": "未設定 TDX_CLIENT_ID / TDX_CLIENT_SECRET"}
    lat = SITE_LAT if lat is None else lat
    lng = SITE_LON if lng is None else lng
    try:
        out = {"enabled": True, "site": {"lat": lat, "lng": lng},
               "pairs": T.discover_by_coord(lat, lng, road_id or None, radius_km)}
        if km is not None:
            out["by_km"] = T.discover(km, road_id or None)
        return out
    except Exception as e:
        return {"enabled": True, "error": f"{type(e).__name__}: {e}"}


# ── 逐時評估(使用者:「每小時都要有」)──────────────────────────────────
# 每個整點 +2 分算前一小時:配對法(控制器 5F03 秒數)、成效核心指標、取樣一致率、
# 當時的 change_cost(看參數有沒有生效)。存 signal_hourly_eval,端點與報告都讀表;
# 沒算過的小時(含當前這一小時)才即時算,並標 partial。
HOURLY_TABLE_SQL = """CREATE TABLE IF NOT EXISTS signal_hourly_eval (
    hour TEXT PRIMARY KEY,            -- 'YYYY-MM-DDTHH'
    computed_at TEXT, partial INTEGER,
    samples INTEGER, agree_rate REAL, queue_avg_m REAL, change_cost_avg REAL,
    runs INTEGER, earlier INTEGER, earlier_meaningful INTEGER, same INTEGER, hold INTEGER, later INTEGER,
    delta_avg REAL, waste_sec REAL, source TEXT,
    delay_per_veh REAL, throughput_vph REAL, queue_eval_m REAL, cycles INTEGER)"""


def _hourly_compute(hour_iso: str) -> dict:
    """算一個小時。hour_iso = 'YYYY-MM-DDTHH'。"""
    since = hour_iso + ":00:00"
    h = datetime.fromisoformat(since)
    until = (h + timedelta(hours=1)).isoformat(timespec="seconds")
    now = datetime.now()
    partial = now < (h + timedelta(hours=1, minutes=1))
    out = {"hour": hour_iso, "partial": partial}
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase,green_elapsed,ours,actual,forced,clearance,queue_m_1,queue_m_2,"
            "switch_gain,keep_gain,change_cost,agree "
            "FROM signal_shadow_log WHERE ts>=? AND ts<? AND control_mode='external_dynamic' ORDER BY ts",
            (since, until)).fetchall()
        conn.close()
    except Exception:
        rows = []
    out["samples"] = len(rows)
    comp = [r for r in rows if r[12] is not None]
    hv = [r for r in comp if (r[7] or 0) > 0 or (r[8] or 0) > 0]
    out["agree_rate"] = round(100.0 * sum(1 for r in hv if r[12]) / len(hv), 1) if hv else None
    out["queue_avg_m"] = round(sum(max(r[7] or 0, r[8] or 0) for r in rows) / len(rows), 2) if rows else None
    ccs = [r[11] for r in rows if r[11] is not None]
    out["change_cost_avg"] = round(sum(ccs) / len(ccs), 2) if ccs else None
    actual = _actual_runs_from_frames(since, until)
    if actual and rows:
        pr = _paired_precise([r[:12] for r in rows], actual)
        out["source"] = "controller_5F03"
    elif rows:
        pr = _paired_runs([r[:9] for r in rows])
        out["source"] = "shadow_sampling_fallback"
    else:
        pr = None
        out["source"] = None
    if pr:
        out.update({"runs": pr["runs_usable"], "earlier": pr["earlier"], "earlier_meaningful": pr["earlier_meaningful"],
                    "same": pr["same"], "hold": pr["hold"], "later": pr["later"],
                    "delta_avg": (pr.get("delta_meaningful") or {}).get("avg"), "waste_sec": pr["waste_sec_total"]})
    else:
        out.update({"runs": 0, "earlier": 0, "earlier_meaningful": 0, "same": 0, "hold": 0, "later": 0,
                    "delta_avg": None, "waste_sec": 0.0})
    # 成效核心(工程報告三項),兩相合併
    try:
        from detection import signal_eval as E
        ev = _eval_window(since, until)
        allc = ev.get(1, []) + ev.get(2, [])
        sm = E.summarize_cycles(allc, "min")["core"]
        out.update({"delay_per_veh": sm["avg_delay_sec"]["value"], "throughput_vph": sm["throughput_vph"]["value"],
                    "queue_eval_m": sm["avg_queue_m"]["value"], "cycles": len(allc)})
    except Exception as e:
        out.update({"delay_per_veh": None, "throughput_vph": None, "queue_eval_m": None, "cycles": 0,
                    "eval_error": str(e)})
    return out


def _hourly_store(row: dict) -> None:
    conn = _db()
    conn.execute(HOURLY_TABLE_SQL)
    conn.execute("INSERT OR REPLACE INTO signal_hourly_eval VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (row["hour"], datetime.now().isoformat(timespec="seconds"), 1 if row.get("partial") else 0,
                  row.get("samples"), row.get("agree_rate"), row.get("queue_avg_m"), row.get("change_cost_avg"),
                  row.get("runs"), row.get("earlier"), row.get("earlier_meaningful"), row.get("same"), row.get("hold"),
                  row.get("later"), row.get("delta_avg"), row.get("waste_sec"), row.get("source"),
                  row.get("delay_per_veh"), row.get("throughput_vph"), row.get("queue_eval_m"), row.get("cycles")))
    conn.commit()
    conn.close()


HOURLY_COLS = ["hour", "computed_at", "partial", "samples", "agree_rate", "queue_avg_m", "change_cost_avg",
               "runs", "earlier", "earlier_meaningful", "same", "hold", "later", "delta_avg", "waste_sec", "source",
               "delay_per_veh", "throughput_vph", "queue_eval_m", "cycles"]


_backfill = {"thread": None, "pending": 0}


def _hourly_backfill(keys: list) -> None:
    """背景回填:一小時要解 3,600 框,十幾個小時同步算會讓端點卡幾十秒。"""
    for k in keys:
        try:
            _hourly_store(_hourly_compute(k))
        except Exception as e:
            _stats["last_error"] = f"hourly backfill {k}: {e}"
        finally:
            _backfill["pending"] = max(0, _backfill["pending"] - 1)


def _start_backfill(keys: list) -> None:
    if not keys:
        return
    t = _backfill["thread"]
    if t is not None and t.is_alive():
        return          # 已經在回填,下次呼叫再補
    _backfill["pending"] = len(keys)
    _backfill["thread"] = threading.Thread(target=_hourly_backfill, args=(keys,), name="signal-hourly-backfill", daemon=True)
    _backfill["thread"].start()


def hourly_rows(day: str, compute_missing: bool = True, max_sync: int = 2) -> dict:
    """一天 24 小時的列。存過的直接讀;沒存的(或 partial 的)最多同步算 max_sync 個
    (最近的優先),其餘丟背景回填 —— 回應裡帶 backfilling,前端隔幾秒再拉一次。"""
    conn = _db()
    conn.execute(HOURLY_TABLE_SQL)
    got = {r[0]: dict(zip(HOURLY_COLS, r)) for r in
           conn.execute("SELECT %s FROM signal_hourly_eval WHERE hour LIKE ?" % ",".join(HOURLY_COLS), (day + "T%",))}
    conn.close()
    now = datetime.now()
    keys = []
    for hh in range(24):
        key = f"{day}T{hh:02d}"
        if datetime.fromisoformat(key + ":00:00") > now:
            break
        keys.append(key)
    missing = [k for k in keys if got.get(k) is None or got[k].get("partial")]
    if compute_missing and missing:
        sync = missing[-max_sync:]          # 最近的先算(當前 partial 那小時一定在裡面)
        for k in sync:
            row = _hourly_compute(k)
            _hourly_store(row)
            got[k] = row
        rest = [k for k in missing if k not in sync]
        _start_backfill(rest)
    out = [got[k] for k in keys if got.get(k) is not None]
    return {"rows": out, "backfilling": bool(_backfill["thread"] and _backfill["thread"].is_alive()),
            "pending": _backfill["pending"]}


_last_hourly = [0.0]


def _hourly_tick() -> None:
    """給影子迴圈每輪呼叫:整點過 2 分且這小時還沒算過前一小時 → 算。"""
    now = datetime.now()
    if now.minute < 2:
        return
    prev = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H")
    if _last_hourly[0] == prev:
        return
    _last_hourly[0] = prev
    try:
        _hourly_store(_hourly_compute(prev))
    except Exception as e:
        _stats["last_error"] = f"hourly eval failed: {e}"


@router.get("/hourly", summary="逐時評估(配對/成效/一致率/參數),每整點自動算前一小時")
async def shadow_hourly(date: str = Query("", description="YYYY-MM-DD,空 = 今天"),
                        _user=Depends(get_current_user)):
    day = date or datetime.now().strftime("%Y-%m-%d")
    out = hourly_rows(day)
    out["date"] = day
    return out


@router.get("/paired", summary="逐次綠燈配對(精確比對:我方會早幾秒切)")
async def shadow_paired(minutes: int = Query(180, ge=5, le=10080),
                        since: str = Query(""), until: str = Query(""),
                        include_runs: int = Query(0, ge=0, le=1),
                        _user=Depends(get_current_user)):
    if since:
        since_iso, until_iso = since, (until or datetime.now().isoformat(timespec="seconds"))
    else:
        since_iso = datetime.fromtimestamp(time.time() - minutes * 60).isoformat(timespec="seconds")
        until_iso = datetime.now().isoformat(timespec="seconds")
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase,green_elapsed,ours,actual,forced,clearance,queue_m_1,queue_m_2,"
            "switch_gain,keep_gain,change_cost "
            "FROM signal_shadow_log WHERE ts>=? AND ts<=? AND control_mode='external_dynamic' "
            "ORDER BY ts", (since_iso, until_iso)).fetchall()
        conn.close()
    except Exception as e:
        return {"since": since_iso, "until": until_iso, "error": str(e), "runs_usable": 0}
    # 🛑 優先用控制器框的精確秒數;抄錄框拿不到才退回取樣法,並在 source 標明。
    actual = _actual_runs_from_frames(since_iso, until_iso)
    if actual:
        out = _paired_precise(rows, actual)
        sp = _frames_spacing(since_iso, until_iso)
        out["frame_interval_sec"] = sp
        out["precision_note"] = ("控制器 5F03 每秒回報,秒數精度 ±1 秒" if (sp is not None and sp <= 1.5)
                                 else "控制器 5F03 每 %s 秒回報一框,秒數精度只有一個框距" % sp)
    else:
        out = _paired_runs([r[:9] for r in rows])
        out["source"] = "shadow_sampling_fallback"
    out["since"], out["until"] = since_iso, until_iso
    if not include_runs:
        out.pop("runs", None)
    return out


@router.get("/local-metrics", summary="局部可觀測指標(不需反事實模型)")
async def shadow_local_metrics(minutes: int = Query(360, ge=30, le=10080),
                               since: str = Query(""), until: str = Query(""),
                               _user=Depends(get_current_user)):
    """三個「當下那一刻可直接觀測」的指標,以及我方引擎在同一刻的判定。

    🛑 為什麼這條路成立而模擬不成立:
       這裡的主張全部是**瞬時、局部**的 —— 例如「這一刻綠燈側沒有車也沒有
       流量,綠燈正在空放,而我方判定應換相」。這個陳述只描述那一刻,
       不需要推演「換相之後車流會如何」,所以不需要反事實模型。

    🛑 它能證明什麼、不能證明什麼:
       能 —— 我方在這些具體時刻的判斷方向較佳(有幾次、佔比多少,都可複核)。
       不能 —— 全時段總延滯較低。那要靠 A/B 交替時段的實績。
       報告裡要寫成「局部佐證」,不可以寫成「整體較優」。
    """
    from detection.signal_timing_lookup import (
        current_base_plan, phase_role, plan_params,
    )
    if since:
        since_iso = since
        until_iso = until or datetime.now().isoformat(timespec="seconds")
    else:
        since_iso = datetime.fromtimestamp(
            time.time() - int(minutes) * 60).isoformat(timespec="seconds")
        until_iso = datetime.now().isoformat(timespec="seconds")
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase,green_elapsed,queue_m_1,queue_m_2,"
            "flow_vpm_1,flow_vpm_2,ours,actual,forced,clearance,reason "
            "FROM signal_shadow_log WHERE ts>=? AND ts<=? "
            "AND control_mode='external_dynamic' ORDER BY ts",
            (since_iso, until_iso)).fetchall()
        conn.close()
    except Exception as e:
        return {"error": str(e)}
    if not rows:
        return {"available": False, "since": since_iso, "until": until_iso,
                "reason": "此區間無樣本"}

    pp = plan_params(current_base_plan()) or {}
    max_green = _max_green(pp)
    storage2 = (phase_role(2) or {}).get("storage_m") or 600
    spill_m = storage2 * DEFAULT_SPILLBACK_RATIO_LOCAL

    waste_n = waste_ours_switch = 0        # 有代價的空放:綠側沒需求、紅側有人等
    waste_flow_known = 0
    idle_both_n = 0                        # 兩側都沒需求 —— 這不算浪費
    max_q2_observed = 0.0
    maxg_n = maxg_ours_switch = 0
    spill_n = spill_ours_keep = 0
    dt = SHADOW_INTERVAL_SEC

    keep_min_green = keep_not_worth = keep_other = 0
    mins_cfg = pp.get("min_green") or [10, 20]
    for (ts, gp, el, q1, q2, f1, f2, ours, actual, forced, clr, reason) in rows:
        if clr:
            continue                      # 清道期間不算,那時本來就在換相
        gq = q1 if gp == 1 else q2
        gf = f1 if gp == 1 else f2
        rq = q2 if gp == 1 else q1
        if q2 is not None:
            max_q2_observed = max(max_q2_observed, float(q2))
        # ① 綠燈空放 —— 🛑 定義要加上「紅側有人在等」。
        #    2026-09-04 第一版只看綠側沒車,結果 59.1% 的取樣都被算成空放,
        #    但我方只有 3.1% 判定應換相 —— 因為那些時刻**兩側都沒車**(夜間
        #    離峰)。兩邊都空的綠燈不是浪費,換相沒有任何好處,判 KEEP 是對的。
        #    真正有代價的空放是「綠側沒需求、紅側有人等」,那才是我方該贏的地方。
        green_idle = (gq is not None) and float(gq) <= 0 and (
            gf is None or float(gf) <= 0)
        red_waiting = (rq is not None) and float(rq) > 0
        if green_idle and not red_waiting:
            idle_both_n += 1
        elif green_idle and red_waiting:
            waste_n += 1
            if gf is not None:
                waste_flow_known += 1
            if ours == "SWITCH":
                waste_ours_switch += 1
            else:
                # 我方也判 KEEP 的原因要拆開 —— 「未滿最小綠」是安全約束
                # (我方遵守規則,不是判斷失準),「成本比較不值得切」才是
                # 演算法的實質選擇,兩者混在一起看不出問題出在哪。
                mg = float(mins_cfg[gp - 1]) if len(mins_cfg) >= gp else 10.0
                if float(el or 0) < mg:
                    keep_min_green += 1
                elif "≤" in (reason or ""):
                    keep_not_worth += 1
                else:
                    keep_other += 1
        # ② 最大綠撞頂:實際被迫換相
        if forced:
            maxg_n += 1
            if ours == "SWITCH":
                maxg_ours_switch += 1
        # ③ 下匝道回堵:排隊達儲車上限比例
        if q2 is not None and float(q2) >= spill_m:
            spill_n += 1
            # 主線保護的正解是「不要把綠燈從下匝道切走」
            if gp == 2 and ours == "KEEP":
                spill_ours_keep += 1

    def pct(a, b):
        return round(a / b * 100, 1) if b else None

    return {
        "available": True, "since": since_iso, "until": until_iso,
        "samples": len(rows), "interval_sec": dt,
        "green_waste": {
            "samples": waste_n,
            "seconds": round(waste_n * dt, 1),
            "share_pct": pct(waste_n, len(rows)),
            "ours_switch": waste_ours_switch,
            "ours_switch_pct": pct(waste_ours_switch, waste_n),
            "flow_known_samples": waste_flow_known,
            "ours_keep_min_green": keep_min_green,
            "ours_keep_not_worth": keep_not_worth,
            "ours_keep_other": keep_other,
            "idle_both_sides": idle_both_n,
            "idle_both_seconds": round(idle_both_n * dt, 1),
            "criteria": "綠燈側無需求(排隊 0 且流量 0)**且紅燈側有人在等**"
                        " —— 兩側都空不算浪費,換相沒有好處",
        },
        "max_green_hit": {
            "samples": maxg_n,
            "max_green_sec": max_green,
            "ours_switch": maxg_ours_switch,
            "ours_switch_pct": pct(maxg_ours_switch, maxg_n),
            "criteria": "實際達最大綠被迫換相;我方在同一刻是否也判定應換相",
        },
        "spillback": {
            "samples": spill_n,
            "threshold_m": round(spill_m, 1),
            "storage_m": storage2,
            "ours_protect": spill_ours_keep,
            "max_observed_m": round(max_q2_observed, 1),
            # 🛑 門檻可能超出量測範圍:ROI 看不到那麼長的隊伍,
            #    這時「0 次回堵」只代表沒量到,不代表沒發生。
            "threshold_reachable": max_q2_observed >= spill_m * 0.6,
            "criteria": f"下匝道排隊 ≥ 儲車上限 {storage2}m 的 "
                        f"{int(DEFAULT_SPILLBACK_RATIO_LOCAL*100)}% = {spill_m:.0f}m",
        },
        "note": "🛑 這些是**局部佐證**:每一項都只描述那一刻可直接觀測的事實,"
                "不需要推演換相後的車流,所以不需要反事實模型。"
                "但它們證明不了全時段總延滯較低 —— 那要靠 A/B 交替時段的實績。",
    }

@router.get("/timeline", summary="時間軸(壓縮平行陣列,給即時總覽畫圖用)")
async def shadow_timeline(minutes: int = Query(15, ge=1, le=1440),
                          since: str = Query(""), until: str = Query(""),
                          max_points: int = Query(900, ge=100, le=5000),
                          _user=Depends(get_current_user)):
    """回傳一段時間的分相、排隊、流量與決策,給前端畫同步時間軸。

    🛑 回**平行陣列**不回物件陣列:15 分鐘 180 筆 × 11 欄,物件陣列會把欄位名
       重複 180 次,體積差三倍以上。這個端點會被前端每幾秒重打一次。

    🛑 帶 `t`(距 t0 的秒數)而**不是**只給 interval_sec 就假設等距:
       抄錄 stale 時影子會跳過取樣,樣本之間會有洞。只給 interval_sec 的話,
       一個 40 秒的洞會被畫成一格 —— 時間軸會被壓扁,換相與排隊的對應位置
       全部跑掉。`gaps` 另外標出洞的區間,前端要畫成斷線不要內插。
       (這個坑先前踩過:凍結的抄錄資料曾讓 green_elapsed 累積出假的長綠。)
    """
    if since:
        since_iso = since
        until_iso = until or datetime.now().isoformat(timespec="seconds")
    else:
        since_iso = datetime.fromtimestamp(
            time.time() - int(minutes) * 60).isoformat(timespec="seconds")
        until_iso = datetime.now().isoformat(timespec="seconds")
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT ts,green_phase,green_elapsed,queue_m_1,queue_m_2,"
            "flow_vpm_1,flow_vpm_2,ours,actual,agree,switch_gain,keep_gain,"
            "change_cost,reason,clearance,step_id "
            "FROM signal_shadow_log WHERE ts>=? AND ts<=? "
            "AND control_mode='external_dynamic' ORDER BY ts",
            (since_iso, until_iso)).fetchall()
        conn.close()
    except Exception as e:
        return {"available": False, "error": str(e)}
    if not rows:
        return {"available": False, "since": since_iso, "until": until_iso,
                "reason": "此區間無樣本"}

    # 抽樣:視窗拉長時筆數會爆(24 小時約 17000 筆)。
    # 🛑 抽樣不能只是每 N 筆取一筆 —— 那會把換相事件抽掉,而換相正是要看的東西。
    #    改成分桶,桶內的換相(ours/actual)用 OR 保留,其餘取桶內最後一筆。
    stride = max(1, (len(rows) + max_points - 1) // max_points)
    t0_dt = datetime.fromisoformat(rows[0][0])
    t0_ts = t0_dt.timestamp()

    T, GP, Q1, Q2, F1, F2, OU, AC, AG, SG, KG, CC, RS, CL = (
        [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    bucket: list = []

    def flush(b):
        if not b:
            return
        last = b[-1]
        T.append(round(datetime.fromisoformat(last[0]).timestamp() - t0_ts, 1))
        GP.append(last[1])
        Q1.append(last[3])
        Q2.append(last[4])
        F1.append(last[5])
        F2.append(last[6])
        # 換相事件用 OR —— 抽樣不可以把它抽掉
        OU.append(1 if any(x[7] == "SWITCH" for x in b) else 0)
        AC.append(1 if any(x[8] == "SWITCH" for x in b) else 0)
        AG.append(last[9])
        SG.append(last[10])
        KG.append(last[11])
        CC.append(last[12])
        CL.append(1 if last[14] else 0)
        # reason 只在歧異點帶字串 —— 一致的點不需要理由,全帶會讓回應肥三倍
        dis = next((x for x in b if x[9] == 0), None)
        RS.append(dis[13] if dis else None)

    for r in rows:
        bucket.append(r)
        if len(bucket) >= stride:
            flush(bucket)
            bucket = []
    flush(bucket)

    # 取樣斷點:間隔超過名目週期 1.6 倍就是洞,前端要畫成斷線不要內插
    gaps = []
    nominal = SHADOW_INTERVAL_SEC * stride
    for i in range(len(T) - 1):
        if T[i + 1] - T[i] > nominal * 1.6:
            gaps.append([T[i], T[i + 1]])

    return {
        "available": True,
        "t0": rows[0][0], "since": since_iso, "until": until_iso,
        "interval_sec": nominal, "stride": stride,
        "samples_raw": len(rows), "points": len(T),
        "t": T,
        "green_phase": GP,
        "queue_m_1": Q1, "queue_m_2": Q2,
        "flow_vpm_1": F1, "flow_vpm_2": F2,
        "ours_switch": OU, "actual_switch": AC, "agree": AG,
        "switch_gain": SG, "keep_gain": KG, "change_cost": CC,
        "clearance": CL,
        "reason": RS,
        "gaps": gaps,
        "note": "t 是距 t0 的秒數,不要假設等距;gaps 標出取樣斷點,"
                "那些區間要畫成斷線不要內插 —— 抄錄過期時影子會跳過取樣。",
    }
