"""演算法運轉:下發路徑唯一且層層把關 + 切換偵測正確性。

2026-09-07 之前這個模組是純影子(絕對不下發),使用者要求「演算法上去」後
改為可下發。舊的「絕對不下發」保證因此換成三條更精確的保證:
  (1) 下發只有 _actuate 一條路,不會有第二個地方偷送;
  (2) 預設關閉,要明確開啟才會動到路口;
  (3) 它走的是與人工下發**同一道**把關(_control_guard),不是自己的簡化版。
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import pytest

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_only_one_send_path():
    """★最重要:下發只能有 _actuate 一條路。

    下發是對 signal_daemon 送 control/send,所以這裡掃的是「誰在呼叫
    _daemon_post」。多一個地方能送,就多一個沒被把關的路口控制入口。
    """
    import ast
    src = (ROOT / "api" / "routes" / "signal_shadow.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    def calls_daemon_post(node):
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "_daemon_post" for n in ast.walk(node))

    senders = [fn.name for fn in ast.walk(tree)
               if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
               and calls_daemon_post(fn)]
    assert senders == ["_actuate"], "下發路徑不只一條:%s" % senders


def test_never_sends_in_process():
    """🛑 不可以自己 import signal_tc3 直接送。

    連著 :1001 的 socket 與控制策略只存在 traffic-signal 那個行程;
    在 traffic-api 裡直接呼叫等於「看起來有送、其實什麼都沒發生」。
    """
    import ast
    src = (ROOT / "api" / "routes" / "signal_shadow.py").read_text(encoding="utf-8")
    called = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call):
            f = n.func
            called.add(f.id if isinstance(f, ast.Name) else
                       (f.attr if isinstance(f, ast.Attribute) else ""))
    forbidden = {"_controller_send", "control_send", "sendall",
                 "_send_frame", "send_frame", "write_frame"}
    assert not (called & forbidden), "不可在本行程直接下發:%s" % (called & forbidden)


def test_actuate_disabled_by_default():
    """預設不下發 —— 要明確開啟才會動到路口號誌。"""
    import importlib
    import api.routes.signal_shadow as m
    os.environ.pop("SIGNAL_SHADOW_ACTUATE", None)
    m = importlib.reload(m)
    assert m.ACTUATE_DEFAULT is False
    assert m._act["enabled"] is False


def _fake_daemon(monkeypatch, m, calls):
    def post(path, body):
        calls.append((path, body))
        if path.endswith("/prepare"):
            return {"token": "T"}
        return {"sent": {"seq": 7, "raw": "AA 5F 1C"}}
    monkeypatch.setattr(m, "_daemon_post", post)


class _D:
    action = "SWITCH"
    reason = "測試"


LIVE_OK = {"control_mode": "external_dynamic", "clearance": False, "stale": False}


def test_actuate_blocked_when_disabled(monkeypatch):
    """關閉時不管引擎判什麼都不送,而且要說得出原因。"""
    import api.routes.signal_shadow as m
    calls = []
    _fake_daemon(monkeypatch, m, calls)
    monkeypatch.setitem(m._act, "enabled", False)
    m._actuate(_D(), 1, LIVE_OK)
    assert calls == []
    assert m._act["blocked"]


def test_actuate_requires_phase_control(monkeypatch):
    """控制策略沒有時相控制(bit4)就不送 —— 送了控制器一定回 NAK。"""
    import api.routes.signal_shadow as m
    calls = []
    _fake_daemon(monkeypatch, m, calls)
    monkeypatch.setitem(m._act, "enabled", True)
    monkeypatch.setitem(m._act, "last_ts", 0.0)
    m._actuate(_D(), 1, dict(LIVE_OK, control_mode="fixtime"))
    assert calls == []
    assert "時相控制" in m._act["blocked"]


def test_actuate_not_during_clearance_or_stale(monkeypatch):
    """清道中不送(會變成連跳兩步階);抄錄過期也不送(看不到現況不動路口)。"""
    import api.routes.signal_shadow as m
    calls = []
    _fake_daemon(monkeypatch, m, calls)
    monkeypatch.setitem(m._act, "enabled", True)
    monkeypatch.setitem(m._act, "last_ts", 0.0)
    m._actuate(_D(), 1, dict(LIVE_OK, clearance=True))
    m._actuate(_D(), 1, dict(LIVE_OK, stale=True))
    assert calls == []


def test_actuate_sends_next_step_and_throttles(monkeypatch):
    """實際送出時:內容是 5F1C info=000000(跳下一步階),而且會節流。

    🛑 為什麼不是指定對向綠燈步階:那會跳過清道(行閃/行紅/黃/全紅)。
    """
    import api.routes.signal_shadow as m
    calls = []
    _fake_daemon(monkeypatch, m, calls)
    monkeypatch.setitem(m._act, "enabled", True)
    monkeypatch.setitem(m._act, "last_ts", 0.0)
    monkeypatch.setitem(m._act, "n", 0)

    m._actuate(_D(), 1, LIVE_OK)
    # 🛑 by=algorithm 是稽核標籤:演算法與人工走同一支下發端點,沒有它
    #    signal_frames 裡兩者長得一模一樣,驗收就答不出「這幾次是誰送的」。
    assert calls[0] == ("/api/signal/control/prepare",
                        {"code": "5F1C", "info_hex": "000000", "by": "algorithm"})
    assert calls[1][0] == "/api/signal/control/send"
    assert m._act["n"] == 1

    # 立刻再判一次 SWITCH:節流要擋下來
    m._actuate(_D(), 1, LIVE_OK)
    assert len(calls) == 2
    assert "節流" in m._act["blocked"]


def test_actuate_reports_daemon_rejection(monkeypatch):
    """daemon 把關擋下(403)時:不可以當成送出成功,原因要留給畫面看。"""
    import api.routes.signal_shadow as m

    def post(path, body):
        raise RuntimeError("daemon 403: 目前限制為「只准查詢」")

    monkeypatch.setattr(m, "_daemon_post", post)
    monkeypatch.setitem(m._act, "enabled", True)
    monkeypatch.setitem(m._act, "last_ts", 0.0)
    monkeypatch.setitem(m._act, "n", 0)
    m._actuate(_D(), 1, LIVE_OK)
    assert m._act["n"] == 0
    assert "只准查詢" in m._act["blocked"]


def test_shadow_disabled_by_default():
    """預設不啟用 —— 要明確開啟才跑。"""
    import importlib
    import api.routes.signal_shadow as m
    os.environ.pop("SIGNAL_SHADOW_ENABLED", None)
    m = importlib.reload(m)
    assert m.SHADOW_ENABLED is False


def test_interval_aligns_with_opac():
    """取樣週期預設 5 秒,與 OPAC 的決策週期對齊才好比對。"""
    import importlib
    import api.routes.signal_shadow as m
    os.environ.pop("SIGNAL_SHADOW_INTERVAL_SEC", None)
    m = importlib.reload(m)
    assert m.SHADOW_INTERVAL_SEC == 5


def test_phase_camera_mapping_matches_baseline():
    """分相→相機的對應要與官方時制表的 constraint_camera 一致。

    baseline: 分相1(上匝道)=ID3、分相2(下匝道)=ID4
    """
    import importlib
    import api.routes.signal_shadow as m
    for k in ("SIGNAL_SHADOW_CAM_PHASE1", "SIGNAL_SHADOW_CAM_PHASE2"):
        os.environ.pop(k, None)
    m = importlib.reload(m)
    assert m.PHASE_CAMERA[1] == 3
    assert m.PHASE_CAMERA[2] == 4

    # 🛑 決策的量測要涵蓋該相的**所有**相機,不是只有基準測點。
    #    現場四台 NE-1 / NE-2 / WN-1 / WN-2(相機 id 2/3/4/5);先前只用 constraint_camera
    #    各取一台,等於少看一半的進場,而 switch_gain 直接由排隊車數算出來。
    assert sorted(m.PHASE_CAMERAS[1]) == [2, 3], "分相1 要含 NE-1 與 NE-2"
    assert sorted(m.PHASE_CAMERAS[2]) == [4, 5], "分相2 要含 WN-1 與 WN-2"
    allcams = sorted(m.PHASE_CAMERAS[1] + m.PHASE_CAMERAS[2])
    assert allcams == [2, 3, 4, 5], f"四台都要對應到,實際 {allcams}"
    # constraint_camera 必須落在該相的相機清單裡,否則兩者對不起來
    for ph in (1, 2):
        assert m.PHASE_CAMERA[ph] in m.PHASE_CAMERAS[ph]


def test_queue_m_returns_none_when_no_data():
    """壅塞沒資料時回 None 不當機(決策端會當 0 處理)。"""
    import api.routes.signal_shadow as m
    assert m._queue_m(99999) is None


def test_live_phase_none_when_no_frames():
    """沒有抄到燈態時回 None,迴圈會跳過該輪而不是亂算。"""
    import api.routes.signal_shadow as m
    r = m._live_phase()
    assert r is None or isinstance(r.get("sub_phase_id"), int)


def test_stop_is_idempotent():
    import api.routes.signal_shadow as m
    m.stop_shadow()
    m.stop_shadow()   # 重複呼叫不可當機


def test_summarize_splits_active_and_idle_samples(tmp_path, monkeypatch):
    """一致率必須分「有車/無車」算。

    夜間兩側排隊都 0、兩邊都 KEEP，一致率會漂到 100%，那個數字沒有資訊量。
    實測 13.5 小時整體 87.4%，但只看有車樣本，尖峰只有 54.7% —— 若不分開算，
    尖峰的真實表現會被夜間的假一致蓋掉。
    """
    import sqlite3
    from api.routes import signal_shadow as ss

    db = tmp_path / "s.db"
    monkeypatch.setattr(ss, "_DB_PATH", str(db))
    monkeypatch.setattr(ss, "_db_ready", False)

    conn = ss._db()
    now = datetime.now()
    rows = []
    # 90 筆無車、全都一致(夜間)
    for i in range(90):
        rows.append((now.isoformat(timespec="seconds"), 1, 30.0, 0.0, 0.0,
                     "KEEP", "KEEP", 1, 0, 0, 12.5, 0, 0, ""))
    # 10 筆有車，其中只有 2 筆一致 → 有車一致率應為 20%
    for i in range(10):
        agree = 1 if i < 2 else 0
        ours = "KEEP" if agree else "SWITCH"
        rows.append((now.isoformat(timespec="seconds"), 1, 30.0, 0.0, 40.0,
                     ours, "KEEP", agree, 100.0, 0.0, 12.5, 0, 0, ""))
    conn.executemany(
        "INSERT INTO signal_shadow_log(ts,green_phase,green_elapsed,queue_m_1,"
        "queue_m_2,ours,actual,agree,switch_gain,keep_gain,change_cost,forced,"
        "blocked,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    s = ss.summarize(minutes=60)
    assert s["samples"] == 100
    assert s["active_samples"] == 10
    assert s["agree_rate"] == 0.92          # 整體被夜間拉高
    assert s["active_agree_rate"] == 0.2    # 有車時的真實表現
    assert s["disagree_switch_early"] == 8  # 岐異全是我方提早切
    assert s["disagree_switch_late"] == 0
    assert s["keep_gain_zero"] == 8         # 綠側價值=0 是根因


def test_summarize_reports_what_it_excluded(tmp_path, monkeypatch):
    """一致率必須說清楚它是拿哪些樣本比出來的。

    三種樣本前提不成立、不能列入:清道(黃燈/全紅,控制器已 committed)、
    非外部動態控制(定時/手動時 actual 不是 OPAC 的決策)、切換瞬間。
    只給一個裸的一致率而不攤開排除量,沒人能判斷那個數字可不可信。
    """
    from api.routes import signal_shadow as ss

    db = tmp_path / "s2.db"
    monkeypatch.setattr(ss, "_DB_PATH", str(db))
    monkeypatch.setattr(ss, "_db_ready", False)
    conn = ss._db()
    ts = datetime.now().isoformat(timespec="seconds")

    def row(agree, clearance, mode, q2=40.0, ours="KEEP"):
        return (ts, 1, 30.0, 0.0, q2, ours, "KEEP", agree,
                0.0, 0.0, 12.5, 0, 0, "", 1, clearance, mode)

    rows = [row(1, 0, "external_dynamic") for _ in range(6)]
    rows += [row(0, 0, "external_dynamic", ours="SWITCH") for _ in range(4)]
    rows += [row(None, 1, "external_dynamic") for _ in range(3)]   # 清道
    rows += [row(None, 0, "fixtime") for _ in range(5)]            # 定時,非 OPAC
    rows += [row(None, 0, "external_dynamic")]                     # 切換瞬間
    conn.executemany(
        "INSERT INTO signal_shadow_log(ts,green_phase,green_elapsed,queue_m_1,"
        "queue_m_2,ours,actual,agree,switch_gain,keep_gain,change_cost,forced,"
        "blocked,reason,step_id,clearance,control_mode) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    s = ss.summarize(minutes=60)
    assert s["samples"] == 19
    assert s["judged_samples"] == 10          # 只有前提成立的才算
    assert s["active_agree_rate"] == 0.6      # 6/10
    assert s["excluded_clearance"] == 3
    assert s["excluded_not_opac"] == 5
    assert s["excluded_switch_instant"] == 1


def test_report_schedule_survives_restart(tmp_path, monkeypatch):
    """上次回報時刻要落地,重啟不能把一小時的計時歸零。

    2026-09-03 部署頻繁,14:04 與 14:56 兩次重啟讓 14 點那個小時整個
    沒有回報 —— 因為 _last_report 只存在行程記憶體裡。
    """
    from api.routes import signal_shadow as ss

    db = tmp_path / "s3.db"
    monkeypatch.setattr(ss, "_DB_PATH", str(db))
    monkeypatch.setattr(ss, "_db_ready", False)

    assert ss._last_report_at() == 0.0        # 全新 DB:從未回報
    ss._mark_reported(1_000_000.0)
    assert ss._last_report_at() == 1_000_000.0

    # 模擬重啟:行程內變數歸零,但從 DB 讀得回來
    monkeypatch.setattr(ss, "_db_ready", False)
    assert ss._last_report_at() == 1_000_000.0


def test_summarize_fixed_window_and_hourly_breakdown(tmp_path, monkeypatch):
    """比對固定時段(如尖峰 06-12)要能指定起訖,而且逐時要拆得開。

    「最近 N 分鐘」會隨查詢時間漂移 —— 早一分鐘晚一分鐘查到的不是同一段,
    兩次結果沒有可比性。而整段平均會被無車時段稀釋:2026-09-03 實測整體
    87.4%,但拆開來 08 時只有 54.7%。
    """
    from api.routes import signal_shadow as ss

    db = tmp_path / "s4.db"
    monkeypatch.setattr(ss, "_DB_PATH", str(db))
    monkeypatch.setattr(ss, "_db_ready", False)
    conn = ss._db()

    def row(ts, agree, q2):
        ours = "KEEP" if agree else "SWITCH"
        return (ts, 1, 30.0, 0.0, q2, ours, "KEEP", agree,
                0.0, 0.0, 12.5, 0, 0, "", 1, 0, "external_dynamic")

    rows = []
    rows += [row("2026-09-04T07:%02d:00" % i, 1, 40.0) for i in range(10)]  # 07 全對
    rows += [row("2026-09-04T08:%02d:00" % i, 0, 40.0) for i in range(8)]   # 08 全錯
    rows += [row("2026-09-04T08:%02d:30" % i, 1, 40.0) for i in range(2)]   # 08 對 2
    rows += [row("2026-09-04T13:%02d:00" % i, 1, 40.0) for i in range(20)]  # 視窗外
    conn.executemany(
        "INSERT INTO signal_shadow_log(ts,green_phase,green_elapsed,queue_m_1,"
        "queue_m_2,ours,actual,agree,switch_gain,keep_gain,change_cost,forced,"
        "blocked,reason,step_id,clearance,control_mode) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()

    s = ss.summarize(since="2026-09-04T06:00:00", until="2026-09-04T12:00:00")
    assert s["samples"] == 20                 # 13 時那 20 筆不在視窗內
    assert s["active_agree_rate"] == 0.6      # 12/20
    by = {b["hour"]: b for b in s["by_hour"]}
    assert set(by) == {"07", "08"}
    assert by["07"]["active_agree_rate"] == 1.0
    assert by["08"]["active_agree_rate"] == 0.2   # 尖峰掉下來,整段平均看不出來


def test_shadow_routes_registered_before_signal_proxy():
    """影子路由必須註冊在 /api/signal/{sub_path:path} 萬用代理之前。

    🛑 2026-09-03 實際事故:api/main.py 把 signal_shadow.router 註冊在那個
       代理之後,Starlette 依註冊順序比對 → /api/signal/shadow/* 整組被代理
       吃掉、轉去 traffic-signal daemon(它沒有這些路由)→ 一律回 404。
       網頁的影子卡因此一直顯示「未啟動/沒有樣本」,而後端資料明明在寫。
       從 localhost 測看到 401 還誤判成「端點存在」—— 那個 401 是
       middleware 擋在路由之前,亂打的路徑也一樣回 401。

    這裡不啟動 app(會載模型),直接讀原始碼比對兩者的出現順序。
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "api" / "main.py").read_text(
        encoding="utf-8")
    inc = src.index("app.include_router(signal_shadow.router)")
    proxy = src.index('@app.api_route("/api/signal/{sub_path:path}"')
    assert inc < proxy, (
        "signal_shadow.router 必須註冊在 /api/signal 萬用代理之前,"
        "否則影子端點會被代理吃掉並回 404")

    # 代理內也要留防呆,萬一順序又被改回去至少報得出原因
    assert re.search(r'sub_path\s*==\s*"shadow"', src), \
        "萬用代理應保留 shadow 防呆,避免靜默轉發成 404"


def test_green_runs_rebuild_and_quality_flags():
    """綠燈長度重建:靠 green_elapsed 變小判斷換相,單取樣段要能挑出來。

    不能靠比對 sub_phase_id —— 分相在 1/2 之間來回,單看編號分不出
    「同一個分相的第二輪」。而只有一個取樣的段等於從沒看它長大過,
    長度是假的(實測 6 小時 531 段中有 1 段 0.0 秒,但分相2 最小綠是 20 秒)。
    """
    from api.routes.signal_shadow import _green_runs, _stat

    rows = [
        ("t1", 1, 5.0, 0, None, None, None, None),
        ("t2", 1, 10.0, 0, None, None, None, None),
        ("t3", 1, 15.0, 0, None, None, None, None),
        ("t4", 2, 0.0, 0, None, None, None, None),   # 換相
        ("t5", 2, 5.0, 0, None, None, None, None),
        ("t6", 1, 0.0, 0, None, None, None, None),   # 又換回分相1(編號重複)
        ("t7", 1, 8.0, 1, None, None, None, None),   # 這段有強制切換
        ("t8", 2, 0.0, 0, None, None, None, None),   # 單取樣段:長度沒觀測到
    ]
    runs = _green_runs(rows)
    assert [r["phase"] for r in runs] == [1, 2, 1, 2]
    assert [r["green_sec"] for r in runs] == [15.0, 5.0, 8.0, 0.0]
    assert [r["samples"] for r in runs] == [3, 2, 2, 1]
    assert runs[2]["forced"] is True
    # 🛑 排除條件要看「長度是不是 0」,不能看「取樣數<2」——
    #    抄錄 stale 跳過時 prev_phase 會清掉,下一筆重新起算 elapsed=0,
    #    連兩筆都落在 0 就會拼出取樣數 2 但長度 0 的假段(現場實測遇到)。
    rows2 = rows + [("t9", 2, 0.0, 0, None, None, None, None)]
    runs2 = _green_runs(rows2)
    zero = [r for r in runs2 if r["green_sec"] <= 0]
    assert zero and zero[-1]["samples"] == 2      # 兩筆取樣但長度仍是 0
    assert [r for r in runs if r["green_sec"] <= 0] == [runs[3]]

    st = _stat([15.0, 5.0, 8.0])
    assert st["n"] == 3
    assert st["avg"] == pytest.approx(9.3, abs=0.05)
    assert _stat([])["avg"] is None      # 沒樣本不用 0 代表


def test_switch_detection_sample_never_counted(tmp_path, monkeypatch):
    """偵測到換相的那一筆一律不列入一致率,不能靠 green_elapsed 門檻判。

    🛑 2026-09-04 回歸:原本用 green_elapsed < 1.0 當「切換瞬間」的代理條件,
       那只在自己推算秒數(切換瞬間必為 0)時成立。改用抄錄器的精確已亮秒數
       後,同一筆變成 1.8 秒,條件失效 —— 每一次換相都被算成岐異
       (6 小時約 530 次),一致率會被整片拉垮。
       actual=SWITCH 代表分相已經變了,是過去事件;我方在該刻評估的是
       「新分相要不要再切」,問的不是同一件事。
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "api" / "routes"
           / "signal_shadow.py").read_text(encoding="utf-8")
    # 排除條件裡不可以再出現「用 green_elapsed 門檻判切換瞬間」
    assert not re.search(r'actual\s*==\s*"SWITCH"\s*and\s*green_elapsed', src), \
        "切換瞬間的排除不可以依賴 green_elapsed 門檻"
    assert re.search(r'None if \(\s*\n\s*actual == "SWITCH"', src), \
        "agree=NULL 的第一個條件應該是單看 actual == 'SWITCH'"


def test_normal_sampling_is_not_flagged_as_truncated():
    """正常取樣間隔不可以被當成「斷點」。

    🛑 2026-09-04 實測:max_inner_gap 原本寫成 gap > 0 才記,但正常取樣本來
       就每 5 秒一筆 —— 結果 417 段全被標成「不確定(斷點)」,這個欄位等於
       完全失去意義。要比的是「異常大的間隔」。
    """
    from api.routes.signal_shadow import _green_runs, _run_after_gap, SHADOW_INTERVAL_SEC

    step = SHADOW_INTERVAL_SEC
    rows = [("2026-09-04T08:00:%02d" % int(i * step), 1, i * step, 0,
             None, None, None, None) for i in range(5)]
    runs = _green_runs(rows)
    assert len(runs) == 1
    assert not runs[0].get("max_inner_gap"), "正常間隔不該被標成段內斷點"
    assert not _run_after_gap(runs[0])

    # 真的有斷點(中間少了好幾筆)才要標
    rows_gap = rows[:2] + [("2026-09-04T08:00:40", 1, 40.0, 0,
                            None, None, None, None)]
    runs_gap = _green_runs(rows_gap)
    assert runs_gap[0].get("max_inner_gap"), "異常大的間隔要標成段內斷點"


def test_measured_saturation_falls_back_when_implausible(monkeypatch):
    """量到的飽和流不合理時必須退回預設,不能讓一次異常量測帶偏控制邏輯。

    🛑 change_cost = 損失時間 × 飽和流 × 損失時間 —— 飽和流直接決定
       「值不值得換相」的門檻。若某次量測因為 ROI 遮蔽或車種異常而算出
       離譜的值,照單全收會讓演算法整段時間亂切或完全不切。
    """
    from api.routes import signal_shadow as ss
    from detection.signal_decision_engine import DEFAULT_SATURATION_VPH

    ss._measured_sat.update({"vph": {}, "ts": None, "source": "default"})
    assert ss._sat_for(1) == DEFAULT_SATURATION_VPH      # 沒量到 → 預設

    ss._measured_sat["vph"] = {1: 700.0}
    assert ss._sat_for(1) == 700.0                       # 量到就用量到的
    assert ss._sat_for(2) == DEFAULT_SATURATION_VPH      # 沒量到的相仍用預設

    # 界限:_refresh_saturation 只收 SAT_MIN_VPH ~ SAT_MAX_VPH 之間的值
    assert ss.SAT_MIN_VPH > 0 and ss.SAT_MAX_VPH > ss.SAT_MIN_VPH
    assert not (ss.SAT_MIN_VPH <= 50 <= ss.SAT_MAX_VPH), "50 vph 應被視為異常"
    assert not (ss.SAT_MIN_VPH <= 9000 <= ss.SAT_MAX_VPH), "9000 vph 應被視為異常"


def _tl_rows(conn, specs):
    """specs: [(ts_iso, green_phase, ours, actual, agree, reason), ...]"""
    conn.executemany(
        "INSERT INTO signal_shadow_log(ts,green_phase,green_elapsed,queue_m_1,"
        "queue_m_2,ours,actual,agree,switch_gain,keep_gain,change_cost,forced,"
        "blocked,reason,step_id,clearance,control_mode,flow_vpm_1,flow_vpm_2) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(ts, gp, 10.0, 5.0, 8.0, ours, actual, agree, 1.0, 2.0, 3.0, 0, 0,
          reason, 1, 0, "external_dynamic", 1.0, 2.0)
         for (ts, gp, ours, actual, agree, reason) in specs])
    conn.commit()


def test_timeline_stride_keeps_switch_events(tmp_path, monkeypatch):
    """🛑 抽樣不可以把換相事件抽掉 —— 換相正是時間軸要看的東西。

    每 N 筆取一筆的做法會讓換相剛好落在被丟掉的位置時整個消失。
    正解是分桶,桶內的 ours/actual 用 OR 保留。
    """
    from api.routes import signal_shadow as ss

    db = tmp_path / "tl.db"
    monkeypatch.setattr(ss, "_DB_PATH", str(db))
    monkeypatch.setattr(ss, "_db_ready", False)
    conn = ss._db()
    # 30 筆,只有第 7 筆有換相 —— stride=10 時它會落在桶內非最後一筆
    specs = []
    for i in range(30):
        ts = "2026-09-05T08:%02d:%02d" % (i // 12, (i % 12) * 5)
        sw = (i == 7)
        specs.append((ts, 1, "SWITCH" if sw else "KEEP",
                      "SWITCH" if sw else "KEEP", None if sw else 1, None))
    _tl_rows(conn, specs)
    conn.close()

    import asyncio
    r = asyncio.new_event_loop().run_until_complete(
        ss.shadow_timeline(minutes=1440, since="2026-09-05T00:00:00",
                           until="2026-09-05T23:59:59", max_points=3,
                           _user=None))
    assert r["available"] and r["stride"] >= 10
    assert sum(r["ours_switch"]) == 1, "抽樣把我方換相抽掉了"
    assert sum(r["actual_switch"]) == 1, "抽樣把實際換相抽掉了"


def test_timeline_marks_sampling_gaps(tmp_path, monkeypatch):
    """🛑 取樣斷點要標出來,不可以讓前端當成等距。

    抄錄 stale 時影子會跳過取樣;只給 interval_sec 的話,一個 40 秒的洞會被
    畫成一格,換相與排隊的對應位置全部跑掉。
    """
    from api.routes import signal_shadow as ss

    db = tmp_path / "tl2.db"
    monkeypatch.setattr(ss, "_DB_PATH", str(db))
    monkeypatch.setattr(ss, "_db_ready", False)
    conn = ss._db()
    specs = [("2026-09-05T08:00:%02d" % (i * 5), 1, "KEEP", "KEEP", 1, None)
             for i in range(4)]
    specs += [("2026-09-05T08:01:%02d" % (i * 5), 2, "KEEP", "KEEP", 1, None)
              for i in range(4)]          # 中間空掉約 40 秒
    _tl_rows(conn, specs)
    conn.close()

    import asyncio
    r = asyncio.new_event_loop().run_until_complete(
        ss.shadow_timeline(minutes=1440, since="2026-09-05T00:00:00",
                           until="2026-09-05T23:59:59", max_points=900,
                           _user=None))
    assert r["gaps"], "取樣斷點沒有被標出來"
    g = r["gaps"][0]
    assert g[1] - g[0] >= 20, f"斷點區間看起來不對: {g}"
    # t 必須是實際偏移,不是索引 × interval
    assert r["t"][-1] >= 75, "t 應該是實際秒數偏移"


def test_saturation_persists_across_restart(tmp_path, monkeypatch):
    """★ 2026-09-05 教訓:量到的飽和流只存記憶體,每次部署重啟就歸零,
    半夜重量又量不到 → 退回預設 1800,早尖峰整段跑假設值。
    量到必須落地、啟動必須載回。"""
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "_SAT_FILE", str(tmp_path / "sat.json"))
    S._measured_sat.update({"vph": {1: 496.0, 2: 506.0}, "ts": "2026-09-04T20:00:00",
                            "source": "measured", "window_hours": 6, "samples": 4247})
    S._save_saturation()
    # 模擬重啟:記憶體清空
    S._measured_sat.update({"vph": {}, "ts": None, "source": "default"})
    assert S._sat_for(1) == 1800.0          # 清空後真的退回預設
    assert S._load_saturation() is True
    assert S._sat_for(1) == 496.0 and S._sat_for(2) == 506.0
    assert S._measured_sat["source"] == "measured(restored)"


def test_saturation_load_rejects_out_of_range(tmp_path, monkeypatch):
    """檔案裡的值超出 [SAT_MIN, SAT_MAX] 不可載回 —— 壞檔不能把控制邏輯帶偏。"""
    import json
    from api.routes import signal_shadow as S
    f = tmp_path / "sat.json"
    f.write_text(json.dumps({"vph": {"1": 50.0, "2": 9999.0}}), encoding="utf-8")
    monkeypatch.setattr(S, "_SAT_FILE", str(f))
    S._measured_sat.update({"vph": {}, "ts": None, "source": "default"})
    assert S._load_saturation() is False
    assert S._sat_for(1) == 1800.0


def test_estimate_saturation_ignores_greens_without_queue():
    """半夜綠燈亮起時沒車在排,那些段不可以拿來量飽和流。"""
    from detection.signal_sim import estimate_saturation
    mk = lambda t, ph, q1, q2: (f"2026-09-05T03:{t//60:02d}:{t%60:02d}", ph, q1, q2)
    # 分相1 綠燈兩段:第一段起始排隊 0m(沒車)、第二段起始 21m(3 台)消退到 0
    rows = ([mk(t, 1, 0.0, 0.0) for t in range(0, 30, 5)]        # 無車綠燈 30s
            + [mk(t, 2, 0.0, 0.0) for t in range(30, 60, 5)]
            + [mk(t, 1, 21.0 - 0.7 * (t - 60), 0.0) for t in range(60, 90, 5)]  # 有車綠燈
            + [mk(t, 2, 0.0, 0.0) for t in range(90, 120, 5)])
    arr = {1: {"veh_per_sec": 0.0}, 2: {"veh_per_sec": 0.0}}
    # 🛑 2026-09-06 起有兩道門檻,要分開驗:
    #    min_start_queue_m —— 綠燈起始沒有隊伍就不採計
    #    min_saturated_sec —— 隊伍幾秒就放完(起動加速主導)也不採計
    #    無車那一段兩道都會擋,所以只關掉 min_saturated_sec 才看得出第一道的作用。
    loose = estimate_saturation(rows, arr, min_start_queue_m=0.0,
                                min_saturated_sec=0.0)[1]
    strict = estimate_saturation(rows, arr, min_start_queue_m=14.0,
                                 min_saturated_sec=0.0)[1]
    assert loose["green_sec"] > strict["green_sec"], "無門檻時無車綠燈會被算進分母"
    assert strict["veh_per_sec"] > loose["veh_per_sec"], "被稀釋的飽和流一定較低"
    # 兩道門檻都開時,無車那一段一定不在採計範圍內
    both = estimate_saturation(rows, arr, min_start_queue_m=14.0,
                               min_saturated_sec=8.0)[1]
    assert both["green_sec"] <= strict["green_sec"]


def test_max_green_fixed_100_forces_switch(monkeypatch):
    """使用者:最大綠固定 100 秒。時制表寫 210 也要在 100 秒強制切;設 0 才退回時制表。"""
    from api.routes import signal_shadow as S
    from detection.signal_decision_engine import ApproachState, decide
    monkeypatch.setattr(S, "MAX_GREEN_SEC", 100.0)
    assert S._max_green({"max_green": 210}) == 100.0
    monkeypatch.setattr(S, "MAX_GREEN_SEC", 0.0)
    assert S._max_green({"max_green": 210}) == 210.0 and S._max_green({}) == 210.0
    # 引擎:綠側還有隊伍、紅側沒車(平常會 KEEP),已亮 100s 仍強制 SWITCH
    g = ApproachState(1, queue_m=40.0, flow_vpm=10.0, storage_m=210, priority=False)
    r = ApproachState(2, queue_m=0.0, flow_vpm=0.0, storage_m=600, priority=True, waiting_sec=100.0)
    d = decide(green_phase=1, green_elapsed_sec=100.0, green_side=g, red_side=r, min_green_sec=10.0, max_green_sec=100.0,
               saturation_vph=500.0)
    assert d.action == "SWITCH" and "最大綠" in d.reason
    d2 = decide(green_phase=1, green_elapsed_sec=99.0, green_side=g, red_side=r, min_green_sec=10.0, max_green_sec=100.0,
                saturation_vph=500.0)
    assert d2.action == "KEEP"


def test_keep_weight_raises_switch_threshold():
    """綠側價值權重是現場校正:權重越大越不願換相,門檻要跟著抬高。"""
    from detection.signal_decision_engine import ApproachState, decide
    kw = dict(green_phase=1, green_elapsed_sec=40.0,
              green_side=ApproachState(1, queue_m=18.0, flow_vpm=12.0, storage_m=210),
              red_side=ApproachState(2, queue_m=30.0, storage_m=600, waiting_sec=40.0),
              min_green_sec=10.0, max_green_sec=100.0,
              saturation_vph=1181.0, meters_per_vehicle=6.0, lost_time_sec=5.0)
    a = decide(**kw, keep_weight=1.0)
    b = decide(**kw, keep_weight=3.0)
    assert b.detail["threshold"] > a.detail["threshold"]
    assert b.detail["keep_weight"] == 3.0
    assert b.detail["keep_gain_weighted"] == round(b.keep_gain * 3.0, 2)
    # 權重只調綠側那一項,紅側延滯與換相成本不動
    assert a.switch_gain == b.switch_gain and a.change_cost == b.change_cost
    # 門檻抬高之後不可能從續綠變成換相
    assert not (a.action == "KEEP" and b.action == "SWITCH")


def test_keep_weight_default_is_pure_model():
    """引擎預設值維持 1.0(純模型);3.0 是部署層的現場校正,不是模型結構。"""
    from detection.signal_decision_engine import DEFAULT_KEEP_WEIGHT, ApproachState, decide
    assert DEFAULT_KEEP_WEIGHT == 1.0
    d = decide(green_phase=1, green_elapsed_sec=40.0,
               green_side=ApproachState(1, queue_m=18.0, storage_m=210),
               red_side=ApproachState(2, queue_m=30.0, storage_m=600, waiting_sec=40.0),
               min_green_sec=10.0, max_green_sec=100.0)
    assert d.detail["keep_weight"] == 1.0


def test_shadow_route_deploys_keep_weight_3():
    """使用者 2026-09-06 決定上線的值,寫在程式裡才會進版控、不會某台機器忘了設。"""
    from api.routes import signal_shadow as S
    assert S.KEEP_WEIGHT == 3.0
    assert S.KEEP_WEIGHT_SINCE == "2026-09-06"


def _sat_rows(green_sec, start_q, clear_at, phase=1):
    """造一段綠燈:start_q 公尺的隊伍在 clear_at 秒清空,綠燈共 green_sec 秒。"""
    from datetime import datetime, timedelta
    base = datetime(2026, 9, 6, 10, 0, 0)
    rows = []
    other = 2 if phase == 1 else 1
    # 前置紅燈(讓上一段收尾)
    for k in range(4):
        rows.append(((base + timedelta(seconds=k * 5)).isoformat(), other, 0.0, 0.0))
    for k in range(0, green_sec, 5):
        q = max(0.0, start_q * (1 - k / clear_at)) if k < clear_at else 0.0
        rows.append(((base + timedelta(seconds=20 + k)).isoformat(), phase,
                     q if phase == 1 else 0.0, q if phase == 2 else 0.0))
    rows.append(((base + timedelta(seconds=20 + green_sec)).isoformat(), other, 0.0, 0.0))
    return rows


def test_saturation_skips_runs_that_clear_too_fast():
    """隊伍兩三秒就放完的綠燈,量到的是起動加速不是穩態放行,整段要丟掉。

    2026-09-06 的病徵:同一路口同一算法,週五晚尖峰量到 1407/1171 vph、
    週日早尖峰只有 758/523 —— 飽和流變成跟著當天車流跑,而它是物理容量。
    """
    from detection.signal_sim import estimate_arrivals, estimate_saturation
    arr = {1: {"veh_per_sec": 0.0}, 2: {"veh_per_sec": 0.0}}
    # 隊伍 20 m 在 5 秒內清空 —— 飽和段太短,應被丟掉
    fast = _sat_rows(green_sec=60, start_q=20.0, clear_at=5)
    s = estimate_saturation(fast, arr, mpv=6.0, min_start_queue_m=14.0,
                            min_saturated_sec=8.0)
    assert s[1]["veh_per_sec"] is None, "飽和段太短的綠燈不可採計"
    assert s[1]["runs_skipped_short"] >= 1 and s[1]["runs_used"] == 0
    # 同一段資料在舊門檻(不丟)下會算出值 —— 證明差別確實來自這個門檻
    s_old = estimate_saturation(fast, arr, mpv=6.0, min_start_queue_m=14.0,
                                min_saturated_sec=0.0)
    assert s_old[1]["veh_per_sec"] is not None


def test_saturation_keeps_runs_with_sustained_queue():
    """隊伍夠長、放行持續 30 秒的綠燈要採計,而且算得出合理的飽和流。"""
    from detection.signal_sim import estimate_saturation
    arr = {1: {"veh_per_sec": 0.0}, 2: {"veh_per_sec": 0.0}}
    # 90 m 隊伍在 30 秒清空 = 15 輛 / 30 秒 = 0.5 輛/秒 = 1800 vph
    slow = _sat_rows(green_sec=60, start_q=90.0, clear_at=30)
    s = estimate_saturation(slow, arr, mpv=6.0, min_start_queue_m=14.0,
                            min_saturated_sec=8.0)
    assert s[1]["runs_used"] >= 1 and s[1]["runs_skipped_short"] == 0
    assert 0.35 <= s[1]["veh_per_sec"] <= 0.65, s[1]["veh_per_sec"]


def test_saturation_window_is_seven_days():
    """飽和流是物理容量,不該每天重算成不同的值 —— 視窗拉長到 7 天。"""
    from api.routes import signal_shadow as S
    assert S.SAT_WINDOW_HOURS == 168.0
    assert S.SAT_MIN_SATURATED_SEC == 8.0


def test_auth_renew_only_when_dynamic_on(monkeypatch):
    """🛑 授權續約只在動態控制開著且 L0 時送。

    關掉總開關就不再續 —— 不送任何「歸還」命令,授權自己會在一分鐘內過期,
    控制器回到定時。這是最重要的 fail-safe,不可以被續約執行緒繞過。
    2026-09-07 實測:EffectTime=1 就是 1 分鐘(中央的覆蓋全被擋掉時看得乾淨,
    23:25:00 策略 10H → 23:26:00 自己變回 01H,期間沒有任何中央命令通過)。
    """
    from api.routes import signal_tc3 as T

    sent = []
    monkeypatch.setattr(T, "_controller_send", lambda b: sent.append(b) or True)
    monkeypatch.setattr(T, "_target_addr", lambda: 1)
    monkeypatch.setattr(T, "_enqueue_frame", lambda rec: None)
    monkeypatch.setitem(T._safety, "strategy", 0x01)

    monkeypatch.setitem(T._dyn, "enabled", False)
    T._do_reassert(kind="續約")
    assert sent == [], "總開關關閉時不可送出續約"

    monkeypatch.setitem(T._dyn, "enabled", True)
    monkeypatch.setitem(T._dyn, "level", "L2")
    T._do_reassert(kind="續約")
    assert sent == [], "降階時不可送出續約"

    monkeypatch.setitem(T._dyn, "level", "L0")
    T._do_reassert(kind="續約")
    assert len(sent) == 1
    # 內容:5F10 + 我方維持的策略 + EffectTime
    # 🛑 2026-09-08 策略值改成可設定 + 持久化(畫面上的控制策略卡就是設它),
    #    所以這裡要讀 reassert_strategy(),不可以再讀寫死常數 ——
    #    續約若沒跟著改,畫面上設完會在 20 秒內被蓋回去。
    assert bytes([0x5F, 0x10, T.reassert_strategy(), T.REASSERT_EFFECT]) in sent[0]


def test_auth_renew_period_shorter_than_effect_time():
    """🛑 續約週期必須短於授權有效期,否則每分鐘都會出現一段空窗。

    而且刻意**不**把 EffectTime 設很大:授權會過期正是 fail-safe ——
    服務掛掉/網路斷/使用者關開關,控制器最多一分鐘後自己回到定時。
    """
    from api.routes import signal_tc3 as T
    assert T.REASSERT_EFFECT == 1, "EffectTime 不應被放大,那會拆掉 fail-safe"
    assert T.AUTH_RENEW_SEC < 60, "續約週期要短於 EffectTime 的 1 分鐘"


def test_prepare_accepts_by_label(monkeypatch):
    """🛑 prepare 要能正常回 token —— 這條測試是被實際事故逼出來的。

    2026-09-08:我把稽核用的 `by` 標籤塞進 _finish_prepare,但 `by` 是
    control_prepare 的區域變數 → NameError → **每一次 prepare 都 500**。
    後果是演算法一則 5F1C 都送不出去,而我部署後只驗了「策略還是 10H」——
    那是續約(_do_reassert)在撐,走的是完全不同的路徑,看不出下發已經全掛。
    所以這裡直接驗兩條下發路徑的共用函式。
    """
    import inspect
    from api.routes import signal_tc3 as T

    sig = inspect.signature(T._finish_prepare)
    assert "by" in sig.parameters, "_finish_prepare 要收 by,否則呼叫端會 NameError"

    out = T._finish_prepare(b"\xaa\xbb\x01", "5F1C", 0x1C, 0x5F, 1, 1, "algorithm")
    assert out.get("token")
    assert T._pending[out["token"]]["by"] == "algorithm"

    # 沒帶 by 也要能用(人工下發不一定給標籤)
    out2 = T._finish_prepare(b"\xaa\xbb\x02", "5F10", 0x10, 0x5F, 1, 2)
    assert out2.get("token")
    assert T._pending[out2["token"]]["by"] == ""


def test_priority_keep_weight_only_when_red_is_priority():
    """🛑 主線保護相在等時才降低切換門檻,其他情況完全不變。

    現場 2026-09-08:「下匝道要放多點,很塞」。實測佐證下匝道排隊 >30m 的時段,
    上匝道拿到 2094 秒綠燈、下匝道只有 1354 秒。
    🛑 但不可以直接調低全域 keep_weight —— 3.0 是參數搜尋 + 五個未調過的驗證
       情境驗出來的(1.0→3.0 讓離最佳解從 +192.7% 收到 +51.8%),動它會讓整體
       延滯變差。所以這條測試守住「只有紅側是優先相時才用低值」。
    """
    from detection.signal_decision_engine import ApproachState, decide

    def run(red_priority: bool, pkw):
        return decide(
            green_phase=1, green_elapsed_sec=60.0,
            green_side=ApproachState(1, queue_m=10.0, flow_vpm=5.0,
                                     storage_m=210, priority=False),
            red_side=ApproachState(2, queue_m=40.0, flow_vpm=20.0,
                                   storage_m=600, priority=red_priority,
                                   waiting_sec=60.0),
            min_green_sec=10, max_green_sec=210,
            keep_weight=3.0, priority_keep_weight=pkw)

    # 紅側不是優先相 → 一定用 3.0,不受 priority_keep_weight 影響
    d = run(False, 2.0)
    assert d.detail["keep_weight"] == 3.0
    assert "priority_red" not in d.detail

    # 紅側是優先相 → 用較低的值,門檻跟著變低
    d2 = run(True, 2.0)
    assert d2.detail["keep_weight"] == 2.0
    assert d2.detail["priority_red"] is True
    assert d2.detail["threshold"] < run(True, None).detail["threshold"]

    # 沒給 priority_keep_weight → 行為與從前完全相同(預設不改變現場行為)
    assert run(True, None).detail["keep_weight"] == 3.0


def test_algorithm_never_switches_timing_plan():
    """🛑 演算法不得自行切換時制計畫(5F18)。

    規格 (C)(a) 要求動態號誌「無固定週期,**只做延長或結束綠燈**」。
    時制計畫是機關核定的號誌設計,演算法自行切換等於改變核定內容,超出授權。
    2026-09-08 我一度提議切到綠燈 40/60 的計畫 23 讓下匝道多放,使用者否決,
    而且他是對的。5F18 的介面保留給**人工**操作(特勤等情境),演算法不得呼叫。

    這條測試守住那條界線:shadow 模組送出的命令碼只能是 5F1C。
    """
    import ast
    src = (ROOT / "api" / "routes" / "signal_shadow.py").read_text(encoding="utf-8")
    fn = [n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "_actuate"][0]
    codes = {n.value for n in ast.walk(fn)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and len(n.value) == 4 and n.value.upper().startswith("5F")}
    assert codes == {"5F1C"}, "演算法只能送 5F1C,實際出現:%s" % codes

    # 🛑 只檢查**送出路徑**。5F18 在本模組其他地方出現是合法的:
    #    統計「調整次數」時要把人工切換時制也算進去,那是 SQL 讀取不是送出。
    #    把「檔案裡不准出現 5F18」當測試會擋掉正當的讀取用途(第一版就踩到)。
    fn_src = ast.get_source_segment(src, fn) or ""
    assert "5F18" not in fn_src, "演算法不得切換時制計畫(5F18)"


def test_no_send_on_first_green_step(monkeypatch):
    """🛑 第一個綠階不送 —— 那會把綠燈推進延長段,與意圖相反。

    2026-09-08 實測:步階結構是 綠1 → 綠2(感應延長) → 黃4 → 全紅5。
    「跳下一步階」從步階1 只跳到步階2,而步階2 沒車時控制器本來會跳過。
    實證:分相1 走 1→2 的比例 有介入 37% / 沒介入 16%;綠燈長度
    有介入 47.1s vs 沒介入 46.2s —— **我方反而讓綠燈變長 0.9 秒**。
    """
    import api.routes.signal_shadow as m
    calls = []
    _fake_daemon(monkeypatch, m, calls)
    monkeypatch.setitem(m._act, "enabled", True)
    monkeypatch.setitem(m._act, "last_ts", 0.0)

    # 第一個綠階 → 不送
    m._actuate(_D(), 1, dict(LIVE_OK, step_id=m.FIRST_GREEN_STEP))
    assert calls == []
    assert "延長段" in m._act["blocked"]

    # 後續綠階 → 才送(跳下一步階才會真的進清道)
    m._actuate(_D(), 1, dict(LIVE_OK, step_id=m.FIRST_GREEN_STEP + 1))
    assert len(calls) == 2      # prepare + send


def test_adjust_log_不得再宣稱_seq_無法配對():
    """🛑 2026-09-08 現場指正:那句說明是錯的。

    舊註解寫「0F80 的 seq 是控制器自己的計數,無法配對」,但當天四則不同命令
    實測 seq 完全對得上(0F12/0F10/0F47/5F10)。既然對得上就該精確配對,
    不可以繼續用「時間鄰近」的推定當唯一手段,更不可以在畫面上這樣寫。
    """
    import inspect
    import api.routes.signal_shadow as m
    src = inspect.getsource(m.adjust_log)
    # 🛑 docstring 可以**引用**那句錯話,但必須同時標明它是錯的 ——
    #    留著更正紀錄有價值,留著沒被推翻的錯誤說明才是問題。
    if "無法配對" in src:
        assert "那是錯的" in src, "引用了那句錯話卻沒有標明它是錯的"
    assert '"seq"' in src, "沒有做 seq 精確配對"
    assert "ack_match" in src, "沒有標明每一筆是精確還是推定"

    import pathlib
    web = (pathlib.Path(__file__).resolve().parents[1] / "web" / "index.html"
           ).read_text(encoding="utf-8")
    assert "seq 是控制器自己的計數" not in web, "畫面上還留著那句錯誤說明"


def test_adjust_log_查詢類不算時制調整():
    """🛑 查詢類(5F40/0F42/0F46…)只讀資料,不改變運轉 ——
    排除的理由跟續約 5F10 一模一樣。當天 0F42/5F40 探測佔了表格一半,
    把真正的調整淹沒。
    """
    import inspect
    import api.routes.signal_shadow as m
    src = inspect.getsource(m.adjust_log)
    assert "include_query" in src, "沒有把查詢類分開"
    assert "_is_query" in src, "沒有判斷查詢類"
    assert "query_excluded" in src, "沒有回報排除了幾筆,使用者會以為資料不見了"


def test_adjust_log_查詢用自己的回報碼配對():
    """🛑 查詢類的回應是它自己的回報碼(0F42 → 0FC2),不是 0F80。
    舊版一律找 0F80,所以每一則查詢都被標成「無回應」—— 那是錯的。
    """
    import inspect
    import api.routes.signal_shadow as m
    src = inspect.getsource(m.adjust_log)
    assert "_query_reply_code" in src, "沒有把查詢碼對到它的回報碼"
    assert "0x40 <= cmd < 0x80" in src, "沒有用指令碼區間判斷查詢類"


def test_adjust_log_依據只掛在換相命令():
    """🛑 把最近的決策理由套到 0F42 對時查詢上會顯示「未滿最小綠 20s」,
    與那則命令毫無關係 —— 那是誤導,不是資訊。
    """
    import inspect
    import api.routes.signal_shadow as m
    src = inspect.getsource(m.adjust_log)
    assert 'if code == "5F1C":' in src, "依據沒有限定只掛在換相命令上"


def test_degrade_span_kind_follows_latest_fault(tmp_path, monkeypatch):
    """同一段降階裡先後發生兩種故障時,類別不可停在第一種。

    🛑 2026-09-08 現場實際看到的矛盾列:
       類別「指令傳輸錯誤」/ 原因「偵測器故障:分相 1、2 的排隊與流量都取不到」。
       成因是 spans 的續接分支更新了 level 與 reason 卻沒更新 kind。
    """
    import asyncio
    from api.routes import signal_shadow as S

    db = tmp_path / "shadow.db"
    monkeypatch.setattr(S, "_DB_PATH", str(db))
    monkeypatch.setattr(S, "_db_ready", False)

    S._degrade_persist("L2", "指令傳輸錯誤:連續 3 次未被接受", "transmit")
    S._degrade_persist("L3", "偵測器故障:分相 1、2 的排隊與流量都取不到", "detector")

    out = asyncio.new_event_loop().run_until_complete(
        S.degrade_log(hours=24, _user=None))
    sp = out["spans"][0]
    assert sp["duration_sec"] is None          # 仍在降階中 → 不是 0 秒
    assert sp["level"] == "L3"
    assert sp["kind"] == "detector"            # 跟著最新的原因走
    assert set(sp["kinds"]) == {"transmit", "detector"}


def test_degrade_bootstrap_closes_open_span_on_restart(tmp_path, monkeypatch):
    """重啟時要把上一段沒關閉的降階補一筆復歸,並標明是重啟關閉的。

    🛑 2026-09-08 現場:統計報表顯示「降階 1 段 · 累計 0 秒 · 進行中」,
       但同一頁的運作狀態是 L0 —— 降階狀態在記憶體,重啟就回 L0,
       DB 那一段卻永遠開著。
    """
    import asyncio
    from api.routes import signal_shadow as S

    db = tmp_path / "shadow.db"
    monkeypatch.setattr(S, "_DB_PATH", str(db))
    monkeypatch.setattr(S, "_db_ready", False)

    S._degrade_persist("L2", "偵測器故障:兩相都取不到", "detector")
    S._degrade_bootstrap()

    out = asyncio.new_event_loop().run_until_complete(
        S.degrade_log(hours=24, _user=None))
    assert out["count"] == 1
    sp = out["spans"][0]
    assert sp["duration_sec"] is not None       # 已關閉,不再是「進行中」
    assert sp["closed_by_restart"] is True      # 但要標明不是量到的恢復時刻
    assert out["ongoing"] is False
    # 再跑一次不可以重複補(最後一筆已經是 L0)
    S._degrade_bootstrap()
    again = asyncio.new_event_loop().run_until_complete(
        S.degrade_log(hours=24, _user=None))
    assert again["count"] == 1


def test_degrade_bootstrap_noop_when_nothing_open(tmp_path, monkeypatch):
    from api.routes import signal_shadow as S
    db = tmp_path / "shadow2.db"
    monkeypatch.setattr(S, "_DB_PATH", str(db))
    monkeypatch.setattr(S, "_db_ready", False)
    S._degrade_bootstrap()                       # 空表
    conn = S._db()
    n = conn.execute("SELECT COUNT(*) FROM signal_degrade_log").fetchone()[0]
    conn.close()
    assert n == 0
