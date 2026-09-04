"""影子模式:只記錄不下發 + 切換偵測正確性。

影子模式是 bypass OPAC 的前置驗證 —— 我方決策全速運轉但不碰控制器,
趁 OPAC 還在跑時累積對照資料。最重要的保證是「絕對不下發」。
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


def test_shadow_never_sends_anything():
    """★最重要:影子模組不可以有任何下發行為。

    用 AST 檢查「實際被呼叫的函式名」,不掃註解與 docstring
    (docstring 裡會提到 control/send,那是在說明「不走那條路」)。
    """
    import ast
    src = (ROOT / "api" / "routes" / "signal_shadow.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    forbidden = {"control_send", "send", "sendall", "_send_frame",
                 "send_frame", "write_frame"}
    hit = called & forbidden
    assert not hit, "影子模組不可呼叫下發相關函式:%s" % hit

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.name)
    assert "control_send" not in imported
    assert "control_prepare" not in imported


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
