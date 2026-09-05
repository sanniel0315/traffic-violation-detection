"""影子精確比對:逐次綠燈配對。

取樣層級的一致率答不了「我方會早幾秒切」。這裡以一次綠燈為單位:
每一段實際綠燈,我方第一次判 SWITCH 落在第幾秒,跟實際換相差多少。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rows(spec):
    """spec: list of (phase, elapsed_list, ours_switch_at_or_None, red_queue)。
    每段以 5 秒取樣產生 (ts, phase, elapsed, ours, actual, forced, clearance, q1, q2)。"""
    from datetime import datetime, timedelta
    t = datetime(2026, 9, 5, 8, 0, 0)
    out = []
    for ph, elapsed_list, sw_at, red_q in spec:
        for i, el in enumerate(elapsed_list):
            ours = "SWITCH" if (sw_at is not None and el >= sw_at) else "KEEP"
            actual = "SWITCH" if i == 0 else "KEEP"
            q1, q2 = (0.0, red_q) if ph == 1 else (red_q, 0.0)
            out.append((t.isoformat(timespec="seconds"), ph, float(el), ours, actual, 0, 0, q1, q2))
            t += timedelta(seconds=5)
    return out


def test_paired_runs_measures_how_much_earlier_we_would_switch():
    from api.routes.signal_shadow import _paired_runs
    rows = _rows([
        (1, [0, 5, 10, 15, 20, 25, 30, 35], 20, 14.0),   # 實際 35s,我方 20s 說切,紅側有 2 台 → 早 15s、有代價
        (2, [0, 5, 10, 15, 20, 25], None, 0.0),           # 整段同意續綠
        (1, [0, 5, 10, 15, 20, 25, 30], 15, 0.0),         # 早 15s 但紅側沒車 → idle(早 5s 會落在容忍內算 same)
        (2, [0, 5, 10, 15, 20, 25, 30, 35, 40], 40, 7.0), # 同一時刻(容忍一個週期)→ same
    ])
    r = _paired_runs(rows, interval=5.0)
    assert r["runs_usable"] == 4 and r["runs_truncated"] == 0
    assert r["earlier"] == 2 and r["earlier_meaningful"] == 1 and r["earlier_idle"] == 1
    assert r["hold"] == 1 and r["same"] == 1 and r["later"] == 0
    first = r["runs"][0]
    assert first["delta_sec"] == -15.0 and first["red_waiting"] is True and first["waste_sec"] == 15.0
    assert r["waste_sec_total"] == 15.0          # 沒車的早切不算代價
    assert r["by_phase"]["1"]["earlier"] == 2 and r["by_phase"]["2"]["hold"] == 1


def test_paired_runs_marks_truncated_and_ignores_clearance():
    from api.routes.signal_shadow import _paired_runs
    rows = _rows([(1, [0, 5, 10, 15, 20], 10, 14.0)])
    # 段內塞一個 40 秒的洞 → truncated,不計入統計
    ts, ph, el, ours, actual, f, c, q1, q2 = rows[-1]
    from datetime import datetime, timedelta
    late = (datetime.fromisoformat(ts) + timedelta(seconds=40)).isoformat(timespec="seconds")
    rows.append((late, ph, 60.0, "SWITCH", "KEEP", 0, 0, q1, q2))
    r = _paired_runs(rows, interval=5.0)
    assert r["runs_truncated"] == 1 and r["runs_usable"] == 0
    # 清道期間的 SWITCH 不算「我方第一次判切」
    rows2 = _rows([(2, [0, 5, 10, 15], None, 14.0)])
    rows2[1] = rows2[1][:3] + ("SWITCH", "KEEP", 0, 1) + rows2[1][7:]   # clearance=1
    r2 = _paired_runs(rows2, interval=5.0)
    assert r2["hold"] == 1 and r2["earlier"] == 0
