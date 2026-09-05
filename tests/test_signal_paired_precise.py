"""精確配對:實際綠燈秒數來自控制器 5F03 框,我方判斷來自影子 log;續綠段也有比對數據。"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

T0 = datetime(2026, 9, 4, 9, 0, 0)


def _seg(phase, start_s, green_s, clear_s=5):
    a = (T0 + timedelta(seconds=start_s)).timestamp()
    return {"phase": phase, "start": a, "green_end": a + green_s, "end": a + green_s + clear_s,
            "green_sec": float(green_s)}


def _rows(spec):
    """spec: (phase, start_s, n_samples, switch_at_elapsed|None, red_q, margin_terms(sg,kg,cc))
    以 5 秒取樣產生影子 log 列。"""
    out = []
    for ph, start_s, n, sw, red_q, (sg, kg, cc) in spec:
        for i in range(n):
            el = i * 5.0
            ts = (T0 + timedelta(seconds=start_s + el)).isoformat(timespec="seconds")
            ours = "SWITCH" if (sw is not None and el >= sw) else "KEEP"
            q1, q2 = (0.0, red_q) if ph == 1 else (red_q, 0.0)
            out.append((ts, ph, el, ours, "KEEP", 0, 0, q1, q2, sg, kg, cc))
    return out


def test_precise_uses_controller_green_seconds_not_sampling():
    from api.routes.signal_shadow import _paired_precise
    # 控制器說綠燈 33 秒;影子取樣只看到 30(區間量測)。我方 15 秒判切,紅側有車。
    actual = [_seg(1, 0, 33), _seg(2, 40, 28)]
    rows = _rows([(1, 0, 7, 15, 14.0, (5.0, 20.0, 3.5)),          # 0..30
                  (2, 40, 6, None, 0.0, (0.0, 80.0, 3.5))])       # 續綠
    r = _paired_precise(rows, actual, interval=5.0)
    assert r["source"] == "controller_5F03" and r["runs_usable"] == 2
    first = r["runs"][0]
    assert first["actual_sec"] == 33.0                    # 取控制器秒數,不是 30
    assert first["delta_sec"] == -18.0 and first["waste_sec"] == 18.0
    assert r["earlier_meaningful"] == 1 and r["hold"] == 1 and r["later"] == 0
    # 續綠段的比對數據:OPAC 切相那一刻我方裕度 = 80 + 3.5 − 0 = 83.5,紅側無車
    hc = r["hold_compare"]
    assert hc["runs"] == 1 and hc["margin_at_switch"]["avg"] == 83.5
    assert hc["red_waiting_at_switch"] == 0 and hc["red_waiting_ratio"] == 0.0


def test_precise_ignores_switch_during_clearance_and_pairs_by_time_window():
    from api.routes.signal_shadow import _paired_precise
    actual = [_seg(2, 0, 20)]
    rows = _rows([(2, 0, 5, 20, 7.0, (30.0, 10.0, 3.5))])
    # 第 20 秒那筆落在綠燈結束點,標成清道 → 不算第一次判切 → 續綠
    ts, ph, el, ours, act, f, c, q1, q2, sg, kg, cc = rows[-1]
    rows[-1] = (ts, ph, el, "SWITCH", act, f, 1, q1, q2, sg, kg, cc)
    r = _paired_precise(rows, actual, interval=5.0)
    assert r["hold"] == 1 and r["earlier"] == 0
    # 續綠但紅側有車(7 m)→ 記為「我方比 OPAC 保守」
    assert r["hold_compare"]["red_waiting_at_switch"] == 1
