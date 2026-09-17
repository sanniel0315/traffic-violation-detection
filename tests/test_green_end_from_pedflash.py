"""綠燈結束改成「黃燈框優先、收不到就用行人綠閃推算」(2026-09-17)。

斷電後控制器的短步階回報大量遺失:黃燈(3 秒)只有 16% 的週期收得到、全紅 7%,
但行人綠閃(5 秒)還有 64%,兩者合起來 81%。行人綠閃的回報帶剩餘秒數,
「這一框時刻 + 剩餘」就是綠燈結束。
"""
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


def _frame(seq, phase, step, remain):
    """組一個 5F03 框(六個方向燈色用固定值,解碼只看分相/步階/剩餘)。"""
    from api.routes.signal_tc3 import build_frame
    info = bytes((0x5F, 0x03, 0x00, 0x5F, 0x06, phase, step,
                  (remain >> 8) & 0xFF, remain & 0xFF,
                  0x44, 0x44, 0x81, 0x81, 0x44, 0x81))
    return build_frame(0xFFFF, seq, info).hex(" ").upper()


@pytest.fixture
def runs(monkeypatch, tmp_path):
    """把一段假的 5F03 寫進暫時的 DB,讓 _actual_runs_from_frames 讀。"""
    import sqlite3
    from api.routes import signal_shadow as S
    from api.routes import signal_tc3 as T
    db = tmp_path / "q.db"
    c = sqlite3.connect(str(db))
    c.execute("CREATE TABLE signal_frames (ts REAL, src TEXT, code TEXT, seq INTEGER,"
              " addr INTEGER, len INTEGER, cks_ok INTEGER, raw TEXT, user TEXT, sent_hw INTEGER)")
    monkeypatch.setattr(T, "_QDB_PATH", str(db))

    def build(rows):
        c.execute("DELETE FROM signal_frames")
        for ts, seq, ph, st, rem in rows:
            c.execute("INSERT INTO signal_frames(ts,src,code,seq,cks_ok,raw) VALUES(?,?,?,?,1,?)",
                      (ts, "controller", "5F03", seq, _frame(seq, ph, st, rem)))
        c.commit()
        a = datetime.fromtimestamp(rows[0][0] - 5).isoformat()
        b = datetime.fromtimestamp(rows[-1][0] + 5).isoformat()
        return S._actual_runs_from_frames(a, b)
    return build


def _cycles(t0, n, with_yellow=True, with_ped=True):
    """n 個週期:步階1 → (行人綠閃) → (黃燈) → 全紅 → 換相。"""
    rows = []
    t, seq, ph = t0, 1, 1
    for _ in range(n):
        rows.append((t, seq, ph, 1, 30)); seq += 1; t += 30
        if with_ped:
            rows.append((t, seq, ph, 2, 5)); seq += 1
        t += 5
        if with_yellow:
            rows.append((t, seq, ph, 4, 3)); seq += 1
        t += 5
        ph = 2 if ph == 1 else 1
    return rows


def test_yellow_frame_is_preferred_when_present(runs):
    out = runs(_cycles(1_700_000_000, 12))
    assert out and len(out) >= 10
    assert all(s.get("green_end_source") == "yellow_frame" for s in out)
    assert 34 <= out[0]["green_sec"] <= 36          # 30 主綠 + 5 行閃


def test_falls_back_to_ped_flash_when_yellow_is_missing(runs):
    """黃燈框全部遺失時,仍要算得出綠燈長度(而不是整段作廢或低估)。"""
    out = runs(_cycles(1_700_000_000, 20, with_yellow=False))
    assert out and len(out) >= 15
    assert all(s.get("green_end_source") == "step2_est" for s in out)
    assert 34 <= out[0]["green_sec"] <= 36


def test_without_yellow_and_ped_it_degrades_but_still_reports(runs):
    out = runs(_cycles(1_700_000_000, 40, with_yellow=False, with_ped=False))
    assert out and all(s.get("green_end_source") == "green_last" for s in out)
    assert out[0]["green_sec"] >= 0                 # 低估但不會是負的


def test_mixed_cycles_pick_the_best_source_each(runs):
    rows = _cycles(1_700_000_000, 6) + _cycles(1_700_000_600, 6, with_yellow=False)
    out = runs(rows)
    srcs = {s.get("green_end_source") for s in out}
    assert "yellow_frame" in srcs and "step2_est" in srcs
    assert all(30 <= s["green_sec"] <= 40 for s in out)
