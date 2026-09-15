"""候選規則「下匝道沒人用就切」—— 只做影子評估,不可下發。"""
import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.signal_decision_engine import offramp_idle_cut  # noqa: E402


def _cut(**kw):
    base = dict(green_is_priority=True, green_elapsed_sec=25, min_green_sec=20,
                green_queue_m=0, prev_green_queue_m=0, red_queue_m=12, meters_per_vehicle=6)
    base.update(kw)
    return offramp_idle_cut(**base)


def test_cuts_when_offramp_idle_and_onramp_waiting():
    would, why = _cut()
    assert would and "2.0 台" in why


def test_never_before_min_green():
    assert _cut(green_elapsed_sec=19)[0] is False


def test_needs_two_consecutive_empty_samples():
    """單次 0 可能只是漏框:上一筆有排隊或量不到都不切。"""
    assert _cut(prev_green_queue_m=6)[0] is False
    assert _cut(prev_green_queue_m=None)[0] is False
    assert _cut(green_queue_m=None)[0] is False


def test_needs_someone_waiting_on_onramp():
    assert _cut(red_queue_m=3)[0] is False
    assert _cut(red_queue_m=None)[0] is False


def test_only_for_offramp_green():
    assert _cut(green_is_priority=False)[0] is False


def test_rule_shadow_never_sends():
    """影子紀錄不可以碰任何下發函式。"""
    src = (ROOT / "api" / "routes" / "signal_shadow.py").read_text(encoding="utf-8")
    fn = [n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "_rule_shadow_record"][0]
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"_actuate", "_daemon_post"}, called


def test_input_shadow_never_sends():
    """到達率影子紀錄不可以碰任何下發函式。"""
    src = (ROOT / "api" / "routes" / "signal_shadow.py").read_text(encoding="utf-8")
    fn = [n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "_input_shadow_record"][0]
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"_actuate", "_daemon_post"}, called


def test_input_shadow_arrival_counts_flips(tmp_path, monkeypatch):
    """上匝道綠燈、綠側沒排隊:NE-2 量到 0、NE-1 量到很多 → 綠側價值變高,切→續綠翻轉。"""
    import sqlite3
    from api.routes import signal_shadow as S
    db = tmp_path / "s.db"
    c = sqlite3.connect(str(db))
    c.execute("""CREATE TABLE signal_input_shadow (id INTEGER PRIMARY KEY, ts TEXT, epoch REAL,
                 green_phase INTEGER, green_elapsed REAL, min_green REAL, max_green REAL,
                 queue_m_1 REAL, queue_m_2 REAL, arr1_stop REAL, arr1_up REAL, arr2 REAL)""")
    from datetime import datetime
    ep = datetime(2026, 9, 15, 10, 0).timestamp()
    c.execute("INSERT INTO signal_input_shadow(ts,epoch,green_phase,green_elapsed,min_green,max_green,"
              "queue_m_1,queue_m_2,arr1_stop,arr1_up,arr2) VALUES('x',?,1,25,10,100,0,30,0,20,3)", (ep,))
    c.commit(); c.close()
    monkeypatch.setattr(S, "_db", lambda: sqlite3.connect(str(db)))
    r = S.input_shadow_arrival(since="2026-09-15T09:00:00", until="2026-09-15T11:00:00", _user=None)
    f = r["decision_flips"]["分相1(上匝道綠燈)"]
    assert f["n"] == 1 and f["SWITCH→KEEP"] == 1
