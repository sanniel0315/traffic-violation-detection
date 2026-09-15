"""決策頻率(規範 K(C)c(a) 每 1~2 秒)與紀錄頻率分開。"""
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_decision_interval_defaults_to_log_interval(monkeypatch):
    """沒設就跟紀錄頻率一樣 —— 行為不變。"""
    monkeypatch.delenv("SIGNAL_DECISION_INTERVAL_SEC", raising=False)
    monkeypatch.delenv("SIGNAL_SHADOW_INTERVAL_SEC", raising=False)
    import api.routes.signal_shadow as m
    m = importlib.reload(m)
    assert m.DECISION_INTERVAL_SEC == m.SHADOW_INTERVAL_SEC == 5


def test_decision_interval_override_keeps_log_interval(monkeypatch):
    """設 2 秒決策時,紀錄仍是 5 秒一筆(分析與歷史資料以一筆 = 5 秒為前提)。"""
    monkeypatch.setenv("SIGNAL_DECISION_INTERVAL_SEC", "2")
    monkeypatch.delenv("SIGNAL_SHADOW_INTERVAL_SEC", raising=False)
    import api.routes.signal_shadow as m
    m = importlib.reload(m)
    assert m.DECISION_INTERVAL_SEC == 2 and m.SHADOW_INTERVAL_SEC == 5
    monkeypatch.delenv("SIGNAL_DECISION_INTERVAL_SEC")
    importlib.reload(m)
