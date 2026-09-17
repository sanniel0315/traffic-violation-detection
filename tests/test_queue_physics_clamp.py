"""排隊量測的物理一致性夾制:演算法自己吸收計數誤差(2026-09-17)。

上下游兩台相機看同一批車,逐時計數差 0~30%。決策方向不受影響(逐筆翻轉 0.5~1%),
但單筆跳動會讓續綠/換相在同一秒來回 —— 進決策前先用物理限制夾住。
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture
def S(monkeypatch):
    from api.routes import signal_shadow as S
    monkeypatch.setattr(S, "_q_hist", {})
    monkeypatch.setattr(S, "_q_clamp_stat", {1: 0, 2: 0})
    monkeypatch.setattr(S, "QUEUE_CLAMP_ENABLED", True)
    monkeypatch.setattr(S, "QUEUE_CLAMP_SLACK_M", 12.0)
    monkeypatch.setattr(S, "_mpv", lambda: 6.0)
    monkeypatch.setattr(S, "_sat_for", lambda p: 900.0)      # 900 vph = 1.5 m/s 的消散
    return S


def test_first_sample_is_never_clamped(S):
    assert S._queue_physics_clamp(1, 80.0, 6.0) == (80.0, False)


def test_impossible_jump_up_is_clamped(S):
    """到達 6 輛/分 = 0.6 m/s;2 秒最多長 1.2 m,加餘裕 12 m。
    量測從 20 跳到 90 是不可能的,夾到 33.2。"""
    S._queue_physics_clamp(1, 20.0, 6.0)
    time.sleep(2)
    out, clamped = S._queue_physics_clamp(1, 90.0, 6.0)
    assert clamped and 30 < out < 36
    assert S._q_clamp_stat[1] == 1


def test_impossible_drop_is_clamped(S):
    """飽和流 900 vph = 1.5 m/s;2 秒最多消 3 m,加餘裕 12 m。"""
    S._queue_physics_clamp(2, 80.0, 0.0)
    time.sleep(2)
    out, clamped = S._queue_physics_clamp(2, 5.0)
    assert clamped and 60 < out < 70


def test_plausible_change_passes_through(S):
    S._queue_physics_clamp(1, 40.0, 6.0)
    time.sleep(1)
    out, clamped = S._queue_physics_clamp(1, 44.0, 6.0)
    assert out == 44.0 and clamped is False


def test_long_gap_restarts_instead_of_dragging(S):
    """中斷太久(>60 秒)就重新起算,不要拿很舊的值去夾現在的量測。"""
    S._q_hist[1] = (time.time() - 300, 10.0)
    out, clamped = S._queue_physics_clamp(1, 200.0, 6.0)
    assert out == 200.0 and clamped is False


def test_disabled_by_env_switch(S, monkeypatch):
    monkeypatch.setattr(S, "QUEUE_CLAMP_ENABLED", False)
    S._q_hist[1] = (time.time() - 1, 10.0)
    assert S._queue_physics_clamp(1, 900.0, 0.0) == (900.0, False)


def test_none_measurement_passes(S):
    assert S._queue_physics_clamp(1, None, 5.0) == (None, False)
