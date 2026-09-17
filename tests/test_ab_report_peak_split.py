"""A/B 成效比較要分尖峰／離峰(使用者 2026-09-17:「要分尖峰 離峰」)。

🛑 60 分鐘的分段很可能跨過 09:00 或 16:30 的尖峰邊界。整段歸一邊會把尖峰的
   塞算進離峰(或反過來),那比不分還糟 —— 看起來有分時段,其實混著。
"""
import os
from datetime import datetime

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _ts(h, m=0):
    d = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    return d.timestamp()


def test_split_inside_one_tier_stays_one_piece():
    from api.routes.signal_shadow import _split_by_peak
    out = _split_by_peak(_ts(13, 0), _ts(14, 0))       # 離峰內
    assert len(out) == 1 and out[0][2] == "離峰"


def test_split_across_peak_start():
    """08:30→09:30 要切成 離峰 30 分 + 尖峰 30 分。"""
    from api.routes.signal_shadow import _split_by_peak
    out = _split_by_peak(_ts(8, 30), _ts(9, 30))
    assert [x[2] for x in out] == ["離峰", "尖峰"]
    assert round((out[0][1] - out[0][0]) / 60) == 30
    assert round((out[1][1] - out[1][0]) / 60) == 30


def test_split_across_peak_end():
    """11:45→12:15 要切成 尖峰 15 分 + 離峰 15 分。"""
    from api.routes.signal_shadow import _split_by_peak
    out = _split_by_peak(_ts(11, 45), _ts(12, 15))
    assert [x[2] for x in out] == ["尖峰", "離峰"]


def test_split_covers_the_whole_window_without_gaps():
    """切出來的片段要首尾相接、總長不變 —— 漏掉一段就是漏掉樣本。"""
    from api.routes.signal_shadow import _split_by_peak
    a, b = _ts(15, 50), _ts(20, 10)      # 跨 16:30 進尖峰、20:00 出尖峰
    out = _split_by_peak(a, b)
    assert out[0][0] == a and out[-1][1] == b
    for i in range(1, len(out)):
        assert out[i][0] == out[i - 1][1]
    assert abs(sum(x[1] - x[0] for x in out) - (b - a)) < 1e-6
    assert [x[2] for x in out] == ["離峰", "尖峰", "離峰"]


def test_report_exposes_both_tiers_and_judges_them_separately():
    import inspect
    from api.routes import signal_shadow as S

    src = inspect.getsource(S.ab_report)
    assert '"by_tier": by_tier' in src
    # 每個時段別各自判樣本夠不夠,不可以共用總表的結論
    assert 'for tier in ("尖峰", "離峰")' in src
    assert '"conclusive": ok' in src


def test_report_labels_metrics_by_ramp_not_number():
    """指標標籤用匝道名(主鍵是匝道),不是「分相2」。"""
    import inspect
    from api.routes import signal_shadow as S

    src = inspect.getsource(S.ab_report)
    assert "下匝道最大排隊" in src and "主線回堵次數(下匝道)" in src
    assert '_pof("off_ramp")' in src, "分相編號要由 role 查出來,不可寫死"
