"""疊加框外推:三道閘門與位移上限。

2026-08-31 87 現場:gap 中位 2.0 秒、偵測間隔 1.8 秒,而 INTERP_MAX 是 0.5、
span 閘門寫死 1.5 —— 三道閘門同時擋住,外推整個失效,框凍在偵測當下的位置,
使用者看到「框標不準」。這裡把可調性與位移上限釘住。
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(**env):
    """用指定 env 重新載入模組(這些門檻是 import 時讀進來的)。"""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        import detection.annotated_streamer as m
        return importlib.reload(m)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _det(cx, cy, w=100, h=80, cls="car"):
    return {"class_name": cls,
            "bbox": {"x1": cx - w // 2, "y1": cy - h // 2,
                     "x2": cx + w // 2, "y2": cy + h // 2}}


def _streamer(m):
    obj = m.AnnotatedStreamer.__new__(m.AnnotatedStreamer)
    obj.width = 1920
    obj._interp_hit = 0
    obj._interp_miss = 0
    obj._interp_hit_narrow = 0
    return obj


def _cx(det):
    b = det["bbox"]
    return (b["x1"] + b["x2"]) / 2


def test_delay_max_env_configurable():
    """上限寫死 1.20 等於把『壓畫面等偵測』的機制關掉。"""
    m = _load(ANNOTATED_STREAM_DELAY_MAX="3.0")
    assert m.DELAY_MAX == 3.0
    m = _load(ANNOTATED_STREAM_DELAY_MAX="")
    assert m.DELAY_MAX == 1.20          # 預設不變


def test_span_gate_env_configurable():
    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="3.0")
    assert m.INTERP_SPAN_MAX_SEC == 3.0
    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="")
    assert m.INTERP_SPAN_MAX_SEC == 1.5


def test_span_over_gate_no_extrapolation():
    """偵測間隔 1.8 秒 > 閘門 1.5 → 速度不可信,維持原位。"""
    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="1.5", ANNOTATED_STREAM_INTERP_MAX="2.0")
    st = _streamer(m)
    prev, cur = [_det(100, 500)], [_det(300, 500)]
    items = [(0.0, prev), (1.8, cur)]
    out = st._extrapolate(cur, 1.8, 2.3, items)
    assert _cx(out[0]) == 300           # 沒有被推


def test_widened_span_gate_extrapolates():
    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="3.0", ANNOTATED_STREAM_INTERP_MAX="2.0",
              ANNOTATED_STREAM_INTERP_MATCH="0.20", ANNOTATED_STREAM_INTERP_MAX_SHIFT="0")
    st = _streamer(m)
    prev, cur = [_det(100, 500)], [_det(300, 500)]
    items = [(0.0, prev), (1.8, cur)]
    # 速度 (300-100)/1.8 = 111.1 px/s，往前推 0.5s → +55.6
    out = st._extrapolate(cur, 1.8, 2.3, items)
    assert _cx(out[0]) == pytest.approx(355, abs=2)


def test_interp_max_is_independent_gate():
    """INTERP_MAX 是另一道獨立閘門,放寬 span 不會繞過它。"""
    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="3.0", ANNOTATED_STREAM_INTERP_MAX="0.5")
    st = _streamer(m)
    prev, cur = [_det(100, 500)], [_det(300, 500)]
    items = [(0.0, prev), (1.8, cur)]
    out = st._extrapolate(cur, 1.8, 4.0, items)   # dt=2.2 > 0.5
    assert _cx(out[0]) == 300


def test_shift_cap_prevents_fling():
    """速度估錯時不讓框飛出去 —— 上限是框自身尺寸的倍數。"""
    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="3.0", ANNOTATED_STREAM_INTERP_MAX="3.0",
              ANNOTATED_STREAM_INTERP_MATCH="0.50", ANNOTATED_STREAM_INTERP_MAX_SHIFT="1.5")
    st = _streamer(m)
    # 兩組之間位移 800px/0.5s = 1600 px/s，往前推 2s 會是 3200px（整個飛出畫面）
    prev, cur = [_det(100, 500)], [_det(900, 500)]
    items = [(0.0, prev), (0.5, cur)]
    out = st._extrapolate(cur, 0.5, 2.5, items)
    # 框寬 100 → 上限 150px
    assert _cx(out[0]) == pytest.approx(900 + 150, abs=1)


def test_shift_cap_zero_disables():
    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="3.0", ANNOTATED_STREAM_INTERP_MAX="3.0",
              ANNOTATED_STREAM_INTERP_MATCH="0.50", ANNOTATED_STREAM_INTERP_MAX_SHIFT="0")
    st = _streamer(m)
    prev, cur = [_det(100, 500)], [_det(900, 500)]
    items = [(0.0, prev), (0.5, cur)]
    out = st._extrapolate(cur, 0.5, 2.5, items)
    assert _cx(out[0]) == pytest.approx(900 + 3200, abs=5)


def test_match_ratio_too_narrow_for_fast_vehicle():
    """0.08 只涵蓋低速車;快車位移超過門檻就配不到前一組,退回不外推。"""
    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="3.0", ANNOTATED_STREAM_INTERP_MAX="3.0",
              ANNOTATED_STREAM_INTERP_MATCH="0.08")
    st = _streamer(m)
    prev, cur = [_det(100, 500)], [_det(500, 500)]   # 位移 400px > 1920*0.08=153.6
    items = [(0.0, prev), (1.0, cur)]
    out = st._extrapolate(cur, 1.0, 1.5, items)
    assert _cx(out[0]) == 500

    m = _load(ANNOTATED_STREAM_INTERP_SPAN_MAX="3.0", ANNOTATED_STREAM_INTERP_MAX="3.0",
              ANNOTATED_STREAM_INTERP_MATCH="0.30", ANNOTATED_STREAM_INTERP_MAX_SHIFT="0")
    st = _streamer(m)
    out = st._extrapolate(cur, 1.0, 1.5, items)      # 門檻 576px,配得到
    assert _cx(out[0]) == pytest.approx(700, abs=2)
