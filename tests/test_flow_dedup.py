"""斷面計數去重:同一台車被 tracker 重新編號後不可以再算一次(2026-09-17)。

現場實測:0.01~1.2 秒內出現框幾乎重疊的重複事件 —— WN-1 8%、NE-2 4%、NE-1 2%、WN-2 1%。
原因是「已算過」記在 track_id 上,而這個路口的 track 在斷面附近會斷掉重編號。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.stream import bbox_iou  # noqa: E402


def _b(x1, y1, w, h):
    return {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h}


def test_same_vehicle_two_frames_overlaps_a_lot():
    """現場真實案例:WN-2 間隔 0.01 秒的兩筆。"""
    assert bbox_iou(_b(183, 474, 210, 149), _b(202, 480, 229, 154)) > 0.6


def test_nearly_identical_boxes():
    """NE-2 間隔 0.06 秒:框幾乎完全一樣。"""
    assert bbox_iou(_b(892, 482, 106, 67), _b(892, 482, 106, 66)) > 0.9


def test_two_different_vehicles_do_not_overlap():
    """前後兩台車即使同車道,框不會重疊。"""
    assert bbox_iou(_b(780, 308, 103, 93), _b(500, 300, 100, 90)) == 0.0
    assert bbox_iou(_b(780, 308, 103, 93), _b(780, 190, 103, 93)) < 0.4


def test_partial_overlap_below_threshold_is_not_deduped():
    """車尾接車頭那種輕微重疊不可以被當成同一台(門檻 0.4)。"""
    assert bbox_iou(_b(100, 100, 100, 100), _b(170, 100, 100, 100)) < 0.4


def test_bad_input_returns_zero():
    assert bbox_iou({}, _b(0, 0, 10, 10)) == 0.0
    assert bbox_iou(None or {}, {}) == 0.0
