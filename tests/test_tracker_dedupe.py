"""VehicleTracker:同一台車的重複偵測框要去重,不可算成兩台。

🛑 現場 bug(2026-09-03 cam3):YOLO 同一幀對同一台車吐出兩個高度重疊的框,
   tracker 只做「偵測↔track」配對,第一個框配到 track A、第二個框因 A 已被
   佔用而新建 track B → 一台車變兩台 → 排隊 ~5m 變 11.5m。
   而排隊正在餵給 OPAC 做號誌配時決策,所以這不只是顯示問題。
   實測值:track 323 與 325 的 IoU=0.93、中心只差 3px。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.violation_detector import VehicleTracker


def _det(x1, y1, x2, y2, cls="car"):
    # 真實偵測的 bbox 都帶 width/height,測試資料要一致
    return {"class_name": cls,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "width": x2 - x1, "height": y2 - y1}}


def test_duplicate_boxes_counted_once():
    """★核心:兩個幾乎重合的框只能算一台車。

    重現現場數值:IoU 0.93、中心差 3px。
    """
    t = VehicleTracker(max_age=30, iou_threshold=0.15)
    out = t.update([_det(800, 470, 970, 590), _det(803, 473, 969, 588)])
    assert len(out) == 1, "重複框應被去重,只留一台"
    assert len(t.tracks) == 1, "不可產生第二個 track"


def test_two_real_vehicles_kept():
    """相鄰但不同的兩台車不可被誤併(前後車 IoU 遠低於門檻)。"""
    t = VehicleTracker(max_age=30, iou_threshold=0.15)
    out = t.update([_det(100, 100, 200, 200), _det(210, 100, 310, 200)])
    assert len(out) == 2
    assert len(t.tracks) == 2


def test_partial_overlap_kept():
    """部分重疊(如車道相鄰)不算重複 —— 只有極高重疊才去重。"""
    t = VehicleTracker(max_age=30, iou_threshold=0.15)
    # IoU 約 0.33,低於 DEDUPE_IOU 0.7
    out = t.update([_det(100, 100, 200, 200), _det(150, 100, 250, 200)])
    assert len(out) == 2


def test_different_class_not_merged():
    """不同類別不合併 —— 避免把車上載的東西或不同車種誤併。"""
    t = VehicleTracker(max_age=30, iou_threshold=0.15)
    out = t.update([_det(100, 100, 200, 200, "car"),
                    _det(102, 102, 198, 198, "truck")])
    assert len(out) == 2


def test_larger_box_kept():
    """去重時保留面積較大的框(被切開時較完整的那個通常較大)。"""
    t = VehicleTracker(max_age=30, iou_threshold=0.15)
    out = t.update([_det(100, 100, 190, 190), _det(100, 100, 200, 200)])
    assert len(out) == 1
    b = out[0]["bbox"]
    assert (b["x2"] - b["x1"]) == 100, "應保留較大的框"


def test_three_duplicates_collapse_to_one():
    """三個重複框也要收斂成一台。"""
    t = VehicleTracker(max_age=30, iou_threshold=0.15)
    out = t.update([_det(800, 470, 970, 590),
                    _det(802, 472, 968, 588),
                    _det(801, 471, 969, 589)])
    assert len(out) == 1


def test_empty_and_single_unchanged():
    t = VehicleTracker()
    assert t.update([]) == []
    out = t.update([_det(100, 100, 200, 200)])
    assert len(out) == 1
