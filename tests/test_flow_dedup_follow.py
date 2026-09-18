"""去重框跟著車走(2026-09-18)。

🛑 WN-1 15:17~15:18 一分鐘記了 6 筆大貨車,線圈同一分鐘大型只有 2 輛 ——
   停在斷面窄帶內的聯結車,框在晃、軌跡斷掉重生,每個片段都再算一次。
   但也不能把時間窗拉長:排隊時前後兩台會在幾秒內經過同一位置。
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _b(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_stopped_trailer_stays_suppressed():
    """實際事件的框:聯結車在同一區晃 8 秒以上,去重框一路跟著 → 仍然有效。"""
    from api.routes.stream import _follow_recent, bbox_iou
    recent = [(0.0, _b(664, 275, 883, 488), 0.0)]            # 15:17:08 計數
    frames = [_b(670, 270, 890, 495), _b(655, 260, 900, 510), _b(640, 250, 910, 530),
              _b(630, 240, 915, 560), _b(620, 235, 920, 580), _b(634, 235, 864, 504)]
    t = 0.0
    for fb in frames:
        t += 1.5
        recent = _follow_recent(recent, [fb], t, follow_iou=0.5, max_sec=20)
    assert recent[0][0] == t, "車一直在,最後看到的時間要一直刷新"
    # 8~9 秒後新生片段(15:17:50 那一筆的框)仍與去重框重疊 → 會被當成同一台
    assert bbox_iou(recent[0][1], _b(688, 246, 928, 600)) >= 0.4


def test_following_car_in_queue_still_counts():
    """排隊:前車計數後往前開,去重框跟著前車走;後車開進同一位置時不可被擋。

    移動量取實際值:6 fps、排隊時速約 5 km/h → 一幀約 0.2 m,畫面上約 8~10 px。
    (第一版測試一步移動 40 px,兩幀重疊只剩 0.40,不像真實的車流。)
    """
    from api.routes.stream import _follow_recent, bbox_iou
    start = _b(740, 340, 860, 446)
    recent = [(0.0, dict(start), 0.0)]
    t = 0.0
    a = dict(start)
    for _ in range(14):                                      # 前車往下(往前)開 14 幀
        t += 1 / 6
        a = {"x1": a["x1"] + 1, "y1": a["y1"] + 9, "x2": a["x2"] + 1, "y2": a["y2"] + 9}
        recent = _follow_recent(recent, [a], t, follow_iou=0.5)
    car_b = dict(start)                                      # 後車開到前車當初被計數的位置
    recent = _follow_recent(recent, [a, car_b], t + 1 / 6, follow_iou=0.5)
    assert recent[0][1] == a, "去重框要跟著前車,不能被後車搶走"
    assert bbox_iou(recent[0][1], car_b) < 0.4, "後車不重疊 → 照常計數"


def test_fast_car_falls_back_to_old_rule():
    """車開快、兩幀重疊不到門檻:框跟不上就不跟,退回原本 1.5 秒過期規則(不會比舊版差)。"""
    from api.routes.stream import _follow_recent
    recent = [(0.0, _b(740, 340, 860, 446), 0.0)]
    out = _follow_recent(recent, [_b(745, 420, 870, 530)], 0.2, follow_iou=0.5)
    assert out[0][0] == 0.0, "沒跟上 → 最後看到時間不刷新"


def test_one_detection_cannot_be_claimed_twice():
    """兩台已計數的車不可以共用同一個偵測框。"""
    from api.routes.stream import _follow_recent
    recent = [(0.0, _b(100, 100, 200, 200), 0.0), (0.0, _b(105, 100, 205, 200), 0.0)]
    out = _follow_recent(recent, [_b(102, 100, 202, 200)], 1.0, follow_iou=0.5)
    followed = [e for e in out if e[0] == 1.0]
    assert len(followed) == 1


def test_follow_is_capped():
    """跟隨上限到了就不再刷新,交給原本的過期規則收掉。"""
    from api.routes.stream import _follow_recent
    recent = [(0.0, _b(100, 100, 200, 200), 0.0)]
    out = _follow_recent(recent, [_b(100, 100, 200, 200)], 25.0, follow_iou=0.5, max_sec=20)
    assert out[0][0] == 0.0, "超過 20 秒不刷新"


def test_flag_default_off():
    import api.routes.stream as S
    orig = S._DEDUP_FOLLOW_RAW
    try:
        S._DEDUP_FOLLOW_RAW = ""
        assert S._dedup_follow_on(4) is False
        S._DEDUP_FOLLOW_RAW = "4"
        assert S._dedup_follow_on(4) and not S._dedup_follow_on(5)
    finally:
        S._DEDUP_FOLLOW_RAW = orig
