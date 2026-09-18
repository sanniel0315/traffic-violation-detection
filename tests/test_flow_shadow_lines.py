"""影子計數線與依車種去重(2026-09-18,使用者:「解好」)。

🛑 下匝道線圈與 WN-1/WN-2 看同一批車:小車少算約 3 成、大車多算 1.4~2.8 倍。
   計數線位置不再用猜的 —— 6 條影子線各自記錄,之後逐條跟線圈比。
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def test_shadow_lines_span_the_zone():
    from api.routes.stream import shadow_line_positions, SHADOW_LINE_FRACS
    out = shadow_line_positions([-200.0, 50.0, 300.0, -100.0])
    assert [k for k, _, _ in out] == list(range(len(SHADOW_LINE_FRACS)))
    ts = [t for _, _, t in out]
    assert ts == sorted(ts), "由軸的一端排到另一端"
    assert -200 < ts[0] < ts[-1] < 300, "都在框的範圍內,不含兩個邊緣"
    assert abs(ts[0] - (-200 + 500 * 0.10)) < 1e-6


def test_shadow_lines_empty_zone():
    from api.routes.stream import shadow_line_positions
    assert shadow_line_positions([]) == []


def test_shadow_flag_default_off():
    import api.routes.stream as S
    orig = S._SHADOW_LINES_RAW
    try:
        S._SHADOW_LINES_RAW = ""
        assert S._shadow_lines_on(4) is False, "預設關閉 —— 部署程式不改變行為"
        S._SHADOW_LINES_RAW = "4,5"
        assert S._shadow_lines_on(5) and not S._shadow_lines_on(2)
    finally:
        S._SHADOW_LINES_RAW = orig


def test_large_dedup_window_only_for_large():
    from api.routes.stream import _dedup_window
    assert _dedup_window((0.0, {}, 0.0, True), 1.5, 25.0) == 25.0
    assert _dedup_window((0.0, {}, 0.0, False), 1.5, 25.0) == 1.5
    assert _dedup_window((0.0, {}, 0.0), 1.5, 25.0) == 1.5, "舊格式(3 欄)視為一般車"
    assert _dedup_window((0.0, {}, 0.0, True), 1.5, 0.0) == 1.5, "0 = 關閉,與一般相同"


def test_follow_keeps_large_flag():
    """去重跟隨更新位置時不可以把「是不是大車」這一欄丟掉。"""
    from api.routes.stream import _follow_recent
    b = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
    out = _follow_recent([(0.0, b, 0.0, True)], [dict(b)], 1.0, follow_iou=0.5)
    assert out[0][3] is True
