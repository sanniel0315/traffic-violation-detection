"""壅塞快照不可以污染「正在跑的那條分析」的狀態。

快照是唯讀的除錯工具。若和執行中的壅塞執行緒共用 camera_key,每呼叫一次
就等於往那台相機插入一張「別的時間點」的畫面:平滑歷史多一筆、tracker 軌跡
被打斷(車被當成新出現)、flow 的「出現又消失」被誤計成通過、排隊持續時間重算。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def cong(monkeypatch):
    import api.routes.congestion as m
    seen = []

    class _FakeDetector:
        def analyze(self, frame, zones, camera_key="default", params=None):
            seen.append(camera_key)
            return {"level": "low", "vehicles": []}

    monkeypatch.setattr(m, "get_detector", lambda: _FakeDetector())
    monkeypatch.setattr(m, "get_effective_params", lambda cid: {})
    return m, seen


def test_default_uses_live_state_key(cong):
    """不給 state_key 時就是那條正在跑的分析,行為不變。"""
    m, seen = cong
    m.analyze_with_lock(None, [], 3)
    assert seen == ["3"]


def test_explicit_state_key_is_isolated(cong):
    """給了 state_key 就用它,不碰 live 狀態。"""
    m, seen = cong
    m.analyze_with_lock(None, [], 3, state_key="3::snapshot")
    assert seen == ["3::snapshot"]
    assert "3" not in seen


def test_snapshot_key_differs_from_live_key(cong):
    """兩者必須不同 —— 相同就等於沒隔離。"""
    m, seen = cong
    m.analyze_with_lock(None, [], 3)
    m.analyze_with_lock(None, [], 3, state_key="3::snapshot")
    assert len(set(seen)) == 2


def test_status_text_never_renders_question_marks():
    """沒有 CJK 字型時要退回英文,不可以畫出一排 ??????(現場實際看到過)。"""
    import api.routes.congestion as m
    img = np.zeros((60, 900, 3), dtype=np.uint8)
    m._draw_status_text(img, "壅塞: 擁擠 | 車輛: 8", (10, 6), (0, 165, 255))
    assert int((img > 0).sum()) > 0          # 有畫出東西,不是空白


def test_status_text_ascii_fallback_has_no_cjk(monkeypatch):
    """退回路徑送進 cv2.putText 的字串不可以含中文 —— 含了就會變問號。"""
    import api.routes.congestion as m
    monkeypatch.setattr(m, "_cjk_font", lambda size=22: None)
    captured = {}

    def _fake_puttext(img, text, *a, **k):
        captured["text"] = text

    monkeypatch.setattr(m.cv2, "putText", _fake_puttext)
    m._draw_status_text(np.zeros((60, 900, 3), dtype=np.uint8),
                        "壅塞: 擁擠 | 車輛: 8 | 停滯: 4 | 排隊: 25.5m | 分數: 62.8%",
                        (10, 6), (0, 165, 255))
    assert all(ord(ch) < 128 for ch in captured["text"]), captured["text"]
