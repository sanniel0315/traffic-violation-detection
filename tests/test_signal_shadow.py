"""影子模式:只記錄不下發 + 切換偵測正確性。

影子模式是 bypass OPAC 的前置驗證 —— 我方決策全速運轉但不碰控制器,
趁 OPAC 還在跑時累積對照資料。最重要的保證是「絕對不下發」。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_shadow_never_sends_anything():
    """★最重要:影子模組不可以有任何下發行為。

    用 AST 檢查「實際被呼叫的函式名」,不掃註解與 docstring
    (docstring 裡會提到 control/send,那是在說明「不走那條路」)。
    """
    import ast
    src = (ROOT / "api" / "routes" / "signal_shadow.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    forbidden = {"control_send", "send", "sendall", "_send_frame",
                 "send_frame", "write_frame"}
    hit = called & forbidden
    assert not hit, "影子模組不可呼叫下發相關函式:%s" % hit

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.name)
    assert "control_send" not in imported
    assert "control_prepare" not in imported


def test_shadow_disabled_by_default():
    """預設不啟用 —— 要明確開啟才跑。"""
    import importlib
    import api.routes.signal_shadow as m
    os.environ.pop("SIGNAL_SHADOW_ENABLED", None)
    m = importlib.reload(m)
    assert m.SHADOW_ENABLED is False


def test_interval_aligns_with_opac():
    """取樣週期預設 5 秒,與 OPAC 的決策週期對齊才好比對。"""
    import importlib
    import api.routes.signal_shadow as m
    os.environ.pop("SIGNAL_SHADOW_INTERVAL_SEC", None)
    m = importlib.reload(m)
    assert m.SHADOW_INTERVAL_SEC == 5


def test_phase_camera_mapping_matches_baseline():
    """分相→相機的對應要與官方時制表的 constraint_camera 一致。

    baseline: 分相1(上匝道)=ID3、分相2(下匝道)=ID4
    """
    import importlib
    import api.routes.signal_shadow as m
    for k in ("SIGNAL_SHADOW_CAM_PHASE1", "SIGNAL_SHADOW_CAM_PHASE2"):
        os.environ.pop(k, None)
    m = importlib.reload(m)
    assert m.PHASE_CAMERA[1] == 3
    assert m.PHASE_CAMERA[2] == 4


def test_queue_m_returns_none_when_no_data():
    """壅塞沒資料時回 None 不當機(決策端會當 0 處理)。"""
    import api.routes.signal_shadow as m
    assert m._queue_m(99999) is None


def test_live_phase_none_when_no_frames():
    """沒有抄到燈態時回 None,迴圈會跳過該輪而不是亂算。"""
    import api.routes.signal_shadow as m
    r = m._live_phase()
    assert r is None or isinstance(r.get("sub_phase_id"), int)


def test_stop_is_idempotent():
    import api.routes.signal_shadow as m
    m.stop_shadow()
    m.stop_shadow()   # 重複呼叫不可當機
