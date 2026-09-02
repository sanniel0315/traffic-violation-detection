"""號誌控制模式判讀 —— 誰在控制。

🛑 核心教訓(2026-09-02 現場實測):不可以只看 roadSideManual 就說是手動。
   OPAC 接管時 takeover-strategy 會同時寫 roadSideManual=1 + phase=1,
   那期間路口是被演算法動態控制,不是現場有人操作控制箱。
   當時就是只看 roadSideManual 而誤判成「被切手動」。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.signal_tc3 import _control_mode

FIXTIME, DYNAMIC, ROADSIDE, CENTER, PHASE = 0x01, 0x02, 0x04, 0x08, 0x10


def test_opac_takeover_is_not_manual():
    """★最重要:OPAC 接管期間 roadSideManual=1 + phase=1,不可判成手動。

    這正是 2026-09-02 現場實際出現的值。
    """
    v = ROADSIDE | PHASE
    r = _control_mode(v)
    assert r["code"] == "external_dynamic"
    assert "手動" not in r["label"]


def test_real_roadside_manual():
    """真的現場手動:只有 roadSideManual、沒有 phase 也沒有 fixTime。"""
    r = _control_mode(ROADSIDE)
    assert r["code"] == "roadside_manual"
    assert r["severity"] == "warn"


def test_fixtime_after_opac_release():
    """OPAC 交還後:phase 歸 0、fixTime=1,但 roadSideManual 位元殘留為 1。

    2026-09-02 實測三次回退,值都是 fixTime=1, roadSideManual=1, phase=0。
    這要判成「定時控制」,不是手動。
    """
    v = FIXTIME | ROADSIDE
    r = _control_mode(v)
    assert r["code"] == "fixtime"


def test_plain_fixtime():
    assert _control_mode(FIXTIME)["code"] == "fixtime"


def test_center_manual():
    assert _control_mode(CENTER)["code"] == "center_manual"


def test_controller_builtin_dynamic():
    """控制器內建動態(dynamic bit),不是外部接管。"""
    r = _control_mode(DYNAMIC)
    assert r["code"] == "controller_dynamic"


def test_phase_takes_priority_over_everything():
    """phase(外部逐步階接管)優先級最高 —— 有它就是外部在控。"""
    for extra in (FIXTIME, ROADSIDE, CENTER, DYNAMIC, FIXTIME | ROADSIDE | CENTER):
        assert _control_mode(PHASE | extra)["code"] == "external_dynamic"


def test_none_and_invalid():
    assert _control_mode(None)["code"] == "unknown"
    assert _control_mode("x")["code"] == "unknown"


def test_zero_is_other_not_crash():
    r = _control_mode(0)
    assert r["code"] == "other"
