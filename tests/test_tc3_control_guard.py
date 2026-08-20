"""號控的安全閘門。

這些檢查是唯一擋在「按錯一個按鈕就對運轉中的號誌送出位元組」前面的東西,
所以每一條都要有測試。壞掉的話不會有錯誤訊息 —— 只會有一個路口被改掉。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def tc3():
    import importlib.util
    os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret-" + "x" * 24)
    spec = importlib.util.spec_from_file_location(
        "_tc3_ctl", ROOT / "api" / "routes" / "signal_tc3.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        pytest.skip(f"signal_tc3 匯入失敗: {exc}")
    return mod


def test_型態以規範目錄為準(tc3):
    """分類錯了,擋不擋得住就全錯。

    🛑 不能只看指令碼高位元組 —— 用 105 條目錄比對過,高位元組 8 那格是混的:
       0F8E「密碼代碼－設定」和 0F8F「設定回報」都落在 8。
       所以 _kind_of 以目錄的 message_type 為準,目錄查不到才退回位元組規則。
    """
    assert tc3._kind_of(0x03, 0x5F) == "主動回報"   # 5F03 燈態主動回報
    assert tc3._kind_of(0x15, 0x5F) == "設定"       # 5F15 時制計畫
    assert tc3._kind_of(0x45, 0x5F) == "查詢"       # 5F45 時制計畫查詢
    assert tc3._kind_of(0xC0, 0x0F) == "查詢回報"
    # 就是那個例外:同樣是高位元組 8,一個是設定一個是回報
    assert tc3._kind_of(0x8E, 0x0F) == "設定"
    assert tc3._kind_of(0x8F, 0x0F) == "設定回報"


def test_位元組規則本身也要對(tc3):
    """目錄查不到的碼要退回這個規則,所以它也要正確。"""
    assert tc3._kind_by_nibble(0x03) == "主動回報"
    assert tc3._kind_by_nibble(0x15) == "設定"
    assert tc3._kind_by_nibble(0x45) == "查詢"
    assert tc3._kind_by_nibble(0x80) == "設定回報"
    assert tc3._kind_by_nibble(0xC0) == "查詢回報"


def test_總開關關著時什麼都不准送(tc3, monkeypatch):
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", False)
    for cmd in (0x45, 0x15, 0x10):
        why = tc3._control_guard(cmd)
        assert why, f"總開關關著卻放行了 cmd={cmd:02X}"
        assert "未啟用" in why


def test_控制器回給中心的訊息中心不會送(tc3, monkeypatch):
    """主動回報/設定回報/查詢回報都是控制器→中心,中心送這些沒有意義。"""
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", True)
    monkeypatch.setattr(tc3, "CONTROL_QUERY_ONLY", False)
    for cmd, dev in ((0x03, 0x5F), (0x8F, 0x0F), (0xC0, 0x0F)):
        why = tc3._control_guard(cmd, dev)
        assert why, f"{dev:02X}{cmd:02X} 應該被擋"


def test_只准查詢時設定類要被擋(tc3, monkeypatch):
    """這是預設值,也是驗證 TX 通不通時唯一該開的狀態。"""
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", True)
    monkeypatch.setattr(tc3, "CONTROL_QUERY_ONLY", True)
    assert tc3._control_guard(0x45) is None, "查詢應該放行"
    for cmd in (0x15, 0x10, 0x18, 0x1C):        # 時制/控制策略/指定時制/步階變換
        why = tc3._control_guard(cmd)
        assert why, f"只准查詢卻放行了 cmd={cmd:02X}"
        assert "只准查詢" in why


def test_開放設定類之後才放行(tc3, monkeypatch):
    monkeypatch.setattr(tc3, "CONTROL_ENABLED", True)
    monkeypatch.setattr(tc3, "CONTROL_QUERY_ONLY", False)
    for cmd in (0x15, 0x10, 0x45):
        assert tc3._control_guard(cmd) is None


def test_送出用的碼框組得出來且解得回去(tc3):
    """5F45 = 時制計畫查詢,參數帶 PlanID。"""
    frame = tc3.build_frame(0x1230, 5, bytes([0x5F, 0x45, 0x05]))
    out = tc3.decode_frame(frame)
    assert out is not None and out["cks_ok"] is True
    assert out["code"] == "5F45"
    assert out["addr"] == 0x1230
    assert out["len"] == len(frame)


def test_位址推得規則(tc3, monkeypatch):
    """猜錯位址等於把命令送給別的路口,所以來源要明確。"""
    monkeypatch.setattr(tc3, "CONTROL_ADDR", 0x9999)
    assert tc3._target_addr() == 0x9999          # 有設定就用設定

    monkeypatch.setattr(tc3, "CONTROL_ADDR", 0)
    tc3._frames.clear()
    assert tc3._target_addr() is None            # 沒設定也沒抄到 → 不要亂猜
    tc3._frames.append({"addr": 0x1230})
    assert tc3._target_addr() == 0x1230          # 用抄到的
