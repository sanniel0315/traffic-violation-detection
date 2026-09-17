"""主鍵是匝道,不是分相編號(2026-09-17 使用者定調)。

🛑 為什麼:分相編號是控制器協定當下的身分,現場改線路就會對調
   (09-11、09-12、09-16 各換過一次)。每一次對調都在某個「寫死編號」的角落
   留下靜靜算錯的數字 —— 當天盤點抓到「主線回堵次數恆為 0」「出口滯留取到
   上匝道」「報表把兩條匝道的數字掛在對方名下」三件。
   對外輸出一律附上以匝道為鍵的那一組,呼叫端就不必自己翻譯。
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def test_by_ramp_keys_are_roles_not_numbers():
    from detection.signal_timing_lookup import by_ramp, phase_of_role

    out = by_ramp(lambda ph: {"q": ph * 10})
    assert set(out) == {"off_ramp", "on_ramp"}
    assert out["off_ramp"]["ramp"] == "下匝道"
    assert out["on_ramp"]["ramp"] == "上匝道"
    # 值要跟著角色走,不是跟著編號走
    assert out["off_ramp"]["value"]["q"] == phase_of_role("off_ramp") * 10
    assert out["on_ramp"]["value"]["q"] == phase_of_role("on_ramp") * 10


def test_by_ramp_carries_what_the_number_used_to_imply():
    """編號以前隱含的東西(儲車、主線保護、目前是哪一相)都要明講。"""
    from detection.signal_timing_lookup import by_ramp

    out = by_ramp(lambda ph: None)
    assert out["off_ramp"]["storage_m"] == 600 and out["off_ramp"]["priority"] is True
    assert out["on_ramp"]["storage_m"] == 210 and out["on_ramp"]["priority"] is False
    # 編號仍要帶著 —— 下控 5F1C 需要它,只是不再當主鍵
    assert out["off_ramp"]["phase_no"] in (1, 2)
    assert out["off_ramp"]["phase_no"] != out["on_ramp"]["phase_no"]


def test_outcome_and_plan_expose_ramp_keyed_block():
    import pathlib
    src = pathlib.Path("api/routes/signal_shadow.py").read_text(encoding="utf-8")
    assert 'out["by_ramp"] = by_ramp(' in src, "成效 KPI 要附匝道鍵"
    assert '"ramp_state": ramp_state' in src, "/plan 要附匝道鍵"


def test_by_ramp_survives_a_swapped_baseline(monkeypatch):
    """對調基準表之後,同一支程式要自動跟著換 —— 這正是編號當主鍵做不到的。"""
    import detection.signal_timing_lookup as LK

    base = LK.load_baseline()
    swapped = {
        "phases": {
            "1": dict(base["phases"]["2"]),      # 分相1 變成上匝道
            "2": dict(base["phases"]["1"]),
        },
        "plans": base.get("plans", {}),
    }
    monkeypatch.setattr(LK, "load_baseline", lambda: swapped)
    out = LK.by_ramp(lambda ph: {"ph": ph})
    assert out["off_ramp"]["value"]["ph"] == 2, "下匝道現在是分相2"
    assert out["off_ramp"]["ramp"] == "下匝道"
    assert out["off_ramp"]["storage_m"] == 600, "儲車上限跟著匝道走,不跟著編號"
