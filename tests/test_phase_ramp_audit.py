"""2026-09-17 全面盤點抓到的兩個活 bug + 反向退路值。

背景:分相編號↔匝道的對應會因現場改線路對調(09-15 斷電施工就對調過一次)。
凡是「寫死分相號碼代表某條匝道」的地方,對調後都會靜靜地算錯數字,
而且不會有任何錯誤訊息 —— 這正是這次盤點的目的。
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def test_outcome_passes_both_storage_limits():
    """🛑 主線回堵次數曾經**恆為 0**:只送了 storage_2。

    evaluate_outcome 要有 storage_1 才會累計 spillback_events_1,而現行主線
    保護相就是分相1(下匝道)——少送一邊,spillback_events_mainline 永遠是 0,
    看起來像「從來沒有回堵過」。
    """
    from detection.signal_decision_engine import evaluate_outcome

    # 兩相都逼近儲車上限(600m 的 95%、210m 的 95%)
    samples = [{"queue_m_1": 570, "queue_m_2": 200,
                "storage_1": 600, "storage_2": 210,
                "interval_sec": 5, "switched": False} for _ in range(3)]
    out = evaluate_outcome(samples)
    assert out["spillback_events_1"] == 3, "有給 storage_1 就要算得出分相1 的回堵"
    assert out["spillback_events_2"] == 3

    # 少送 storage_1 就會漏算 —— 這正是修掉的 bug 的形狀
    missing = [dict(s, storage_1=None) for s in samples]
    assert evaluate_outcome(missing)["spillback_events_1"] == 0


def test_outcome_query_sends_storage_for_both_phases():
    """查詢端(signal_shadow)要把兩相的儲車上限都放進 samples。"""
    import pathlib
    src = pathlib.Path("api/routes/signal_shadow.py").read_text(encoding="utf-8")
    assert '"storage_1": st1' in src, "少了 storage_1 → 主線回堵永遠 0"
    assert '"storage_2": st2' in src


def test_exit_queue_follows_off_ramp_role():
    """出口滯留要跟著 off_ramp 這個角色走,不可寫死 queue_m_2。"""
    import pathlib
    src = pathlib.Path("api/routes/signal_shadow.py").read_text(encoding="utf-8")
    i = src.index("出口(下匝道)滯留")
    seg = src[i:i + 900]
    assert '_phase_of_role("off_ramp")' in seg, "欄位要由 role 決定"
    assert "exit_queue_phase" in seg, "要回報用的是哪一相,事後才查得出來"


def test_fallback_constants_match_the_baseline():
    """設定讀不到時的退路值必須與基準表同向 —— 寫反的退路比沒有更危險。"""
    from api.routes import signal_shadow as S
    from detection.signal_timing_lookup import phase_of_role

    off, on = phase_of_role("off_ramp"), phase_of_role("on_ramp")
    assert S.PHASE_CAMERA[off] == 4 and S.PHASE_CAMERA[on] == 3
    assert S.PHASE_STOPLINE[off] == 5 and S.PHASE_STOPLINE[on] == 3
    assert S.APPROACH_LEN_M[off] == 16.0 and S.APPROACH_LEN_M[on] == 52.7


def test_report_ramp_name_comes_from_lookup():
    """報告的匝道名要查表,不可寫死 ph=='1' → 上匝道。"""
    from detection.signal_report_md import _ramp_name
    from detection.signal_timing_lookup import ramp_name, phase_of_role

    assert _ramp_name("1") == ramp_name(1)
    assert _ramp_name(str(phase_of_role("off_ramp"))) == "下匝道"
