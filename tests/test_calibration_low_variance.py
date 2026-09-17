"""校準門檻:相關係數只在「排隊真的有在動」時才有意義。

🛑 2026-09-18 逐窗稽核 35 個時段窗的發現:夜間窗(20:00-24:00)的 MAE 只有
   2~3 公尺(模型其實很準),但 r 只有 0.15~0.4 —— 因為整晚排隊都貼近 0,
   幾乎沒有變異可以相關。用 r 擋掉這種窗,擋掉的是「沒東西好預測」,
   不是「模型不準」,結果是可用樣本被無謂地砍掉七成。

   但低變異窗**不可以**拿來宣稱模型在尖峰也準,所以結果要標明它是低變異窗。
"""
import os

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _stats(mae, sd, r, n=500):
    return {"n": n, "mae": mae, "sd_m": sd, "r": r}


def test_low_variance_window_passes_on_mae_alone():
    from detection.signal_sim import _calib_verdict

    # 夜間:排隊幾乎不動(sd 3m)、模型很準(MAE 2.5m)、r 低
    s = _stats(2.5, 3.0, 0.2)
    out = _calib_verdict(s, s, method="cycle")
    assert out["usable"] is True
    assert out["low_variance"], "要標明這是低變異窗"
    assert "低變異" in out["reason"]


def test_low_variance_still_rejected_when_mae_is_large():
    """低變異不是免死金牌 —— 絕對誤差過大照樣擋。"""
    from detection.signal_sim import _calib_verdict

    s = _stats(20.0, 3.0, 0.9)
    assert _calib_verdict(s, s, method="cycle")["usable"] is False


def test_high_variance_window_still_needs_correlation():
    """排隊有在動的時段,r 門檻照舊 —— 這是原本就該擋的情況。"""
    from detection.signal_sim import _calib_verdict

    s = _stats(8.0, 20.0, 0.3)
    out = _calib_verdict(s, s, method="cycle")
    assert out["usable"] is False
    assert "相關係數" in out["reason"]

    good = _stats(8.0, 20.0, 0.62)
    assert _calib_verdict(good, good, method="cycle")["usable"] is True


def test_fit_stats_reports_sd():
    """sd_m 要真的算出來,不然上面的判斷沒有依據。"""
    from detection.signal_sim import _fit_stats

    xs = [0, 10, 20, 30, 40] * 4
    ys = [1, 11, 19, 31, 39] * 4
    st = _fit_stats(xs, ys)
    assert st["sd_m"] > 12, "這組資料的標準差約 14m"
    assert st["mae"] < 2


def test_one_phase_low_variance_other_not():
    """兩相分開判:一相夜間沒車、另一相仍在動,不可以互相豁免。"""
    from detection.signal_sim import _calib_verdict

    quiet = _stats(2.0, 2.0, 0.1)          # 低變異 → 以 MAE 過
    busy_bad = _stats(9.0, 25.0, 0.2)      # 有變異但相關不足 → 擋
    out = _calib_verdict(quiet, busy_bad, method="cycle")
    assert out["usable"] is False
    assert "分相2 相關係數" in out["reason"]
