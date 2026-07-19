#!/usr/bin/env python3
"""車速估算精度報告 + 建議校正係數。

用法：
    python scripts/speed_calibration_report.py [路徑]
    # 路徑可為 JSONL 檔或目錄，預設 ./output/speed_calib/

前置：
  1. 在 web 車速設定頁開「精度校正記錄」(detection_config.speed_calib_log=true)
  2. 該相機設好 trip-wire 兩條線 (speed_line_in / speed_line_out) + 車道距離，
     讓車輛跨線產生 ground truth
  3. 收集一段車流後跑本工具

判讀：
  - bias 正值 = 系統性高估、負值 = 低估
  - 純 pixel 組會給「建議係數」，直接填回車速設定頁的 speed_kmh_per_pxps
  - 已校正(homography) 組不給係數；誤差大代表校正矩形的實際寬長要重量
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detection.speed_calib import CALIB_DIR, analyze, load_samples  # noqa: E402


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    samples = load_samples(path)
    if not samples:
        print(f"⚠️ 沒有校正資料（查看 {path or CALIB_DIR}）。")
        print("   請先在車速設定頁開啟「精度校正記錄」並設好 trip-wire 兩條線。")
        return 1

    rep = analyze(samples)
    print(f"\n=== 車速估算精度報告（總 sample={rep['n_total']}）===\n")
    if not rep["groups"]:
        print("有資料但無有效配對（gt 與 est 需皆 > 0）。")
        return 1

    for g in rep["groups"]:
        tag = "已校正 homography" if g["calibrated"] else "純 pixel"
        print(f"[相機 {g['camera_id']} · {tag}]  n={g['n']}")
        print(f"  ground truth 平均 : {g['gt_mean_kmh']:.1f} km/h")
        print(f"  估算平均          : {g['est_mean_kmh']:.1f} km/h")
        print(f"  MAE 平均絕對誤差  : {g['mae_kmh']:.1f} km/h")
        print(f"  bias 系統偏差     : {g['bias_kmh']:+.1f} km/h（正=高估）")
        print(f"  MAPE 平均百分誤差 : {g['mape_pct']:.1f} %")
        if not g["calibrated"] and g.get("suggested_coeff"):
            print(f"  目前 speed_kmh_per_pxps = {g['current_coeff']}")
            print(f"  建議校正為              = {g['suggested_coeff']}  （×{g['ratio_median']}）")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
