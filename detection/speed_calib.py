#!/usr/bin/env python3
"""車速估算精度驗證 + 調參的資料層。

以 trip-wire 跨線速度為 ground truth（實測物理速度：車道距離 ÷ 跨線秒數），
記錄同一 track 當下的視覺估速（pixel / homography / kalman），供離線比對
誤差（MAE / bias / MAPE）並反推 speed_kmh_per_pxps 校正係數。

流程：
- `log_sample()`：trip-wire 觸發時呼叫，append 一筆 JSONL。
  由 `detection_config.speed_calib_log` 開關控制（預設關，不影響 production）。
- `load_samples()` / `analyze()`：給 `scripts/speed_calibration_report.py`
  與單元測試用的純函式，無副作用。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 校正 sample 輸出目錄（與違規截圖同一 output 樹下，可用環境變數覆寫）
CALIB_DIR = Path(os.environ.get("SPEED_CALIB_DIR", "./output/speed_calib"))


def log_sample(
    camera_id: int,
    track_id: Any,
    gt_kmh: float,
    est_kmh: Optional[float],
    *,
    calibrated: bool,
    unit: str,
    dist_m: float,
    dt_cross: float,
    coeff: float,
    lane: str = "",
    ts: Optional[float] = None,
) -> None:
    """append 一筆校正 sample 到當日 JSONL。

    失敗一律吞掉（絕不可影響偵測主流程）。

    Args:
        gt_kmh: trip-wire 跨線實測速度（ground truth）
        est_kmh: 同一 track 當下的視覺估速（可能為 None）
        calibrated: 該估速是否用 homography 世界座標（True）或純 pixel（False）
        unit: 'world_m' 或 'pixel'
        coeff: 當下生效的 speed_kmh_per_pxps（只有純 pixel 才有意義）
    """
    try:
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        _ts = ts if ts is not None else time.time()
        rec = {
            "ts": round(float(_ts), 3),
            "camera_id": int(camera_id),
            "track_id": track_id,
            "lane": str(lane),
            "gt_kmh": round(float(gt_kmh), 2),
            "est_kmh": (round(float(est_kmh), 2)
                        if isinstance(est_kmh, (int, float)) else None),
            "calibrated": bool(calibrated),
            "unit": str(unit),
            "dist_m": round(float(dist_m), 3),
            "dt_cross": round(float(dt_cross), 4),
            "coeff": round(float(coeff), 5),
        }
        day = time.strftime("%Y%m%d", time.localtime(_ts))
        path = CALIB_DIR / f"speed_calib_{day}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_samples(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """讀入 JSONL sample。path 可為單一檔或目錄（預設整個 CALIB_DIR）。"""
    target = Path(path) if path else CALIB_DIR
    if target.is_dir():
        files = sorted(target.glob("*.jsonl"))
    elif target.is_file():
        files = [target]
    else:
        files = []
    out: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    return out


def _median(xs: List[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def analyze(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """依 (camera_id, calibrated) 分組算誤差統計 + 建議係數。

    僅納入 gt_kmh > 0 且 est_kmh > 0 的有效配對。
    純 pixel 組（calibrated=False）因估速與係數成線性關係，可反推建議係數：
        suggested_coeff = current_coeff × median(gt / est)
    """
    groups: Dict[Tuple[Any, bool], List[Dict[str, Any]]] = {}
    for s in samples:
        gt = s.get("gt_kmh")
        est = s.get("est_kmh")
        if not isinstance(gt, (int, float)) or gt <= 0:
            continue
        if not isinstance(est, (int, float)) or est <= 0:
            continue
        key = (s.get("camera_id"), bool(s.get("calibrated")))
        groups.setdefault(key, []).append(s)

    results: List[Dict[str, Any]] = []
    for (cam, calibrated), rows in sorted(
        groups.items(), key=lambda kv: ((kv[0][0] if kv[0][0] is not None else -1), kv[0][1])
    ):
        n = len(rows)
        errs = [r["est_kmh"] - r["gt_kmh"] for r in rows]
        abs_errs = [abs(e) for e in errs]
        pct_errs = [abs(r["est_kmh"] - r["gt_kmh"]) / r["gt_kmh"] * 100.0 for r in rows]
        ratios = [r["gt_kmh"] / r["est_kmh"] for r in rows]  # est × ratio = gt
        gt_mean = sum(r["gt_kmh"] for r in rows) / n
        est_mean = sum(r["est_kmh"] for r in rows) / n
        entry: Dict[str, Any] = {
            "camera_id": cam,
            "calibrated": calibrated,
            "n": n,
            "gt_mean_kmh": round(gt_mean, 2),
            "est_mean_kmh": round(est_mean, 2),
            "mae_kmh": round(sum(abs_errs) / n, 2),
            "bias_kmh": round(sum(errs) / n, 2),
            "mape_pct": round(sum(pct_errs) / n, 1),
        }
        # 只有純 pixel 的係數可線性反推
        if not calibrated:
            coeffs = [r["coeff"] for r in rows
                      if isinstance(r.get("coeff"), (int, float)) and r["coeff"] > 0]
            cur_coeff = _median(coeffs) if coeffs else None
            corr = _median(ratios)
            entry["current_coeff"] = round(cur_coeff, 5) if cur_coeff else None
            entry["suggested_coeff"] = (round(cur_coeff * corr, 5) if cur_coeff else None)
            entry["ratio_median"] = round(corr, 4)
        results.append(entry)

    return {"n_total": sum(len(r) for r in groups.values()), "groups": results}
