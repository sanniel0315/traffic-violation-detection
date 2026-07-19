#!/usr/bin/env python3
"""speed_calib.analyze() / log_sample() 單元測試

執行：
    python -m detection.test_speed_calib
"""
import json
import tempfile
import unittest
from pathlib import Path

from detection.speed_calib import analyze, load_samples, log_sample


def _s(cam, gt, est, calibrated=False, coeff=0.12):
    return {"camera_id": cam, "gt_kmh": gt, "est_kmh": est,
            "calibrated": calibrated, "coeff": coeff, "unit": "pixel"}


class TestAnalyze(unittest.TestCase):
    def test_pixel_low_estimate_suggests_higher_coeff(self):
        # 估算一律為 GT 的 0.8 倍 → 需要把係數放大 1.25 倍
        rows = [_s(7, 50, 40), _s(7, 60, 48), _s(7, 40, 32)]
        rep = analyze(rows)
        self.assertEqual(len(rep["groups"]), 1)
        g = rep["groups"][0]
        self.assertEqual(g["n"], 3)
        self.assertFalse(g["calibrated"])
        # bias 為負（低估）
        self.assertLess(g["bias_kmh"], 0)
        # ratio median = 1.25 → 建議係數 = 0.12 × 1.25 = 0.15
        self.assertAlmostEqual(g["ratio_median"], 1.25, places=3)
        self.assertAlmostEqual(g["suggested_coeff"], 0.15, places=4)

    def test_filters_invalid_pairs(self):
        rows = [_s(1, 50, 40), _s(1, 0, 40), _s(1, 50, 0), _s(1, 50, None)]
        rep = analyze(rows)
        self.assertEqual(rep["n_total"], 1)  # 只有第一筆有效
        self.assertEqual(rep["groups"][0]["n"], 1)

    def test_calibrated_group_has_no_coeff_suggestion(self):
        rows = [_s(3, 50, 51, calibrated=True), _s(3, 60, 59, calibrated=True)]
        rep = analyze(rows)
        g = rep["groups"][0]
        self.assertTrue(g["calibrated"])
        self.assertNotIn("suggested_coeff", g)
        self.assertLess(g["mae_kmh"], 2.0)

    def test_groups_split_by_camera_and_calibrated(self):
        rows = [_s(1, 50, 40), _s(2, 50, 40), _s(1, 50, 49, calibrated=True)]
        rep = analyze(rows)
        keys = {(g["camera_id"], g["calibrated"]) for g in rep["groups"]}
        self.assertEqual(keys, {(1, False), (2, False), (1, True)})

    def test_empty(self):
        rep = analyze([])
        self.assertEqual(rep["n_total"], 0)
        self.assertEqual(rep["groups"], [])


class TestLogAndLoadRoundTrip(unittest.TestCase):
    def test_log_sample_writes_loadable_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            import detection.speed_calib as sc
            orig = sc.CALIB_DIR
            sc.CALIB_DIR = Path(d)
            try:
                log_sample(7, 12, 42.3, 38.1, calibrated=False, unit="pixel",
                           dist_m=10.0, dt_cross=0.85, coeff=0.12, lane="1",
                           ts=1720000000.0)
                rows = load_samples(d)
                self.assertEqual(len(rows), 1)
                r = rows[0]
                self.assertEqual(r["camera_id"], 7)
                self.assertEqual(r["gt_kmh"], 42.3)
                self.assertEqual(r["est_kmh"], 38.1)
                self.assertFalse(r["calibrated"])
                # 確認寫出的是合法 JSONL
                files = list(Path(d).glob("*.jsonl"))
                self.assertEqual(len(files), 1)
                json.loads(files[0].read_text(encoding="utf-8").strip())
            finally:
                sc.CALIB_DIR = orig

    def test_log_sample_handles_none_estimate(self):
        with tempfile.TemporaryDirectory() as d:
            import detection.speed_calib as sc
            orig = sc.CALIB_DIR
            sc.CALIB_DIR = Path(d)
            try:
                log_sample(1, 3, 50.0, None, calibrated=True, unit="world_m",
                           dist_m=10.0, dt_cross=0.7, coeff=0.12, ts=1720000000.0)
                rows = load_samples(d)
                self.assertEqual(len(rows), 1)
                self.assertIsNone(rows[0]["est_kmh"])
                # est=None 會被 analyze 濾掉
                self.assertEqual(analyze(rows)["n_total"], 0)
            finally:
                sc.CALIB_DIR = orig


if __name__ == "__main__":
    unittest.main()
