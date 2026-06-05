#!/usr/bin/env python3
"""sensor_fusion + radar_track 單元測試

跑法 (本機):
    python -m detection.test_sensor_fusion
"""
import unittest
from detection.radar_track import MockRadarDriver, RadarDetection
from detection.sensor_fusion import fuse, summarize, FusedTrack


class TestMockRadarDriver(unittest.TestCase):
    def test_feed_and_get(self):
        driver = MockRadarDriver(camera_id=7, seed=42)
        visual_tracks = [
            {"track_id": 1, "world_x": 25.0, "world_y": -1.2, "vx": 16.7, "vy": 0.0, "class_name": "car"},
            {"track_id": 2, "world_x": 40.0, "world_y": 2.0, "vx": 18.0, "vy": 0.1, "class_name": "truck"},
        ]
        driver.feed_visual_tracks(visual_tracks)
        dets = driver.get_detections()
        self.assertGreaterEqual(len(dets), 1, "should produce at least 1 detection (drop 5%)")
        for d in dets:
            self.assertIsInstance(d, RadarDetection)
            # 加雜訊後位置不會完全等於 visual
            self.assertNotEqual(d.x, 25.0)
            # RCS 應該是有意義的範圍
            self.assertGreater(d.rcs, 0)
            # speed 計算 ok
            self.assertGreater(d.speed_kmh, 30.0)


class TestFusionMatching(unittest.TestCase):
    def test_perfect_match_visual_radar(self):
        """雷達跟視覺位置完全一致 → fused"""
        radar = [RadarDetection(track_id=101, x=25.0, y=-1.2, vx=16.7, vy=0.0,
                                rcs=12.0, timestamp=1000.0)]
        visual = [{"track_id": 1, "world_x": 25.0, "world_y": -1.2, "vx": 16.7,
                  "vy": 0.0, "class_name": "car", "confidence": 0.92,
                  "bbox": {"x1": 100, "y1": 200, "x2": 150, "y2": 240}}]
        out = fuse(radar, visual)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "fused")
        self.assertEqual(out[0].vehicle_class, "car")
        self.assertEqual(out[0].radar_track_id, 101)
        self.assertEqual(out[0].visual_track_id, 1)
        # speed 用雷達
        self.assertAlmostEqual(out[0].vx, 16.7, places=2)
        # position 平均 (兩邊一致所以就是該值)
        self.assertAlmostEqual(out[0].world_x, 25.0, places=2)
        self.assertEqual(out[0].bbox, {"x1": 100, "y1": 200, "x2": 150, "y2": 240})

    def test_radar_only(self):
        """雷達看到但視覺沒 → radar_only"""
        radar = [RadarDetection(track_id=99, x=50.0, y=0.0, vx=5.0, vy=0.0,
                                rcs=8.0, timestamp=1000.0)]
        visual = []
        out = fuse(radar, visual)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "radar_only")
        self.assertEqual(out[0].vehicle_class, "unknown")

    def test_visual_only(self):
        """視覺看到但雷達沒 → visual_only"""
        radar = []
        visual = [{"track_id": 5, "world_x": 30.0, "world_y": 1.5, "vx": 14.0,
                  "vy": 0.0, "class_name": "motorcycle", "confidence": 0.85,
                  "bbox": {"x1": 200, "y1": 300, "x2": 230, "y2": 350}}]
        out = fuse(radar, visual)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "visual_only")
        self.assertEqual(out[0].vehicle_class, "motorcycle")

    def test_distance_threshold(self):
        """雷達跟視覺差超過 threshold → 各自分開"""
        radar = [RadarDetection(track_id=10, x=10.0, y=0.0, vx=10.0, vy=0.0,
                                rcs=10.0, timestamp=1000.0)]
        # 視覺位置差 10m → > threshold 3m → 不 match
        visual = [{"track_id": 20, "world_x": 30.0, "world_y": 0.0, "vx": 10.0,
                  "vy": 0.0, "class_name": "car", "confidence": 0.9}]
        out = fuse(radar, visual)
        self.assertEqual(len(out), 2)
        sources = {t.source for t in out}
        self.assertSetEqual(sources, {"radar_only", "visual_only"})

    def test_hungarian_multiple(self):
        """多 radar + 多 visual,要正確配對 (最近的配最近的)"""
        radar = [
            RadarDetection(track_id=1, x=10.0, y=0.0, vx=10.0, vy=0.0, rcs=10.0, timestamp=1000.0),
            RadarDetection(track_id=2, x=30.0, y=0.0, vx=15.0, vy=0.0, rcs=12.0, timestamp=1000.0),
            RadarDetection(track_id=3, x=50.0, y=0.0, vx=8.0, vy=0.0, rcs=20.0, timestamp=1000.0),
        ]
        visual = [
            {"track_id": 11, "world_x": 50.5, "world_y": 0.0, "vx": 8.0, "vy": 0.0, "class_name": "truck", "confidence": 0.9},
            {"track_id": 12, "world_x": 10.2, "world_y": 0.2, "vx": 10.0, "vy": 0.0, "class_name": "car", "confidence": 0.95},
            {"track_id": 13, "world_x": 30.3, "world_y": -0.1, "vx": 15.0, "vy": 0.0, "class_name": "car", "confidence": 0.88},
        ]
        out = fuse(radar, visual)
        # 應該 3 個 fused (順序不一定但配對正確)
        fused_only = [t for t in out if t.source == "fused"]
        self.assertEqual(len(fused_only), 3)
        # 確認配對:radar 1 ↔ visual 12, radar 2 ↔ visual 13, radar 3 ↔ visual 11
        pairs = {(t.radar_track_id, t.visual_track_id) for t in fused_only}
        self.assertIn((1, 12), pairs)
        self.assertIn((2, 13), pairs)
        self.assertIn((3, 11), pairs)

    def test_summarize(self):
        out = [
            FusedTrack(track_id=1, world_x=10, world_y=0, vx=10, vy=0,
                       vehicle_class="car", source="fused"),
            FusedTrack(track_id=2, world_x=30, world_y=0, vx=15, vy=0,
                       vehicle_class="truck", source="fused"),
            FusedTrack(track_id=3, world_x=50, world_y=0, vx=8, vy=0,
                       vehicle_class="unknown", source="radar_only"),
        ]
        s = summarize(out)
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["by_class"]["car"], 1)
        self.assertEqual(s["by_class"]["truck"], 1)
        self.assertEqual(s["by_source"]["fused"], 2)
        self.assertEqual(s["radar_visual_fused_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
