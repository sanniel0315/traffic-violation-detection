#!/usr/bin/env python3
"""類別名稱映射:不可把非車輛類別當成車。

回歸案例:COCO 第 51 類是 'carrot'，舊版第二輪用裸子字串比對，
"car" in "carrot" 成立 → 紅蘿蔔被歸類成小客車。路面的白/黃色標線
正好長得像紅蘿蔔，於是使用者看到「地上標線被辨識成小客車」。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from detection.vehicle_detector import VehicleDetector  # noqa: E402

match = VehicleDetector._match_canonical_label

# Ultralytics 內建的 COCO 80 類完整名單
COCO80 = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]

# COCO 之中「應該」被視為交通類的，就這 6 個
EXPECTED_TRAFFIC = {
    'person': 'person', 'bicycle': 'bicycle', 'car': 'car',
    'motorcycle': 'motorcycle', 'bus': 'bus', 'truck': 'truck',
}

fails = []


def check(desc, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{desc}  got={got!r} want={want!r}")
    if not ok:
        fails.append(desc)


print("[1] 回歸:carrot 絕不可變成 car")
check("carrot", match('carrot'), None)
check("Carrot 大小寫", match('Carrot'), None)

print("\n[2] 掃過完整 COCO 80 類,只有 6 個交通類別可被認出")
wrong = []
for name in COCO80:
    got = match(name)
    want = EXPECTED_TRAFFIC.get(name)
    if got != want:
        wrong.append((name, got, want))
check("COCO80 全類別映射", wrong, [])

print("\n[3] 原本就要支援的別名不能被改壞")
for raw, want in [
    ('pickup_truck', 'truck'), ('pickup truck', 'truck'), ('pickup', 'truck'),
    ('lorry', 'truck'), ('trailer', 'truck'),
    ('sedan', 'car'), ('SUV', 'car'), ('van', 'car'), ('taxi', 'car'),
    ('automobile', 'car'), ('jeep', 'car'), ('vehicle', 'car'),
    ('motorbike', 'motorcycle'), ('scooter', 'motorcycle'), ('moped', 'motorcycle'),
    ('minibus', 'bus'), ('coach', 'bus'),
    ('pedestrian', 'person'), ('people', 'person'),
    ('cycle', 'bicycle'),
]:
    check(f"{raw!r}", match(raw), want)

print("\n[4] motorcycle 不可因為含 'cycle' 而被判成 bicycle")
check("motorcycle", match('motorcycle'), 'motorcycle')
check("motor_cycle", match('motor_cycle'), 'motorcycle')

print("\n[5] 其他容易誤傷的詞")
for raw in ['carton', 'cargo', 'caravan', 'scarf', 'card', 'carpet',
            'busy', 'trucker', 'personal']:
    check(f"{raw!r} 不該是車", match(raw), None)

print("\n[6] 空值/雜訊")
for raw in ['', '   ', '???']:
    check(f"{raw!r}", match(raw), None)

print("\n" + (f"FAILED ({len(fails)}): {fails}" if fails else "ALL PASS"))
sys.exit(1 if fails else 0)
