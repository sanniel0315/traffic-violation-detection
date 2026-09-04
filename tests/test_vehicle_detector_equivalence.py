#!/usr/bin/env python3
"""證明「縮小 GPU 鎖範圍」沒有改變偵測結果。

2026-08-18 改了 VehicleDetector 兩件事:
  ① tensor→CPU 搬運從「每框三次」改成「整批三次」
  ② 純 CPU 的機車誤判過濾移出 GPU_INFERENCE_LOCK
兩件都不該影響輸出 —— 但「不該」不算數,這裡用模擬的 ultralytics Boxes
(逐框索引與整批索引都支援,跟真物件同語意)把新路徑與「改動前的參考實作」
逐項比對:類別、信心、bbox 四角與長寬。

刻意不載入真模型:在現場機上再開一個 VehicleDetector 會重複佔用 TensorRT
engine 與顯存(104 Orin NX 8GB 實測直接崩)。要驗的是解析邏輯,不是模型本身。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np  # noqa: E402

from detection.vehicle_detector import VehicleDetector  # noqa: E402


class _Tensor:
    """夠像 torch tensor 的替身:支援索引、.cpu()、.numpy()、len()。"""

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def __getitem__(self, i):
        v = self._a[i]
        return _Tensor(v) if isinstance(v, np.ndarray) else float(v)

    def __len__(self):
        return len(self._a)

    def __float__(self):
        return float(self._a)

    def __int__(self):
        return int(self._a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = _Tensor(cls)
        self.conf = _Tensor(conf)
        self.xyxy = _Tensor(xyxy)

    def __len__(self):
        return len(self.cls)

    def __iter__(self):
        for i in range(len(self)):
            yield _Boxes(self.cls.numpy()[i:i + 1],
                         self.conf.numpy()[i:i + 1],
                         self.xyxy.numpy()[i:i + 1])


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


def reference_parse(det, frame, results):
    """改動前的寫法:逐框 .cpu()、機車過濾在同一個函式內。對照組。"""
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id not in det.vehicle_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            detections.append({
                'class_id': class_id,
                'class_name': det.vehicle_classes[class_id],
                'confidence': confidence,
                'bbox': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                         'width': int(x2 - x1), 'height': int(y2 - y1)},
            })
    ih, iw = frame.shape[:2]
    edge = int(os.getenv("MOTO_EDGE_MARGIN_PX", "6"))
    mc = float(os.getenv("MOTO_MIN_CONF", "0.30"))
    # 🛑 對照組要跟著「刻意的」後續改動走,否則測到的是行為變更而不是重構。
    #    ceca34d 之後多了低信心行人過濾(國道點位幾乎沒有真行人,靜態路標
    #    常被判成 person)。這個測試要驗的是 2026-08-18 縮小 GPU 鎖範圍的
    #    重構有沒有改變輸出,不是去反對後來那個刻意加的過濾 —— 所以補進來。
    pmc = float(os.getenv("PERSON_MIN_CONF", "0.6"))
    out = []
    for d in detections:
        if d['class_name'] == 'motorcycle':
            b = d['bbox']
            if (b['x1'] <= edge or b['y1'] <= edge
                    or b['x2'] >= iw - edge or b['y2'] >= ih - edge):
                continue
            if d['confidence'] < mc:
                continue
        elif d['class_name'] == 'person':
            if pmc > 0 and d['confidence'] < pmc:
                continue
        out.append(d)
    return out


def key(d):
    b = d['bbox']
    return (d['class_id'], d['class_name'], round(d['confidence'], 6),
            b['x1'], b['y1'], b['x2'], b['y2'], b['width'], b['height'])


def main() -> int:
    # 不跑 __init__,避免載入模型
    det = VehicleDetector.__new__(VehicleDetector)
    det.conf_threshold = 0.5
    det.runtime_device = 'cpu'
    det.vehicle_classes = dict(VehicleDetector.DEFAULT_VEHICLE_CLASSES)
    det.truck_classifier = None

    rng = np.random.default_rng(20260818)
    W, H = 1920, 1080
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    cases = 0
    bad = []
    for trial in range(200):
        n = int(rng.integers(0, 25))
        # 類別刻意含非車輛(如 9)驗過濾、含 motorcycle 驗邊緣與信心兩道防護
        cls = rng.choice([0, 1, 2, 3, 5, 7, 9, 11], size=n)
        conf = rng.uniform(0.05, 0.99, size=n)
        x1 = rng.uniform(0, W - 10, size=n)
        y1 = rng.uniform(0, H - 10, size=n)
        # 有些框刻意貼邊(x1=0 或 x2=W),觸發出畫殘影那條規則
        edge_hit = rng.random(n) < 0.3
        x1 = np.where(edge_hit, 0.0, x1)
        x2 = np.minimum(W, x1 + rng.uniform(5, 300, size=n))
        x2 = np.where(rng.random(n) < 0.2, float(W), x2)
        y2 = np.minimum(H, y1 + rng.uniform(5, 300, size=n))
        boxes = _Boxes(cls, conf, np.stack([x1, y1, x2, y2], axis=1))
        results = [_Result(boxes)]

        det.model = lambda *a, **k: results          # noqa: B023
        new = det.detect(frame)
        ref = reference_parse(det, frame, results)
        cases += len(ref)
        if sorted(map(key, new)) != sorted(map(key, ref)):
            bad.append((trial, len(new), len(ref)))

    print(f"  200 組隨機偵測結果(共 {cases} 個保留框)")
    if bad:
        print(f"  ❌ {len(bad)} 組不同,例:{bad[:3]}")
        return 1
    print("  ✅ 新舊路徑輸出完全相同(類別/信心/bbox 四角與長寬逐項比對)")

    # 邊界:空結果、boxes=None 都不可以炸
    det.model = lambda *a, **k: [_Result(_Boxes([], [], np.zeros((0, 4))))]
    assert det.detect(frame) == []
    det.model = lambda *a, **k: [_Result(None)]
    assert det.detect(frame) == []
    print("  ✅ 空偵測 / boxes=None 邊界正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
