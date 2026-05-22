"""Process-wide GPU inference lock.

Jetson Orin 上 ultralytics + TensorRT 多 thread 並發 inference 不同 detector
instance 仍會共用 GPU，CUDA stream race 觸發 `torch.cuda.synchronize` SEGV
(~每 1-2 分鐘一次)。所有 detector inference call 都序列化過這個 lock，
trade throughput for stability。

涵蓋：
  - VehicleDetector.detect (vehicle_detector.py)
  - TruckClassifier.classify (truck_classifier.py)
  - PlateDetector.detect (recognition/plate_detector.py)

用 `RLock` 而非 `Lock`：VehicleDetector.detect 在 lock 內會 call truck_classifier
做大型車細分類，同一 thread 嵌套 acquire 必須允許。
"""
import threading

GPU_INFERENCE_LOCK = threading.RLock()
