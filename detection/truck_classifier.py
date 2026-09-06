#!/usr/bin/env python3
"""
大貨車/小貨車/大客車 分類器
===========================
對 YOLO 偵測出的 truck / bus 做二階段細分類。

類別對應:
  0: bus         (大客車)
  1: heavy_truck (大貨車)
  2: light_truck (小貨車)
  3: non_truck   (非目標)
"""

import os
import time
import warnings
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

from ultralytics import YOLO
from model_paths import get_model_dir
from detection.gpu_lock import GPU_INFERENCE_LOCK


# 分類結果 → 顯示用中文標籤 & 車輛等效長度（停等評估用）
# group: large=大車（大貨車+大客車）, small=小車（其他全部）
CLASS_META = {
    "heavy_truck": {"label": "大貨車", "length_m": 12.0, "group": "large"},
    "light_truck": {"label": "小貨車", "length_m": 6.0,  "group": "small"},
    "bus":         {"label": "大客車", "length_m": 12.0, "group": "large"},
    "non_truck":   {"label": "小客車", "length_m": 5.0,  "group": "small"},
}


def get_truck_cls_model_path() -> str:
    model_dir = get_model_dir()
    value = os.getenv("TRUCK_CLS_MODEL", "truck_cls_yolo26s.pt")
    if os.path.isabs(value):
        return value
    return os.path.join(model_dir, value)


# 細分類的最小框短邊(px)。低於此值直接判定「判不出來」,不送 GPU。
# 0 = 不啟用(全部都送,舊行為)。
MIN_CROP_PX = int(os.getenv("TRUCK_CLS_MIN_CROP_PX", "48") or 0)


class TruckClassifier:
    """
    大型車輛細分類器

    用法:
        classifier = TruckClassifier()
        result = classifier.classify(frame, bbox)
        # result = {"class_name": "heavy_truck", "label": "大貨車",
        #           "confidence": 0.92, "group": "large", "length_m": 12.0}
    """

    # 🛑 類別層級預設:__init__ 在模型檔不存在時會提早 return,那條路徑不會設這兩個
    #    屬性;測試也可能用 __new__ 直接建物件。放在類別上,任何建構路徑都拿得到,
    #    且預設 = 不啟用仲裁 = 與加這功能之前完全相同的行為。
    primary = None
    primary_names = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        imgsz: int = 224,
    ):
        model_path = model_path or get_truck_cls_model_path()
        if not os.path.exists(model_path):
            print(f"⚠️  分類模型不存在: {model_path}，TruckClassifier 停用")
            self.model = None
            return

        # 優先用 TensorRT engine（~3x speedup）
        engine_path = os.path.splitext(model_path)[0] + ".engine"
        if os.path.exists(engine_path) and os.getenv("DISABLE_TRT", "").lower() not in ("1", "true", "yes"):
            print(f"⚡ TruckClassifier 切換到 TensorRT engine: {engine_path}", flush=True)
            model_path = engine_path

        self.model = YOLO(model_path, task='classify')
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        self.device = os.getenv("DEVICE", "cuda:0")
        if not model_path.endswith(".engine"):
            try:
                self.model.to(self.device)
            except Exception:
                self.device = "cpu"

        # 建立 class index → name 的映射
        self.class_names = self.model.names  # {0: 'bus', 1: 'heavy_truck', ...}
        print(f"✅ 大型車分類器初始化完成 (模型: {model_path}, 類別: {self.class_names})")

        # ── 大小貨車仲裁(規則 A) ──────────────────────────────────────
        # TRUCK_CLS_PRIMARY_MODEL 設了才啟用。主模型先判,只有它判「小貨」時
        # 才叫上面這顆(現行模型)當仲裁,若仲裁說大貨就採大貨。
        # 依據:2026-09-06 用 695 條盲標的歧異抽樣檢定,主模型 v4 單獨上線是
        # -2.46pp(已回滾),但它幾乎每一格都贏,只輸「線上大貨→v4小貨」這一格
        # (246 條中線上對 226、v4 只對 19)。擋掉那一格 = +1.62pp,
        # 95%CI[+273,+557],留半驗證前半+393/後半+437(後半更高,非過擬合)。
        # 🛑 只在主模型判小貨時才跑第二顆(實測佔 15.8% 的車),成本才壓得住:
        #    truck_cls 佔 GPU 通道 5.4% → 額外約 0.9%,而非兩顆全跑的 5.4%。
        # 🛑 載不起來就退回「只用現行模型」= 與啟用前完全相同的行為,
        #    絕不會變成「只跑主模型」(那是實測更差的狀態)。
        self.primary = None
        self.primary_names = None
        primary_path = os.getenv("TRUCK_CLS_PRIMARY_MODEL", "").strip()
        if primary_path:
            try:
                if not os.path.isabs(primary_path):
                    primary_path = os.path.join(get_model_dir(), primary_path)
                p_engine = os.path.splitext(primary_path)[0] + ".engine"
                if os.path.exists(p_engine) and os.getenv("DISABLE_TRT", "").lower() not in ("1", "true", "yes"):
                    primary_path = p_engine
                if not os.path.exists(primary_path):
                    raise FileNotFoundError(primary_path)
                self.primary = YOLO(primary_path, task='classify')
                if not primary_path.endswith(".engine"):
                    self.primary.to(self.device)
                # 🛑 兩顆模型的 index→name 未必相同,各用各的,不可共用 class_names
                self.primary_names = self.primary.names
                print(f"⚖️  大小貨車仲裁啟用 — 主模型 {primary_path} 類別 {self.primary_names}", flush=True)
            except Exception as exc:
                self.primary = None
                self.primary_names = None
                print(f"⚠️  主模型載入失敗({exc}),退回單一模型(行為同啟用前)", flush=True)

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def classify(
        self,
        frame: np.ndarray,
        bbox: dict,
        pad_ratio: float = 0.1,
    ) -> dict:
        """
        對 bounding box 區域做分類

        Args:
            frame: 完整影像 (BGR)
            bbox: {"x1": int, "y1": int, "x2": int, "y2": int}
            pad_ratio: bbox 外擴比例，避免裁太緊

        Returns:
            {"class_name": str, "label": str, "confidence": float,
             "group": str, "length_m": float}
        """
        if not self.enabled:
            return self._default_result()

        # 裁切 + padding
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        bw, bh = x2 - x1, y2 - y1
        # 🛑 太小的框不送 GPU。imgsz 是 224,一個 30px 的遠處框放大 7 倍只有插值出來的
        #    糊圖,實測 top1conf 幾乎都低於 conf_threshold → 回 _default_result(unknown)
        #    → 依 vehicle_detector 的規則 unknown 不進快取 → 下一幀再算一次,永遠白花。
        #    2026-08-31 87 實測:GPU 鎖飽和度 1.0、細分類佔 52% GPU 時間、
        #    每幀 2.27 次呼叫但快取命中率只有 15.4%,分析率掉到 0.3 fps
        #    (疊加框比畫面舊 6.4 秒)。擋掉這些必定判不出來的呼叫直接換回分析率。
        #    門檻取短邊 —— 遠處車是等比縮小,短邊比面積更能代表「還剩多少細節」。
        if min(bw, bh) < MIN_CROP_PX:
            return self._default_result()
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._default_result()

        if self.primary is not None:
            # 主模型先判
            class_name, top1_conf = self._infer(self.primary, self.primary_names, crop, "truck_cls")
            # 🛑 只有主模型判「小貨」時才叫仲裁 —— 這是把成本從 5.4% 壓到 0.9% 的關鍵。
            #    主模型判其他三類時實測都比現行模型好,不需要也不該去問第二顆。
            if class_name == "light_truck":
                arb_name, arb_conf = self._infer(
                    self.model, self.class_names, crop, "truck_cls_arbiter")
                if arb_name == "heavy_truck":
                    class_name, top1_conf = arb_name, arb_conf
        else:
            class_name, top1_conf = self._infer(self.model, self.class_names, crop, "truck_cls")

        if class_name is None:
            return self._default_result()

        # 信心度不足 → 回傳預設
        if top1_conf < self.conf_threshold:
            return self._default_result()

        meta = CLASS_META.get(class_name, CLASS_META["non_truck"])
        return {
            "class_name": class_name,
            "label": meta["label"],
            "confidence": round(top1_conf, 3),
            "group": meta["group"],
            "length_m": meta["length_m"],
        }

    def _infer(self, model, names, crop, tag: str):
        """跑一顆模型,回 (class_name, conf);失敗回 (None, 0.0)。

        過 process-wide GPU lock 避免跟其他 detector 並發踩到 CUDA stream race。
        這裡通常巢狀在 VehicleDetector.detect 的 lock 內,時間會被算進 detection,
        所以自報一次讓 gpu-lock-stats 的 nested 能單獨看到各模型的通道佔用。
        """
        with GPU_INFERENCE_LOCK:
            _t0 = time.perf_counter()
            results = model.predict(
                source=crop,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
            GPU_INFERENCE_LOCK.record_nested(tag, time.perf_counter() - _t0)

        if not results or results[0].probs is None:
            return None, 0.0
        probs = results[0].probs
        return names[probs.top1], probs.top1conf.item()

    def classify_batch(
        self,
        frame: np.ndarray,
        bboxes: list[dict],
        pad_ratio: float = 0.1,
    ) -> list[dict]:
        """批次分類多個 bbox"""
        return [self.classify(frame, bb, pad_ratio) for bb in bboxes]

    @staticmethod
    def _default_result() -> dict:
        return {
            "class_name": "unknown",
            "label": "未知",
            "confidence": 0.0,
            "group": "other",
            "length_m": 6.0,
        }
