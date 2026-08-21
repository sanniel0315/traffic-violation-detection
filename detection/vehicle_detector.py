#!/usr/bin/env python3
"""車輛偵測模組 - 使用 YOLOv8"""
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import numpy as np
import os
import threading as _th_stats
import time as _time_stats
from collections import deque as _deque_stats

from model_paths import get_detect_model_pt
from detection.gpu_lock import GPU_INFERENCE_LOCK


# ── 鎖內細部計時 ────────────────────────────────────────────────────────
# GPU_INFERENCE_LOCK 的 hold 時間是全 process 分析率的分母,但「hold 37ms」
# 不足以決定下一步:如果是模型 forward 就得換模型/降解析度,如果是 ultralytics
# 的 CPU letterbox 或結果搬運,那是可以搬到鎖外的浪費。所以把鎖內再拆三段。
_STAT_LOCK = _th_stats.Lock()
_STAT_BUF = _deque_stats()          # (ts, model_sec, parse_sec, truck_sec, n_box)
_STAT_WINDOW = 60.0


# model 時間與 parse/truck 是分開量的(前者在 detect,後者在 _parse_result),
# 要把同一次偵測的兩段接起來。
# 🛑 必須是 thread-local:四台相機各有自己的 worker,共用一個 list 會張冠李戴
#    —— 實測過,會出現 model_ms(49.6) 比 hold(40.2) 還大的不可能數字。
_MODEL_LAST = _th_stats.local()


def _record_model_time(model_s: float) -> None:
    _MODEL_LAST.value = model_s


def _record_detect_stats(parse_s: float, truck_s: float, n_box: int,
                         tc_call: int = 0, tc_hit: int = 0) -> None:
    now = _time_stats.time()
    with _STAT_LOCK:
        _STAT_BUF.append((now, getattr(_MODEL_LAST, "value", 0.0),
                          parse_s, truck_s, n_box, tc_call, tc_hit))
        cut = now - _STAT_WINDOW
        while _STAT_BUF and _STAT_BUF[0][0] < cut:
            _STAT_BUF.popleft()


def detect_timing_stats() -> dict:
    """鎖內時間的組成。給 /api/stream/gpu-lock-stats 用。"""
    with _STAT_LOCK:
        items = list(_STAT_BUF)
    if len(items) < 2:
        return {"samples": len(items)}
    span = items[-1][0] - items[0][0]
    n = len(items)
    if span <= 0:
        return {"samples": n}
    tot = sum(i[1] + i[2] + i[3] for i in items)
    return {
        "samples": n,
        "window_sec": round(span, 1),
        # model = ultralytics 那一呼叫(含 CPU letterbox + GPU forward + NMS)
        "model_ms_avg": round(sum(i[1] for i in items) / n * 1000, 1),
        # parse = tensor→CPU 搬運 + 組 dict
        "parse_ms_avg": round(sum(i[2] for i in items) / n * 1000, 1),
        # truck = 大型車細分類(只有出現 truck/bus 才會跑)
        "truck_ms_avg": round(sum(i[3] for i in items) / n * 1000, 1),
        "boxes_avg": round(sum(i[4] for i in items) / n, 1),
        # 細分類被要求幾次、其中幾次靠快取免掉了推論
        "truck_calls_per_frame": round(sum(i[5] for i in items) / n, 2),
        "truck_cache_hit_rate": round(
            sum(i[6] for i in items) / max(1, sum(i[5] for i in items)), 3),
        "model_share": round(sum(i[1] for i in items) / max(1e-9, tot), 3),
        "parse_share": round(sum(i[2] for i in items) / max(1e-9, tot), 3),
        "truck_share": round(sum(i[3] for i in items) / max(1e-9, tot), 3),
    }


# ── 大型車細分類的重複呼叫快取 ─────────────────────────────────────────
# 🛑 為什麼需要:一台車的車種不會變,但先前每一幀都重跑一次細分類。
#    一台車在畫面裡待 5 秒、分析率 2.5 fps → 同一台重複分類 12 次,
#    後 11 次的結果必然跟第 1 次相同,純粹是浪費 GPU 通道。
#    2026-08-20 在 87 量到:細分類佔 detection 鎖持有時間的 24%(11.7/49.8ms),
#    而 GPU 鎖飽和度 0.99 —— 省下來的直接變成分析率。
# 🛑 這裡沒有 track_id 可用(走的是 model(frame) 不是 model.track),
#    所以用 bbox 重疊比對來認「同一台車」。命中時把 bbox 更新成新位置
#    (跟著車走),但**不更新時間戳** —— 每台車至少每 TTL 秒重驗一次,
#    萬一第一次分錯也不會永久留著。
# 順帶的好處:同一台車的標籤在連續幀之間會穩定下來,不會忽大貨車忽小客車。
TRUCK_CLS_CACHE = os.getenv("TRUCK_CLS_CACHE", "1") != "0"
TRUCK_CLS_CACHE_SEC = float(os.getenv("TRUCK_CLS_CACHE_SEC", "2.0") or 2.0)
TRUCK_CLS_CACHE_IOU = float(os.getenv("TRUCK_CLS_CACHE_IOU", "0.6") or 0.6)
# 影子稽核:命中快取時「照樣真的跑一次分類」,比對沿用的標籤跟現算的一不一致。
# 這是唯一能證明快取在真實車流上安全的方法 —— 單元測試只能證明它照邏輯運作,
# 證明不了 IoU 0.6 / 2 秒這組參數不會把「另一台車」誤認成同一台。
# 🛑 開著會讓細分類的工作量加倍(命中也要算),只在驗證時開,驗完要關。
TRUCK_CLS_AUDIT = os.getenv("TRUCK_CLS_AUDIT", "0") != "0"
# by_conf: 以「存快取時的信心」分桶(0.0~0.1 為 0,依此類推)→ [一致, 不一致]
_TC_AUDIT = {"agree": 0, "differ": 0, "samples": [], "by_conf": {}, "by_iou": {}}
_TC_AUDIT_LOCK = _th_stats.Lock()


def truck_cls_audit_stats() -> dict:
    """影子稽核結果:快取沿用的標籤有多少比例跟現算的一致。"""
    with _TC_AUDIT_LOCK:
        a, d = _TC_AUDIT["agree"], _TC_AUDIT["differ"]
        samples = list(_TC_AUDIT["samples"])
        by_conf = {k: list(v) for k, v in _TC_AUDIT["by_conf"].items()}
        by_iou = {k: list(v) for k, v in _TC_AUDIT["by_iou"].items()}
    n = a + d
    return {
        "enabled": TRUCK_CLS_AUDIT,
        "checked": n,
        "agree": a,
        "differ": d,
        "agree_rate": round(a / n, 4) if n else None,
        # 不一致的實例,給人判斷是「快取抓錯車」還是「分類器本身在跳」
        "differ_samples": samples[-20:],
        # 依「存快取時的信心」分桶的一致率 —— 用來判斷加信心門檻有沒有用
        "agree_by_cached_conf": {
            f"{k/10:.1f}-{(k+1)/10:.1f}": {
                "checked": sum(v), "agree_rate": round(v[0] / sum(v), 3) if sum(v) else None}
            for k, v in sorted(by_conf.items())
        },
        # 依「配對時的 IoU」分桶 —— IoU 低代表可能配到別台車,用來訂門檻
        "agree_by_iou": {
            f"{k/20:.2f}-{(k+1)/20:.2f}": {
                "checked": sum(v), "agree_rate": round(v[0] / sum(v), 3) if sum(v) else None}
            for k, v in sorted(by_iou.items())
        },
    }


def _bbox_iou(a: dict, b: dict) -> float:
    """兩個 bbox 的交集/聯集。給「這是不是同一台車」用。"""
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = ((a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
             + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter)
    return inter / union if union > 0 else 0.0


class VehicleDetector:
    """車輛偵測器"""

    # COCO 預設類別對照；若模型有自帶 names，會動態覆蓋。
    DEFAULT_VEHICLE_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    # 需要做二階段細分類的類別
    _RECLASSIFY_CLASSES = {'truck', 'bus'}

    # 車種中文標籤對照（統一所有車種顯示）
    CLASS_LABEL_ZH = {
        'car':         '小客車',
        'motorcycle':  '機車',
        'bicycle':     '自行車',
        'person':      '行人',
        'truck':       '貨車',
        'bus':         '大客車',
        'heavy_truck': '大貨車',
        'light_truck': '小貨車',
        'non_truck':   '小客車',
    }

    @classmethod
    def get_zh_label(cls, class_name: str) -> str:
        return cls.CLASS_LABEL_ZH.get(str(class_name or ''), str(class_name or ''))

    # 共用 truck classifier（一個 instance 給所有 cam），classify() 有內部 lock
    _shared_truck_classifier = None
    _shared_truck_classifier_lock = None

    @classmethod
    def _get_shared_truck_classifier(cls):
        import threading as _th
        if cls._shared_truck_classifier_lock is None:
            cls._shared_truck_classifier_lock = _th.Lock()
        if cls._shared_truck_classifier is not None:
            return cls._shared_truck_classifier
        with cls._shared_truck_classifier_lock:
            if cls._shared_truck_classifier is not None:
                return cls._shared_truck_classifier
            from detection.truck_classifier import TruckClassifier
            tc = TruckClassifier()
            if tc.enabled:
                cls._shared_truck_classifier = tc
                print("♻️  共用 TruckClassifier 載入完成", flush=True)
            return cls._shared_truck_classifier

    CLASS_NAME_ALIASES = {
        'person': {'person', 'pedestrian', 'people'},
        'bicycle': {'bicycle', 'cycle'},
        'car': {'car', 'vehicle', 'sedan', 'suv', 'van', 'auto', 'automobile', 'taxi', 'jeep'},
        'motorcycle': {'motorcycle', 'motorbike', 'scooter', 'moped'},
        'bus': {'bus', 'coach', 'minibus'},
        'truck': {'truck', 'lorry', 'pickup', 'pickup truck', 'pickup_truck', 'trailer'},
    }
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5,
                 enable_truck_cls: bool = True,
                 cls_cache: bool = False):
        """
        初始化偵測器

        Args:
            model_path: YOLOv8 模型路徑
            conf_threshold: 信心度閾值
            enable_truck_cls: 是否啟用大型車細分類
        """
        model_path = model_path or get_detect_model_pt()
        # 優先使用同名 .engine（TensorRT 加速 ~3-4x），存在才切換
        engine_path = os.path.splitext(model_path)[0] + ".engine"
        if os.path.exists(engine_path) and os.getenv("DISABLE_TRT", "").lower() not in ("1", "true", "yes"):
            print(f"⚡ 偵測到 TensorRT engine，切換到 {engine_path}")
            model_path = engine_path
        self.model = YOLO(model_path, task='detect')
        self.conf_threshold = conf_threshold
        self.device = os.getenv("DEVICE", "cuda:0")
        self.runtime_device = "cpu"
        # TensorRT engine 已綁定 device，不需 .to()
        if not model_path.endswith(".engine"):
            try:
                self.model.to(self.device)
                self.runtime_device = self.device
            except Exception:
                self.runtime_device = "cpu"
        else:
            self.runtime_device = self.device
        self.vehicle_classes = self._resolve_vehicle_classes()

        # 大型車細分類器（可選）— 所有 VehicleDetector 共用單一 instance，避免 4 cam 各一份吃 GPU
        self.truck_classifier = None
        if enable_truck_cls:
            try:
                self.truck_classifier = VehicleDetector._get_shared_truck_classifier()
            except Exception as e:
                print(f"⚠️  大型車分類器載入失敗: {e}")

        # 細分類快取:每台相機各自一份。
        # 🛑 預設關閉,只有「一個 instance 固定服務一個來源」的偵測器才可以開
        #    (api/routes/stream.py 的 _get_per_cam_detector)。
        #    共用的 fallback singleton 會被多台相機輪流呼叫,兩台在同一像素位置
        #    的車會互相沿用標籤 —— 那是錯的,所以那條路徑維持原本每幀重算。
        self._tc_cache = []
        self._tc_cache_on = bool(cls_cache)

        print(f"✅ 車輛偵測器初始化完成 (模型: {model_path}, device: {self.runtime_device})")
        print(f"✅ 車種類別映射: {self.vehicle_classes}")
        if self.truck_classifier:
            print(f"✅ 大型車細分類: 啟用")

    def _truck_cls_cached(self, frame, bbox: dict):
        """細分類,同一台車在 TTL 內重複出現就沿用上次結果。

        回傳 (結果, 是否命中快取)。
        """
        if not (TRUCK_CLS_CACHE and self._tc_cache_on):
            return self._truck_cls_raw(frame, bbox), False
        now = _time_stats.perf_counter()
        # 掃一次:順手丟掉過期的,並找重疊最大的那一筆
        keep, best, best_iou = [], None, 0.0
        for ent in self._tc_cache:
            if now - ent["ts"] > TRUCK_CLS_CACHE_SEC:
                continue
            keep.append(ent)
            iou = _bbox_iou(ent["bbox"], bbox)
            if iou > best_iou:
                best, best_iou = ent, iou
        self._tc_cache = keep
        if best is not None and best_iou >= TRUCK_CLS_CACHE_IOU:
            if TRUCK_CLS_AUDIT:
                self._audit_hit(frame, bbox, best["res"], best_iou)
            # bbox 跟著車走,但 ts 不動 —— 保證每 TTL 秒重驗一次
            best["bbox"] = dict(bbox)
            return best["res"], True
        res = self._truck_cls_raw(frame, bbox)
        # 🛑 沒有結論的不要存。_default_result() 回的是 class_name="unknown"、
        #    confidence=0(信心不足或裁切失敗),那是「這次判不出來」,不是判斷結果。
        #    存了會讓這台車接下來 TTL 秒內都拿不到細分類 —— 下一幀角度稍微變一下
        #    本來就可能判得出來。2026-08-20 影子稽核實測:信心 0.0~0.1 這一桶
        #    一致率是 0.000(14 次全部不一致),就是這個 bug。
        if (res or {}).get("class_name") not in (None, "unknown"):
            self._tc_cache.append({"bbox": dict(bbox), "res": res, "ts": now})
        return res, False

    def _audit_hit(self, frame, bbox: dict, cached_res, iou: float):
        """影子稽核:這次命中如果改成現算,結果會不會不一樣。"""
        try:
            fresh = self._truck_cls_raw(frame, bbox)
        except Exception:
            return
        same = (fresh or {}).get("class_name") == (cached_res or {}).get("class_name")
        conf_bucket = int(min(0.999, float((cached_res or {}).get("confidence") or 0)) * 10)
        iou_bucket = int(min(0.999, float(iou)) * 20)      # 0.05 一格,門檻要訂得準
        with _TC_AUDIT_LOCK:
            _TC_AUDIT["by_conf"].setdefault(conf_bucket, [0, 0])[0 if same else 1] += 1
            _TC_AUDIT["by_iou"].setdefault(iou_bucket, [0, 0])[0 if same else 1] += 1
            if same:
                _TC_AUDIT["agree"] += 1
            else:
                _TC_AUDIT["differ"] += 1
                if len(_TC_AUDIT["samples"]) < 200:
                    _TC_AUDIT["samples"].append({
                        "cached": (cached_res or {}).get("class_name"),
                        "cached_conf": (cached_res or {}).get("confidence"),
                        "fresh": (fresh or {}).get("class_name"),
                        "fresh_conf": (fresh or {}).get("confidence"),
                        "iou": round(iou, 3),
                    })

    def _truck_cls_raw(self, frame, bbox: dict):
        """真的跑一次細分類(共用 instance,要加鎖避免多 cam 併發)。"""
        _tc_lock = VehicleDetector._shared_truck_classifier_lock
        if _tc_lock is not None:
            with _tc_lock:
                return self.truck_classifier.classify(frame, bbox)
        return self.truck_classifier.classify(frame, bbox)

    @classmethod
    def _normalize_label(cls, value: str) -> str:
        text = str(value or "").strip().lower()
        for ch in ("_", "-"):
            text = text.replace(ch, " ")
        return " ".join(text.split())

    @classmethod
    def _match_canonical_label(cls, raw_name: str) -> Optional[str]:
        name = cls._normalize_label(raw_name)
        compact = name.replace(" ", "")
        # 第一輪：精確比對（避免 bicycle 的 alias "cycle" 子字串命中 "motorcycle"）
        for canonical, aliases in cls.CLASS_NAME_ALIASES.items():
            for alias in aliases:
                alias_norm = cls._normalize_label(alias)
                alias_compact = alias_norm.replace(" ", "")
                if name == alias_norm or compact == alias_compact:
                    return canonical
        # 第二輪：以「詞」為單位比對（容忍 "pickup_truck" 之類的變體）。
        # 不能用裸子字串——COCO 第 51 類是 'carrot'，"car" in "carrot" 會成立，
        # 導致紅蘿蔔類被當成小客車：路面的白/黃色標線正好長得像，於是「地上標線
        # 被辨識成小客車」。_normalize_label 已把 _ - 換成空白，所以
        # "pickup_truck" → {"pickup","truck"} 仍能正確命中 truck。
        tokens = set(name.split())
        for canonical, aliases in cls.CLASS_NAME_ALIASES.items():
            for alias in aliases:
                alias_norm = cls._normalize_label(alias)
                if not alias_norm:
                    continue
                if " " in alias_norm:
                    # 多詞別名（如 "pickup truck"）仍用片語比對
                    if alias_norm in name:
                        return canonical
                elif alias_norm in tokens:
                    return canonical
        return None

    def _resolve_vehicle_classes(self) -> Dict[int, str]:
        names = getattr(self.model, "names", None)
        items = []
        if isinstance(names, dict):
            items = list(names.items())
        elif isinstance(names, (list, tuple)):
            items = list(enumerate(names))

        resolved: Dict[int, str] = {}
        for class_id, class_name in items:
            canonical = self._match_canonical_label(str(class_name))
            if canonical:
                resolved[int(class_id)] = canonical

        if resolved:
            return resolved
        return dict(self.DEFAULT_VEHICLE_CLASSES)
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        偵測影像中的車輛

        Args:
            frame: BGR 影像 (numpy array)

        Returns:
            偵測結果列表
        """
        # inference + tensor→cpu 轉換 + truck_classifier 序列化過 GPU lock,
        # 避免多 cam VehicleDetector instance 同時打 GPU 造成 CUDA stream race SEGV。
        #
        # 🛑 鎖裡只放「碰得到 GPU」的東西。這把鎖是全 process 唯一的推論通道,
        #    2026-08-18 實測 87:飽和度 1.0、每台等鎖 160~200ms,而 GPU 只跑到
        #    34~85% —— 鎖住不需要鎖的純 CPU 工作,等於直接把分析率砍掉。
        #    機車誤判過濾只讀 python dict,搬到鎖外,結果一模一樣。
        # ── 🛑 不要再嘗試「多相機合批推論」(2026-08-18 兩次實測 + 一次線上事故)──
        # 最硬的理由:現場用的 TensorRT engine 是「batch=1 固定」建出來的。
        #     AssertionError: input size torch.Size([2,3,640,640])
        #                     not equal to max model size (1,3,640,640)
        #   線上只要真的湊成 2 張就直接丟例外。2026-08-18 14:44 在 87 實際發生,
        #   55 分鐘內 1680 筆錯誤,LPR 與壅塞的分析同時被打斷、即時影像抖動。
        #   要合批得先把 engine 用 dynamic batch 重建 —— 那是模型的決定,不是這裡。
        #
        # 就算 engine 支援,收益也幾乎沒有。實測單張 1920x1080 的組成:
        #     preprocess(CPU letterbox) 6.10ms + inference(TensorRT GPU) 7.29ms
        #     + postprocess(NMS) 3.97ms = 17.36ms,wall 18.00ms
        #   → ultralytics Python 層的固定開銷只有 0.64ms(4%)。
        #   先前量到「批次 1.78x」是拿 .pt 檔量的(沒走 engine 替換),量錯對象。
        #   合批能攤掉的就是那 4%,不值得。
        #
        # 也試過中央批次執行緒:把 N 條偵測執行緒收斂成 1 條,跟 LPR 搶鎖時
        # 從 N 對 1 變 1 對 1,104 A/B 分析率反而掉約 35%。
        # 離線量測批次確實划算:4 張分開跑 121.0ms、一次批次 68.1ms(1.78x),
        # 因為 ultralytics 每次呼叫有固定開銷(predictor 設定/Results 物件/NMS)。
        # 但實際接上去之後在 104 A/B(負載相當,飽和度 0.708 vs 0.696):
        #     批次開啟  分析率 3.17 / 3.01   detection 佔通道 30%
        #     批次關閉  分析率 5.66 / 4.49   detection 佔通道 39%
        # 反而掉了約 35%。原因有兩個,都是架構性的:
        #   ① 批次器把所有 VehicleDetector 呼叫收斂成「一條」執行緒,它跟 LPR
        #      搶 GPU_INFERENCE_LOCK 時從 2 條對 1 條變成 1 條對 1 條,
        #      detection 分到的通道直接變少。
        #   ② 到達率低於處理量時佇列根本不會積,批次大小實測就是 1.0,
        #      沒有任何合批來補償①。
        # 要做對得改成 leader-follower(先拿到鎖的那條執行緒順手撈走其他待處理
        # 的畫面),才能同時保留 N 個競爭者又拿到合批 —— 那是另一個題目。
        with GPU_INFERENCE_LOCK:
            # 🛑 計時起點一定要在拿到鎖「之後」。放在 with 之前會把等鎖時間算進
            #    model,量出來的 model_ms 會比 hold_ms 還大(實測 54.4 > 38.1),
            #    那是不可能的數字,也會讓人誤判瓶頸在模型而不是排隊。
            _t0 = _time_stats.perf_counter()
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                device=self.runtime_device,
            )
            # 先記 model 時間,_parse_result 內才會把同一次的兩段接起來
            _record_model_time(_time_stats.perf_counter() - _t0)
            detections: List[Dict[str, Any]] = []
            for result in results:
                detections = self._parse_result(result, frame)
        return self._filter_motorcycle_artifacts(detections, frame.shape[:2])

    def _parse_result(self, result, frame: np.ndarray) -> List[Dict[str, Any]]:
        """把單一張的推論結果轉成 detection dict。

        由批次閘門在 GPU_INFERENCE_LOCK 內逐張呼叫 —— 這裡會讀 GPU tensor,
        而且大型車細分類要再進 GPU,所以不能搬到鎖外。
        """
        _t_truck = 0.0
        _n_box = 0
        _n_tc_call = 0      # 這一幀要求了幾次細分類
        _n_tc_hit = 0       # 其中幾次靠快取免掉推論
        _t1 = _time_stats.perf_counter()

        detections = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            # 🛑 一次把整批搬回 CPU,不要每個框各搬一次。
            #    原本 int(box.cls[0]) / box.xyxy[0].cpu() / float(box.conf[0])
            #    是「每框三次」GPU→CPU 同步,10 台車就是 30 次,而且全在
            #    GPU_INFERENCE_LOCK 裡面 —— 排隊的其他相機全都在等這個。
            #    改成整批三次,取值與原本逐框完全相同。
            _n_box += len(boxes)
            _cls = boxes.cls.cpu().numpy()
            _xyxy = boxes.xyxy.cpu().numpy()
            _conf = boxes.conf.cpu().numpy()
            for _i in range(len(_cls)):
                class_id = int(_cls[_i])

                # 只保留交通相關類別
                if class_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = _xyxy[_i]
                confidence = float(_conf[_i])
                
                det = {
                    'class_id': class_id,
                    'class_name': self.vehicle_classes[class_id],
                    'confidence': confidence,
                    'bbox': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    }
                }

                # 大型車細分類（共用 instance，加 lock 保護避免多 cam 併發）
                if (self.truck_classifier
                        and det['class_name'] in self._RECLASSIFY_CLASSES):
                    _tt0 = _time_stats.perf_counter()
                    cls_result, _tc_hit_one = self._truck_cls_cached(frame, det['bbox'])
                    _t_truck += _time_stats.perf_counter() - _tt0
                    _n_tc_call += 1
                    if _tc_hit_one:
                        _n_tc_hit += 1
                    if cls_result['class_name'] == 'non_truck':
                        det['class_name'] = 'car'
                        det['truck_cls'] = cls_result
                    elif cls_result['class_name'] != 'unknown':
                        det['class_name'] = cls_result['class_name']
                        det['truck_cls'] = cls_result

                detections.append(det)

        _t_parse = _time_stats.perf_counter() - _t1 - _t_truck
        _record_detect_stats(_t_parse, _t_truck, _n_box, _n_tc_call, _n_tc_hit)
        return detections

    def _filter_motorcycle_artifacts(self, detections: List[Dict[str, Any]],
                                     shape: tuple) -> List[Dict[str, Any]]:
        """機車類誤判防護 (只動 motorcycle,不影響其他車種計數)。

        國8(demo 影片)實測:車輛駛出畫面邊緣時,殘缺車尾+紅尾燈被誤判成機車,
        bbox 緊貼邊緣、信心 0.13~0.75。兩道防護:
          ① 貼畫面邊緣(出畫殘影,分類不可靠) → 丟
          ② 信心低於門檻 → 丟
        門檻可用環境變數 MOTO_EDGE_MARGIN_PX / MOTO_MIN_CONF 調整。
        台62 等有真機車的點位:完整在畫面內且信心足者仍保留。

        純 python dict 運算,碰不到 GPU,所以在 GPU_INFERENCE_LOCK 外面跑。
        """
        ih, iw = shape[:2]
        _edge = int(os.getenv("MOTO_EDGE_MARGIN_PX", "6"))
        _moto_min_conf = float(os.getenv("MOTO_MIN_CONF", "0.30"))
        # 標誌/桿柱誤判成行人:國道點位幾乎沒有真行人,靜態路標卻常被判 person。
        # 用信心門檻先擋(env 可調);預設 0.6 高於一般 0.5。0=停用此過濾。
        _person_min_conf = float(os.getenv("PERSON_MIN_CONF", "0.6"))
        filtered = []
        for det in detections:
            if det['class_name'] == 'motorcycle':
                b = det['bbox']
                if (b['x1'] <= _edge or b['y1'] <= _edge
                        or b['x2'] >= iw - _edge or b['y2'] >= ih - _edge):
                    continue  # 出畫殘影(貼邊)
                if det['confidence'] < _moto_min_conf:
                    continue  # 低信心機車誤判
            elif det['class_name'] == 'person':
                if _person_min_conf > 0 and det['confidence'] < _person_min_conf:
                    continue  # 低信心行人(多半是路標/桿柱誤判)
            filtered.append(det)

        return filtered
    
    def detect_with_draw(self, frame: np.ndarray) -> tuple:
        """
        偵測並繪製標註框
        
        Returns:
            (標註後的影像, 偵測結果)
        """
        import cv2
        
        detections = self.detect(frame)
        annotated = frame.copy()
        
        # 顏色定義 (BGR)
        colors = {
            'person': (0, 255, 0),        # 綠色
            'car': (255, 0, 0),           # 藍色
            'motorcycle': (0, 255, 255),  # 黃色
            'bus': (255, 165, 0),         # 橙色
            'truck': (128, 0, 128),       # 紫色
            'bicycle': (0, 128, 255),     # 橘色
            'heavy_truck': (0, 0, 255),   # 紅色
            'light_truck': (255, 128, 0), # 淺藍
            'non_truck': (180, 180, 180), # 灰色
        }
        
        for det in detections:
            bbox = det['bbox']
            color = colors.get(det['class_name'], (255, 255, 255))
            
            # 繪製矩形框
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color, 2
            )
            
            # 繪製標籤（統一顯示中文，不帶信心度）
            truck_cls = det.get('truck_cls')
            if truck_cls:
                label = str(truck_cls['label'])
            else:
                label = self.get_zh_label(det['class_name'])
            cv2.putText(
                annotated, label,
                (bbox['x1'], bbox['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        return annotated, detections


# 測試
if __name__ == '__main__':
    detector = VehicleDetector()
    print(f"支援類別: {list(detector.VEHICLE_CLASSES.values())}")
