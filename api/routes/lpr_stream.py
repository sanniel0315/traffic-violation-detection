#!/usr/bin/env python3
"""車牌辨識串流 API - YOLO + Tesseract"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from collections import deque
import cv2
import numpy as np
import pytesseract
import time
import threading
import asyncio
import os
from typing import Dict, Any, List, Optional
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import re

sys.path.insert(0, '/workspace')

from model_paths import get_detect_model_engine, get_detect_model_pt
from api.models import SessionLocal, Camera, LPRCameraStat, LPRRecord
from api.utils.feature_state import get_feature_state, set_feature_state
from api.utils.camera_stream import resolve_analysis_source
from detection.violation_detector import VehicleTracker

router = APIRouter(prefix="/api/lpr/stream", tags=["lpr-stream"])

_lpr_tasks: Dict[int, 'LPRStreamTask'] = {}
_yolo_model = None
_yolo_lock = threading.Lock()
_recognizer = None
_plate_detector = None
_OCR_FAST_MODE = os.getenv("LPR_OCR_MODE", "accurate").strip().lower() in {"fast", "quick", "speed"}
_PLATE_INSET_SCALE = 3.0
_PLATE_DETECT_CONF = 0.30
_PLATE_CROP_PAD_X_RATIO = 0.18
_PLATE_CROP_PAD_Y_RATIO = 0.25
_PLATE_MIN_WIDTH = 40
_PLATE_MIN_HEIGHT = 14
_PLATE_MIN_AREA = 600
_PLATE_BLUR_TOO_LOW = 35.0
# === 最佳距離窗口 — 車牌夠大夠清楚才做 OCR ===
_PLATE_OCR_MIN_CROP_W = 40        # 車牌裁切最小寬度（px）才值得 OCR
_PLATE_OCR_MIN_CROP_H = 12        # 車牌裁切最小高度（px）
_PLATE_OCR_MIN_BLUR = 15.0        # 最低清晰度（Laplacian variance）
_VEHICLE_MIN_AREA_RATIO = 0.003   # 車輛面積佔畫面比例 — 太小不做 OCR
_PLATE_BLUR_THRESHOLD = 85.0
_PLATE_BLUR_GOOD = 120.0
_PLATE_BRIGHTNESS_MIN = 45.0
_PLATE_BRIGHTNESS_MAX = 215.0
_PLATE_CONTRAST_MIN = 28.0
_PLATE_ASPECT_MIN = 2.0
_PLATE_ASPECT_MAX = 6.5
_PLATE_NORM_W = 256
_PLATE_NORM_H = 80
_PLATE_LIGHT_ROTATE_MAX_DEG = 8.0
_PLATE_VOTE_BUCKET_SIZE = 160
_PLATE_VOTE_TTL_SEC = 3.5
_PLATE_VOTE_STRONG_SINGLE_CONF = 0.40         # 0.50 → 0.40
_PLATE_VOTE_VERY_STRONG_SINGLE_CONF = 0.62    # 0.72 → 0.62
# === best-of-track 追蹤式 commit ===
_TRACK_PLATE_TTL_SEC = 5.0                    # track 超時（最後偵測後多久算結束）
_TRACK_MIN_OCR_COUNT = 1                      # 允許單次 OCR commit — 字元投票在多次時自動修正
_PLATE_MAIN_OCR_CONF_THRESHOLD = 0.55         # 0.68 → 0.55
_PLATE_MAIN_OCR_MIN_LEN = 5
_PLATE_MAIN_OCR_MAX_LEN = 8
_PLATE_CONFIRM_MIN_COUNT = 2                  # 3 → 2
_PLATE_CONFIRM_MIN_SCORE = 1.8                # 2.4 → 1.8
_PLATE_COMMIT_MIN_SCORE = 1.5                 # 2.0 → 1.5
_PLATE_COMMIT_COOLDOWN_SEC = 20.0             # 同車牌冷卻 20 秒
_PLATE_COMMIT_MIN_CONF = 0.40                 # 提高最低信心度
_PLATE_COMMIT_MIN_QUALITY = 0.08              # 0.12 → 0.08
_PLATE_SINGLE_FRAME_MIN_CONF = 0.45           # 0.55 → 0.45
_PLATE_SINGLE_FRAME_MIN_SCORE = 2.50          # 3.10 → 2.50
_PLATE_SINGLE_FRAME_MIN_LAYOUT = 1.00         # 1.20 → 1.00
_PLATE_ADAPTIVE_BLOCK_SIZE = 19
_PLATE_ADAPTIVE_C = 6
_PLATE_CHAR_MIN_W = 8
_PLATE_CHAR_MAX_W = 54
_PLATE_CHAR_MIN_H = 22
_PLATE_CHAR_MAX_H = 78
_PLATE_CHAR_MIN_AREA = 130
_PLATE_CHAR_ASPECT_MIN = 0.15
_PLATE_CHAR_ASPECT_MAX = 1.05
_PLATE_CHAR_TOPK = 3
_PLATE_CHAR_BEAM_WIDTH = 12
_PLATE_REAR_ROI_ENABLE = False
_PLATE_REAR_ROI_X_MIN = 0.20
_PLATE_REAR_ROI_X_MAX = 0.80
_PLATE_REAR_ROI_Y_MIN = 0.45
_PLATE_REAR_ROI_Y_MAX = 0.92
_PLATE_MAX_RANKED_CANDIDATES = 5
_PLATE_KEEP_SECONDARY_CANDIDATES = True
_VEHICLE_UNKNOWN_MIN_SEEN_FRAMES = 2
_PLATE_CONFUSION_MAP = {
    "O": ("0", "Q"),
    "0": ("O", "Q"),
    "I": ("1", "L"),
    "1": ("I", "L"),
    "H": ("M",),
    "M": ("H",),
    "B": ("8",),
    "8": ("B",),
    "Z": ("2",),
    "2": ("Z",),
    "S": ("5",),
    "5": ("S",),
    "D": ("0",),
    "9": ("8",),
    "3": ("8",),
    "G": ("6",),
    "6": ("G",),
    "Q": ("0", "O"),
}

SNAPSHOT_DIR = '/workspace/storage/lpr_snapshots'
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
TZ_TAIPEI = ZoneInfo("Asia/Taipei")
_LPR_CUMULATIVE_COUNTER_FIELDS = (
    "total_frames",
    "vehicles_detected",
    "plate_boxes_detected",
    "ocr_candidates_detected",
    "vote_candidates_detected",
    "confirmed_candidates",
    "committed_candidates",
)


def _derive_plate_snapshot_name(snapshot_name: Optional[str]) -> Optional[str]:
    if not snapshot_name:
        return None
    name = str(snapshot_name)
    full_name = os.path.join(SNAPSHOT_DIR, name)
    if os.path.exists(full_name):
        if "_plate" in os.path.splitext(name)[0]:
            return name
    base, ext = os.path.splitext(name)
    ext = ext or ".jpg"
    candidate = f"{base}_plate{ext}"
    full = os.path.join(SNAPSHOT_DIR, candidate)
    return candidate if os.path.exists(full) else None


def _open_capture(source: str):
    src = str(source or "").strip()
    src_lc = src.lower()
    is_rtsp = src_lc.startswith("rtsp://")
    if is_rtsp:
        # 強制 RTSP TCP transport 避免封包遺失，加 socket timeout 避免 read 卡死
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|buffer_size;65536"

    backends = []
    if src_lc.startswith("http://") or src_lc.startswith("https://"):
        mjpeg_backend = getattr(cv2, "CAP_OPENCV_MJPEG", None)
        if mjpeg_backend is not None:
            backends.append(mjpeg_backend)
        backends.append(None)
    ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
    gst_backend = getattr(cv2, "CAP_GSTREAMER", None)
    if ffmpeg_backend is not None:
        backends.append(ffmpeg_backend)
    if gst_backend is not None and gst_backend not in backends:
        backends.append(gst_backend)
    if None not in backends:
        backends.append(None)

    last_cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
        except Exception:
            cap = None
        if cap is not None and cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # 設 read/open timeout（OpenCV 4.x 支援）
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except Exception:
                pass
            return cap
        if cap is not None:
            last_cap = cap
            try:
                cap.release()
            except Exception:
                pass
    return last_cap if last_cap is not None else cv2.VideoCapture(src)


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _normalize_plate_edge_angle(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    return angle


def _rotate_plate_roi(roi, quad: np.ndarray, angle_deg: float):
    if roi is None or getattr(roi, "size", 0) == 0:
        return roi, None
    try:
        h, w = roi.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, -float(angle_deg), 1.0)
        rotated = cv2.warpAffine(
            roi,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        rotated_pts = cv2.transform(np.array([quad], dtype=np.float32), M)[0]
        bx, by, bw, bh = cv2.boundingRect(rotated_pts.astype(np.float32))
        x1 = max(0, min(w - 1, int(bx)))
        y1 = max(0, min(h - 1, int(by)))
        x2 = max(0, min(w, int(bx + bw)))
        y2 = max(0, min(h, int(by + bh)))
        if x2 <= x1 or y2 <= y1:
            return rotated, None
        return rotated[y1:y2, x1:x2], [x1, y1, x2, y2]
    except Exception:
        return roi, None


def _flatten_plate_roi_with_bbox(roi):
    """從候選 ROI 找車牌框，小角度只旋轉，大角度才做透視攤平。"""
    if roi is None or getattr(roi, "size", 0) == 0:
        return roi, None
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return roi, None

        h, w = gray.shape[:2]
        img_area = float(max(1, h * w))
        best = None
        best_score = 0.0

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            area = float(cv2.contourArea(cnt))
            if area < img_area * 0.01:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
            else:
                rect = cv2.minAreaRect(cnt)
                pts = cv2.boxPoints(rect).astype(np.float32)
            ordered = _order_quad_points(pts)

            w1 = np.linalg.norm(ordered[1] - ordered[0])
            w2 = np.linalg.norm(ordered[2] - ordered[3])
            h1 = np.linalg.norm(ordered[3] - ordered[0])
            h2 = np.linalg.norm(ordered[2] - ordered[1])
            ww = max(w1, w2)
            hh = max(h1, h2)
            if ww < 28 or hh < 10:
                continue
            aspect = ww / max(1.0, hh)
            if not (1.5 <= aspect <= 8.5):
                continue

            score = area * min(3.0, aspect)
            if score > best_score:
                best_score = score
                best = ordered

        if best is None:
            return roi, None

        bx, by, bw, bh = cv2.boundingRect(best.astype(np.float32))
        x1 = max(0, min(w - 1, int(bx)))
        y1 = max(0, min(h - 1, int(by)))
        x2 = max(0, min(w, int(bx + bw)))
        y2 = max(0, min(h, int(by + bh)))
        local_bbox = [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None

        top_edge_angle = np.degrees(np.arctan2(best[1][1] - best[0][1], best[1][0] - best[0][0]))
        norm_angle = _normalize_plate_edge_angle(top_edge_angle)
        if abs(norm_angle) <= _PLATE_LIGHT_ROTATE_MAX_DEG:
            rotated_crop, rotated_bbox = _rotate_plate_roi(roi, best, norm_angle)
            if rotated_crop is not None and getattr(rotated_crop, "size", 0) > 0:
                return rotated_crop, rotated_bbox

        w1 = np.linalg.norm(best[1] - best[0])
        w2 = np.linalg.norm(best[2] - best[3])
        h1 = np.linalg.norm(best[3] - best[0])
        h2 = np.linalg.norm(best[2] - best[1])
        out_w = int(max(40, max(w1, w2)))
        out_h = int(max(14, max(h1, h2)))
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(best, dst)
        flat = cv2.warpPerspective(roi, M, (out_w, out_h))
        if out_w < 280:
            scale = 280 / max(1, out_w)
            flat = cv2.resize(flat, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return flat, local_bbox
    except Exception:
        return roi, None


def _recognize_plate_fast(roi, recognizer) -> Dict[str, Any]:
    """已改用 YOLO 字元偵測，直接走 recognize_easy"""
    return recognizer.recognize_easy(roi) if hasattr(recognizer, 'recognize_easy') else recognizer.recognize(roi)
    best: Dict[str, Any] = {"plate_number": None, "confidence": 0.0, "valid": False, "raw": ""}
    if roi is None or getattr(roi, "size", 0) == 0:
        return best
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        h, w = gray.shape[:2]
        if w < 220:
            scale = 260 / max(1, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = getattr(
            recognizer,
            "config_single",
            "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        )
        clean_fn = getattr(recognizer, "_clean", None)
        valid_fn = getattr(recognizer, "_validate", None)

        for p_img in (gray, binary):
            data = pytesseract.image_to_data(p_img, config=config, output_type=pytesseract.Output.DICT)
            texts = []
            confs = []
            for i, text in enumerate(data.get("text", [])):
                t = str(text or "").strip()
                if len(t) < 2:
                    continue
                texts.append(t)
                raw_conf = str(data.get("conf", ["0"])[i])
                try:
                    c = float(raw_conf)
                except Exception:
                    c = 0.0
                if c < 0:
                    c = 0.0
                confs.append(c / 100.0)
            if not texts:
                continue

            raw = "".join(texts)
            if callable(clean_fn):
                plate, _ = clean_fn(raw)
            else:
                plate = re.sub(r"[^A-Z0-9-]", "", raw.upper())
            conf = max(confs) if confs else 0.0
            valid = bool(valid_fn(plate)) if callable(valid_fn) else bool(re.match(r"^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$", plate))
            if valid and (not best["valid"] or conf > best["confidence"]):
                best = {"plate_number": plate, "confidence": conf, "valid": True, "raw": raw}
            elif (not best["valid"]) and 5 <= len(str(plate or "")) <= 9 and conf > best["confidence"]:
                best = {"plate_number": plate, "confidence": conf, "valid": valid, "raw": raw}
    except Exception:
        return best
    return best


def _clean_plate_text(raw: str, recognizer) -> str:
    clean_fn = getattr(recognizer, "_clean", None)
    if callable(clean_fn):
        plate, _ = clean_fn(raw)
        return str(plate or "")
    return re.sub(r"[^A-Z0-9-]", "", str(raw or "").upper())


def _validate_plate_text(plate: str, recognizer) -> bool:
    valid_fn = getattr(recognizer, "_validate", None)
    if callable(valid_fn):
        return bool(valid_fn(plate))
    return bool(re.match(r"^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$", str(plate or "")))


def _plate_text_core_len(text: str) -> int:
    return len(re.sub(r"[^A-Z0-9]", "", str(text or "").upper()))


def _plate_text_candidates(raw: str, recognizer, limit: int = 16) -> List[str]:
    text = re.sub(r"[^A-Z0-9]", "", str(raw or "").upper())
    if not text:
        return []

    seeds = {text}
    for idx, ch in enumerate(text):
        for repl in _PLATE_CONFUSION_MAP.get(ch, ()):
            seeds.add(f"{text[:idx]}{repl}{text[idx + 1:]}")

    candidates: List[str] = []
    seen = set()
    gen_fn = getattr(recognizer, "_generate_plate_candidates", None)
    for seed in seeds:
        expanded = gen_fn(seed) if callable(gen_fn) else []
        if not expanded:
            expanded = [_clean_plate_text(seed, recognizer)]
        for cand in expanded:
            normalized = _clean_plate_text(cand, recognizer)
            if not normalized or normalized in seen:
                continue
            core_len = _plate_text_core_len(normalized)
            if 5 <= core_len <= 8:
                seen.add(normalized)
                candidates.append(normalized)
                if len(candidates) >= limit:
                    return candidates
    return candidates


def _plate_binary_variants(gray) -> List[np.ndarray]:
    if gray is None or getattr(gray, "size", 0) == 0:
        return []
    variants: List[np.ndarray] = []
    try:
        work = gray.copy()
        work = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(work)
        blur = cv2.GaussianBlur(work, (3, 3), 0)

        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.extend([otsu, 255 - otsu])

        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            _PLATE_ADAPTIVE_BLOCK_SIZE,
            _PLATE_ADAPTIVE_C,
        )
        variants.extend([adaptive, 255 - adaptive])

        blackhat = cv2.morphologyEx(
            blur,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
        )
        _, bh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.extend([bh, 255 - bh])

        cleaned = []
        seen = set()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        for img in variants:
            v = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
            key = v.tobytes()[:128]
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(v)
        return cleaned
    except Exception:
        return []


def _build_plate_ocr_variants(roi, original_roi=None) -> List[tuple[str, np.ndarray]]:
    variants: List[tuple[str, np.ndarray]] = []
    seen = set()

    def add(name: str, img, resize: bool = True) -> None:
        if img is None or getattr(img, "size", 0) == 0:
            return
        try:
            work = img
            if len(work.shape) == 3:
                gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            else:
                gray = work.copy()
            if resize:
                gray = cv2.resize(gray, (_PLATE_NORM_W, _PLATE_NORM_H), interpolation=cv2.INTER_CUBIC)
            key = (gray.shape[1], gray.shape[0], gray.tobytes()[:256])
            if key in seen:
                return
            seen.add(key)
            variants.append((name, gray))
        except Exception:
            return

    # 不變形原圖（讓 recognizer 自己預處理 - 對小車牌特別重要）
    add("raw_native", original_roi, resize=False)
    add("enhanced_native", roi, resize=False)
    # 標準化版本
    add("enhanced", roi)
    add("raw", original_roi)

    try:
        base = roi if roi is not None and getattr(roi, "size", 0) > 0 else original_roi
        if base is not None and getattr(base, "size", 0) > 0:
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if len(base.shape) == 3 else base.copy()
            gray = cv2.resize(gray, (_PLATE_NORM_W, _PLATE_NORM_H), interpolation=cv2.INTER_CUBIC)
            add("gray", gray)
            clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
            add("gray_clahe", clahe)
            denoise = cv2.medianBlur(clahe, 3)
            add("gray_clahe_denoise", denoise)
            blur = cv2.GaussianBlur(denoise, (0, 0), 1.0)
            sharpen = cv2.addWeighted(denoise, 1.22, blur, -0.22, 0)
            add("gray_unsharp", sharpen)
            for idx, mask in enumerate(_plate_binary_variants(clahe)[:4]):
                add(f"binary_{idx}", mask)
    except Exception:
        pass

    return variants


def _segment_plate_characters(plate_img) -> List[np.ndarray]:
    if plate_img is None or getattr(plate_img, "size", 0) == 0:
        return []
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img.copy()
        h, w = gray.shape[:2]
        if h < 1 or w < 1:
            return []

        scale = 96.0 / max(1.0, float(h))
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        wh, ww = work.shape[:2]

        best_boxes = []
        best_score = -1.0
        for mask in _plate_binary_variants(work):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < _PLATE_CHAR_MIN_W or cw > _PLATE_CHAR_MAX_W:
                    continue
                if ch < _PLATE_CHAR_MIN_H or ch > _PLATE_CHAR_MAX_H:
                    continue
                area = cw * ch
                if area < _PLATE_CHAR_MIN_AREA:
                    continue
                h_ratio = ch / max(1.0, float(wh))
                w_ratio = cw / max(1.0, float(ww))
                aspect = cw / max(1.0, float(ch))
                if not (0.35 <= h_ratio <= 0.95):
                    continue
                if not (0.015 <= w_ratio <= 0.22):
                    continue
                if not (_PLATE_CHAR_ASPECT_MIN <= aspect <= _PLATE_CHAR_ASPECT_MAX):
                    continue
                if y + ch < int(wh * 0.28) or y > int(wh * 0.88):
                    continue
                if x <= 1 or (x + cw) >= (ww - 1):
                    continue
                boxes.append((x, y, cw, ch))

            filtered = []
            for box in sorted(boxes, key=lambda b: (b[0], -b[2] * b[3])):
                x, y, cw, ch = box
                nested = False
                for ox, oy, ow, oh in boxes:
                    if box == (ox, oy, ow, oh):
                        continue
                    if x >= ox and y >= oy and (x + cw) <= (ox + ow) and (y + ch) <= (oy + oh):
                        if (ow * oh) >= (cw * ch):
                            nested = True
                            break
                if not nested:
                    filtered.append(box)

            filtered.sort(key=lambda b: b[0])
            if len(filtered) < 4:
                continue

            count_score = max(0.0, 1.0 - (abs(len(filtered) - 6) / 6.0))
            mean_height = float(np.mean([b[3] for b in filtered])) / max(1.0, float(wh))
            coverage = float((filtered[-1][0] + filtered[-1][2]) - filtered[0][0]) / max(1.0, float(ww))
            score = (count_score * 0.45) + (min(1.0, mean_height / 0.72) * 0.30) + (min(1.0, coverage / 0.72) * 0.25)
            if score > best_score:
                best_score = score
                best_boxes = filtered

        if len(best_boxes) < 4:
            return []

        chars = []
        for x, y, cw, ch in best_boxes[:8]:
            pad_x = max(1, int(cw * 0.12))
            pad_y = max(1, int(ch * 0.10))
            x1 = max(0, int((x - pad_x) / scale))
            y1 = max(0, int((y - pad_y) / scale))
            x2 = min(w, int((x + cw + pad_x) / scale))
            y2 = min(h, int((y + ch + pad_y) / scale))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = gray[y1:y2, x1:x2]
            if crop is not None and getattr(crop, "size", 0) > 0:
                chars.append(crop)
        return chars
    except Exception:
        return []


def _ocr_single_character_candidates(char_img, top_k: int = _PLATE_CHAR_TOPK) -> List[tuple[str, float]]:
    if char_img is None or getattr(char_img, "size", 0) == 0:
        return []
    try:
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY) if len(char_img.shape) == 3 else char_img.copy()
        h, w = gray.shape[:2]
        scale = 72.0 / max(1.0, float(h))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.copyMakeBorder(gray, 12, 12, 10, 10, cv2.BORDER_CONSTANT, value=255)
        variants = [gray]
        variants.extend(_plate_binary_variants(gray))
        config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        scored: Dict[str, float] = {}
        for img in variants:
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            texts = data.get("text", [])
            confs = data.get("conf", [])
            for idx, text in enumerate(texts):
                token = re.sub(r"[^A-Z0-9]", "", str(text or "").upper())
                if not token:
                    continue
                raw_conf = str(confs[idx]) if idx < len(confs) else "0"
                try:
                    conf = max(0.0, float(raw_conf)) / 100.0
                except Exception:
                    conf = 0.0
                char = token[0]
                scored[char] = max(float(scored.get(char, 0.0)), conf)
                for repl in _PLATE_CONFUSION_MAP.get(char, ()):
                    scored[repl] = max(float(scored.get(repl, 0.0)), conf * 0.82)
        return sorted(scored.items(), key=lambda item: float(item[1]), reverse=True)[:max(1, int(top_k))]
    except Exception:
        return []


def _ocr_single_character(char_img) -> tuple[str, float]:
    candidates = _ocr_single_character_candidates(char_img, top_k=1)
    if not candidates:
        return "", 0.0
    return candidates[0]


def _rebuild_plate_from_char_candidates(char_candidates: List[List[tuple[str, float]]], recognizer) -> Dict[str, Any]:
    empty = {"plate_number": None, "confidence": 0.0, "valid": False, "raw": "", "char_candidates": []}
    if not char_candidates:
        return empty

    beam: List[tuple[str, float]] = [("", 0.0)]
    packed_candidates: List[List[List[Any]]] = []
    for slot in char_candidates:
        filtered = [(ch, float(conf)) for ch, conf in slot if ch]
        if not filtered:
            continue
        packed_candidates.append([[ch, round(conf, 3)] for ch, conf in filtered])
        next_beam: List[tuple[str, float]] = []
        for prefix, score in beam:
            for ch, conf in filtered[:_PLATE_CHAR_TOPK]:
                next_beam.append((prefix + ch, score + conf))
        next_beam.sort(key=lambda item: float(item[1]), reverse=True)
        beam = next_beam[:_PLATE_CHAR_BEAM_WIDTH]

    if not beam:
        return empty

    best_plate = None
    best_score = -10.0
    best_conf = 0.0
    best_raw = ""
    for raw_text, raw_score in beam:
        for cand in _plate_text_candidates(raw_text, recognizer):
            conf = raw_score / max(1, len(raw_text))
            valid = _validate_plate_text(cand, recognizer)
            score = (conf * 1.7) + (2.2 if valid else 0.0)
            if re.match(r"^[A-Z]{2,4}-[0-9]{2,4}$", cand):
                score += 0.9
            elif re.match(r"^[0-9]{2,4}-[A-Z]{2,4}$", cand):
                score += 0.5
            if score > best_score:
                best_score = score
                best_plate = cand
                best_conf = conf
                best_raw = raw_text

    if not best_plate:
        return empty
    return {
        "plate_number": best_plate,
        "confidence": best_conf,
        "valid": _validate_plate_text(best_plate, recognizer),
        "raw": best_raw,
        "char_candidates": packed_candidates,
    }


def _recognize_plate_by_characters(plate_img, recognizer) -> Dict[str, Any]:
    empty = {"plate_number": None, "confidence": 0.0, "valid": False, "raw": "", "char_candidates": []}
    if plate_img is None or getattr(plate_img, "size", 0) == 0:
        return empty
    char_rois = _segment_plate_characters(plate_img)
    if len(char_rois) < 4:
        return empty

    char_candidates: List[List[tuple[str, float]]] = []
    for roi in char_rois:
        candidates = _ocr_single_character_candidates(roi)
        if not candidates:
            continue
        char_candidates.append(candidates)

    if len(char_candidates) < 4:
        return empty

    rebuilt = _rebuild_plate_from_char_candidates(char_candidates, recognizer)
    if rebuilt.get("plate_number"):
        return rebuilt

    chars = [slot[0][0] for slot in char_candidates if slot]
    confs = [float(slot[0][1]) for slot in char_candidates if slot]
    raw = "".join(chars)
    for plate in _plate_text_candidates(raw, recognizer):
        valid = _validate_plate_text(plate, recognizer)
        if not valid and len(raw) not in {5, 6, 7, 8}:
            continue
        return {
            "plate_number": plate,
            "confidence": float(sum(confs) / len(confs)) if confs else 0.0,
            "valid": valid,
            "raw": raw,
            "char_candidates": [[[ch, round(conf, 3)] for ch, conf in slot] for slot in char_candidates],
        }
    return empty


def _plate_result_score(result: Dict[str, Any], recognizer) -> float:
    plate = _clean_plate_text(result.get("plate_number") or result.get("raw") or "", recognizer)
    if not plate:
        return -10.0
    conf = float(result.get("confidence") or 0.0)
    valid = bool(result.get("valid")) or _validate_plate_text(plate, recognizer)
    plain = plate.replace("-", "")
    score = conf * 1.8
    if valid:
        score += 2.6
    if re.match(r"^[A-Z]{2,4}-[0-9]{2,4}$", plate):
        score += 1.1
    elif re.match(r"^[0-9]{2,4}-[A-Z]{2,4}$", plate):
        score += 0.6
    else:
        score -= 0.3
    if re.search(r"([A-Z0-9])\1\1", plain):
        score -= 1.0
    return score


def _plate_quality_metrics(img) -> Dict[str, float]:
    if img is None or getattr(img, "size", 0) == 0:
        return {"width": 0.0, "height": 0.0, "blur": 0.0, "brightness": 0.0, "contrast": 0.0, "aspect": 0.0, "angle": 0.0}
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape[:2]
        angle = 0.0
        try:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(255 - bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
                raw_angle = float(rect[-1])
                angle = raw_angle if raw_angle >= -45.0 else (90.0 + raw_angle)
        except Exception:
            angle = 0.0
        return {
            "width": float(w),
            "height": float(h),
            "blur": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "brightness": float(np.mean(gray)),
            "contrast": float(np.std(gray)),
            "aspect": float(w) / max(1.0, float(h)),
            "angle": abs(angle),
        }
    except Exception:
        return {"width": 0.0, "height": 0.0, "blur": 0.0, "brightness": 0.0, "contrast": 0.0, "aspect": 0.0, "angle": 0.0}


def _plate_quality_score(metrics: Dict[str, float]) -> float:
    area_score = min(
        1.0,
        (float(metrics.get("width") or 0.0) * float(metrics.get("height") or 0.0)) / float(max(1.0, _PLATE_MIN_AREA)),
    )
    width_score = min(1.0, float(metrics.get("width") or 0.0) / float(_PLATE_MIN_WIDTH))
    height_score = min(1.0, float(metrics.get("height") or 0.0) / float(_PLATE_MIN_HEIGHT))
    blur_score = min(1.0, float(metrics.get("blur") or 0.0) / max(1.0, _PLATE_BLUR_GOOD))
    brightness = float(metrics.get("brightness") or 0.0)
    if _PLATE_BRIGHTNESS_MIN <= brightness <= _PLATE_BRIGHTNESS_MAX:
        brightness_score = 1.0
    else:
        dist = min(abs(brightness - _PLATE_BRIGHTNESS_MIN), abs(brightness - _PLATE_BRIGHTNESS_MAX))
        brightness_score = max(0.0, 1.0 - (dist / 80.0))
    contrast_score = min(1.0, float(metrics.get("contrast") or 0.0) / max(1.0, _PLATE_CONTRAST_MIN * 1.5))
    aspect = float(metrics.get("aspect") or 0.0)
    if _PLATE_ASPECT_MIN <= aspect <= _PLATE_ASPECT_MAX:
        aspect_score = 1.0
    else:
        aspect_score = max(0.0, 1.0 - min(abs(aspect - _PLATE_ASPECT_MIN), abs(aspect - _PLATE_ASPECT_MAX)) / 2.8)
    angle_score = max(0.0, 1.0 - (float(metrics.get("angle") or 0.0) / 25.0))
    return max(
        0.0,
        min(
            1.0,
            (width_score * 0.10)
            + (height_score * 0.10)
            + (area_score * 0.10)
            + (blur_score * 0.20)
            + (brightness_score * 0.15)
            + (contrast_score * 0.15)
            + (aspect_score * 0.20)
            + (angle_score * 0.10),
        ),
    )


def _plate_quality_level(img) -> str:
    m = _plate_quality_metrics(img)
    if (
        m["width"] < 30
        or m["height"] < 10
        or (m["width"] * m["height"]) < 300
        or m["aspect"] < 1.2
        or m["aspect"] > 8.0
    ):
        return "low"
    if m["angle"] > 30.0:
        return "low"
    if m["blur"] <= 25.0:
        return "low"
    if (
        m["blur"] >= _PLATE_BLUR_GOOD
        and _PLATE_BRIGHTNESS_MIN <= m["brightness"] <= _PLATE_BRIGHTNESS_MAX
        and m["contrast"] >= _PLATE_CONTRAST_MIN
        and m["angle"] <= 12.0
    ):
        return "high"
    return "medium"


def _should_try_character_fallback(result: Dict[str, Any]) -> bool:
    plate = str(result.get("plate_number") or "").strip()
    conf = float(result.get("confidence") or 0.0)
    valid = bool(result.get("valid", False))
    if not plate:
        return True
    if valid and conf >= 0.68:
        return False
    return (not valid) or conf < 0.68


def _recognize_plate_on_crop(plate_crop, recognizer) -> Dict[str, Any]:
    """車牌辨識 — 直接用 YOLO 字元偵測微服務"""
    empty = {
        "plate_number": None,
        "confidence": 0.0,
        "valid": False,
        "raw": "",
        "quality_level": "low",
        "quality_score": 0.0,
        "quality_metrics": _plate_quality_metrics(plate_crop),
        "char_candidates": [],
    }
    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
        return empty
    quality_metrics = _plate_quality_metrics(plate_crop)
    quality_level = _plate_quality_level(plate_crop)
    quality_score = _plate_quality_score(quality_metrics)
    result = recognizer.recognize_easy(plate_crop)
    result["quality_level"] = quality_level
    result["quality_score"] = quality_score
    result["quality_metrics"] = quality_metrics
    result["char_candidates"] = []
    return result
    # === 以下為舊 Tesseract 邏輯（停用）===
    original_roi = plate_crop.copy()
    try:
        flat, _ = _flatten_plate_roi_with_bbox(plate_crop)
        roi = flat if flat is not None and getattr(flat, "size", 0) > 0 else plate_crop
    except Exception:
        roi = plate_crop

    # 只在大圖時才 tighten 與正規化 resize，避免破壞小車牌
    rh, rw = roi.shape[:2]
    if rw >= 120 and rh >= 30:
        try:
            tightened = _tighten_plate_crop(roi)
            if tightened is not None and getattr(tightened, "size", 0) > 0:
                roi = tightened
        except Exception:
            pass

    try:
        roi = _enhance_plate_snapshot(roi)
    except Exception:
        pass

    # 不再強制 resize 到固定尺寸（會變形小車牌），讓 recognizer 自己處理

    quality = _plate_snapshot_quality_score(roi)
    quality_metrics = _plate_quality_metrics(roi)
    quality_level = _plate_quality_level(roi)
    quality_score = _plate_quality_score(quality_metrics)

    # ---- 1. 字元分割 OCR（快速）----
    if hasattr(recognizer, 'recognize_chars'):
        try:
            char_res = recognizer.recognize_chars(original_roi)
            cp = str(char_res.get('plate_number') or '')
            cv_flag = bool(char_res.get('valid'))
            if cp and cv_flag:
                char_res['quality_level'] = quality_level
                char_res['quality_score'] = quality_score
                char_res['quality_metrics'] = quality_metrics
                char_res['variant'] = 'char_segment'
                return char_res
        except Exception:
            pass

    # ---- 2. EasyOCR GPU 加速辨識 ----
    if hasattr(recognizer, 'easy_reader') and recognizer.easy_reader is not None:
        try:
            easy_res = recognizer.recognize_easy(original_roi)
            ep = str(easy_res.get('plate_number') or '')
            if ep and len(ep) >= 4:
                easy_res['quality_level'] = quality_level
                easy_res['quality_score'] = quality_score
                easy_res['quality_metrics'] = quality_metrics
                easy_res['variant'] = 'easyocr_gpu'
                return easy_res
        except Exception:
            pass

    def _run_variant_ocr(img, variant_name: str = ""):
        try:
            ocr_img = img
            if ocr_img is not None and getattr(ocr_img, "size", 0) > 0 and len(ocr_img.shape) == 2:
                ocr_img = cv2.cvtColor(ocr_img, cv2.COLOR_GRAY2BGR)
            res = _recognize_plate_fast(ocr_img, recognizer) if _OCR_FAST_MODE else recognizer.recognize(ocr_img)
        except Exception:
            res = dict(empty)
        plate_text = _clean_plate_text(res.get("plate_number") or res.get("raw") or "", recognizer)
        if plate_text:
            res["plate_number"] = plate_text
            res["valid"] = bool(res.get("valid")) or _validate_plate_text(plate_text, recognizer)
        if variant_name:
            res["variant"] = variant_name
        return res

    ocr_variants = _build_plate_ocr_variants(roi, original_roi=original_roi)
    variant_results: List[Dict[str, Any]] = []
    for variant_name, variant_img in ocr_variants:
        res = _run_variant_ocr(variant_img, variant_name)
        res["quality_level"] = quality_level
        res["quality_score"] = quality_score
        res["quality_metrics"] = quality_metrics
        variant_results.append(res)

    if variant_results:
        main_res = max(
            variant_results,
            key=lambda item: _plate_result_score(item, recognizer)
            + (0.12 if str(item.get("variant") or "").startswith("gray") else 0.0)
            + (0.05 if str(item.get("variant") or "") == "enhanced" else 0.0),
        )
    else:
        main_res = dict(empty)
    main_res["quality_level"] = quality_level
    main_res["quality_score"] = quality_score
    main_res["quality_metrics"] = quality_metrics
    main_len = len(str(main_res.get("plate_number") or "").replace("-", ""))

    # 放寬品質門檻 — 即使低品質也嘗試 OCR，讓投票機制決定
    if quality_level == "low" and quality_score < 0.02 and quality < 0.10:
        return empty

    if (
        main_res.get("valid")
        and float(main_res.get("confidence") or 0.0) >= _PLATE_MAIN_OCR_CONF_THRESHOLD
        and _PLATE_MAIN_OCR_MIN_LEN <= main_len <= _PLATE_MAIN_OCR_MAX_LEN
    ):
        return main_res

    if quality_level == "high" and not _should_try_character_fallback(main_res):
        return main_res

    char_candidates: List[Dict[str, Any]] = []
    char_inputs: List[tuple[str, Any]] = []
    if roi is not None and getattr(roi, "size", 0) > 0:
        char_inputs.append(("enhanced", roi))
    for variant_name, variant_img in ocr_variants:
        if variant_name.startswith("binary_") or variant_name in {"gray_clahe", "gray_unsharp", "raw"}:
            char_inputs.append((variant_name, variant_img))

    seen_char_variants = set()
    for variant_name, variant_img in char_inputs:
        if variant_name in seen_char_variants:
            continue
        seen_char_variants.add(variant_name)
        try:
            res = _recognize_plate_by_characters(variant_img, recognizer)
        except Exception:
            res = empty
        plate_text = _clean_plate_text(res.get("plate_number") or res.get("raw") or "", recognizer)
        if plate_text:
            res["plate_number"] = plate_text
            res["valid"] = bool(res.get("valid")) or _validate_plate_text(plate_text, recognizer)
        res["quality_level"] = quality_level
        res["quality_score"] = quality_score
        res["quality_metrics"] = quality_metrics
        res["variant"] = variant_name
        char_candidates.append(res)

    char_res = max(char_candidates, key=lambda item: _plate_result_score(item, recognizer), default=empty)

    char_plate = _clean_plate_text(char_res.get("plate_number") or char_res.get("raw") or "", recognizer)
    if char_plate:
        char_res["plate_number"] = char_plate
        char_res["valid"] = bool(char_res.get("valid")) or _validate_plate_text(char_plate, recognizer)
    char_res["quality_level"] = quality_level
    char_res["quality_score"] = quality_score
    char_res["quality_metrics"] = quality_metrics

    main_score = _plate_result_score(main_res, recognizer)
    char_score = _plate_result_score(char_res, recognizer)
    if char_score > (main_score + 0.08):
        return char_res
    if main_res.get("plate_number"):
        return main_res
    return char_res


def _plate_mount_hole_score(gray) -> float:
    if gray is None or getattr(gray, "size", 0) == 0:
        return 0.0
    try:
        h, w = gray.shape[:2]
        if h < 18 or w < 60:
            return 0.0

        band_h = max(10, min(h - 1, int(round(h * 0.38))))
        top_band = gray[:band_h, :]
        blur = cv2.GaussianBlur(top_band, (5, 5), 0)

        # 螺絲凹點通常是上緣附近較暗、近圓形的小區塊。
        mean_v = float(np.mean(blur))
        std_v = float(np.std(blur))
        thr = max(0, min(255, int(round(mean_v - max(8.0, std_v * 0.65)))))
        _, mask = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        band_area = float(max(1, band_h * w))
        spots = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < band_area * 0.0008 or area > band_area * 0.035:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            if y > band_h * 0.72:
                continue
            aspect = cw / max(1.0, float(ch))
            if not (0.45 <= aspect <= 1.9):
                continue
            peri = max(1.0, float(cv2.arcLength(cnt, True)))
            circularity = float(4.0 * np.pi * area / (peri * peri))
            if circularity < 0.25:
                continue
            cx = x + (cw / 2.0)
            cy = y + (ch / 2.0)
            spots.append({
                "cx": cx,
                "cy": cy,
                "area": area,
                "aspect": aspect,
                "circularity": circularity,
            })

        if len(spots) < 2:
            return 0.0

        center_x = w / 2.0
        best = 0.0
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                a = spots[i]
                b = spots[j]
                left, right = (a, b) if a["cx"] <= b["cx"] else (b, a)
                if left["cx"] >= center_x or right["cx"] <= center_x:
                    continue

                sep = (right["cx"] - left["cx"]) / max(1.0, float(w))
                if not (0.22 <= sep <= 0.72):
                    continue

                symmetry = 1.0 - min(1.0, abs((center_x - left["cx"]) - (right["cx"] - center_x)) / max(6.0, w * 0.22))
                level = 1.0 - min(1.0, abs(left["cy"] - right["cy"]) / max(4.0, band_h * 0.35))
                size = 1.0 - min(1.0, abs(left["area"] - right["area"]) / max(8.0, max(left["area"], right["area"])))
                shape = min(1.0, ((left["circularity"] + right["circularity"]) / 2.0) / 0.85)
                pair_score = (symmetry * 0.42) + (level * 0.23) + (size * 0.20) + (shape * 0.15)
                if pair_score > best:
                    best = pair_score

        return max(0.0, min(1.0, best))
    except Exception:
        return 0.0


def _plate_feature_score(patch) -> float:
    if patch is None or getattr(patch, "size", 0) == 0:
        return -1.0
    try:
        h, w = patch.shape[:2]
        if h < 16 or w < 40:
            return -1.0
        area = float(max(1, h * w))
        aspect = w / max(1.0, float(h))
        if not (1.8 <= aspect <= 7.2):
            return -1.0

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 60, 180)
        edge_density = float(cv2.countNonZero(edges)) / area

        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg = cv2.countNonZero(255 - bw)
        fg_ratio = float(fg) / area

        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        stroke_energy = float(np.mean(np.abs(sobel_x))) / 255.0
        mount_score = _plate_mount_hole_score(gray)

        # 車牌特徵：中高水平長寬比 + 適中邊緣密度 + 適中文字前景比例
        aspect_score = max(0.0, 1.0 - abs(aspect - 3.2) / 3.2)
        edge_score = max(0.0, 1.0 - abs(edge_density - 0.16) / 0.16)
        fg_score = max(0.0, 1.0 - abs(fg_ratio - 0.34) / 0.34)
        stroke_score = min(1.0, stroke_energy / 0.22)
        return (
            (aspect_score * 0.31)
            + (edge_score * 0.27)
            + (fg_score * 0.18)
            + (stroke_score * 0.14)
            + (mount_score * 0.10)
        )
    except Exception:
        return -1.0


def _expand_plate_bbox(
    bbox: List[int],
    full_w: int,
    full_h: int,
    pad_x_ratio: float = _PLATE_CROP_PAD_X_RATIO,
    pad_y_ratio: float = _PLATE_CROP_PAD_Y_RATIO,
) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = max(4, int(bw * pad_x_ratio))
    pad_y = max(4, int(bh * pad_y_ratio))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(full_w, x2 + pad_x),
        min(full_h, y2 + pad_y),
    ]


def _crop_vehicle_rear_roi(vehicle_crop):
    if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
        return vehicle_crop, [0, 0, 0, 0]
    h, w = vehicle_crop.shape[:2]
    x1 = max(0, min(w - 1, int(round(w * _PLATE_REAR_ROI_X_MIN))))
    x2 = max(x1 + 1, min(w, int(round(w * _PLATE_REAR_ROI_X_MAX))))
    y1 = max(0, min(h - 1, int(round(h * _PLATE_REAR_ROI_Y_MIN))))
    y2 = max(y1 + 1, min(h, int(round(h * _PLATE_REAR_ROI_Y_MAX))))
    return vehicle_crop[y1:y2, x1:x2], [x1, y1, x2, y2]


def _fallback_plate_bbox_from_vehicle(vehicle_crop) -> List[int]:
    if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
        return [0, 0, 0, 0]
    h, w = vehicle_crop.shape[:2]
    x1 = max(0, min(w - 1, int(round(w * 0.22))))
    x2 = max(x1 + 1, min(w, int(round(w * 0.78))))
    y1 = max(0, min(h - 1, int(round(h * 0.56))))
    y2 = max(y1 + 1, min(h, int(round(h * 0.88))))
    return [x1, y1, x2, y2]


def _translate_plate_candidates(candidates: List[Dict[str, Any]], offset_x: int, offset_y: int) -> List[Dict[str, Any]]:
    translated: List[Dict[str, Any]] = []
    for item in candidates or []:
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        translated.append({
            **item,
            "bbox": [x1 + int(offset_x), y1 + int(offset_y), x2 + int(offset_x), y2 + int(offset_y)],
        })
    return translated


def _rank_plate_candidates(candidates: List[Dict[str, Any]], full_w: int, full_h: int) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    vehicle_area = float(max(1, full_w * full_h))
    for item in candidates or []:
        bbox = item.get("bbox") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = ((x1 + x2) / 2.0) / max(1.0, float(full_w))
        cy = ((y1 + y2) / 2.0) / max(1.0, float(full_h))
        area_ratio = float(bw * bh) / vehicle_area
        normalized_area = min(1.0, area_ratio / 0.06)
        # 車牌在車輛下方 (cy≈0.85)；中上方的疑似文字（如車身電話號碼）要被壓低
        lower_position_score = max(0.0, 1.0 - abs(cy - 0.85) / 0.22)
        # cy < 0.45 幾乎不可能是車牌 → 重罰
        upper_half_penalty = max(0.0, (0.45 - cy) / 0.45) if cy < 0.45 else 0.0
        center_x_score = max(0.0, 1.0 - abs(cx - 0.50) / 0.30)
        aspect = float(bw) / max(1.0, float(bh))
        plate_aspect_score = max(0.0, 1.0 - abs(aspect - 3.6) / 2.2)
        candidate_score = (
            0.28 * float(item.get("confidence") or 0.0)
            + 0.22 * normalized_area
            + 0.32 * lower_position_score
            + 0.10 * plate_aspect_score
            + 0.08 * center_x_score
            - 0.60 * upper_half_penalty
        )
        ranked.append({
            **item,
            "rank_score": float(candidate_score),
            "rank_area": float(normalized_area),
            "lower_position_score": float(lower_position_score),
            "center_x_score": float(center_x_score),
            "plate_aspect_score": float(plate_aspect_score),
        })

    ranked.sort(key=lambda item: float(item.get("rank_score") or 0.0), reverse=True)
    return ranked[:_PLATE_MAX_RANKED_CANDIDATES]


def _propose_plate_bboxes(vehicle_crop, max_candidates: int = 3) -> List[Dict[str, Any]]:
    if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
        return []
    try:
        h, w = vehicle_crop.shape[:2]
        if h < 40 or w < 80:
            return []

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY) if len(vehicle_crop.shape) == 3 else vehicle_crop
        proposals: List[Dict[str, Any]] = []
        scales = (
            (0.22, 0.09),
            (0.26, 0.11),
            (0.30, 0.12),
            (0.34, 0.14),
            (0.40, 0.16),
            (0.46, 0.18),
        )
        y_centers = (0.52, 0.60, 0.68, 0.76, 0.84)
        x_centers = (0.28, 0.40, 0.50, 0.60, 0.72)

        for width_ratio, height_ratio in scales:
            win_w = max(40, int(round(w * width_ratio)))
            win_h = max(14, int(round(h * height_ratio)))
            if win_w >= w or win_h >= h:
                continue
            for yc_ratio in y_centers:
                cy = int(round(h * yc_ratio))
                for xc_ratio in x_centers:
                    cx = int(round(w * xc_ratio))
                    x1 = max(0, min(w - 1, cx - (win_w // 2)))
                    y1 = max(0, min(h - 1, cy - (win_h // 2)))
                    x2 = min(w, x1 + win_w)
                    y2 = min(h, y1 + win_h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    patch = vehicle_crop[y1:y2, x1:x2]
                    score = _plate_feature_score(patch)
                    if score < 0.17:
                        continue
                    # 偏好車體下半部中央附近的候選。
                    center_bias = max(0.0, 1.0 - abs((cx / max(1.0, w)) - 0.5) / 0.38)
                    lower_bias = max(0.0, 1.0 - abs((cy / max(1.0, h)) - 0.72) / 0.26)
                    final_score = score + (center_bias * 0.12) + (lower_bias * 0.14)
                    proposals.append(
                        {
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(min(0.99, final_score)),
                            "backend": "vehicle-heuristic",
                        }
                    )

        proposals.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        kept: List[Dict[str, Any]] = []
        for item in proposals:
            bbox = item["bbox"]
            skip = False
            for prev in kept:
                ax1, ay1, ax2, ay2 = bbox
                bx1, by1, bx2, by2 = prev["bbox"]
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
                    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
                    iou = inter / max(1.0, area_a + area_b - inter)
                    if iou > 0.45:
                        skip = True
                        break
            if skip:
                continue
            kept.append(item)
            if len(kept) >= max_candidates:
                break
        return kept
    except Exception:
        return []


def _tighten_plate_crop_with_bbox(plate_crop):
    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
        return plate_crop, None
    try:
        h, w = plate_crop.shape[:2]
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = float(max(1, h * w))
        base_bbox = [0, 0, w, h]
        base_score = _plate_feature_score(plate_crop)
        best_patch = plate_crop
        best_bbox = base_bbox
        best_score = base_score

        def _update_candidate(x1: int, y1: int, x2: int, y2: int, bonus: float = 0.0):
            nonlocal best_patch, best_bbox, best_score
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w, int(x2)))
            y2 = max(0, min(h, int(y2)))
            if x2 <= x1 or y2 <= y1:
                return
            area_ratio = float((x2 - x1) * (y2 - y1)) / img_area
            if area_ratio < 0.10 or area_ratio > 0.98:
                return
            patch = plate_crop[y1:y2, x1:x2]
            score = _plate_feature_score(patch) + bonus
            # 只有明顯優於原框時才採用，避免把 detector 正確框縮壞。
            if score >= max(best_score + 0.05, 0.22):
                best_score = score
                best_patch = patch
                best_bbox = [x1, y1, x2, y2]

        # 候選 1：輪廓矩形（依車牌特徵評分）
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:25]:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 30 or ch < 12:
                continue
            area = float(cw * ch)
            if area < img_area * 0.03 or area > img_area * 0.92:
                continue
            aspect = cw / max(1.0, float(ch))
            if not (1.8 <= aspect <= 7.2):
                continue

            pad_x = int(cw * 0.03)
            pad_y = int(ch * 0.08)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            _update_candidate(x1, y1, x2, y2)

        # 候選 2：既有 flatten 偵測區
        roi, local_bbox = _flatten_plate_roi_with_bbox(plate_crop)
        if isinstance(local_bbox, (list, tuple)) and len(local_bbox) == 4:
            _update_candidate(local_bbox[0], local_bbox[1], local_bbox[2], local_bbox[3], bonus=0.03)

        if best_patch is not None and getattr(best_patch, "size", 0) > 0:
            return best_patch, best_bbox
    except Exception:
        pass
    return plate_crop, [0, 0, plate_crop.shape[1], plate_crop.shape[0]]


def _tighten_plate_crop(plate_crop):
    crop, _ = _tighten_plate_crop_with_bbox(plate_crop)
    return crop


def _enhance_plate_snapshot(plate_crop, upscale: bool = True):
    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
        return plate_crop
    try:
        out = plate_crop.copy()
        h, w = out.shape[:2]
        if h <= 0 or w <= 0:
            return plate_crop

        target_w = max(280, min(520, int(w * 2.2)))
        if upscale and w < target_w:
            scale = target_w / max(1.0, float(w))
            interp = cv2.INTER_LANCZOS4 if scale <= 3.2 else cv2.INTER_CUBIC
            out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=interp)

        # Keep color while improving local contrast on the luminance channel.
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # Daylight trailer scenes benefit from slightly stronger local contrast recovery.
        out = cv2.medianBlur(out, 3)
        blurred = cv2.GaussianBlur(out, (0, 0), 1.1)
        out = cv2.addWeighted(out, 1.30, blurred, -0.30, 0)
        return out
    except Exception:
        return plate_crop


def _expand_saved_plate_bbox(bbox: List[int], full_w: int, full_h: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = max(4, int(round(bw * _PLATE_CROP_PAD_X_RATIO)))
    pad_y = max(4, int(round(bh * _PLATE_CROP_PAD_Y_RATIO)))
    return [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(full_w, x2 + pad_x),
        min(full_h, y2 + pad_y),
    ]


def _plate_snapshot_quality_score(img) -> float:
    if img is None or getattr(img, "size", 0) == 0:
        return -1.0
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape[:2]
        if h < 10 or w < 24:
            return -1.0
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        stroke = float(np.mean(np.abs(sobel_x)))
        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))
        area_bonus = min(1.0, float(w * h) / 18000.0)
        aspect = w / max(1.0, float(h))
        aspect_bonus = max(0.0, 1.0 - abs(aspect - 3.2) / 3.2)
        brightness_bonus = 1.0 if _PLATE_BRIGHTNESS_MIN <= brightness <= _PLATE_BRIGHTNESS_MAX else max(
            0.0,
            1.0 - min(abs(brightness - _PLATE_BRIGHTNESS_MIN), abs(brightness - _PLATE_BRIGHTNESS_MAX)) / 90.0,
        )
        return (
            (lap_var * 0.040)
            + (stroke * 0.34)
            + (contrast * 0.18)
            + (area_bonus * 8.5)
            + (aspect_bonus * 3.8)
            + (brightness_bonus * 6.0)
        )
    except Exception:
        return -1.0


def _extract_native_plate_snapshot(frame, plate_bbox: List[int], full_w: int, full_h: int):
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    if not (isinstance(plate_bbox, (list, tuple)) and len(plate_bbox) == 4):
        return None
    sx1, sy1, sx2, sy2 = _expand_saved_plate_bbox([int(v) for v in plate_bbox], full_w, full_h)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    return frame[sy1:sy2, sx1:sx2].copy()


def _select_best_native_plate_snapshot(frames: List[Any], plate_bbox: List[int], full_w: int, full_h: int):
    best_img = None
    best_score = -1.0
    for frm in frames:
        crop = _extract_native_plate_snapshot(frm, plate_bbox, full_w, full_h)
        score = _plate_snapshot_quality_score(crop)
        if score > best_score:
            best_score = score
            best_img = crop
    return best_img


def _draw_plate_inset(snapshot, plate_crop, plate_text: str, conf: float) -> int:
    if snapshot is None or getattr(snapshot, "size", 0) == 0:
        return 0
    if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
        return 0
    plate_crop = _enhance_plate_snapshot(plate_crop)
    ih, iw = snapshot.shape[:2]
    ch, cw = plate_crop.shape[:2]
    if cw <= 0 or ch <= 0:
        return 0

    # 固定放在左上角，並以畫面可用區塊等比縮放，不再用 plate 自身倍率硬放大。
    max_w = max(120, int(iw * 0.28))
    max_h = max(40, int(ih * 0.18))
    scale = min(max_w / max(1.0, float(cw)), max_h / max(1.0, float(ch)))
    target_w = max(80, int(round(cw * scale)))
    target_h = max(24, int(round(ch * scale)))
    target_w = min(target_w, max(80, iw - 20))
    target_h = min(target_h, max(24, ih - 50))
    if target_w <= 0 or target_h <= 0:
        return 0

    inset = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    x1, y1 = 10, 10
    x2, y2 = x1 + target_w, y1 + target_h
    if x2 >= iw:
        x1 = max(2, iw - target_w - 2)
        x2 = x1 + target_w
    if y2 >= ih:
        y1 = max(2, ih - target_h - 30)
        y2 = y1 + target_h

    panel_bottom = min(ih - 2, y2 + 8)
    cv2.rectangle(snapshot, (max(0, x1 - 6), max(0, y1 - 6)), (min(iw - 1, x2 + 6), panel_bottom), (0, 0, 0), -1)
    snapshot[y1:y2, x1:x2] = inset
    cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return panel_bottom


def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from detection.vehicle_detector import VehicleDetector
        _yolo_model = VehicleDetector(conf_threshold=0.12)
    return _yolo_model


def get_plate_detector():
    global _plate_detector
    if _plate_detector is None:
        from recognition.plate_detector import PlateDetector
        _plate_detector = PlateDetector()
    return _plate_detector


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        from recognition.plate_recognizer import PlateRecognizer
        _recognizer = PlateRecognizer()
    return _recognizer


def _fmt_taipei(dt: datetime) -> str:
    if not dt:
        return ""
    # Existing DB rows are naive UTC; normalize then convert.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_taipei_iso(dt: datetime) -> str:
    if not dt:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(TZ_TAIPEI).isoformat(timespec="seconds")


def _get_lpr_camera_stat_row(db, camera_id: int) -> Optional[LPRCameraStat]:
    return db.query(LPRCameraStat).filter(LPRCameraStat.camera_id == int(camera_id)).first()


def _read_lpr_cumulative_stats(camera_id: int) -> dict:
    stats = {
        "total_frames": 0,
        "vehicles_detected": 0,
        "plate_boxes_detected": 0,
        "ocr_candidates_detected": 0,
        "vote_candidates_detected": 0,
        "confirmed_candidates": 0,
        "committed_candidates": 0,
        "last_history_at": "",
    }
    db = SessionLocal()
    try:
        row = _get_lpr_camera_stat_row(db, camera_id)
        if row:
            for field in _LPR_CUMULATIVE_COUNTER_FIELDS:
                stats[field] = int(getattr(row, field, 0) or 0)
            stats["last_history_at"] = _fmt_taipei_iso(getattr(row, "last_history_at", None))
        committed_count = (
            db.query(LPRRecord)
            .filter(LPRRecord.camera_id == int(camera_id))
            .count()
        )
        stats["committed_candidates"] = max(int(stats.get("committed_candidates", 0) or 0), int(committed_count or 0))
        latest_history_row = (
            db.query(LPRRecord.created_at)
            .filter(LPRRecord.camera_id == int(camera_id))
            .order_by(LPRRecord.created_at.desc(), LPRRecord.id.desc())
            .first()
        )
        latest_history_at = latest_history_row[0] if latest_history_row else None
        if latest_history_at:
            stats["last_history_at"] = _fmt_taipei_iso(latest_history_at)
        return stats
    finally:
        db.close()


class LPRStreamTask:
    """LPR 串流辨識"""
    
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    def __init__(self, camera_id: int, source: str, camera_name: str):
        self.camera_id = camera_id
        self.source = source
        self.camera_name = camera_name
        self.running = False
        self.thread = None
        self.results = []
        self.max_results = 300
        self.last_plates = {}
        self.last_committed_plates = {}
        self.last_plate_number = None
        self.last_candidate_plate = None
        self.last_confirmed_plate = None
        self.last_rejected_plate = None
        self.last_rejected_reason = None
        self.cooldown = _PLATE_COMMIT_COOLDOWN_SEC
        self.total_frames = 0
        self.total_detections = 0
        self.vehicles_detected = 0
        self.plate_boxes_detected = 0
        self.ocr_candidates_detected = 0
        self.vote_candidates_detected = 0
        self.confirmed_candidates = 0
        self.committed_candidates = 0
        self.zones = []
        self.last_error = None
        self.last_frame_at = 0.0
        self.started_at = time.time()
        self.recent_frames = deque(maxlen=10)
        self.pending_plate_votes: Dict[str, Dict[str, Any]] = {}
        self.track_plate_candidates: Dict[int, Dict[str, Any]] = {}
        self.vehicle_tracker = VehicleTracker(max_age=8, iou_threshold=0.2)
        self.vehicle_track_states: Dict[int, Dict[str, Any]] = {}
        self._stats_lock = threading.Lock()
        self._stats_pending = {field: 0 for field in _LPR_CUMULATIVE_COUNTER_FIELDS}
        self._stats_last_flush = 0.0

    def _increment_debug_counter(self, field: str, delta: int = 1) -> None:
        if not hasattr(self, field):
            return
        next_value = int(getattr(self, field, 0) or 0) + int(delta)
        setattr(self, field, next_value)
        if field in self._stats_pending:
            with self._stats_lock:
                self._stats_pending[field] = int(self._stats_pending.get(field, 0) or 0) + int(delta)
            self._flush_cumulative_stats()

    def _pending_stat_delta(self, field: str) -> int:
        with self._stats_lock:
            return int(self._stats_pending.get(field, 0) or 0)

    def _flush_cumulative_stats(self, force: bool = False, *, latest_history_at: Optional[datetime] = None) -> None:
        now = time.time()
        with self._stats_lock:
            has_pending = any(int(v or 0) for v in self._stats_pending.values()) or latest_history_at is not None
            if not has_pending:
                return
            if not force and (now - self._stats_last_flush) < 1.0:
                return
            pending = {field: int(self._stats_pending.get(field, 0) or 0) for field in _LPR_CUMULATIVE_COUNTER_FIELDS}
            self._stats_pending = {field: 0 for field in _LPR_CUMULATIVE_COUNTER_FIELDS}
            self._stats_last_flush = now
        db = SessionLocal()
        try:
            row = _get_lpr_camera_stat_row(db, self.camera_id)
            if not row:
                row = LPRCameraStat(camera_id=self.camera_id, camera_name=self.camera_name)
            row.camera_name = self.camera_name
            for field, delta in pending.items():
                if delta:
                    setattr(row, field, int(getattr(row, field, 0) or 0) + int(delta))
            if latest_history_at is not None:
                row.last_history_at = latest_history_at
            row.updated_at = datetime.utcnow()
            db.add(row)
            db.commit()
        except Exception as e:
            self.last_error = f"stats_flush_failed: {e}"
            with self._stats_lock:
                for field, delta in pending.items():
                    if delta:
                        self._stats_pending[field] = int(self._stats_pending.get(field, 0) or 0) + int(delta)
        finally:
            db.close()

    def _save_record_db(self, record: dict):
        try:
            from api.models import SessionLocal, LPRRecord
            db = SessionLocal()
            try:
                row = LPRRecord(
                    camera_id=self.camera_id,
                    camera_name=self.camera_name,
                    plate_number=record.get("plate_number"),
                    confidence=float(record.get("confidence", 0)),
                    valid=bool(record.get("valid", False)),
                    vehicle_type=record.get("vehicle_type"),
                    snapshot=record.get("snapshot"),
                    raw=record.get("raw"),
                    created_at=datetime.utcnow(),
                )
                db.add(row)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            self.last_error = f"db_write_failed: {e}"

    def _store_history_record(
        self,
        *,
        frame,
        vehicle_bbox: List[int],
        vehicle_type: str,
        plate_number: Optional[str] = None,
        confidence: float = 0.0,
        valid: bool = False,
        raw: str = "",
        plate_bbox: Optional[List[int]] = None,
        plate_crop=None,
    ) -> dict:
        self.total_detections += 1
        plate_text = str(plate_number or "UNKNOWN").strip() or "UNKNOWN"
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        safe_plate = re.sub(r"[^A-Z0-9]+", "", plate_text.upper()) or "UNKNOWN"
        snapshot_name = f"{timestamp}_{safe_plate}_{self.total_detections}.jpg"
        snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_name)
        plate_snapshot_name = f"{timestamp}_{safe_plate}_{self.total_detections}_plate.png"
        plate_snapshot_path = os.path.join(SNAPSHOT_DIR, plate_snapshot_name)

        snapshot = frame.copy()
        vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]
        cv2.rectangle(snapshot, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)

        saved_plate_snapshot = None
        if isinstance(plate_bbox, (list, tuple)) and len(plate_bbox) == 4:
            px1, py1, px2, py2 = [int(v) for v in plate_bbox]
            cv2.rectangle(snapshot, (px1, py1), (px2, py2), (0, 255, 255), 2)
            if plate_crop is not None and getattr(plate_crop, "size", 0) > 0:
                # 存攤平+二值化的截圖（顯示用）
                try:
                    from recognition.plate_recognizer import PlateRecognizer
                    _display_crop = PlateRecognizer.enhance_plate_static(plate_crop)
                except Exception:
                    _display_crop = plate_crop
                cv2.imwrite(
                    plate_snapshot_path,
                    _display_crop,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                )
                saved_plate_snapshot = plate_snapshot_name
                if plate_text != "UNKNOWN":
                    _draw_plate_inset(snapshot, plate_crop, plate_text, float(confidence or 0.0))

        cv2.imwrite(snapshot_path, snapshot, [cv2.IMWRITE_JPEG_QUALITY, 96])

        record = {
            'id': self.total_detections,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'plate_number': plate_text,
            'confidence': round(float(confidence or 0.0), 3),
            'valid': bool(valid),
            'vehicle_type': vehicle_type,
            'camera_id': self.camera_id,
            'camera_name': self.camera_name,
            'snapshot': snapshot_name,
            'plate_snapshot': saved_plate_snapshot,
            'plate_bbox': list(plate_bbox) if isinstance(plate_bbox, (list, tuple)) and len(plate_bbox) == 4 else None,
            'raw': raw or ('vehicle_only' if plate_text == "UNKNOWN" else ""),
        }
        self.results.insert(0, record)
        if len(self.results) > self.max_results:
            self.results.pop()
        self._save_record_db(record)
        return record

    def _update_vehicle_track_state(self, track_id: int, vehicle_type: str, bbox: List[int], frame) -> None:
        state = self.vehicle_track_states.get(track_id)
        now_ts = time.time()
        bbox_list = [int(v) for v in bbox]
        if state is None:
            state = {
                "vehicle_type": vehicle_type,
                "bbox": bbox_list,
                "frame": frame.copy(),
                "first_seen": now_ts,
                "last_seen": now_ts,
                "seen_frames": 1,
                "plate_recorded": False,
                "unknown_recorded": False,
            }
            self.vehicle_track_states[track_id] = state
            return
        state["vehicle_type"] = vehicle_type
        state["bbox"] = bbox_list
        state["frame"] = frame.copy()
        state["last_seen"] = now_ts
        state["seen_frames"] = int(state.get("seen_frames", 0)) + 1

    def _mark_vehicle_track_plate_recorded(self, track_id: Optional[int]) -> None:
        if track_id is None:
            return
        state = self.vehicle_track_states.get(int(track_id))
        if state is not None:
            state["plate_recorded"] = True

    def _flush_inactive_vehicle_tracks(self, *, force: bool = False) -> None:
        active_track_ids = set(int(track_id) for track_id in self.vehicle_tracker.tracks.keys())
        stale_ids = [
            int(tid) for tid in self.vehicle_track_states
            if force or int(tid) not in active_track_ids
        ]
        for track_id in stale_ids:
            self.vehicle_track_states.pop(track_id, None)

    def _plate_vote_bucket(self, bbox: List[int], track_id: Optional[int] = None) -> str:
        if track_id is not None and int(track_id) > 0:
            return f"track:{int(track_id)}"
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        return f"{cx // _PLATE_VOTE_BUCKET_SIZE}:{cy // _PLATE_VOTE_BUCKET_SIZE}"

    def _cleanup_plate_votes(self, now_ts: float) -> None:
        expired = []
        for key, item in self.pending_plate_votes.items():
            if (now_ts - float(item.get("last_seen", 0.0))) > _PLATE_VOTE_TTL_SEC:
                expired.append(key)
        for key in expired:
            self.pending_plate_votes.pop(key, None)

    def _syntax_score(self, plate: str, valid: bool = False) -> float:
        score = self._plate_layout_score(plate)
        if valid:
            score += 1.2
        if re.match(r"^[A-Z]{2,4}-[0-9]{2,4}$", plate):
            score += 0.8
        elif re.match(r"^[0-9]{2,4}-[A-Z]{2,4}$", plate):
            score += 0.45
        return max(0.0, score)

    def _vote_weight(self, *, conf: float, det_conf: float, quality_score: float, syntax_score: float, center_score: float) -> float:
        return max(
            0.1,
            (0.40 * float(conf or 0.0))
            + (0.20 * float(det_conf or 0.0))
            + (0.15 * float(quality_score or 0.0))
            + (0.15 * min(1.0, float(syntax_score or 0.0) / 3.0))
            + (0.10 * float(center_score or 0.0)),
        )

    def _history_plate_gate(self, plate: str, *, conf: float, quality_score: float, vote_count: int) -> bool:
        normalized = self._normalize_plate_candidate(plate)
        core = normalized.replace("-", "")
        if len(core) < 4:
            return False
        # 所有合法台灣車牌格式統一門檻
        tw_patterns = [
            r"^[A-Z]{2,4}-\d{2,4}$",  # ABC-1234, AB-12, ABCD-12
            r"^\d{2,4}-[A-Z]{2,4}$",  # 1234-AB, 12-ABCD
            r"^[A-Z]\d{2}-\d{3,4}$",  # A12-3456
            r"^[A-Z]{3}-\d{4}$",      # 新式: ABC-1234
            r"^\d{4}-[A-Z]{2}$",      # 舊式: 1234-AB
            r"^[A-Z]{2}\d-\d{4}$",    # 特殊: AB1-2345
            r"^[A-Z]\d-\d{4}$",       # 營業: A1-2345
            r"^\d{3}-[A-Z]{2,3}$",    # 舊式機車: 123-AB, 123-ABC
        ]
        for pat in tw_patterns:
            if re.match(pat, normalized):
                return True  # 格式合法直接通過（OCR 品質由 _effective_plate_conf 把關）
        return False

    def _reject_commit(self, plate: str, reason: str) -> bool:
        self.last_rejected_plate = str(plate or "") or None
        self.last_rejected_reason = reason
        return False

    def _should_commit_plate(
        self,
        plate: str,
        vote_score: float,
        raw: str,
        *,
        conf: float,
        valid: bool,
        quality_score: float,
        vote_count: int,
    ) -> bool:
        raw_text = str(raw or "").strip().lower()
        if not plate or plate == "UNKNOWN" or raw_text == "vehicle_only":
            return self._reject_commit(plate, "empty_or_unknown")
        if not self._is_plausible_plate(plate):
            return self._reject_commit(plate, "implausible")
        if not self._history_plate_gate(plate, conf=conf, quality_score=quality_score, vote_count=vote_count):
            return self._reject_commit(plate, "history_gate")
        normalized = self._normalize_plate_candidate(plate)
        standard_pattern = bool(re.match(r"^[A-Z]{2,4}-\d{2,4}$", normalized))
        if not valid and not (standard_pattern and float(conf or 0.0) >= 0.30):
            return self._reject_commit(plate, "invalid")
        if float(vote_score or 0.0) < _PLATE_COMMIT_MIN_SCORE:
            return self._reject_commit(plate, "vote_score")
        if float(conf or 0.0) < _PLATE_COMMIT_MIN_CONF:
            return self._reject_commit(plate, "confidence")
        if float(quality_score or 0.0) < 0.05 and float(conf or 0.0) < 0.35:
            return self._reject_commit(plate, "quality")
        if int(vote_count or 0) < 2 and float(conf or 0.0) < 0.30:
            return self._reject_commit(plate, "vote_count")
        self.last_rejected_plate = None
        self.last_rejected_reason = None
        return True

    # === best-of-track 追蹤式 commit ===

    def _update_track_plate_candidate(
        self,
        track_id: int,
        plate: str,
        effective_conf: float,
        valid: bool,
        raw: str,
        quality_score: float,
        plate_bbox: List[int],
        vehicle_bbox: List[int],
        vehicle_type: str,
        frame,
        plate_crop,
        iw: int,
        ih: int,
    ) -> None:
        """累積 track 的所有 OCR 結果，做字元級投票選最佳。"""
        if track_id is None or track_id <= 0:
            return
        now = time.time()
        entry = self.track_plate_candidates.get(track_id)
        if entry is None:
            entry = {
                "track_id": track_id,
                "vehicle_type": vehicle_type,
                "first_seen": now,
                "last_seen": now,
                "best_plate": plate,
                "best_conf": effective_conf,
                "best_valid": valid,
                "best_raw": raw,
                "best_quality_score": quality_score,
                "best_frame": frame.copy() if frame is not None else None,
                "best_plate_crop": plate_crop.copy() if plate_crop is not None and hasattr(plate_crop, 'copy') else plate_crop,
                "best_plate_bbox": list(plate_bbox),
                "best_vehicle_bbox": list(vehicle_bbox),
                "iw": iw,
                "ih": ih,
                "ocr_count": 1,
                "committed": False,
                "all_raws": [],       # 所有 OCR 原始結果
                "char_slots": {},     # 字元投票 {length: [{char: weight, ...}, ...]}
            }
            self.track_plate_candidates[track_id] = entry
        else:
            entry["last_seen"] = now
            entry["ocr_count"] += 1

        # 累積所有原始 OCR 結果用於字元投票
        raw_clean = re.sub(r"[^A-Z0-9]", "", str(raw or "").upper())
        if raw_clean and len(raw_clean) >= 4:
            entry["all_raws"].append({"raw": raw_clean, "conf": effective_conf, "quality": quality_score})
            # 字元級投票：按 raw 長度分組，每個位置投票
            length = len(raw_clean)
            weight = effective_conf + quality_score * 0.3
            slots = entry["char_slots"].setdefault(length, [dict() for _ in range(length)])
            if len(slots) == length:
                for idx, ch in enumerate(raw_clean):
                    slots[idx][ch] = slots[idx].get(ch, 0.0) + weight

        # 保留最高 confidence+quality 的結果（作為 fallback）
        new_score = effective_conf + quality_score * 0.3
        old_score = entry["best_conf"] + entry.get("best_quality_score", 0) * 0.3
        if new_score > old_score:
            entry["best_plate"] = plate
            entry["best_conf"] = effective_conf
            entry["best_valid"] = valid
            entry["best_raw"] = raw
            entry["best_quality_score"] = quality_score
            entry["best_frame"] = frame.copy() if frame is not None else None
            entry["best_plate_crop"] = plate_crop.copy() if plate_crop is not None and hasattr(plate_crop, 'copy') else plate_crop
            entry["best_plate_bbox"] = list(plate_bbox)
            entry["best_vehicle_bbox"] = list(vehicle_bbox)
            entry["iw"] = iw
            entry["ih"] = ih

    def _resolve_track_best_plate(self, entry: dict) -> Optional[str]:
        """從 track 的字元投票中組合最佳車牌。"""
        char_slots = entry.get("char_slots", {})
        if not char_slots:
            return None
        # 選出票數最多的長度
        best_length = max(char_slots.keys(), key=lambda L: sum(
            max(s.values()) for s in char_slots[L] if s
        ) if char_slots[L] else 0)
        slots = char_slots.get(best_length, [])
        if not slots:
            return None
        # 每個位置取最高票的字元
        voted_chars = []
        total_weight = 0.0
        for slot in slots:
            if not slot:
                return None
            best_ch = max(slot, key=slot.get)
            voted_chars.append(best_ch)
            total_weight += slot[best_ch]
        voted_raw = "".join(voted_chars)
        if len(voted_raw) < 4:
            return None
        # 用 _plate_variants 格式化成標準車牌
        variants = self._plate_variants(voted_raw)
        if not variants:
            return None
        # 從 variants 中選 layout score 最高的
        best_variant = max(variants, key=lambda v: self._plate_layout_score(v))
        return best_variant

    def _flush_completed_tracks(self, *, force: bool = False) -> None:
        """追蹤結束的 track，commit 最佳結果。"""
        now = time.time()
        active_ids = set()
        try:
            active_ids = set(int(t) for t in self.vehicle_tracker.tracks.keys())
        except Exception:
            pass

        to_remove = []
        for tid, entry in self.track_plate_candidates.items():
            if entry.get("committed"):
                # 已 commit，等結束後清理
                if tid not in active_ids or force:
                    to_remove.append(tid)
                continue
            # 判斷追蹤是否結束
            # track 結束：離開畫面 OR 超過 TTL OR 追蹤超過 15 秒
            track_ended = (tid not in active_ids) or ((now - entry["last_seen"]) > _TRACK_PLATE_TTL_SEC) or ((now - entry["first_seen"]) > 15.0) or force
            if not track_ended:
                continue
            _bp = entry.get("best_plate", "?")
            _oc = entry.get("ocr_count", 0)
            # OCR 次數不足，不 commit
            if entry["ocr_count"] < _TRACK_MIN_OCR_COUNT:
                print(f"[LPR-flush] track={tid} plate={_bp} REJECT ocr_count={_oc}", flush=True)
                to_remove.append(tid)
                continue
            # 車牌截圖品質過濾
            _crop = entry.get("best_plate_crop")
            if _crop is not None and hasattr(_crop, 'shape') and len(_crop.shape) >= 2:
                _ch, _cw = _crop.shape[:2]
                _aspect = _cw / max(_ch, 1)
                if _cw < 40 or _ch < 15 or _cw > 400 or _ch > 200 or _aspect > 4.5 or _aspect < 1.5:
                    print(f"[LPR-flush] track={tid} plate={_bp} REJECT size={_cw}x{_ch} aspect={_aspect:.1f}", flush=True)
                    to_remove.append(tid)
                    continue
                _gray = cv2.cvtColor(_crop, cv2.COLOR_BGR2GRAY) if len(_crop.shape) == 3 else _crop
                _edges = cv2.Canny(_gray, 50, 150)
                _edge_density = float(_edges.sum()) / max(1, _cw * _ch * 255)
                if _edge_density < 0.08:
                    print(f"[LPR-flush] track={tid} plate={_bp} REJECT edges={_edge_density:.3f}", flush=True)
                    to_remove.append(tid)
                    continue
            # 直接用最高 confidence 的 OCR 結果（不做字元投票）
            plate = entry.get("best_plate")
            conf = float(entry.get("best_conf", 0))
            valid = entry.get("best_valid", False)
            raw = entry.get("best_raw", "")
            # (字元投票已停用)
            quality_score = float(entry.get("best_quality_score", 0))
            if not plate or plate == "UNKNOWN":
                print(f"[LPR-flush] track={tid} REJECT empty plate", flush=True)
                to_remove.append(tid)
                continue
            # 格式驗證
            if not self._is_plausible_plate(plate):
                print(f"[LPR-flush] track={tid} plate={plate} REJECT implausible", flush=True)
                to_remove.append(tid)
                continue
            if not self._history_plate_gate(plate, conf=conf, quality_score=quality_score, vote_count=entry["ocr_count"]):
                to_remove.append(tid)
                continue
            normalized = self._normalize_plate_candidate(plate)
            standard_pattern = bool(re.match(r"^[A-Z]{2,4}-\d{2,4}$", normalized))
            if not valid and not (standard_pattern and conf >= 0.30):
                to_remove.append(tid)
                continue
            if conf < _PLATE_COMMIT_MIN_CONF:
                to_remove.append(tid)
                continue
            # cooldown 檢查（車牌 + 位置區域，避免同車重複 commit）
            cooldown_key = plate
            bbox = entry.get("best_plate_bbox", [0, 0, 0, 0])
            area_key = f"{bbox[0]//200}:{bbox[1]//200}"
            if cooldown_key in self.last_committed_plates and (now - self.last_committed_plates[cooldown_key]) <= self.cooldown:
                to_remove.append(tid)
                continue
            if area_key in self.last_committed_plates and (now - self.last_committed_plates[area_key]) <= 5.0:
                to_remove.append(tid)
                continue
            # commit
            self.last_committed_plates[cooldown_key] = now
            self.last_committed_plates[area_key] = now
            frame = entry.get("best_frame")
            plate_crop = entry.get("best_plate_crop")
            self._store_history_record(
                frame=frame,
                vehicle_bbox=entry.get("best_vehicle_bbox", [0, 0, 0, 0]),
                vehicle_type=entry.get("vehicle_type", "car"),
                plate_number=plate,
                confidence=conf,
                valid=valid,
                raw=raw,
                plate_bbox=entry.get("best_plate_bbox"),
                plate_crop=plate_crop,
            )
            entry["committed"] = True
            self.committed_candidates += 1
            self._increment_debug_counter("committed_candidates")
            self.last_confirmed_plate = plate
            self.last_plate_number = plate
            self._mark_vehicle_track_plate_recorded(tid)
            to_remove.append(tid)

        for tid in to_remove:
            self.track_plate_candidates.pop(tid, None)
        # 防止記憶體洩漏：超過 200 entries 時清理最舊的
        if len(self.track_plate_candidates) > 200:
            oldest = sorted(self.track_plate_candidates.keys(), key=lambda t: self.track_plate_candidates[t].get("first_seen", 0))
            for tid in oldest[:50]:
                self.track_plate_candidates.pop(tid, None)

    def _register_plate_vote(
        self,
        plate: str,
        conf: float,
        valid: bool,
        raw: str,
        plate_bbox: List[int],
        track_id: Optional[int],
        vehicle_type: str,
        frame,
        plate_crop,
        iw: int,
        ih: int,
        det_conf: float = 0.0,
        quality_score: float = 0.0,
        center_score: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        now_ts = time.time()
        self._cleanup_plate_votes(now_ts)
        bucket = self._plate_vote_bucket(plate_bbox, track_id=track_id)
        entry = self.pending_plate_votes.get(bucket)
        if entry is None:
            entry = {
                "last_seen": now_ts,
                "votes": {},
                "char_votes": {},
                "plate_bbox": list(plate_bbox),
                "vehicle_type": vehicle_type,
                "frame": frame.copy(),
                "plate_crop": plate_crop.copy() if plate_crop is not None and getattr(plate_crop, "size", 0) > 0 else None,
                "iw": iw,
                "ih": ih,
            }
            self.pending_plate_votes[bucket] = entry

        entry["last_seen"] = now_ts
        entry["plate_bbox"] = list(plate_bbox)
        entry["vehicle_type"] = vehicle_type
        entry["frame"] = frame.copy()
        entry["plate_crop"] = plate_crop.copy() if plate_crop is not None and getattr(plate_crop, "size", 0) > 0 else None
        entry["iw"] = iw
        entry["ih"] = ih

        syntax_score = self._syntax_score(plate, valid)
        vote_weight = self._vote_weight(
            conf=conf,
            det_conf=det_conf,
            quality_score=quality_score,
            syntax_score=syntax_score,
            center_score=center_score,
        )
        vote = entry["votes"].setdefault(
            plate,
            {"count": 0, "best_conf": 0.0, "best_quality": 0.0, "valid": False, "raw": raw, "score": 0.0, "syntax_score": 0.0},
        )
        vote["count"] += 1
        vote["best_conf"] = max(float(vote.get("best_conf") or 0.0), float(conf or 0.0))
        vote["best_quality"] = max(float(vote.get("best_quality") or 0.0), float(quality_score or 0.0))
        vote["valid"] = bool(vote.get("valid")) or bool(valid)
        vote["score"] = float(vote.get("score") or 0.0) + vote_weight
        vote["syntax_score"] = max(float(vote.get("syntax_score") or 0.0), syntax_score)
        if raw and float(conf or 0.0) >= float(vote.get("best_conf") or 0.0):
            vote["raw"] = raw
        self._register_char_votes(entry, plate, conf, valid, vote_weight)

        best_plate = ""
        best_score = -1.0
        best_vote = None
        for cand, item in entry["votes"].items():
            vote_score = float(item.get("score") or 0.0)
            vote_score += min(1.6, float(item.get("count") or 0) * 0.45)
            vote_score += min(0.8, float(item.get("best_conf") or 0.0))
            vote_score += min(0.8, float(item.get("syntax_score") or 0.0) * 0.18)
            if vote_score > best_score:
                best_score = vote_score
                best_plate = cand
                best_vote = item

        if not best_plate or best_vote is None:
            return None
        vote_count = int(best_vote.get("count") or 0)
        best_conf = float(best_vote.get("best_conf") or 0.0)
        vote_valid = bool(best_vote.get("valid"))
        char_voted = self._compose_char_vote_candidate(entry)
        if char_voted and (not self._is_plausible_plate(best_plate) or float(char_voted.get("score") or 0.0) > best_score + 0.12):
            best_plate = str(char_voted.get("plate") or best_plate)
            best_score = max(best_score, float(char_voted.get("score") or 0.0))
        enough_votes = vote_count >= _PLATE_CONFIRM_MIN_COUNT or best_score >= _PLATE_CONFIRM_MIN_SCORE
        strong_single = vote_valid and best_conf >= _PLATE_VOTE_STRONG_SINGLE_CONF
        very_strong_single = best_conf >= _PLATE_VOTE_VERY_STRONG_SINGLE_CONF
        if not (enough_votes or strong_single or very_strong_single):
            return None

        self.pending_plate_votes.pop(bucket, None)
        self.last_confirmed_plate = best_plate
        self._increment_debug_counter("confirmed_candidates")
        return {
            "plate": best_plate,
            "confidence": float(best_vote.get("best_conf") or 0.0),
            "valid": bool(best_vote.get("valid")),
            "raw": str(best_vote.get("raw") or ""),
            "plate_bbox": list(entry["plate_bbox"]),
            "vehicle_type": str(entry.get("vehicle_type") or vehicle_type),
            "frame": entry.get("frame"),
            "plate_crop": entry.get("plate_crop"),
            "iw": int(entry.get("iw") or iw),
            "ih": int(entry.get("ih") or ih),
            "vote_count": vote_count,
            "vote_score": float(best_score),
            "quality_score": float(best_vote.get("best_quality") or 0.0),
            "confirmed": True,
        }

    def _register_char_votes(self, entry: Dict[str, Any], plate: str, conf: float, valid: bool, weight: float) -> None:
        normalized = self._normalize_plate_candidate(plate)
        if not normalized:
            return
        length = len(normalized)
        slots = entry.setdefault("char_votes", {}).setdefault(length, [dict() for _ in range(length)])
        weight = max(weight, (float(conf or 0.0) * 0.6) + (0.4 if valid else 0.0))
        for idx, ch in enumerate(normalized):
            slot = slots[idx]
            slot[ch] = float(slot.get(ch, 0.0)) + weight

    def _compose_char_vote_candidate(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        char_votes = entry.get("char_votes") or {}
        best_plate = ""
        best_score = -10.0
        for _length, slots in char_votes.items():
            chars = []
            weight_sum = 0.0
            for slot in slots:
                if not slot:
                    chars = []
                    break
                ch, score = max(slot.items(), key=lambda item: float(item[1]))
                chars.append(ch)
                weight_sum += float(score)
            if not chars:
                continue
            plate = self._normalize_plate_candidate("".join(chars))
            if not self._is_plausible_plate(plate):
                continue
            score = self._plate_layout_score(plate) + (weight_sum / max(1, len(chars)))
            if score > best_score:
                best_score = score
                best_plate = plate
        if not best_plate:
            return None
        return {"plate": best_plate, "score": best_score}

    def _normalize_plate_candidate(self, plate: str) -> str:
        if not plate:
            return ""
        t = plate.strip().upper().replace(" ", "")
        t = re.sub(r"[^A-Z0-9-]", "", t)
        return t

    def _plate_variants(self, plate: str) -> List[str]:
        plate = self._normalize_plate_candidate(plate)
        if not plate:
            return []

        to_digit = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
        to_alpha = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
        variants = set()
        variants.add(plate)

        plain = plate.replace("-", "")
        if 4 <= len(plain) <= 9:
            for cut in (2, 3, 4):
                if 2 <= cut < len(plain) - 1:
                    variants.add(f"{plain[:cut]}-{plain[cut:]}")
                    left = plain[:cut]
                    right = plain[cut:]
                    variants.add(f"{''.join(to_alpha.get(ch, ch) for ch in left)}-{''.join(to_digit.get(ch, ch) for ch in right)}")
                    variants.add(f"{''.join(to_digit.get(ch, ch) for ch in left)}-{''.join(to_alpha.get(ch, ch) for ch in right)}")

        for cand in list(variants):
            if "-" not in cand:
                continue
            left, right = cand.split("-", 1)
            fixed = f"{''.join(to_alpha.get(ch, ch) for ch in left)}-{''.join(to_digit.get(ch, ch) for ch in right)}"
            variants.add(fixed)

        out = []
        for v in variants:
            n = self._normalize_plate_candidate(v)
            core_len = len(n.replace("-", ""))
            if 4 <= core_len <= 9:
                out.append(n)
        return out

    def _plate_layout_score(self, plate: str) -> float:
        plate = self._normalize_plate_candidate(plate)
        if not plate:
            return -3.0
        plain = plate.replace("-", "")
        if len(plain) < 4 or len(plain) > 9:
            return -2.5

        digits = sum(ch.isdigit() for ch in plain)
        letters = sum("A" <= ch <= "Z" for ch in plain)
        score = 0.0

        if digits == 0:
            return -3.0
        if digits == 1:
            score -= 1.5
        elif digits >= 2:
            score += 0.8

        if letters == 0:
            score -= 1.2
        else:
            score += 0.4

        if re.match(r"^[A-Z]{2,4}-[0-9]{2,4}$", plate):
            score += 1.4
        elif re.match(r"^[0-9]{2,4}-[A-Z]{2,4}$", plate):
            score += 0.9

        if "-" in plate:
            left, right = plate.split("-", 1)
            left_letters = sum("A" <= ch <= "Z" for ch in left)
            left_digits = sum(ch.isdigit() for ch in left)
            right_letters = sum("A" <= ch <= "Z" for ch in right)
            right_digits = sum(ch.isdigit() for ch in right)
            left_minor = min(left_letters, left_digits)
            right_minor = min(right_letters, right_digits)
            if left_letters >= max(1, len(left) - 1) and right_digits >= max(1, len(right) - 1):
                score += 0.9
            elif left_digits >= max(1, len(left) - 1) and right_letters >= max(1, len(right) - 1):
                score += 0.45
            else:
                score -= 0.55
            if left_minor:
                score -= 0.45 * left_minor
            if right_minor:
                score -= 0.45 * right_minor
            if left_minor > 1:
                score -= 0.8
            if right_minor > 1:
                score -= 0.8
            if left_digits == len(left) and right_digits == len(right):
                score -= 1.1
            if left_letters == len(left) and right_letters == len(right):
                score -= 1.1

        if re.search(r"([A-Z0-9])\1\1", plain):
            score -= 1.2
        if len(set(plain)) <= 2:
            score -= 1.0
        return score

    def _is_plausible_plate(self, plate: str) -> bool:
        # 台灣車牌字元總數（去掉 dash）至少 6 字：XX-1234 / 1234-XX / ABC-1234 / LAB-123
        n = self._normalize_plate_candidate(plate)
        plain = n.replace("-", "")
        if len(plain) < 6:
            return False
        digits = sum(ch.isdigit() for ch in plain)
        letters = sum("A" <= ch <= "Z" for ch in plain)
        # 全字母或全數字、或字母/數字少於 2 個都拒絕
        if digits < 2 or letters < 2:
            return False
        return self._plate_layout_score(plate) >= -0.5

    def _score_ocr_result(self, result: Dict[str, Any], plate: str) -> float:
        conf = float(result.get("confidence") or 0.0)
        valid = bool(result.get("valid", False))
        score = (conf * 1.5) + self._plate_layout_score(plate)
        if valid:
            score += 2.5
        if re.match(r"^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$", plate):
            score += 0.3
        if re.match(r"^[A-Z]{2,4}-[0-9]{2,4}$", plate):
            score += 0.5
        return score

    def _effective_plate_conf(
        self,
        plate: str,
        conf: float,
        *,
        valid: bool,
        quality_score: float,
        raw: str,
    ) -> float:
        """回傳真實 OCR 信心度，不再硬塞 0.58 防止假陽性"""
        base_conf = max(0.0, float(conf or 0.0))
        # 檢查 plate 與 raw 的相似度（避免 raw="ABUTS" 變成 plate="ABO-75" 還拿到高 conf）
        normalized = self._normalize_plate_candidate(plate)
        plain = normalized.replace("-", "")
        raw_clean = re.sub(r"[^A-Z0-9]", "", str(raw or "").upper())

        if not raw_clean or not plain:
            return base_conf

        # 計算 raw 與 plate 的字元相似度
        common = sum(1 for ch in plain if ch in raw_clean)
        similarity = common / max(1, len(plain))

        # 相似度太低（OCR 讀的跟最終 plate 對不上）→ 不給信心度加成
        if similarity < 0.4:
            return min(base_conf, 0.20)

        if valid and similarity >= 0.8:
            return max(base_conf, 0.45)

        return base_conf

    def _merge_crop_ocr(self, base: Dict[str, Any], crop_res: Dict[str, Any]) -> Dict[str, Any]:
        if not crop_res:
            return base
        out = dict(base or {})
        current_plate = self._normalize_plate_candidate(out.get("plate_number") or "")
        current_score = self._score_ocr_result(out, current_plate) if current_plate else -1.0

        candidates = set()
        candidates.update(self._plate_variants(crop_res.get("plate_number") or ""))
        candidates.update(self._plate_variants(crop_res.get("raw") or ""))
        for cand in candidates:
            score = self._score_ocr_result(crop_res, cand)
            if score > (current_score + 0.05):
                current_score = score
                out["plate_number"] = cand
                out["confidence"] = float(crop_res.get("confidence") or 0.0)
                out["valid"] = bool(crop_res.get("valid", False))
                out["raw"] = crop_res.get("raw", "")
        return out

    def _recognize_plate_best(self, frame, x1: int, y1: int, x2: int, y2: int, recognizer) -> Dict[str, Any]:
        best: Dict[str, Any] = {
            "plate_number": None,
            "confidence": 0.0,
            "valid": False,
            "raw": "",
            "plate_bbox": None,
            "det_conf": 0.0,
            "center_score": 0.0,
            "quality_score": 0.0,
            "primary_plate_area": 0.0,
            "secondary_plate_exists": False,
            "secondary_plate_text": "",
            "selected_candidate": "",
            "fallback_only": False,
            "_score": -1.0,
        }
        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop is None or getattr(vehicle_crop, "size", 0) == 0:
            return best

        detector = get_plate_detector()
        vh, vw = vehicle_crop.shape[:2]
        fallback_bbox = _fallback_plate_bbox_from_vehicle(vehicle_crop)
        detections: List[Dict[str, Any]] = []
        if _PLATE_REAR_ROI_ENABLE:
            rear_roi, rear_bounds = _crop_vehicle_rear_roi(vehicle_crop)
            rx1, ry1, _, _ = rear_bounds
            detections = _translate_plate_candidates(detector.detect(rear_roi, conf=_PLATE_DETECT_CONF), rx1, ry1)
            if not detections:
                detections = _translate_plate_candidates(_propose_plate_bboxes(rear_roi), rx1, ry1)
        if not detections:
            detections = detector.detect(vehicle_crop, conf=_PLATE_DETECT_CONF)
        if not detections:
            detections = _propose_plate_bboxes(vehicle_crop)
        ranked_detections = _rank_plate_candidates(detections, vw, vh)
        if not ranked_detections:
            fx1, fy1, fx2, fy2 = fallback_bbox
            best["plate_bbox"] = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
            best["selected_candidate"] = "vehicle_fallback_bbox"
            best["center_score"] = 0.5
            best["fallback_only"] = True
            return best

        primary_bbox = ranked_detections[0].get("bbox") or fallback_bbox
        if len(primary_bbox) == 4:
            pbx1, pby1, pbx2, pby2 = [int(v) for v in primary_bbox]
            # 無論 OCR 結果如何，先保留 detector 找到的 plate_bbox
            best["plate_bbox"] = [x1 + pbx1, y1 + pby1, x1 + pbx2, y1 + pby2]
            best["selected_candidate"] = "detector_primary_bbox"
            best["det_conf"] = float(ranked_detections[0].get("confidence") or 0.0)

        secondary_candidates: List[str] = []
        primary_area = float(ranked_detections[0].get("rank_area") or 0.0)
        for rank_idx, det in enumerate(ranked_detections):
            bx1, by1, bx2, by2 = det["bbox"]
            ex1, ey1, ex2, ey2 = _expand_plate_bbox([bx1, by1, bx2, by2], vw, vh)
            plate_crop = detector.crop(vehicle_crop, [ex1, ey1, ex2, ey2])
            plate_crop, local_bbox = _tighten_plate_crop_with_bbox(plate_crop)
            if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
                continue

            crop_res = _recognize_plate_on_crop(plate_crop, recognizer)
            candidates = set()
            candidates.update(self._plate_variants(crop_res.get("plate_number") or ""))
            candidates.update(self._plate_variants(crop_res.get("raw") or ""))
            if not candidates:
                continue

            if isinstance(local_bbox, (list, tuple)) and len(local_bbox) == 4:
                lx1, ly1, lx2, ly2 = [int(v) for v in local_bbox]
            else:
                lx1, ly1, lx2, ly2 = 0, 0, max(1, ex2 - ex1), max(1, ey2 - ey1)

            gbx1 = x1 + ex1 + lx1
            gby1 = y1 + ey1 + ly1
            gbx2 = x1 + ex2 if lx2 <= lx1 else x1 + ex1 + lx2
            gby2 = y1 + ey2 if ly2 <= ly1 else y1 + ey1 + ly2
            det_w = max(1.0, float(gbx2 - gbx1))
            det_h = max(1.0, float(gby2 - gby1))
            cx = ((gbx1 + gbx2) / 2.0 - x1) / max(1.0, float(vw))
            cy = ((gby1 + gby2) / 2.0 - y1) / max(1.0, float(vh))
            area_ratio = (det_w * det_h) / max(1.0, float(vw * vh))
            lower_bias = max(0.0, 1.0 - abs(cy - 0.85) / 0.22)
            upper_half_penalty = max(0.0, (0.45 - cy) / 0.45) if cy < 0.45 else 0.0
            center_bias = max(0.0, 1.0 - abs(cx - 0.5) / 0.35)
            aspect = det_w / det_h
            aspect_bias = max(0.0, 1.0 - abs(aspect - 3.2) / 2.4)
            area_bias = max(0.0, 1.0 - abs(area_ratio - 0.035) / 0.04)
            geom_score = (lower_bias * 1.30) + (center_bias * 0.25) + (aspect_bias * 0.25) + (area_bias * 0.20) - (upper_half_penalty * 2.0)
            for cand in candidates:
                if not self._is_plausible_plate(cand):
                    continue
                rank_score = float(det.get("rank_score") or 0.0)
                secondary_penalty = 0.0 if rank_idx == 0 else (0.30 * rank_idx)
                score = self._score_ocr_result(crop_res, cand) + rank_score + geom_score - secondary_penalty
                if rank_idx > 0 and cand not in secondary_candidates:
                    secondary_candidates.append(cand)
                if score > best["_score"]:
                    best = {
                        "plate_number": cand,
                        "confidence": float(crop_res.get("confidence") or 0.0),
                        "valid": bool(crop_res.get("valid", False)),
                        "raw": crop_res.get("raw", ""),
                        "plate_bbox": [gbx1, gby1, gbx2, gby2],
                        "det_conf": float(det.get("confidence") or 0.0),
                        "center_score": max(0.0, min(1.0, (center_bias * 0.5) + (lower_bias * 0.5))),
                        "quality_score": float(crop_res.get("quality_score") or 0.0),
                        "primary_plate_area": primary_area,
                        "secondary_plate_exists": len(ranked_detections) > 1,
                        "secondary_plate_text": secondary_candidates[0] if secondary_candidates else "",
                        "selected_candidate": "primary_white_plate" if rank_idx == 0 else f"secondary_candidate_{rank_idx}",
                        "_score": score,
                    }

        best.pop("_score", None)
        return best

    def load_zones(self):
        try:
            from api.models import SessionLocal, Camera
            db = SessionLocal()
            cam = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if cam and cam.zones:
                self.zones = cam.zones
                print(f"[LPR] 載入 {len(self.zones)} 個 ROI 區域")
            db.close()
        except Exception as e:
            print(f"[LPR] 載入 zones 失敗: {e}")

    def in_any_zone(self, cx, cy):
        if not self.zones:
            return True
        import numpy as np
        for z in self.zones:
            pts = z.get("points", [])
            if len(pts) < 3:
                continue
            polygon = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0:
                return True
        return False
        
    def start(self):
        if self.running:
            return
        self.running = True
        self.load_zones()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"[LPR] 開始: {self.camera_name}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self._flush_cumulative_stats(force=True)
        print(f"[LPR] 停止: {self.camera_name}")
            
    def _run(self):
        # Watchdog: 從外部 thread 監控 last_frame_at，超時 forcibly release cap
        cap_holder = {"cap": None}
        watchdog_stop = threading.Event()
        def _watchdog():
            while not watchdog_stop.is_set():
                time.sleep(2.0)
                last = self.last_frame_at
                if last and (time.time() - last > 8.0):
                    # cap.read() 卡死，強制 release（會讓 read() 拋例外或回 ret=False）
                    c = cap_holder.get("cap")
                    if c is not None:
                        try:
                            c.release()
                        except Exception:
                            pass
                        self.last_frame_at = 0.0
        wd_thread = threading.Thread(target=_watchdog, daemon=True)
        wd_thread.start()

        try:
            yolo = get_yolo()
            recognizer = get_recognizer()
            get_plate_detector()

            cap = None
            for _retry in range(5):
                cap = _open_capture(self.source)
                if cap.isOpened():
                    break
                print(f"[LPR] 來源連線失敗，{3}秒後重試 ({_retry+1}/5): {self.source}")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(3)

            if cap is None or not cap.isOpened():
                self.last_error = f"無法開啟來源: {self.source}"
                print(f"[LPR] {self.last_error}")
                self.running = False
                watchdog_stop.set()
                return

            cap_holder["cap"] = cap
            self.last_frame_at = time.time()  # 啟動 watchdog 計時
            consecutive_fail = 0

            frame_skip = 1
            while self.running:
                # 主迴圈 watchdog：若 read 連續失敗或長時間沒幀就重連
                need_reconnect = (
                    consecutive_fail >= 10
                    or self.last_frame_at == 0.0
                    or (self.last_frame_at and time.time() - self.last_frame_at > 8)
                )
                if need_reconnect:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    cap = _open_capture(self.source)
                    cap_holder["cap"] = cap
                    consecutive_fail = 0
                    if not (cap and cap.isOpened()):
                        time.sleep(2.0)
                        self.last_frame_at = time.time()
                        continue
                    self.last_frame_at = time.time()

                try:
                    ret, frame = cap.read()
                except Exception:
                    ret, frame = False, None
                if not ret:
                    consecutive_fail += 1
                    time.sleep(0.2)
                    continue
                consecutive_fail = 0

                self.last_frame_at = time.time()
                self._increment_debug_counter("total_frames")
                self.recent_frames.append(frame.copy())
                if self.total_frames % frame_skip != 0:
                    continue

                try:
                    with _yolo_lock:
                        detections = yolo.detect(frame)

                    tracked_inputs = []
                    for det in detections:
                        vehicle_type = str(det.get("class_name") or "")
                        if vehicle_type not in ("car", "motorcycle", "bus", "truck", "heavy_truck", "light_truck"):
                            continue

                        bbox = det.get("bbox", {}) or {}
                        x1 = int(bbox.get("x1", 0))
                        y1 = int(bbox.get("y1", 0))
                        x2 = int(bbox.get("x2", 0))
                        y2 = int(bbox.get("y2", 0))
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if not self.in_any_zone(cx, cy):
                            continue
                        tracked_inputs.append({
                            "class_name": vehicle_type,
                            "confidence": float(det.get("confidence") or 0.0),
                            "bbox": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "width": max(0, int(x2) - int(x1)),
                                "height": max(0, int(y2) - int(y1)),
                            }
                        })

                    tracked_vehicles = self.vehicle_tracker.update(tracked_inputs) or []

                    for det in tracked_vehicles:
                        vehicle_type = str(det.get("class_name") or "")
                        bbox = det.get("bbox", {}) or {}
                        x1 = int(bbox.get("x1", 0))
                        y1 = int(bbox.get("y1", 0))
                        x2 = int(bbox.get("x2", 0))
                        y2 = int(bbox.get("y2", 0))
                        track_id = int(det.get("track_id") or 0)
                        self._increment_debug_counter("vehicles_detected")

                        # === 最佳距離窗口：車輛太小時跳過 OCR，等它靠近 ===
                        ih_pre, iw_pre = frame.shape[:2]
                        veh_w = max(1, x2 - x1)
                        veh_h = max(1, y2 - y1)
                        veh_area_ratio = (veh_w * veh_h) / max(1, iw_pre * ih_pre)
                        if veh_area_ratio < _VEHICLE_MIN_AREA_RATIO:
                            self._update_vehicle_track_state(track_id, vehicle_type, [x1, y1, x2, y2], frame)
                            print(f"[LPR] skip: cam{self.camera_id} 車太小 {veh_w}x{veh_h} ratio={veh_area_ratio:.4f}", flush=True)
                            continue

                        # 多裁切候選 + OCR 投票
                        result = self._recognize_plate_best(frame, x1, y1, x2, y2, recognizer)
                        ih, iw = frame.shape[:2]
                        vx1 = max(0, min(iw - 1, int(x1)))
                        vy1 = max(0, min(ih - 1, int(y1)))
                        vx2 = max(0, min(iw - 1, int(x2)))
                        vy2 = max(0, min(ih - 1, int(y2)))
                        if vx2 <= vx1 or vy2 <= vy1:
                            continue
                        self._update_vehicle_track_state(track_id, vehicle_type, [vx1, vy1, vx2, vy2], frame)

                        pb = result.get("plate_bbox")
                        if not (isinstance(pb, (list, tuple)) and len(pb) == 4):
                            print(f"[LPR] skip: cam{self.camera_id} no plate_bbox, vehicle={veh_w}x{veh_h}, selected={result.get('selected_candidate','?')}, fallback={result.get('fallback_only')}", flush=True)
                            continue
                        self._increment_debug_counter("plate_boxes_detected")
                        px1 = max(0, min(iw - 1, int(pb[0])))
                        py1 = max(0, min(ih - 1, int(pb[1])))
                        px2 = max(0, min(iw - 1, int(pb[2])))
                        py2 = max(0, min(ih - 1, int(pb[3])))
                        if px2 <= px1 or py2 <= py1:
                            continue
                        result["plate_bbox"] = [px1, py1, px2, py2]
                        fallback_only = bool(result.get("fallback_only"))

                        # 即使是 fallback bbox 也嘗試 OCR — 每台車��要辨識
                        # if fallback_only:
                        #     continue

                        plate_crop_raw = frame[py1:py2, px1:px2]
                        plate_crop = _select_best_native_plate_snapshot(
                            list(self.recent_frames) or [frame],
                            [px1, py1, px2, py2],
                            iw,
                            ih,
                        )
                        if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
                            plate_crop = plate_crop_raw
                        if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
                            continue

                        # === 車牌裁切品質檢查：太小太糊不做 OCR，等下一幀更好的 ===
                        _crop_h, _crop_w = plate_crop.shape[:2]
                        if _crop_w < _PLATE_OCR_MIN_CROP_W or _crop_h < _PLATE_OCR_MIN_CROP_H:
                            print(f"[LPR] skip: cam{self.camera_id} 車牌太小 {_crop_w}x{_crop_h}", flush=True)
                            continue
                        _crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
                        _crop_blur = float(cv2.Laplacian(_crop_gray, cv2.CV_32F).var())
                        if _crop_blur < _PLATE_OCR_MIN_BLUR:
                            print(f"[LPR] skip: cam{self.camera_id} 車牌太糊 blur={_crop_blur:.1f} size={_crop_w}x{_crop_h}", flush=True)
                            continue

                        crop_res = _recognize_plate_on_crop(plate_crop, recognizer)
                        crop_candidates = set()
                        crop_candidates.update(self._plate_variants(crop_res.get("plate_number") or ""))
                        crop_candidates.update(self._plate_variants(crop_res.get("raw") or ""))
                        if not crop_candidates:
                            continue
                        self._increment_debug_counter("ocr_candidates_detected")

                        detector_w = max(0, px2 - px1)
                        detector_h = max(0, py2 - py1)
                        final_h, final_w = plate_crop.shape[:2]
                        width_ratio = (final_w / detector_w) if detector_w > 0 else 0.0
                        height_ratio = (final_h / detector_h) if detector_h > 0 else 0.0
                        area_ratio = ((final_w * final_h) / float(max(1, detector_w * detector_h)))
                        print(
                            "[LPR] crop "
                            f"detector_bbox_size={detector_w}x{detector_h} "
                            f"final_plate_crop_size={final_w}x{final_h} "
                            f"crop_ratio=w:{width_ratio:.2f},h:{height_ratio:.2f},area:{area_ratio:.2f}"
                        )

                        best_crop_plate = ""
                        best_crop_score = -1.0
                        best_crop_layout = -99.0
                        for cand in crop_candidates:
                            if not self._is_plausible_plate(cand):
                                continue
                            s = self._score_ocr_result(crop_res, cand)
                            if s > best_crop_score:
                                best_crop_score = s
                                best_crop_layout = self._plate_layout_score(cand)
                                best_crop_plate = cand
                        if not best_crop_plate:
                            continue

                        plate = best_crop_plate
                        conf = float(crop_res.get("confidence", 0) or 0)
                        raw = str(crop_res.get("raw", "") or "")
                        crop_valid = bool(crop_res.get("valid", False))
                        quality_score = float(crop_res.get("quality_score") or result.get("quality_score") or 0.0)
                        det_conf = float(result.get("det_conf") or 0.0)
                        center_score = float(result.get("center_score") or 0.0)
                        effective_conf = self._effective_plate_conf(
                            plate,
                            conf,
                            valid=crop_valid,
                            quality_score=quality_score,
                            raw=raw,
                        )
                        self.last_candidate_plate = plate
                        result["plate_number"] = plate
                        result["confidence"] = conf
                        result["effective_confidence"] = effective_conf
                        result["raw"] = raw
                        result["valid"] = crop_valid
                        if raw:
                            print(
                                f"[LPR] OCR原始: {raw} → 格式化: {plate} "
                                f"信心度: {conf:.2f} 有效信心: {effective_conf:.2f}"
                            )

                        # best-of-track：累積 OCR 結果，追蹤結束時才 commit
                        pre_vote_ok = (
                            crop_valid
                            or effective_conf > 0.10
                            or best_crop_layout >= 1.0
                            or quality_score >= 0.08
                        )

                        if plate and self._is_plausible_plate(plate) and 4 <= len(plate) <= 10 and pre_vote_ok:
                            self._update_track_plate_candidate(
                                track_id=track_id,
                                plate=plate,
                                effective_conf=effective_conf,
                                valid=crop_valid,
                                raw=raw,
                                quality_score=quality_score,
                                plate_bbox=[px1, py1, px2, py2],
                                vehicle_bbox=[vx1, vy1, vx2, vy2],
                                vehicle_type=vehicle_type,
                                frame=frame,
                                plate_crop=plate_crop,
                                iw=iw,
                                ih=ih,
                            )
                            self._increment_debug_counter("vote_candidates_detected")
                            self.last_candidate_plate = plate
                        elif plate:
                            self.last_rejected_plate = plate
                            self.last_rejected_reason = "pre_vote_gate"

                    # 追蹤結束的 track → commit 最佳結果
                    self._flush_completed_tracks()
                    self._flush_inactive_vehicle_tracks()

                except Exception as e:
                    self.last_error = str(e)
                    print(f"[LPR] 處理錯誤: {e}")

                time.sleep(0.05)

            cap.release()
            self._flush_completed_tracks(force=True)
            self._flush_inactive_vehicle_tracks(force=True)
            self._flush_cumulative_stats(force=True)
        except Exception as e:
            self.last_error = str(e)
            self.running = False
            print(f"[LPR] 執行緒異常結束: {e}")


@router.post("/start/{camera_id}")
async def start_lpr_stream(camera_id: int):
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="攝影機不存在")
        if not bool(camera.enabled):
            raise HTTPException(status_code=409, detail="攝影機已關閉")
        if camera_id in _lpr_tasks and _lpr_tasks[camera_id].running:
            camera.status = "online"
            camera.detection_enabled = True
            db.commit()
            set_feature_state("lpr", camera_id, True)
            return {"status": "running", "message": "已在運行中"}

        try:
            get_plate_detector()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"LPR plate detector unavailable: {e}")
        started = _start_lpr_task(camera)
        if not started:
            camera.status = "online"
            camera.detection_enabled = True
            db.commit()
            set_feature_state("lpr", camera_id, True)
            return {"status": "running", "message": "已在運行中"}
        await asyncio.sleep(1.2)
        task = _lpr_tasks.get(camera_id)
        if task is None or not (task.thread and task.thread.is_alive()):
            raise HTTPException(status_code=503, detail=f"LPR worker failed to start: {getattr(task, 'last_error', None) or 'thread not alive'}")
        camera.status = "online"
        camera.detection_enabled = True
        db.commit()
        set_feature_state("lpr", camera_id, True)
        return {"status": "started", "message": f"開始辨識: {camera.name}"}
    finally:
        db.close()


@router.post("/stop/{camera_id}")
async def stop_lpr_stream(camera_id: int):
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if camera:
            camera.detection_enabled = False
            db.commit()
    finally:
        db.close()
    if camera_id in _lpr_tasks:
        _lpr_tasks[camera_id].stop()
        del _lpr_tasks[camera_id]
        set_feature_state("lpr", camera_id, False)
        return {"status": "stopped"}
    set_feature_state("lpr", camera_id, False)
    return {"status": "not_running"}


@router.get("/status/{camera_id}")
async def get_stream_status(camera_id: int):
    cumulative = _read_lpr_cumulative_stats(camera_id)
    if camera_id in _lpr_tasks:
        task = _lpr_tasks[camera_id]
        thread_alive = bool(task.thread and task.thread.is_alive())
        if task.running and not thread_alive:
            task.running = False
        for field in _LPR_CUMULATIVE_COUNTER_FIELDS:
            cumulative[field] = int(cumulative.get(field, 0) or 0) + task._pending_stat_delta(field)
        return {
            "running": task.running,
            "thread_alive": thread_alive,
            "camera_id": camera_id,
            "camera_name": task.camera_name,
            "total_frames": task.total_frames,
            "vehicles_detected": task.vehicles_detected,
            "plate_boxes_detected": task.plate_boxes_detected,
            "ocr_candidates_detected": task.ocr_candidates_detected,
            "vote_candidates_detected": task.vote_candidates_detected,
            "confirmed_candidates": task.confirmed_candidates,
            "committed_candidates": task.committed_candidates,
            "total_detections": task.total_detections,
            "last_plate": task.last_plate_number,
            "last_candidate_plate": task.last_candidate_plate,
            "last_confirmed_plate": task.last_confirmed_plate,
            "last_rejected_plate": task.last_rejected_plate,
            "last_rejected_reason": task.last_rejected_reason,
            "last_error": task.last_error,
            "cumulative_total_frames": cumulative.get("total_frames", 0),
            "cumulative_vehicles_detected": cumulative.get("vehicles_detected", 0),
            "cumulative_plate_boxes_detected": cumulative.get("plate_boxes_detected", 0),
            "cumulative_ocr_candidates_detected": cumulative.get("ocr_candidates_detected", 0),
            "cumulative_vote_candidates_detected": cumulative.get("vote_candidates_detected", 0),
            "cumulative_confirmed_candidates": cumulative.get("confirmed_candidates", 0),
            "cumulative_committed_candidates": cumulative.get("committed_candidates", 0),
            "latest_history_at": cumulative.get("last_history_at", ""),
        }
    return {
        "running": False,
        "camera_id": camera_id,
        "cumulative_total_frames": cumulative.get("total_frames", 0),
        "cumulative_vehicles_detected": cumulative.get("vehicles_detected", 0),
        "cumulative_plate_boxes_detected": cumulative.get("plate_boxes_detected", 0),
        "cumulative_ocr_candidates_detected": cumulative.get("ocr_candidates_detected", 0),
        "cumulative_vote_candidates_detected": cumulative.get("vote_candidates_detected", 0),
        "cumulative_confirmed_candidates": cumulative.get("confirmed_candidates", 0),
        "cumulative_committed_candidates": cumulative.get("committed_candidates", 0),
        "latest_history_at": cumulative.get("last_history_at", ""),
    }


@router.get("/results/{camera_id}")
async def get_stream_results(camera_id: int, limit: int = Query(100, ge=1, le=500), include_unknown: bool = False):
    from api.models import SessionLocal, LPRRecord

    db = SessionLocal()
    try:
        query = db.query(LPRRecord).filter(LPRRecord.camera_id == camera_id)
        if not include_unknown:
            query = query.filter(LPRRecord.plate_number.isnot(None)).filter(LPRRecord.plate_number != "UNKNOWN")
            query = query.filter((LPRRecord.raw.is_(None)) | (LPRRecord.raw != "vehicle_only"))
        rows = query.order_by(LPRRecord.created_at.desc(), LPRRecord.id.desc()).limit(limit).all()
        results = [
            {
                "id": r.id,
                "time": _fmt_taipei(r.created_at),
                "time_iso": _fmt_taipei_iso(r.created_at),
                "created_at": _fmt_taipei(r.created_at),
                "plate_number": r.plate_number,
                "confidence": float(r.confidence or 0),
                "valid": bool(r.valid),
                "vehicle_type": r.vehicle_type,
                "camera_id": r.camera_id,
                "camera_name": r.camera_name,
                "snapshot": r.snapshot,
                "plate_snapshot": _derive_plate_snapshot_name(r.snapshot),
                "raw": r.raw or "",
            }
            for r in rows
        ]
    finally:
        db.close()

    # 若 DB 暫時無資料且任務運作中，回退記憶體快取（避免剛啟動時空白）
    if not results and camera_id in _lpr_tasks:
        fallback_rows = _lpr_tasks[camera_id].results
        if not include_unknown:
            fallback_rows = [
                item for item in fallback_rows
                if str(item.get("plate_number") or "").strip().upper() != "UNKNOWN"
                and str(item.get("raw") or "").strip().lower() != "vehicle_only"
            ]
        return {"results": fallback_rows[:limit]}
    return {"results": results}


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@router.get("/history")
async def get_history(
    camera_id: Optional[int] = None,
    q: Optional[str] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = Query(300, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    include_unknown: bool = False,
):
    from sqlalchemy import or_
    from api.models import SessionLocal, LPRRecord

    start_dt = _parse_iso_datetime(start)
    end_dt = _parse_iso_datetime(end)
    keyword = str(q or "").strip()

    db = SessionLocal()
    try:
        query = db.query(LPRRecord)
        if camera_id is not None:
            query = query.filter(LPRRecord.camera_id == int(camera_id))
        if min_confidence > 0:
            query = query.filter(LPRRecord.confidence >= float(min_confidence))
        if start_dt is not None:
            query = query.filter(LPRRecord.created_at >= start_dt)
        if end_dt is not None:
            query = query.filter(LPRRecord.created_at <= end_dt)
        if not include_unknown:
            query = query.filter(LPRRecord.plate_number.isnot(None)).filter(LPRRecord.plate_number != "UNKNOWN")
            query = query.filter((LPRRecord.raw.is_(None)) | (LPRRecord.raw != "vehicle_only"))
        if keyword:
            like_kw = f"%{keyword}%"
            # 中文車種關鍵字對應到 DB 裡的英文 label
            _ZH_TO_EN_VEHICLE = {
                "小客車": ["car"],
                "小貨車": ["light_truck"],
                "大貨車": ["heavy_truck"],
                "大客車": ["bus"],
                "公車": ["bus"],
                "機車": ["motorcycle"],
                "自行車": ["bicycle"],
                "貨車": ["truck", "light_truck", "heavy_truck"],
                "小車": ["car", "light_truck", "motorcycle"],
                "大車": ["heavy_truck", "bus"],
            }
            clauses = [
                LPRRecord.plate_number.ilike(like_kw),
                LPRRecord.vehicle_type.ilike(like_kw),
                LPRRecord.camera_name.ilike(like_kw),
            ]
            # 取最具體的中文詞（較長的優先，避免 "大貨車" 又命中 "貨車"）
            matched_en: set[str] = set()
            remaining = keyword
            for zh, en_list in sorted(_ZH_TO_EN_VEHICLE.items(), key=lambda kv: -len(kv[0])):
                if zh in remaining:
                    for en in en_list:
                        matched_en.add(en)
                    remaining = remaining.replace(zh, "")
            for en in matched_en:
                clauses.append(LPRRecord.vehicle_type.ilike(f"%{en}%"))
            query = query.filter(or_(*clauses))
        total = int(query.count())
        ordered_query = query.order_by(LPRRecord.created_at.desc(), LPRRecord.id.desc())
        latest_row = ordered_query.first()
        rows = ordered_query.offset(int(offset)).limit(int(limit)).all()
        results = [
            {
                "id": r.id,
                "time": _fmt_taipei(r.created_at),
                "time_iso": _fmt_taipei_iso(r.created_at),
                "created_at": _fmt_taipei(r.created_at),
                "plate_number": r.plate_number,
                "confidence": float(r.confidence or 0),
                "valid": bool(r.valid),
                "vehicle_type": r.vehicle_type,
                "camera_id": r.camera_id,
                "camera_name": r.camera_name,
                "snapshot": r.snapshot,
                "plate_snapshot": _derive_plate_snapshot_name(r.snapshot),
                "raw": r.raw or "",
            }
            for r in rows
        ]
        return {
            "results": results,
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
            "latest_id": int(latest_row.id) if latest_row is not None else 0,
        }
    finally:
        db.close()


@router.get("/camera-options")
async def get_lpr_history_camera_options():
    from api.models import SessionLocal, Camera, LPRRecord

    db = SessionLocal()
    try:
        options: Dict[int, Dict[str, Any]] = {}
        for cam in db.query(Camera).order_by(Camera.id.asc()).all():
            cam_id = int(cam.id)
            options[cam_id] = {
                "id": cam_id,
                "name": str(cam.name or f"攝影機 {cam_id}"),
                "enabled": bool(cam.enabled),
            }

        history_rows = (
            db.query(LPRRecord.camera_id, LPRRecord.camera_name)
            .filter(LPRRecord.camera_id.isnot(None))
            .distinct()
            .all()
        )
        for row in history_rows:
            cam_id = int(row.camera_id or 0)
            if cam_id <= 0 or cam_id in options:
                continue
            options[cam_id] = {
                "id": cam_id,
                "name": str(row.camera_name or f"攝影機 {cam_id}"),
                "enabled": False,
            }

        return {"items": sorted(options.values(), key=lambda item: int(item["id"]))}
    finally:
        db.close()


@router.get("/snapshot/{filename}")
async def get_snapshot(filename: str):
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="截圖不存在")


@router.get("/all")
async def get_all_streams():
    return {"streams": [{"camera_id": k, "running": v.running, "detections": v.total_detections} for k, v in _lpr_tasks.items()]}


def _start_lpr_task(camera: Camera) -> bool:
    camera_id = int(camera.id)
    if camera_id in _lpr_tasks and _lpr_tasks[camera_id].running:
        return False
    task = LPRStreamTask(camera_id, resolve_analysis_source(camera), camera.name)
    task.start()
    _lpr_tasks[camera_id] = task
    return True


def resume_lpr_streams() -> dict:
    wants = get_feature_state("lpr")
    if not wants:
        return {"total": 0, "resumed": 0}
    try:
        get_plate_detector()
    except Exception:
        return {"total": 0, "resumed": 0}
    db = SessionLocal()
    resumed = 0
    total = 0
    try:
        for camera_id, enabled in wants.items():
            if not enabled:
                continue
            total += 1
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                continue
            if _start_lpr_task(camera):
                camera.status = "online"
                camera.detection_enabled = True
                resumed += 1
        db.commit()
    finally:
        db.close()
    return {"total": total, "resumed": resumed}
