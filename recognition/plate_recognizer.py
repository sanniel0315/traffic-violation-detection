#!/usr/bin/env python3
"""台灣車牌辨識模組 - 增強版 Tesseract"""
import re
from typing import Any, Dict, List

import cv2
import numpy as np
import pytesseract


class PlateRecognizer:
    """車牌辨識器 - YOLO 字元偵測 + Tesseract fallback"""

    _shared_char_detector = None
    _char_detector_tried = False

    TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8"}
    TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}

    PLATE_PATTERNS = [
        r"^[A-Z]{3}-\d{4}$",
        r"^[A-Z]{2}-\d{4}$",
        r"^[A-Z]{2}\d-\d{4}$",
        r"^\d{4}-[A-Z]{2}$",
        r"^[A-Z]{3}-\d{3}$",
        r"^[A-Z]{2}-\d{3}$",
        r"^\d{3}-[A-Z]{3}$",
        r"^\d{3}-[A-Z]{2}$",
        r"^[A-Z]\d{2}-\d{3}$",
    ]

    def __init__(self, use_gpu: bool = True):
        self.config_single = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        self.config_block = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        self.config_word = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        self.config_raw = "--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

        self.easy_reader = None

        # YOLO 字元偵測透過微服務呼叫（避免跟主 YOLO 搶 GPU）
        self.char_detector = None
        print("✅ PlateRecognizer 初始化完成")

    def _ocr_easy(self, img: np.ndarray):
        """透過 YOLO 字元偵測微服務辨識車牌"""
        if img is None or img.size == 0:
            return None, 0.0
        try:
            import requests as _req
            _, buf = cv2.imencode('.png', img)
            resp = _req.post('http://127.0.0.1:8010/', data=buf.tobytes(), timeout=10)
            if resp.ok:
                d = resp.json()
                text = d.get('text')
                conf = float(d.get('confidence', 0))
                if text and len(text) >= 4:
                    return text, conf
        except Exception:
            pass
        return None, 0.0

    def find_plate_region(self, img: np.ndarray) -> List[np.ndarray]:
        """偵測可能的車牌區域"""
        regions = []

        if img is None or img.size == 0:
            return [img] if img is not None else regions

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0
            area = cw * ch

            if 1.5 < aspect < 6 and area > (w * h * 0.01) and area < (w * h * 0.5):
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + pad)
                regions.append(img[y1:y2, x1:x2])

        if not regions:
            regions.append(img)

        return regions

    def _is_planar_plate(self, pts: np.ndarray, w: int, h: int) -> bool:
        """檢查 4 點是否構成合理的車牌平面（非極端透視）"""
        if pts is None or len(pts) != 4:
            return False
        # 排序四角：tl, tr, br, bl
        rect = self._order_corners(pts)
        # 上邊長 vs 下邊長 比例（透視太大就 invalid）
        top_w = np.linalg.norm(rect[1] - rect[0])
        bot_w = np.linalg.norm(rect[2] - rect[3])
        left_h = np.linalg.norm(rect[3] - rect[0])
        right_h = np.linalg.norm(rect[2] - rect[1])
        if top_w < 1 or bot_w < 1 or left_h < 1 or right_h < 1:
            return False
        # 上下邊長差不能太大（>50% 表示嚴重透視）
        w_ratio = min(top_w, bot_w) / max(top_w, bot_w)
        h_ratio = min(left_h, right_h) / max(left_h, right_h)
        if w_ratio < 0.5 or h_ratio < 0.5:
            return False
        # 整體 aspect 應該接近車牌（2.0~6.0）
        avg_w = (top_w + bot_w) / 2
        avg_h = (left_h + right_h) / 2
        if avg_h < 1: return False
        aspect = avg_w / avg_h
        if not (1.8 <= aspect <= 7.0):
            return False
        # 四角應該佔圖片大部分區域（>30%）
        area = cv2.contourArea(rect)
        if area < (w * h * 0.20):
            return False
        return True

    @staticmethod
    def _order_corners(pts: np.ndarray) -> np.ndarray:
        """4 點排序為 tl, tr, br, bl"""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # top-left
        rect[2] = pts[np.argmax(s)]      # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # top-right
        rect[3] = pts[np.argmax(diff)]   # bottom-left
        return rect

    def perspective_transform(self, img: np.ndarray) -> np.ndarray:
        """透視變換校正傾斜車牌 - 多策略找四角並 warp 攤平
        - 如果原圖已經是 plate 形狀 (aspect 2~6)，跳過避免破壞
        - 先找最大近似四邊形
        - 檢查是否平面（aspect、邊長比、面積）
        - 不平面 → 用 minAreaRect 取角度做 affine 校正
        """
        if img is None or img.size == 0:
            return img

        H, W = img.shape[:2]
        # ⚡ 關鍵：如果輸入已經是車牌形狀且夠小，直接 return（避免破壞已裁切的車牌）
        if H > 0:
            in_aspect = W / H
            if 1.8 <= in_aspect <= 7.0 and W <= 400 and H <= 150:
                return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # ----- 策略 1：找近似四邊形 contour -----
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        # 形態學閉合讓邊緣連續
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_quad = None
        best_area = 0
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            peri = cv2.arcLength(cnt, True)
            for eps_factor in [0.02, 0.04, 0.06]:
                approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype(np.float32)
                    if self._is_planar_plate(pts, W, H):
                        area = cv2.contourArea(approx)
                        if area > best_area:
                            best_area = area
                            best_quad = pts
                    break

        if best_quad is not None:
            rect = self._order_corners(best_quad)
            w1 = np.linalg.norm(rect[1] - rect[0])
            w2 = np.linalg.norm(rect[2] - rect[3])
            h1 = np.linalg.norm(rect[3] - rect[0])
            h2 = np.linalg.norm(rect[2] - rect[1])
            out_w = int(max(w1, w2))
            out_h = int(max(h1, h2))
            if out_w >= 50 and out_h >= 15:
                dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, M, (out_w, out_h))
                return warped

        # ----- 策略 2：minAreaRect 取角度做旋轉校正 -----
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(binary) > 127:
                binary = 255 - binary
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) > 50:
                rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
                angle = rect[-1]
                # 角度正規化到 -45 ~ 45
                if angle < -45:
                    angle = 90 + angle
                elif angle > 45:
                    angle = angle - 90
                # 只在傾斜明顯時才校正（避免過度處理）
                if 2 <= abs(angle) <= 25:
                    M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
                    rotated = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated
        except Exception:
            pass

        return img

    def preprocess(self, img: np.ndarray) -> List[np.ndarray]:
        """多重預處理 - 優化版

        針對台灣車牌不同光照、模糊度產生多個變體：
        - 大幅放大（小車牌也能辨識）
        - 多種對比度增強
        - 多種二值化方法
        - 去噪 + 銳化組合
        """
        results = []
        if img is None or img.size == 0:
            return results

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        # 更大的目標尺寸，提升小車牌辨識
        if w < 360:
            scale = 480 / max(1, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif w > 800:
            scale = 800 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ----- 1. 原始灰階 -----
        results.append(gray)

        # ----- 2. CLAHE 強對比度（多種參數）-----
        for clip in [2.0, 3.5, 5.0]:
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            results.append(enhanced)

        # ----- 3. 去噪後二值化 -----
        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        _, denoised_bin = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(denoised_bin)
        results.append(255 - denoised_bin)

        # ----- 4. CLAHE + 二值化 -----
        enhanced = clahe.apply(gray)
        _, enh_bin = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(enh_bin)
        results.append(255 - enh_bin)

        # ----- 5. 自適應二值化（多種視窗）-----
        for block in [11, 19, 25]:
            adaptive = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                block, 5
            )
            results.append(adaptive)

        # ----- 6. 銳化 -----
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        results.append(sharpened)

        # ----- 7. 形態學去雜訊 -----
        kernel_morph = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(enh_bin, cv2.MORPH_OPEN, kernel_morph)
        results.append(opened)

        # ----- 8. 直方圖均衡化 -----
        equalized = cv2.equalizeHist(gray)
        results.append(equalized)
        _, eq_bin = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(eq_bin)

        return results

    def _generate_plate_candidates(self, raw: str) -> List[str]:
        text = re.sub(r"[^A-Z0-9]", "", str(raw or "").upper())
        if not text:
            return []

        candidates = set()
        cleaned, _ = self._clean(text)
        if cleaned:
            candidates.add(cleaned)

        if 5 <= len(text) <= 8:
            for cut in (2, 3, 4):
                if 2 <= cut < len(text) - 1 and len(text[:cut]) <= 4 and len(text[cut:]) <= 4:
                    candidates.add(f"{text[:cut]}-{text[cut:]}")
                    left = text[:cut]
                    right = text[cut:]
                    candidates.add(
                        f"{''.join(self.TO_ALPHA.get(ch, ch) for ch in left)}-{''.join(self.TO_DIGIT.get(ch, ch) for ch in right)}"
                    )
                    candidates.add(
                        f"{''.join(self.TO_DIGIT.get(ch, ch) for ch in left)}-{''.join(self.TO_ALPHA.get(ch, ch) for ch in right)}"
                    )

        normalized = set()
        for cand in candidates:
            plate, _ = self._clean(cand)
            if plate:
                normalized.add(plate)
        return list(normalized)

    def _matches_plate_pattern(self, plate: str) -> bool:
        plate = re.sub(r"[^A-Z0-9-]", "", str(plate or "").upper())
        if not plate or "-" not in plate:
            return False
        return any(re.match(pattern, plate) for pattern in self.PLATE_PATTERNS)

    def _segment_profile_score(self, plate: str) -> float:
        plate = re.sub(r"[^A-Z0-9-]", "", str(plate or "").upper())
        if not plate or "-" not in plate:
            return -3.0
        left, right = plate.split("-", 1)
        if not (2 <= len(left) <= 4 and 2 <= len(right) <= 4):
            return -2.5

        plain = plate.replace("-", "")
        left_letters = sum(ch.isalpha() for ch in left)
        left_digits = sum(ch.isdigit() for ch in left)
        right_letters = sum(ch.isalpha() for ch in right)
        right_digits = sum(ch.isdigit() for ch in right)
        left_minor = min(left_letters, left_digits)
        right_minor = min(right_letters, right_digits)

        score = 0.0
        if left_letters >= max(1, len(left) - 1) and right_digits >= max(1, len(right) - 1):
            score += 1.25
        elif left_digits >= max(1, len(left) - 1) and right_letters >= max(1, len(right) - 1):
            score += 0.85
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
        # ❌ 移除：台灣車牌很常有連續相同數字 (8886, 0000 等)
        # if re.search(r"([A-Z0-9])\1\1", plain):
        #     score -= 1.2
        if len(set(plain)) <= 2:
            score -= 1.0
        return score

    def _score_candidate(self, plate: str, conf: float, raw: str = "") -> float:
        if not plate:
            return -10.0
        plain = plate.replace("-", "")
        profile_score = self._segment_profile_score(plate)
        score = float(conf or 0.0) * 1.8
        score += profile_score
        if self._validate(plate):
            score += 2.8
        if re.match(r"^[A-Z]{2,4}-\d{2,4}$", plate):
            score += 1.2
        elif re.match(r"^\d{2,4}-[A-Z]{2,4}$", plate):
            score += 0.7

        digits = sum(ch.isdigit() for ch in plain)
        letters = sum(ch.isalpha() for ch in plain)
        if digits >= 2:
            score += 0.4
        if letters >= 2:
            score += 0.4
        if len(plain) < 5 or len(plain) > 8:
            score -= 1.5
        if re.search(r"([A-Z0-9])\1\1", plain):
            score -= 0.8
        if raw and len(re.sub(r"[^A-Z0-9]", "", raw.upper())) >= len(plain):
            score += 0.1
        return score

    def _enhance_plate(self, img: np.ndarray) -> np.ndarray:
        """放大 + 強化車牌圖（給 OCR 用）"""
        if img is None or img.size == 0:
            return img
        h, w = img.shape[:2]
        # 1. 放大到目標寬度（4 倍 or 至少 400px）
        target_w = max(400, w * 4)
        if target_w > w:
            scale = target_w / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # 2. 灰階 + CLAHE 對比強化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # 3. 雙邊濾波去噪保留邊緣
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        # 4. Unsharp mask 銳化
        blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        return gray

    def _segment_chars(self, plate_img: np.ndarray) -> List[np.ndarray]:
        """分割車牌字元 - 找出每個獨立字元的 bbox"""
        if plate_img is None or plate_img.size == 0:
            return []
        gray = plate_img if len(plate_img.shape) == 2 else cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if h < 20 or w < 60:
            return []

        # Otsu 二值化（depending on bg color, try both）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 黑底白字翻轉
        if np.mean(binary) > 127:
            binary = 255 - binary

        # 形態學去雜訊
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 找連通元件
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes = []
        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]
            # 過濾：字元高度應該佔車牌高度的 35~95%，寬度合理
            if ch < h * 0.30 or ch > h * 0.95:
                continue
            if cw < 3 or cw > w * 0.30:
                continue
            if area < 30:
                continue
            aspect = cw / max(1, ch)
            if aspect > 1.2:  # 太寬 = 不是字元（橫線）
                continue
            boxes.append((x, y, cw, ch))

        # 由左到右排序
        boxes.sort(key=lambda b: b[0])

        # 合併 X 重疊的字元（避免單一字元被切成兩個）
        merged = []
        for box in boxes:
            x, y, cw, ch = box
            if merged and x < merged[-1][0] + merged[-1][2] - 2:
                # 與前一個重疊，merge
                px, py, pw, ph = merged[-1]
                nx = min(px, x)
                ny = min(py, y)
                nx2 = max(px + pw, x + cw)
                ny2 = max(py + ph, y + ch)
                merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
            else:
                merged.append(box)

        # 萃取每個字元（含 padding）
        chars = []
        for x, y, cw, ch in merged[:8]:  # 最多 8 字元
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)
            chars.append(gray[y1:y2, x1:x2])
        return chars

    def _ocr_single_char(self, char_img: np.ndarray) -> tuple:
        """單字元 OCR - 用 Tesseract PSM 10 (single character)"""
        if char_img is None or char_img.size == 0:
            return "", 0.0
        try:
            # 放大字元到固定大小
            ch, cw = char_img.shape[:2]
            target_h = 60
            scale = target_h / max(1, ch)
            resized = cv2.resize(char_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # 加白邊
            padded = cv2.copyMakeBorder(resized, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            config = "--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            data = pytesseract.image_to_data(padded, config=config, output_type=pytesseract.Output.DICT)
            best_t, best_c = "", 0.0
            for i, t in enumerate(data["text"]):
                t = str(t or "").strip().upper()
                if not t or not t.isalnum() or len(t) != 1:
                    continue
                try:
                    c = max(0.0, float(data["conf"][i]))
                except Exception:
                    c = 0.0
                if c > best_c:
                    best_t, best_c = t, c / 100.0
            return best_t, best_c
        except Exception:
            return "", 0.0

    def recognize_chars(self, img: np.ndarray) -> Dict[str, Any]:
        """已由 YOLO 字元偵測取代，不再呼叫 Tesseract"""
        return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        # 以下為舊 Tesseract 邏輯（停用）
        if img is None or img.size == 0:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        # 1. 校正透視
        corrected = self.perspective_transform(img)
        # 2. 放大強化
        enhanced = self._enhance_plate(corrected)
        # 3. 分割字元
        chars = self._segment_chars(enhanced)
        if len(chars) < 5:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        # 4. 個別 OCR
        text_parts = []
        confs = []
        for ch_img in chars:
            t, c = self._ocr_single_char(ch_img)
            if t:
                text_parts.append(t)
                confs.append(c)
        if not text_parts:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        raw = "".join(text_parts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        plate, ptype = self._clean(raw)
        valid = bool(plate and self._validate(plate))
        return {
            "plate_number": plate or raw,
            "confidence": avg_conf,
            "valid": valid,
            "type": ptype,
            "raw": raw,
        }

    def enhance_for_display(self, img: np.ndarray) -> np.ndarray:
        """產生攤平+二值化的車牌截圖（存檔顯示用，不用於 OCR）"""
        if img is None or img.size == 0:
            return img
        try:
            corrected = self.perspective_transform(img)
            enhanced = self._enhance_plate(corrected)
            return enhanced
        except Exception:
            return img

    @staticmethod
    def enhance_plate_static(img: np.ndarray) -> np.ndarray:
        """靜態版 enhance（不需要實例化）"""
        if img is None or img.size == 0:
            return img
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            h, w = gray.shape[:2]
            if w < 200:
                scale = 250 / max(w, 1)
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        except Exception:
            return img

    def recognize_easy(self, img: np.ndarray) -> Dict[str, Any]:
        """YOLO 字元偵測辨識車牌"""
        if img is None or img.size == 0:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        try:
            # 原圖直送 OCR 微服務（不做前處理）
            easy_text, easy_conf = self._ocr_easy(img)
            if easy_text and len(easy_text) >= 4:
                plate, ptype = self._clean(easy_text)
                valid = bool(plate and self._validate(plate))
                if valid or easy_conf >= 0.3:
                    return {
                        "plate_number": plate or easy_text,
                        "confidence": easy_conf,
                        "valid": valid,
                        "type": ptype,
                        "raw": easy_text,
                    }
            # EasyOCR 失敗 → fallback 到字元分割 OCR
            char_res = self.recognize_chars(img)
            if char_res.get("plate_number"):
                return char_res
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        except Exception:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}

    def recognize(self, img: np.ndarray) -> Dict[str, Any]:
        """辨識車牌 — 改用 YOLO 字元偵測微服務"""
        if img is None or img.size == 0:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        # 直接走 recognize_easy（YOLO 字元偵測）
        return self.recognize_easy(img)
        # 以下為舊 Tesseract 邏輯（停用）
        best = {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        aggregate: Dict[str, Dict[str, Any]] = {}

        corrected = self.perspective_transform(img)
        plate_regions = self.find_plate_region(corrected)

        for region in plate_regions:
            processed = self.preprocess(region)
            for p_img in processed:
                for config in [self.config_single, self.config_word, self.config_raw, self.config_block]:
                    try:
                        data = pytesseract.image_to_data(p_img, config=config, output_type=pytesseract.Output.DICT)
                        texts = []
                        confs = []
                        for i, text in enumerate(data["text"]):
                            t = str(text or "").strip()
                            if not t or len(t) < 2:
                                continue
                            texts.append(t)
                            raw_conf = str(data["conf"][i]) if i < len(data["conf"]) else "0"
                            try:
                                conf = max(0.0, float(raw_conf))
                            except Exception:
                                conf = 0.0
                            confs.append(conf / 100.0)

                        if not texts:
                            continue

                        raw = "".join(texts)
                        conf = max(confs) if confs else 0.0
                        for plate in self._generate_plate_candidates(raw):
                            valid = self._validate(plate)
                            score = self._score_candidate(plate, conf, raw)
                            _, ptype = self._clean(plate.replace("-", ""))
                            cur = aggregate.get(plate)
                            if not cur:
                                aggregate[plate] = {
                                    "plate_number": plate,
                                    "confidence": conf,
                                    "valid": valid,
                                    "type": ptype,
                                    "raw": raw,
                                    "_score": score,
                                    "_hits": 1,
                                }
                            else:
                                cur["_hits"] += 1
                                cur["confidence"] = max(cur["confidence"], conf)
                                cur["valid"] = cur["valid"] or valid
                                if score >= cur["_score"]:
                                    cur["_score"] = score
                                    cur["raw"] = raw
                                    cur["type"] = ptype
                    except Exception:
                        pass

        for item in aggregate.values():
            final_score = item["_score"] + ((item["_hits"] - 1) * 0.22)
            if best["plate_number"] is None or final_score > best.get("_score", -10):
                best = {
                    "plate_number": item["plate_number"],
                    "confidence": item["confidence"],
                    "valid": item["valid"],
                    "type": item["type"],
                    "raw": item["raw"],
                    "_score": final_score,
                }

        final_score = float(best.get("_score", -10.0))
        best.pop("_score", None)
        if not best.get("plate_number"):
            return best
        if not best.get("valid") and final_score < 1.2:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        if float(best.get("confidence") or 0.0) < 0.10 and final_score < 1.8:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        return best

    def _clean(self, text: str) -> tuple:
        """清理格式 - 嚴格模式，只接受 OCR 結果與台灣車牌格式高度吻合的"""
        if not text:
            return "", "一般"

        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not text:
            return "", "一般"

        # 長度限制
        if len(text) < 5 or len(text) > 8:
            return text, "一般"

        # 嚴格模式：只接受少量字元替換（單個方向：左邊應該是字母，右邊應該是數字）
        # 不允許每個字元都被暴力替換
        candidates = set()

        for cut in (2, 3, 4):
            if not (2 <= cut < len(text) - 1 and len(text[:cut]) <= 4 and len(text[cut:]) <= 4):
                continue
            left = text[:cut]
            right = text[cut:]

            # 1. 完全不替換
            candidates.add(f"{left}-{right}")

            # 2. 左邊字母化（只替換明顯的數字→字母混淆，最多 1 個字元）
            #    右邊數字化（只替換明顯的字母→數字混淆，最多 1 個字元）
            def light_alpha(s):
                """只替換 1~2 個明顯混淆字元，且原本是數字"""
                out = []
                replaced = 0
                for ch in s:
                    if ch.isdigit() and ch in self.TO_ALPHA and replaced < 2:
                        out.append(self.TO_ALPHA[ch])
                        replaced += 1
                    else:
                        out.append(ch)
                return "".join(out)

            def light_digit(s):
                out = []
                replaced = 0
                for ch in s:
                    if ch.isalpha() and ch in self.TO_DIGIT and replaced < 2:
                        out.append(self.TO_DIGIT[ch])
                        replaced += 1
                    else:
                        out.append(ch)
                return "".join(out)

            candidates.add(f"{light_alpha(left)}-{light_digit(right)}")
            candidates.add(f"{light_digit(left)}-{light_alpha(right)}")

        best_plate = ""
        best_score = -10.0
        for cand in candidates:
            cand = re.sub(r"[^A-Z0-9-]", "", cand.upper())
            if "-" not in cand:
                continue
            score = self._segment_profile_score(cand)
            if self._matches_plate_pattern(cand):
                score += 2.0
            if score > best_score:
                best_score = score
                best_plate = cand

        # 嚴格門檻：分數太低不接受（避免亂猜）
        if best_plate and best_score >= 1.0:
            left, right = best_plate.split("-", 1)
            ptype = "機車" if len(best_plate.replace("-", "")) <= 6 else "一般"
            if left.isdigit() and right.isalpha():
                ptype = "一般"
            return best_plate, ptype

        return text, "一般"

    def _validate(self, plate: str) -> bool:
        """驗證台灣車牌格式"""
        if not plate:
            return False
        plate = re.sub(r"[^A-Z0-9-]", "", str(plate or "").upper())
        if len(plate) < 5 or len(plate) > 9:
            return False
        if "-" not in plate:
            return False

        parts = plate.split("-")
        if len(parts) != 2:
            return False

        left, right = parts
        if len(left) > 4 or len(right) > 4:
            return False
        if len(left) < 2 or len(right) < 2:
            return False

        return self._matches_plate_pattern(plate) and self._segment_profile_score(plate) >= 0.4
