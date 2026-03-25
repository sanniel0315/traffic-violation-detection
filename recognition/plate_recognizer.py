#!/usr/bin/env python3
"""台灣車牌辨識模組 - 增強版 Tesseract"""
import re
from typing import Any, Dict, List

import cv2
import numpy as np
import pytesseract


class PlateRecognizer:
    """車牌辨識器 - 多重預處理 + Tesseract"""

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

    def __init__(self, use_gpu: bool = False):
        self.config_single = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        self.config_block = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        self.config_word = "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        self.config_raw = "--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        print("✅ Tesseract OCR 初始化完成")

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

    def perspective_transform(self, img: np.ndarray) -> np.ndarray:
        """透視變換校正傾斜車牌"""
        if img is None or img.size == 0:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)

                rect = np.zeros((4, 2), dtype=np.float32)
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                w1 = np.linalg.norm(rect[1] - rect[0])
                w2 = np.linalg.norm(rect[2] - rect[3])
                h1 = np.linalg.norm(rect[3] - rect[0])
                h2 = np.linalg.norm(rect[2] - rect[1])

                w = int(max(w1, w2))
                h = int(max(h1, h2))

                if w > 50 and h > 15:
                    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(rect, dst)
                    return cv2.warpPerspective(img, M, (w, h))

        return img

    def preprocess(self, img: np.ndarray) -> List[np.ndarray]:
        """多重預處理"""
        results = []
        if img is None or img.size == 0:
            return results

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        _, w = img.shape[:2]
        if w < 260:
            scale = 360 / max(1, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results.append(gray)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(enhanced)

        denoised = cv2.bilateralFilter(enhanced, 5, 35, 35)
        results.append(denoised)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(binary)
        results.append(255 - binary)

        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        results.append(adaptive)
        results.append(255 - adaptive)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        results.append(sharpened)

        _, enhanced_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(enhanced_binary)
        results.append(255 - enhanced_binary)

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
        if re.search(r"([A-Z0-9])\1\1", plain):
            score -= 1.2
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

    def recognize(self, img: np.ndarray) -> Dict[str, Any]:
        """辨識車牌"""
        if img is None or img.size == 0:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}

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
        if not best.get("valid") and final_score < 2.6:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        if float(best.get("confidence") or 0.0) < 0.22 and final_score < 3.2:
            return {"plate_number": None, "confidence": 0, "valid": False, "type": None, "raw": ""}
        return best

    def _clean(self, text: str) -> tuple:
        """清理格式"""
        if not text:
            return "", "一般"

        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not text:
            return "", "一般"

        candidates = set()
        if 5 <= len(text) <= 8:
            for cut in (2, 3, 4):
                if not (2 <= cut < len(text) - 1 and len(text[:cut]) <= 4 and len(text[cut:]) <= 4):
                    continue
                left = text[:cut]
                right = text[cut:]
                candidates.add(f"{''.join(self.TO_ALPHA.get(ch, ch) for ch in left)}-{''.join(self.TO_DIGIT.get(ch, ch) for ch in right)}")
                candidates.add(f"{''.join(self.TO_DIGIT.get(ch, ch) for ch in left)}-{''.join(self.TO_ALPHA.get(ch, ch) for ch in right)}")
                candidates.add(f"{left}-{right}")
        else:
            candidates.add(text)

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

        if best_plate:
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
