#!/usr/bin/env python3
"""台灣車牌辨識模組 - 增強版 Tesseract"""
import re
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Any

class PlateRecognizer:
    """車牌辨識器 - 多重預處理 + Tesseract"""

    PLATE_PATTERNS = [
        r'^[A-Z]{3}-\d{4}$',
        r'^[A-Z]{2}-\d{4}$',
        r'^[A-Z]{2}\d-\d{4}$',
        r'^\d{4}-[A-Z]{2}$',
        r'^[A-Z]{3}-\d{3}$',
        r'^[A-Z]{2}-\d{3}$',
        r'^\d{3}-[A-Z]{3}$',
        r'^\d{3}-[A-Z]{2}$',
        r'^[A-Z]\d{2}-\d{3}$',
        r'^[A-Z0-9]{2,3}-[A-Z0-9]{3,4}$',
    ]

    def __init__(self, use_gpu: bool = False):
        self.config_single = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        self.config_block = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        print("✅ Tesseract OCR 初始化完成")

    def find_plate_region(self, img: np.ndarray) -> List[np.ndarray]:
        """偵測可能的車牌區域"""
        regions = []
        
        if img is None or img.size == 0:
            return [img] if img is not None else regions
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # 邊緣偵測找矩形
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # 膨脹連接邊緣
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape[:2]
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / ch if ch > 0 else 0
            area = cw * ch
            
            # 車牌比例約 2:1 到 5:1
            if 1.5 < aspect < 6 and area > (w * h * 0.01) and area < (w * h * 0.5):
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + cw + pad)
                y2 = min(h, y + ch + pad)
                regions.append(img[y1:y2, x1:x2])
        
        # 如果沒找到，返回原圖
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
                # 找到四邊形，進行透視變換
                pts = approx.reshape(4, 2).astype(np.float32)
                
                # 排序點：左上、右上、右下、左下
                rect = np.zeros((4, 2), dtype=np.float32)
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]  # 左上
                rect[2] = pts[np.argmax(s)]  # 右下
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]  # 右上
                rect[3] = pts[np.argmax(diff)]  # 左下
                
                # 計算目標尺寸
                w1 = np.linalg.norm(rect[1] - rect[0])
                w2 = np.linalg.norm(rect[2] - rect[3])
                h1 = np.linalg.norm(rect[3] - rect[0])
                h2 = np.linalg.norm(rect[2] - rect[1])
                
                w = int(max(w1, w2))
                h = int(max(h1, h2))
                
                if w > 50 and h > 15:
                    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(rect, dst)
                    warped = cv2.warpPerspective(img, M, (w, h))
                    return warped
        
        return img

    def preprocess(self, img: np.ndarray) -> List[np.ndarray]:
        """多重預處理"""
        results = []
        if img is None or img.size == 0:
            return results

        # 確保彩色
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h, w = img.shape[:2]
        
        # 放大小圖
        if w < 200:
            scale = 300 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. 原圖灰階
        results.append(gray)
        
        # 2. CLAHE 增強
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(enhanced)
        
        # 3. 二值化 (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(binary)
        
        # 4. 反轉二值化
        results.append(255 - binary)
        
        # 5. 自適應二值化
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        results.append(adaptive)
        
        # 6. 銳化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        results.append(sharpened)

        return results

    def recognize(self, img: np.ndarray) -> Dict[str, Any]:
        """辨識車牌"""
        if img is None or img.size == 0:
            return {'plate_number': None, 'confidence': 0, 'valid': False, 'type': None, 'raw': ''}

        best = {'plate_number': None, 'confidence': 0, 'valid': False, 'type': None, 'raw': ''}
        
        # 先嘗試透視校正
        corrected = self.perspective_transform(img)
        
        # 然後找車牌區域
        plate_regions = self.find_plate_region(corrected)
        
        for region in plate_regions:
            processed = self.preprocess(region)
            
            for p_img in processed:
                for config in [self.config_single, self.config_block]:
                    try:
                        data = pytesseract.image_to_data(p_img, config=config, output_type=pytesseract.Output.DICT)
                        
                        texts = []
                        confs = []
                        for i, text in enumerate(data['text']):
                            t = text.strip()
                            if t and len(t) >= 2:
                                texts.append(t)
                                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                                confs.append(conf / 100.0)
                        
                        if not texts:
                            continue
                            
                        raw = ''.join(texts)
                        plate, ptype = self._clean(raw)
                        conf = max(confs) if confs else 0
                        valid = self._validate(plate)
                        
                        # 優先選擇有效格式的結果
                        if valid and (not best['valid'] or conf > best['confidence']):
                            best = {'plate_number': plate, 'confidence': conf, 'valid': True, 'type': ptype, 'raw': raw}
                        elif not best['valid'] and 5 <= len(plate) <= 9 and conf > best['confidence']:
                            best = {'plate_number': plate, 'confidence': conf, 'valid': valid, 'type': ptype, 'raw': raw}
                            
                    except Exception as e:
                        pass

        return best

    def _clean(self, text: str) -> tuple:
        """清理格式"""
        if not text:
            return "", "一般"
        
        # 移除非字母數字
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # OCR 錯誤修正
        chars = list(text)
        for i, c in enumerate(chars):
            if i < 3:  # 前面通常是字母
                if c == '0': chars[i] = 'O'
                elif c == '1': chars[i] = 'I'
                elif c == '5': chars[i] = 'S'
                elif c == '8': chars[i] = 'B'
            else:  # 後面通常是數字
                if c == 'O': chars[i] = '0'
                elif c == 'I': chars[i] = '1'
                elif c == 'S': chars[i] = '5'
                elif c == 'B': chars[i] = '8'
        text = ''.join(chars)
        
        # 格式化加入分隔符
        if len(text) == 7:
            if text[:3].isalpha():
                return f"{text[:3]}-{text[3:]}", "一般"
            elif text[:2].isalpha():
                return f"{text[:3]}-{text[3:]}", "一般"
            else:
                return f"{text[:4]}-{text[4:]}", "一般"
        elif len(text) == 6:
            if text[:3].isalpha():
                return f"{text[:3]}-{text[3:]}", "機車"
            elif text[:2].isalpha():
                return f"{text[:2]}-{text[2:]}", "一般"
            elif text[:4].isdigit():
                return f"{text[:4]}-{text[4:]}", "一般"
            else:
                return f"{text[:3]}-{text[3:]}", "一般"
        elif len(text) == 5:
            if text[:2].isalpha():
                return f"{text[:2]}-{text[2:]}", "機車"
            else:
                return f"{text[:3]}-{text[3:]}", "一般"
                
        return text, "一般"

    def _validate(self, plate: str) -> bool:
        """驗證台灣車牌格式"""
        if not plate:
            return False
        
        # 長度限制 (含分隔符 5-8 字元)
        if len(plate) < 5 or len(plate) > 9:
            return False
        
        # 必須包含分隔符
        if '-' not in plate:
            return False
        
        parts = plate.split('-')
        if len(parts) != 2:
            return False
        
        left, right = parts
        
        # 左右兩邊都不能太長
        if len(left) > 4 or len(right) > 4:
            return False
        if len(left) < 2 or len(right) < 2:
            return False
        
        # 標準格式檢查
        for p in self.PLATE_PATTERNS:
            if re.match(p, plate):
                return True
        
        return False
