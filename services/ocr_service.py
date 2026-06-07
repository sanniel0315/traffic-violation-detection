#!/usr/bin/env python3
"""OCR 微服務 — YOLO 字元偵測 + 台灣車牌格式修復"""
import cv2
import numpy as np
import re
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

_char_model = None


# ─── 台灣車牌格式修復 ───
# OCR 字元相似誤判修正 (字母 → 數字 / 數字 → 字母)
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0",
             "I": "1", "L": "1", "Z": "2", "S": "5",
             "G": "6", "T": "7", "B": "8"}
_TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "4": "A",
             "5": "S", "6": "G", "7": "T", "8": "B"}

# 台灣車牌常見格式 (len, [letter_positions, digit_positions])
# 格式: (total_len, (字母位置 set, 數字位置 set))，dash 位置由 split_idx 決定
_PLATE_FORMATS = [
    # 7 碼: ABC-1234 / 1234-ABC / AB1-2345 / 12-3456 等
    (7, (frozenset({0,1,2}), frozenset({3,4,5,6})), 3),  # AAA-DDDD (新式 3+4)
    (7, (frozenset({4,5,6}), frozenset({0,1,2,3})), 4),  # DDDD-AAA
    (7, (frozenset({0,1}),   frozenset({2,3,4,5,6})), 2), # AA-DDDDD (機車部分)
    # 6 碼: AB-1234 / 1234-AB / ABC-123 / 123-ABC / A1-2345 / A12-345
    (6, (frozenset({0,1,2}), frozenset({3,4,5})), 3),  # AAA-DDD
    (6, (frozenset({3,4,5}), frozenset({0,1,2})), 3),  # DDD-AAA
    (6, (frozenset({0,1}),   frozenset({2,3,4,5})), 2),  # AA-DDDD
    (6, (frozenset({4,5}),   frozenset({0,1,2,3})), 4),  # DDDD-AA
    (6, (frozenset({0}),     frozenset({1,2,3,4,5})), 1),  # A-DDDDD (少見)
    # 5 碼: AB-123 / 123-AB / 機車舊式
    (5, (frozenset({0,1,2}), frozenset({3,4})), 3),  # AAA-DD
    (5, (frozenset({0,1}),   frozenset({2,3,4})), 2),  # AA-DDD
    (5, (frozenset({3,4}),   frozenset({0,1,2})), 3),  # DDD-AA
]


_PLATE_LETTERS_BLACKLIST = frozenset('IOQ')  # 台灣車牌不使用 I/O/Q


def _repair_plate(text: str) -> tuple:
    """嘗試把 raw OCR text 修復成符合台灣車牌格式的形式。
    回傳 (修復後 text with dash, score 0-1, matched)。
    matched=True 條件:
      1. 對到 _PLATE_FORMATS 內一個格式
      2. swap 次數 <= 2
      3. letter 部分不含 I/O/Q (台灣車牌規範)
    """
    if not text:
        return (text, 0.0, False)
    raw = re.sub(r'[^A-Z0-9]', '', text.upper())
    n = len(raw)
    best = None
    for total, (letters_set, digits_set), split in _PLATE_FORMATS:
        if n != total:
            continue
        repaired_chars = []
        swap_count = 0
        ok = True
        for i, ch in enumerate(raw):
            target_is_letter = i in letters_set
            target_is_digit  = i in digits_set
            is_letter = ch.isalpha()
            is_digit  = ch.isdigit()
            if target_is_letter and is_letter:
                repaired_chars.append(ch)
            elif target_is_digit and is_digit:
                repaired_chars.append(ch)
            elif target_is_letter and is_digit:
                mapped = _TO_ALPHA.get(ch)
                if mapped:
                    repaired_chars.append(mapped); swap_count += 1
                else:
                    ok = False; break
            elif target_is_digit and is_letter:
                mapped = _TO_DIGIT.get(ch)
                if mapped:
                    repaired_chars.append(mapped); swap_count += 1
                else:
                    ok = False; break
            else:
                ok = False; break
        if not ok:
            continue
        # letter 部分檢查 I/O/Q
        letter_chars = [repaired_chars[i] for i in letters_set]
        has_blacklist = any(c in _PLATE_LETTERS_BLACKLIST for c in letter_chars)
        # score: swap=0 → 1.0；每 swap 扣 0.25；blacklist 扣 0.4
        score = 1.0 - swap_count * 0.25 - (0.4 if has_blacklist else 0.0)
        score = max(0.0, score)
        # matched 條件: 不含 blacklist letters 且 swap <= 2
        is_matched = (not has_blacklist) and (swap_count <= 2)
        repaired = ''.join(repaired_chars[:split]) + '-' + ''.join(repaired_chars[split:])
        if best is None or score > best[1]:
            best = (repaired, score, is_matched)
    if best:
        return best
    # 沒對到任何格式 — 回傳原始 (加 dash 嘗試切 3-4 / 4-3 / 2-4 等)
    if n == 7:
        return (raw[:3] + '-' + raw[3:], 0.0, False)
    if n == 6:
        return (raw[:3] + '-' + raw[3:], 0.0, False)
    return (raw, 0.0, False)


def get_char_model():
    global _char_model
    if _char_model is None:
        from ultralytics import YOLO
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'lpr', 'Charcter-LP.pt')
        _char_model = YOLO(model_path, task='detect')
        dummy = np.zeros((100, 300, 3), dtype=np.uint8)
        _char_model(dummy, conf=0.3, verbose=False)
        print(f"✅ YOLO 字元偵測 ready (classes={len(_char_model.names)})", flush=True)
    return _char_model


def ocr_plate(img_bytes: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"text": None, "confidence": 0.0}
    h, w = img.shape[:2]
    if w < 150:
        scale = 200 / max(w, 1)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    model = get_char_model()
    t0 = time.time()
    results = model(img, conf=0.3, verbose=False)
    dt = time.time() - t0

    raw_chars = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            ch = model.names.get(cls, '?')
            raw_chars.append((x1, y1, x2, y2, ch, conf))

    if not raw_chars:
        return {"text": None, "confidence": 0.0, "time": dt}

    # y 座標過濾：只保留主要行的字元
    y_centers = [(c[1] + c[3]) / 2 for c in raw_chars]
    median_y = sorted(y_centers)[len(y_centers) // 2]
    char_h = np.median([c[3] - c[1] for c in raw_chars])
    tolerance = max(char_h * 0.8, 10)
    chars = [(c[0], c[4], c[5]) for c in raw_chars if abs((c[1] + c[3]) / 2 - median_y) < tolerance]

    if not chars:
        return {"text": None, "confidence": 0.0, "time": dt}

    # 過濾 '-' 分隔線（由 _clean 處理格式）
    chars = [(x, ch, cf) for x, ch, cf in chars if ch != '-']
    if not chars:
        return {"text": None, "confidence": 0.0, "time": dt}

    # 個別字元 conf < 0.4 過濾 — 避免單一糊字拖低整體判斷
    chars = [(x, ch, cf) for x, ch, cf in chars if cf >= 0.4]
    if not chars:
        return {"text": None, "confidence": 0.0, "time": dt}

    chars.sort(key=lambda c: c[0])
    raw_text = ''.join(c[1] for c in chars)
    avg_conf = sum(c[2] for c in chars) / len(chars)

    # 漏字偵測 — 從 raw_chars 算字符寬度,跟 chars span 比推算「理論字符數」
    # 例 BJX-9202 漏 J: chars=6 但 span/char_width ≈ 7 → 推測漏 1 字符
    missing_char_penalty = 1.0
    if len(chars) >= 3 and len(raw_chars) >= 3:
        # 用 raw_chars 算 char 寬度 (純英數,排除 dash)
        char_widths = [c[2] - c[0] for c in raw_chars if c[4] != '-']
        if char_widths:
            import statistics as _st
            char_w_med = _st.median(char_widths)
            xs = sorted(c[0] for c in chars)
            span = xs[-1] - xs[0]
            if char_w_med > 0:
                # 推算理論字符數 = span / (char_width × ~1.1 含字間距)
                expected_chars = span / (char_w_med * 1.1) + 1
                # 實際字符數 比推算少 >= 0.6 → 漏字
                shortfall = expected_chars - len(chars)
                if shortfall >= 0.6:
                    # 扣分 — shortfall=1 (漏 1 字) penalty 0.5,shortfall=2 penalty 0.35
                    missing_char_penalty = max(0.35, 1.0 - shortfall * 0.4)

    # 台灣車牌格式修復 + 字元相似性 swap
    repaired_text, repair_score, matched = _repair_plate(raw_text)
    if matched:
        # 符合格式 — 用修復後 text，conf 加權 (原 OCR conf × repair score)
        final_text = repaired_text
        final_conf = avg_conf * (0.6 + 0.4 * repair_score) * missing_char_penalty
    else:
        # 不合台灣車牌格式 — 大幅降權，pipeline 端 min_confidence filter 會自動丟
        final_text = repaired_text  # 仍含 dash 方便人眼判讀
        final_conf = avg_conf * 0.35 * missing_char_penalty

    return {
        "text": final_text,
        "raw_text": raw_text,
        "confidence": round(final_conf, 3),
        "raw_confidence": round(avg_conf, 3),
        "matched_format": matched,
        "missing_char_suspected": missing_char_penalty < 1.0,
        "time": round(dt, 3),
        "chars": len(chars),
    }


class OCRHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        result = ocr_plate(body)
        resp = json.dumps(result).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(resp))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self):
        resp = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    port = 8010
    print(f"🚀 OCR 微服務（YOLO 字元偵測）啟動 port={port}", flush=True)
    get_char_model()
    server = HTTPServer(('127.0.0.1', port), OCRHandler)
    server.serve_forever()
