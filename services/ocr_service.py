#!/usr/bin/env python3
"""OCR 微服務 — YOLO 字元偵測"""
import cv2
import numpy as np
import re
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

_char_model = None


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
    chars.sort(key=lambda c: c[0])
    text = ''.join(c[1] for c in chars)
    avg_conf = sum(c[2] for c in chars) / len(chars)
    return {"text": text, "confidence": round(avg_conf, 3), "time": round(dt, 3), "chars": len(chars)}


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
