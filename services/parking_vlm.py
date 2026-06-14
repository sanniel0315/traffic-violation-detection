"""
VLM 整合 — Qwen2-VL-2B-Instruct 本地 deploy (Jetson Orin).

用途:
- PKLot/YOLO 不一致時 VLM 仲裁 (auto trigger)
- user 手動 query 「為什麼這個位置看不到車?」
- 處理複雜場景 (跨線違停 / 雜物佔用 / 光影遮蔽)

對 ROI crop 後問模型:
"以下圖片是停車場其中一個車位的區域,請判斷:
 1. 該車位狀態 (有車 / 空 / 無法判定)
 2. 若空,為什麼? (純空 / 被腳踏車 / 雜物 / 光線太暗)
 3. 若有車,車輛類型?"

回傳結構化 dict + raw response.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np

_MODEL = None
_PROCESSOR = None
_LOAD_LOCK = threading.Lock()
_LOAD_STATE = {"loading": False, "loaded": False, "error": ""}
# 全域推論鎖 — VLM (torch) generate 與其他 GPU op 併發在 Jetson 上易 native SEGV,
# 所有 VLM 推論 (query_slot / chat) 必須序列化
_INFER_LOCK = threading.Lock()

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_PROMPT_ZH = (
    "這張圖是停車場畫面,目標車位已用紅色粗框標出.請只判斷紅框內的車位狀態,"
    "周圍其他車輛/車位不要管.\n"
    "請用以下格式回答 (只回 3 行不要加其他文字):\n"
    "狀態: 有車 / 空 / 無法判定\n"
    "原因: (空寫 純空地/雜物/腳踏車佔用/光線太暗;有車寫 -)\n"
    "信心: 0-100 整數"
)


def is_loaded() -> bool:
    return _LOAD_STATE.get("loaded", False)


def get_load_state() -> Dict:
    return dict(_LOAD_STATE)


def _ensure_loaded() -> bool:
    """背景 load model (idempotent).Return True 表示 ready,False 表示 loading 中或失敗"""
    global _MODEL, _PROCESSOR
    if _LOAD_STATE.get("loaded"):
        return True
    if _LOAD_STATE.get("loading"):
        return False
    with _LOAD_LOCK:
        if _LOAD_STATE.get("loaded"):
            return True
        if _LOAD_STATE.get("loading"):
            return False
        _LOAD_STATE["loading"] = True
    # 跑 in background thread (不卡 web request)
    def _bg():
        global _MODEL, _PROCESSOR
        try:
            t0 = time.time()
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            # Jetson torch 版本舊 (< 2.5),不支援 SDPA 的 enable_gqa,強制 eager attention
            _MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype="auto", device_map="auto",
                attn_implementation="eager",
            )
            _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
            _LOAD_STATE["loaded"] = True
            _LOAD_STATE["loading"] = False
            print(f"[vlm] Qwen2-VL-2B loaded in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            _LOAD_STATE["error"] = str(e)
            _LOAD_STATE["loading"] = False
            print(f"[vlm] load err: {e}", flush=True)
    threading.Thread(target=_bg, daemon=True, name="vlm_load").start()
    return False


def trigger_load() -> Dict:
    """手動觸發 load (UI 點按鈕用)"""
    _ensure_loaded()
    return get_load_state()


def query_slot(crop_bgr: np.ndarray, prompt: Optional[str] = None,
               max_new_tokens: int = 80) -> Dict:
    """對單一 slot 的 crop image query VLM,return 結構化結果."""
    if not _ensure_loaded():
        return {"ok": False, "error": "VLM 載入中或未載入", "state": get_load_state()}
    if _MODEL is None or _PROCESSOR is None:
        return {"ok": False, "error": "model 未初始化"}
    try:
        from qwen_vl_utils import process_vision_info
        # BGR → RGB → PIL
        from PIL import Image
        if crop_bgr is None or crop_bgr.size == 0:
            return {"ok": False, "error": "crop is empty"}
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt or DEFAULT_PROMPT_ZH},
            ],
        }]
        text = _PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _PROCESSOR(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(_MODEL.device)

        t0 = time.time()
        import torch
        with _INFER_LOCK:
            with torch.no_grad():
                generated_ids = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        # trim input
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output_text = _PROCESSOR.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        dt = time.time() - t0

        parsed = _parse_response(output_text)
        return {
            "ok": True,
            "raw": output_text,
            "latency_sec": round(dt, 2),
            **parsed,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


CHAT_SYSTEM_ZH = (
    "你是交通監控影像助理。根據攝影機畫面,用繁體中文簡潔、具體回答,直接描述你看到的內容。\n"
    "- 數量問題: 若訊息開頭有 [系統偵測] 數據(YOLO 結果,比目測準),直接以它為準回答;"
    "沒有就給目測估計(例「約 5-8 台」)。\n"
    "- 車種 / 有無機車 / 壅塞 / 異常 等: 就畫面實際看到的給具體答案。\n"
    "- 這是監控影像,本來就有一定距離,請以現有畫面盡力回答。**不要**用「看不清楚」「太遠」"
    "「無法判斷」當作回答或開頭。只有畫面完全沒有該資訊時才簡短說明,且不要編造。"
)


def chat(image_bgr: "np.ndarray", question: str,
         history: Optional[List[Dict]] = None,
         detection_hint: Optional[str] = None,
         max_new_tokens: int = 256) -> Dict:
    """通用視覺問答助理 — 一張當下畫面 + 問題 (+ 多輪文字 history) → 自由文字回答.
    history: [{"role": "user"/"assistant", "text": "..."}] (不含本次 question).
    detection_hint: YOLO 偵測 ground truth (例「YOLO 偵測到共 6 台 (汽車 5, 機車 1)」),
        會以 [系統偵測] 前綴注入本次問題,讓數量類問題有準確依據.
    回答 grounded 在傳入的 image (每次都帶當下快照,確保針對最新畫面)."""
    if not _ensure_loaded():
        return {"ok": False, "error": "VLM 載入中或未載入", "state": get_load_state()}
    if _MODEL is None or _PROCESSOR is None:
        return {"ok": False, "error": "model 未初始化"}
    q = (question or "").strip()
    if not q:
        return {"ok": False, "error": "問題為空"}
    try:
        from qwen_vl_utils import process_vision_info
        from PIL import Image
        if image_bgr is None or image_bgr.size == 0:
            return {"ok": False, "error": "畫面抓取失敗 (來源無影像)"}
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        messages = [{"role": "system", "content": [{"type": "text", "text": CHAT_SYSTEM_ZH}]}]
        # 多輪: 先放純文字 history (最多保留最近 8 則,避免 2B context 爆)
        for h in (history or [])[-8:]:
            role = h.get("role")
            txt = (h.get("text") or "").strip()
            if role in ("user", "assistant") and txt:
                messages.append({"role": role, "content": [{"type": "text", "text": txt}]})
        # 本次 user turn: 帶當下畫面 + 問題 (+ YOLO 偵測 ground truth)
        user_text = q
        if detection_hint:
            user_text = f"[系統偵測] {detection_hint}\n\n問題: {q}"
        messages.append({"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": user_text},
        ]})

        text = _PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _PROCESSOR(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(_MODEL.device)

        t0 = time.time()
        import torch
        with _INFER_LOCK:
            with torch.no_grad():
                generated_ids = _MODEL.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        answer = _PROCESSOR.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        # 清掉低解析度偶發的 broken byte
        answer = answer.replace("�", "").strip()
        return {"ok": True, "answer": answer, "latency_sec": round(time.time() - t0, 2)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def _parse_response(text: str) -> Dict:
    """parse 「狀態: / 原因: / 信心:」三行格式.
    Qwen2VL 2B 在低解析度小 crop 偶爾吐 \\ufffd broken byte,parse 時清掉.
    """
    def _clean(v: str) -> str:
        # 去 U+FFFD replacement char (model output 不完整 UTF-8)
        return (v or "").replace("�", "").strip()
    out = {"status": None, "reason": None, "confidence": None}
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("狀態:") or line.startswith("狀態：") or line.lower().startswith("status:"):
            out["status"] = _clean(line.split(":", 1)[-1].split("：", 1)[-1])
        elif line.startswith("原因:") or line.startswith("原因：") or line.lower().startswith("reason:"):
            out["reason"] = _clean(line.split(":", 1)[-1].split("：", 1)[-1])
        elif line.startswith("信心:") or line.startswith("信心：") or line.lower().startswith("confidence:"):
            v = _clean(line.split(":", 1)[-1].split("：", 1)[-1])
            import re
            m = re.search(r"\d+", v)
            if m:
                try:
                    out["confidence"] = int(m.group()) / 100.0
                except Exception:
                    pass
    # 空字串視為 None
    if not out["status"]: out["status"] = None
    if not out["reason"]: out["reason"] = None
    # derive boolean occupied
    s = (out["status"] or "").lower()
    if "有車" in s or "occupied" in s:
        out["occupied"] = True
    elif "空" in s or "empty" in s:
        out["occupied"] = False
    else:
        out["occupied"] = None
    # 低信心 + 矛盾標記: 邏輯不自洽 (e.g. status=有車 + reason=無法判定 + confidence<5%)
    # → 標 unreliable=True 給 UI 顯示「VLM 對小目標信心不足」而非誤導文字
    conf = out["confidence"] or 0
    reason_says_unknown = bool(out["reason"]) and ("無法判定" in out["reason"] or "其他" in out["reason"])
    out["unreliable"] = (conf < 0.05) or (out["occupied"] is True and reason_says_unknown)
    return out


def crop_slot_from_frame(frame: np.ndarray, polygon: List[List[int]],
                          padding: float = 1.0,
                          highlight: bool = True) -> Optional[np.ndarray]:
    """從 frame 把 slot 區域 crop 出來,並用紅色粗框標出 target polygon.

    padding=1.0 表示往外擴 1 倍 polygon 尺寸 — VLM 才有足夠 context (周圍其他車位)
    可以對比.highlight=True 在 crop 上畫紅框讓 VLM 知道要看哪一格.
    """
    if frame is None or not polygon:
        return None
    xs = [float(p[0]) for p in polygon]; ys = [float(p[1]) for p in polygon]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    w = x2 - x1; h = y2 - y1
    dx = w * padding; dy = h * padding
    H, W = frame.shape[:2]
    cx1 = int(max(0, x1 - dx)); cy1 = int(max(0, y1 - dy))
    cx2 = int(min(W, x2 + dx)); cy2 = int(min(H, y2 + dy))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2].copy()
    if highlight:
        # 把 polygon 座標轉成 crop 內相對座標,畫紅色粗框
        pts = np.array([[int(float(p[0]) - cx1), int(float(p[1]) - cy1)] for p in polygon],
                       dtype=np.int32)
        cv2.polylines(crop, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    return crop
