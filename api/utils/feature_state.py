#!/usr/bin/env python3
"""功能啟停狀態持久化（重啟後恢復）"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict

STATE_PATH = Path("/workspace/config/system/feature_state.json")
_LOCK = threading.Lock()


def _default_state() -> dict:
    return {
        "updated_at": None,
        "features": {
            "detection": {},
            "congestion": {},
            "lpr": {},
        },
    }


def _load() -> dict:
    try:
        if STATE_PATH.exists():
            raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                state = _default_state()
                feats = raw.get("features", {})
                if isinstance(feats, dict):
                    for key in ("detection", "congestion", "lpr"):
                        val = feats.get(key, {})
                        if isinstance(val, dict):
                            state["features"][key] = {
                                str(k): bool(v) for k, v in val.items()
                            }
                state["updated_at"] = raw.get("updated_at")
                return state
    except Exception:
        pass
    return _default_state()


def _save(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.utcnow().isoformat()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def set_feature_state(feature: str, camera_id: int, enabled: bool) -> None:
    key = str(int(camera_id))
    with _LOCK:
        state = _load()
        feats = state.setdefault("features", {})
        if feature not in feats or not isinstance(feats.get(feature), dict):
            feats[feature] = {}
        feats[feature][key] = bool(enabled)
        _save(state)


def get_feature_state(feature: str) -> Dict[int, bool]:
    with _LOCK:
        state = _load()
    feats = state.get("features", {})
    val = feats.get(feature, {}) if isinstance(feats, dict) else {}
    if not isinstance(val, dict):
        return {}
    out: Dict[int, bool] = {}
    for k, v in val.items():
        try:
            out[int(k)] = bool(v)
        except Exception:
            continue
    return out


def get_feature_enabled(feature: str, camera_id: int, default: bool = False) -> bool:
    return get_feature_state(feature).get(int(camera_id), bool(default))
