#!/usr/bin/env python3
"""Model path resolver utilities."""
import os
from pathlib import Path


def _resolve_path(value: str, model_dir: str) -> str:
    if os.path.isabs(value):
        return value
    return str(Path(model_dir) / value)


def get_model_dir() -> str:
    return os.getenv("MODEL_DIR", "/workspace/models")


def get_detect_model_engine() -> str:
    model_dir = get_model_dir()
    value = os.getenv("DETECT_MODEL_ENGINE", "yolov8n.engine")
    return _resolve_path(value, model_dir)


def get_detect_model_pt() -> str:
    model_dir = get_model_dir()
    value = os.getenv("DETECT_MODEL_PT", "yolov8n.pt")
    return _resolve_path(value, model_dir)
