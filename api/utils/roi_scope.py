#!/usr/bin/env python3
"""ROI scope helpers: keep per-feature zones isolated."""
from typing import Iterable, List, Optional

SCOPE_TRAFFIC = "traffic_flow_settings"
SCOPE_SPEED = "speed_detection"
SCOPE_CONGESTION = "congestion_detection"


def _norm_scope(v: Optional[str]) -> str:
    return str(v or "").strip()


def _norm_type(v: Optional[str]) -> str:
    t = str(v or "").strip()
    if t == "detection":
        return "flow_detection"
    if t == "speed":
        return "speed_roi"
    return t


def select_zones(
    zones: Optional[Iterable[dict]],
    *,
    scope: str,
    allowed_types: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Select zones by scope with legacy fallback.

    Rule:
    - If scoped zones exist for the target scope, use only those.
    - Else fallback to legacy zones (no scope).
    """
    items = list(zones or [])
    allow = {_norm_type(t) for t in (allowed_types or [])}

    def _ok_type(z: dict) -> bool:
        if not allow:
            return True
        return _norm_type(z.get("type")) in allow

    scoped = [z for z in items if _norm_scope(z.get("scope")) == scope and _ok_type(z)]
    if scoped:
        return scoped
    return [z for z in items if not _norm_scope(z.get("scope")) and _ok_type(z)]
