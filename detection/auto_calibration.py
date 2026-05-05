# -*- coding: utf-8 -*-
"""Vanishing-point based auto-calibration for road speed estimation.
Outputs 4 calibration points + Homography matrix without manual marking.

Uses cv2.HoughLinesP on edge-detected frame, filters lane-direction lines,
estimates vanishing point as robust median of pairwise intersections,
projects an assumed rectangle (lane width × known length) onto image plane.
"""
from __future__ import annotations
import math
from typing import Optional, List, Tuple

import cv2
import numpy as np


def detect_road_lines(frame: np.ndarray, min_length_frac: float = 0.10) -> List[Tuple[int, int, int, int]]:
    """Hough lines on Canny edges. Returns [(x1,y1,x2,y2),...] in pixel coords."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 50, 150)
    min_len = max(20, int(min(h, w) * min_length_frac))
    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=min_len, maxLineGap=20,
    )
    if raw is None:
        return []
    return [tuple(int(v) for v in seg[0]) for seg in raw]


def filter_lane_direction(lines, min_angle_deg=20, max_angle_deg=80):
    """Keep lines whose angle from vertical is in [min,max] (typical lane-line angle range from cam view)."""
    out = []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        # angle from vertical (90 = horizontal, 0 = vertical)
        ang = abs(math.degrees(math.atan2(abs(dx), abs(dy))))
        if min_angle_deg <= ang <= max_angle_deg:
            out.append((x1, y1, x2, y2))
    return out


def line_intersect(l1, l2) -> Optional[Tuple[float, float]]:
    """Intersection of two infinite lines (each given by 2 points)."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (float(px), float(py))


def vanishing_point_from_lines(lines, frame_w: int, frame_h: int) -> Optional[Tuple[float, float]]:
    """Robust VP estimate as median of pairwise line intersections.
    VP must be reasonably above frame center (sky/horizon area)."""
    if len(lines) < 2:
        return None
    xs, ys = [], []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            p = line_intersect(lines[i], lines[j])
            if p is None:
                continue
            # Reject ridiculous outliers (>2x frame size away)
            if abs(p[0]) > 2 * frame_w or abs(p[1]) > 2 * frame_h:
                continue
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        return None
    return float(np.median(xs)), float(np.median(ys))


def calibration_from_vp(
    vp: Tuple[float, float],
    frame_w: int,
    frame_h: int,
    lane_width_m: float = 3.5,
    rect_length_m: float = 10.0,
):
    """Derive 4 pixel points + 4 world points (rectangle on road) from VP.
    Strategy:
      - Place rectangle on road centered horizontally at frame
      - Top of rectangle near 0.55*frame_h (mid-distance)
      - Bottom of rectangle near 0.92*frame_h (close to camera)
      - Top width narrower than bottom (perspective). Use VP to interpolate.
    Returns dict {points_pixel, width_m, length_m, vp}.
    """
    vx, vy = vp
    # Sanity: VP should be roughly above frame middle (horizon area)
    if vy > frame_h * 0.7:
        # VP too low — calibration unreliable
        return None
    # Choose two y-rows on road
    y_far = int(frame_h * 0.55)
    y_near = int(frame_h * 0.90)
    # Horizontal extent at each y, scaled by distance-from-VP factor
    # Closer to VP = narrower. Linearly interpolate width.
    cx = frame_w / 2.0
    def width_at_y(y):
        # If y == vy, width = 0; if y == frame_h, width = full
        denom = max(1.0, frame_h - vy)
        ratio = (y - vy) / denom
        ratio = max(0.0, min(1.0, ratio))
        return frame_w * 0.50 * ratio  # 50% of frame width at bottom
    w_far = width_at_y(y_far)
    w_near = width_at_y(y_near)
    # 4 corners (TL, TR, BR, BL) - clockwise from top-left
    tl = (int(cx - w_far / 2), y_far)
    tr = (int(cx + w_far / 2), y_far)
    br = (int(cx + w_near / 2), y_near)
    bl = (int(cx - w_near / 2), y_near)
    return {
        "points_pixel": [list(tl), list(tr), list(br), list(bl)],
        "width_m": float(lane_width_m),
        "length_m": float(rect_length_m),
        "vp": [int(vx), int(vy)],
    }


def auto_calibrate(frame: np.ndarray, lane_width_m: float = 3.5, rect_length_m: float = 10.0):
    """Top-level: returns calibration dict or None if estimation failed."""
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    raw_lines = detect_road_lines(frame)
    lane_lines = filter_lane_direction(raw_lines)
    vp = vanishing_point_from_lines(lane_lines, w, h)
    if vp is None:
        return None
    calib = calibration_from_vp(vp, w, h, lane_width_m, rect_length_m)
    if calib is None:
        return None
    calib["frame_size"] = [w, h]
    calib["n_lines_detected"] = len(raw_lines)
    calib["n_lane_lines_used"] = len(lane_lines)
    return calib
