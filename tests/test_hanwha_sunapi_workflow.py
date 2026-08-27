from services.hanwha_sunapi import BBox, PtzPosition, PtzStopWindow, build_plate_lpr_workflow


def test_plate_bbox_too_small_triggers_area_zoom():
    result = build_plate_lpr_workflow(
        plate_bbox=BBox(900, 500, 980, 526),
        frame_width=1920,
        frame_height=1080,
        min_lpr_plate_width=160,
        min_lpr_plate_height=48,
        zoom_padding_ratio=0.25,
    )

    assert result["state"] == "zooming"
    assert result["actions"] == ["area_zoom"]
    assert result["lpr_ready"] is False
    assert result["zoom_bbox"] == [880, 494, 1000, 532]


def test_large_plate_is_ready_for_lpr_without_zoom():
    result = build_plate_lpr_workflow(
        plate_bbox=BBox(700, 420, 900, 480),
        frame_width=1920,
        frame_height=1080,
        min_lpr_plate_width=160,
        min_lpr_plate_height=48,
    )

    assert result["state"] == "lpr_ready"
    assert result["actions"] == ["lpr_ready"]
    assert result["lpr_ready"] is True


def test_plate_center_inside_stop_zone_stops_tracking_first():
    result = build_plate_lpr_workflow(
        plate_bbox=BBox(1420, 720, 1500, 748),
        frame_width=1920,
        frame_height=1080,
        stop_zone=BBox(1300, 650, 1700, 900),
    )

    assert result["state"] == "stopped"
    assert result["actions"] == ["stop_tracking"]
    assert result["reached_stop_zone"] is True


def test_ptz_position_inside_stop_window_stops_tracking_first():
    result = build_plate_lpr_workflow(
        plate_bbox=BBox(900, 500, 980, 526),
        frame_width=1920,
        frame_height=1080,
        ptz_position=PtzPosition(pan=178.5, tilt=24.2, zoom=8.0, zoom_pulse=1700),
        ptz_stop_window=PtzStopWindow(pan=180.0, tilt=25.0, pan_tolerance=2.0, tilt_tolerance=1.0),
    )

    assert result["state"] == "stopped"
    assert result["actions"] == ["stop_tracking"]
    assert result["reached_ptz_stop"] is True
