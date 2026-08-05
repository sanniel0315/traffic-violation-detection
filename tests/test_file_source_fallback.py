#!/usr/bin/env python3
"""上傳影片檔來源不可以套用 frigate/go2rtc 的 cam_{id} fallback。

回歸案例:某台 clone 機 camera id=2 綁了上傳的 .mov,畫面卻一直播出
frigate cam_2(隧道口)的即時影像 —— 因為 fallback 只看 camera_id、
不看 source。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AUTH_SECRET", "test-only-secret-not-for-production-use-01234567")

from api.routes.stream import _is_file_backed_source, _try_frigate_snapshot  # noqa: E402

FILE_SOURCES = [
    "/files/camera_sources/2020_0710_085818_006_f769d2ba.mov",
    "/files/camera_sources/SIG_8_7fa820f1.mkv",
    "http://127.0.0.1:8000/files/camera_sources/clip.mp4",
    "/home/ubuntu/traffic-violation-detection/output/camera_sources/a.mov",
    "/mnt/nvme/traffic/output/camera_sources/b.MP4",      # 大寫副檔名
    "/files/camera_sources/c.webm?t=10",                  # 帶 query
]

STREAM_SOURCES = [
    "rtsp://127.0.0.1:8554/cam_2",
    "rtsp://admin:pw@111.70.34.184:6554/profile2/media.smp",
    "http://127.0.0.1:1984/api/stream.mjpeg?src=cam_6",
    "/api/nx/stream/abc123",
    "",
    None,
]


def test_file_sources_detected():
    for s in FILE_SOURCES:
        assert _is_file_backed_source(s) is True, f"{s!r} 應判定為檔案來源"


def test_stream_sources_not_file():
    for s in STREAM_SOURCES:
        assert _is_file_backed_source(s) is False, f"{s!r} 不該判定為檔案來源"


def test_frigate_snapshot_skipped_for_file_source():
    """檔案來源要直接回 None,而且不能發出任何 HTTP 請求。"""
    import api.routes.stream as st

    called = []
    original = st.requests.get

    def _spy(*a, **kw):
        called.append(a[0] if a else kw.get("url"))
        raise AssertionError("檔案來源不該打 frigate/go2rtc")

    st.requests.get = _spy
    try:
        for s in FILE_SOURCES:
            assert _try_frigate_snapshot(s, camera_id=2) is None, f"{s!r} 應跳過"
        assert not called, f"不該有 HTTP 請求,實際打了 {called}"
    finally:
        st.requests.get = original


def test_frigate_snapshot_still_tries_for_rtsp():
    """RTSP 來源必須維持原本行為 —— 仍會去打 frigate(這裡讓它失敗即可)。"""
    import api.routes.stream as st

    called = []
    original = st.requests.get

    def _spy(url, **kw):
        called.append(url)
        raise RuntimeError("connection refused (模擬 frigate 不在)")

    st.requests.get = _spy
    try:
        assert _try_frigate_snapshot("rtsp://127.0.0.1:8554/cam_2", camera_id=2) is None
        assert called, "RTSP 來源仍應嘗試 frigate fallback"
        assert "cam_2" in called[0], f"應打 cam_2,實際 {called[0]}"
    finally:
        st.requests.get = original


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{'全部通過' if not failed else str(failed) + ' 項失敗'}")
    sys.exit(1 if failed else 0)
