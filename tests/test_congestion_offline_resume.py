"""相機斷線自動停掉壅塞後,影像回來要自動續跑(2026-09-16 NE-1 停了 35 分鐘沒人拉)。"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture
def M(monkeypatch):
    from api import main as M
    from api.routes import stream as ST
    monkeypatch.setattr(ST, "_shared_frames", {})
    return M


def test_frames_alive_needs_fresh_frame(M):
    from api.routes import stream as ST
    assert M._camera_frames_alive(2) is False              # 完全沒畫面
    ST._shared_frames[2] = {"frame": object(), "ts": time.time() - 60}
    assert M._camera_frames_alive(2) is False              # 舊畫面不算
    ST._shared_frames[2] = {"frame": object(), "ts": time.time()}
    assert M._camera_frames_alive(2) is True


def test_offline_service_resumes_only_when_frames_return(M, monkeypatch):
    """離線標記的服務:沒畫面不動它(避免死循環重啟),畫面回來才續跑。"""
    from api.routes import congestion as C
    from api.routes import stream as ST
    started = []
    monkeypatch.setattr(C, "_start_congestion_service", lambda cam: started.append(cam.id))

    class _Cam:
        id, enabled, detection_enabled = 2, True, False

    def _run_once():
        svc = C.congestion_services.get(2, {})
        ct = svc.get("_thread")
        dead = (ct is not None and not ct.is_alive()) or (ct is None and svc and not svc.get("running"))
        if dead and svc.get("offline") and M._camera_frames_alive(2):
            C.congestion_services.pop(2, None)
            C._start_congestion_service(_Cam())

    C.congestion_services[2] = {"running": False, "offline": True, "_thread": None}
    _run_once()
    assert started == [], "相機還沒回來就不要重啟"

    ST._shared_frames[2] = {"frame": object(), "ts": time.time()}
    _run_once()
    assert started == [2], "影像回來要自動續跑"
