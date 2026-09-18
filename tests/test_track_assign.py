"""一對一軌跡配對 —— 線圈地面實況揭露的漏車(2026-09-18)。

🛑 WN 下匝道線圈與 WN-1 看的是同一條路:12 分鐘線圈 84 輛、攝影機 60 輛(−29%),
   進框事件 180 筆(每台車約 2.1 個軌跡)。舊的配對讓每個偵測框各自找最近軌跡,
   車貼車時兩台共用一個軌跡(少算),快車超出配對距離就另開新軌跡(斷段)。
"""
import os
import time

os.environ.setdefault("AUTH_SECRET", "test-only-not-a-real-secret")


def _tracks(*items, t=None):
    t = time.time() - 0.4 if t is None else t
    return {tid: {"center": c, "t": t, "class_name": cls, **({"vel": v} if v else {})}
            for tid, c, cls, v in items}


def test_old_matcher_merges_adjacent_cars():
    """對照組:舊配法會把相鄰兩台車都配到同一個軌跡 —— 這就是少算的形狀。"""
    from api.routes.stream import _nearest_track_id
    tracks = _tracks((1, (500, 300), "car", None))
    a = _nearest_track_id((500, 330), "car", tracks, max_dist=260)
    b = _nearest_track_id((500, 420), "car", tracks, max_dist=260)
    assert a == b == 1, "舊配法:兩台車共用軌跡 1"


def test_adjacent_cars_get_distinct_tracks():
    from api.routes.stream import _assign_tracks
    tracks = _tracks((1, (500, 300), "car", None))
    out = _assign_tracks([(500, 330, "car"), (500, 420, "car")], tracks, 260, time.time())
    assert out[0] == 1, "近的那台接續舊軌跡"
    assert out[1] is None, "遠的那台要開新軌跡,不能共用"


def test_two_tracks_two_cars_no_swap():
    """兩台排隊的車各自接續,不因為距離相近而交換或合併。"""
    from api.routes.stream import _assign_tracks
    tracks = _tracks((1, (500, 300), "car", None), (2, (500, 420), "car", None))
    out = _assign_tracks([(500, 310, "car"), (500, 432, "car")], tracks, 260, time.time())
    assert out == [1, 2]


def test_velocity_prediction_keeps_fast_car_on_its_track():
    """快車一格移動超過配對距離:沒有預測會斷軌,有速度預測要接得上。"""
    from api.routes.stream import _assign_tracks
    now = time.time()
    tracks = _tracks((7, (100, 300), "car", (500.0, 0.0)), t=now - 0.6)   # 500 px/s 往右
    out = _assign_tracks([(400, 300, "car")], tracks, 120, now)             # 實際移動 300 px
    assert out == [7]
    assert tracks[7]["vel"][0] > 0, "配到後要更新速度"


def test_no_extrapolation_for_stale_track():
    """久沒看到(> 1.5 秒)的軌跡不外推 —— 外推會飄到別台車身上。"""
    from api.routes.stream import _assign_tracks
    now = time.time()
    tracks = _tracks((7, (100, 300), "car", (500.0, 0.0)), t=now - 3.0)
    out = _assign_tracks([(1600, 300, "car")], tracks, 120, now)
    assert out == [None]


def test_cross_class_uses_tighter_distance():
    """車型跳動(car ↔ light_truck)仍接得上,但只在較嚴的距離內。"""
    from api.routes.stream import _assign_tracks
    tracks = _tracks((3, (500, 300), "car", None))
    assert _assign_tracks([(500, 340, "light_truck")], tracks, 100, time.time()) == [3]
    tracks = _tracks((3, (500, 300), "car", None))
    assert _assign_tracks([(500, 380, "light_truck")], tracks, 100, time.time()) == [None]


def test_flag_parsing():
    import api.routes.stream as S
    orig = S._TRACK_ASSIGN_RAW
    try:
        S._TRACK_ASSIGN_RAW = ""
        assert S._track_assign_on(4) is False, "預設關閉 —— 部署程式不改變行為"
        S._TRACK_ASSIGN_RAW = "4,5"
        assert S._track_assign_on(4) and S._track_assign_on("5") and not S._track_assign_on(2)
        S._TRACK_ASSIGN_RAW = "all"
        assert S._track_assign_on(9)
    finally:
        S._TRACK_ASSIGN_RAW = orig
