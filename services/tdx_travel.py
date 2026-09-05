"""TDX(交通部運輸資料流通服務)eTag 站間旅行時間 —— 給成效報告的「平均旅行時間」用。

使用者:「旅行時間抓 TDX」。現場自己量的旅行時間是進場道長度 ÷ 區間車速 + 停等
延滯(近似、低信心);TDX 的 eTag 站間旅行時間是高公局用 ETC 門架實測的,
是**實測值**。兩者量的路段不同:TDX 量的是國道主線門架之間,現場量的是匝道
端進場道,報告裡要分開標示,不可以混成同一個「旅行時間」。

資料源(TDX v2 Road Traffic):
  靜態 eTag 配對  GET /api/basic/v2/Road/Traffic/ETagPair/Freeway
  即時站間旅行   GET /api/basic/v2/Road/Traffic/Live/ETagPairLive/Freeway
  認證           POST /auth/realms/TDXConnect/protocol/openid-connect/token
                 grant_type=client_credentials(client_id / client_secret)
環境變數:
  TDX_CLIENT_ID / TDX_CLIENT_SECRET   TDX 會員金鑰(沒有就整個模組閒置,不報錯)
  TDX_ROAD_ID        國道編號,預設 000080(國道 8 號)
  TDX_ETAG_PAIRS     要抓的配對 ID,逗號分隔;空 = 抓該國道全部,由 discover() 挑
  TDX_FETCH_SEC      抓取週期秒數,預設 300(TDX 更新頻率約 5 分鐘)
  TRAFFIC_VIOL_DB    存放位置,預設 data/violations.db(表 tdx_travel_time)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

TDX_BASE = "https://tdx.transportdata.tw/api/basic"
TDX_TOKEN_URL = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"

CLIENT_ID = os.getenv("TDX_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("TDX_CLIENT_SECRET", "").strip()
ROAD_ID = os.getenv("TDX_ROAD_ID", "000080").strip()
PAIRS = [p.strip() for p in os.getenv("TDX_ETAG_PAIRS", "").split(",") if p.strip()]
FETCH_SEC = float(os.getenv("TDX_FETCH_SEC", "300") or 300)
DB_PATH = os.getenv("TRAFFIC_VIOL_DB", "data/violations.db")

_token = {"value": None, "exp": 0.0}
_state = {"enabled": bool(CLIENT_ID and CLIENT_SECRET), "last_fetch": None, "last_error": None,
          "rows_total": 0, "pairs": PAIRS}
_thread: Optional[threading.Thread] = None
_stop = threading.Event()


def enabled() -> bool:
    return _state["enabled"]


def status() -> dict:
    return dict(_state)


# ── HTTP ─────────────────────────────────────────────────────────────
def _http(url: str, data: Optional[bytes] = None, headers: Optional[dict] = None, timeout: float = 20) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_token(force: bool = False) -> str:
    """client_credentials 換 access token,快取到期前 60 秒。"""
    if not force and _token["value"] and time.time() < _token["exp"] - 60:
        return _token["value"]
    body = urllib.parse.urlencode({"grant_type": "client_credentials",
                                   "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}).encode()
    raw = _http(TDX_TOKEN_URL, data=body, headers={"content-type": "application/x-www-form-urlencoded"})
    d = json.loads(raw)
    _token["value"] = d["access_token"]
    _token["exp"] = time.time() + float(d.get("expires_in", 1800))
    return _token["value"]


def _get_json(path: str, params: dict) -> list:
    url = TDX_BASE + path + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    tok = get_token()
    try:
        raw = _http(url, headers={"authorization": "Bearer " + tok, "accept": "application/json"})
    except urllib.error.HTTPError as e:      # token 過期 → 換一次再試
        if e.code == 401:
            tok = get_token(force=True)
            raw = _http(url, headers={"authorization": "Bearer " + tok, "accept": "application/json"})
        else:
            raise
    d = json.loads(raw)
    # v2 有的端點回 list,有的包在 {"ETagPairs": [...]} 這類 key 底下
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, list):
                return v
        return []
    return d


# ── 靜態:找站點附近的配對 ──────────────────────────────────────────────
def list_pairs(road_id: str = None) -> list:
    """該國道所有 eTag 配對(含起訖門架里程、方向)。"""
    road_id = road_id or ROAD_ID
    rows = _get_json("/v2/Road/Traffic/ETagPair/Freeway",
                     {"$filter": f"RoadID eq '{road_id}'", "$format": "JSON", "$top": 1000})
    out = []
    for r in rows:
        s, e = r.get("StartETag") or {}, r.get("EndETag") or {}
        out.append({
            "pair_id": r.get("ETagPairID"),
            "road": r.get("RoadName") or road_id,
            "direction": r.get("Direction"),
            "start_id": s.get("ETagGantryID"), "start_km": _km(s.get("LocationMile")),
            "start_lat": _f(s.get("PositionLat")), "start_lon": _f(s.get("PositionLon")),
            "end_id": e.get("ETagGantryID"), "end_km": _km(e.get("LocationMile")),
            "end_lat": _f(e.get("PositionLat")), "end_lon": _f(e.get("PositionLon")),
            "length_km": r.get("Distance"),
        })
    return out


def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def _dist_km(lat1, lon1, lat2, lon2) -> Optional[float]:
    """兩點距離(公里),等距近似 —— 站點附近幾公里內夠用。"""
    if None in (lat1, lon1, lat2, lon2):
        return None
    dy = (lat2 - lat1) * 110.9
    dx = (lon2 - lon1) * 102.5
    return (dx * dx + dy * dy) ** 0.5


def discover_by_coord(lat: float, lon: float, road_id: str = None, radius_km: float = 8.0) -> list:
    """用站點座標(使用者在地圖頁定的)找門架配對 —— 不靠里程猜。

    每個配對算:兩端門架到站點的距離、站點是否落在兩門架之間(沿門架連線投影
    0~1 之間)。「跨過站點」的配對排最前,其次依最近門架距離。
    """
    out = []
    for p in list_pairs(road_id):
        d1 = _dist_km(lat, lon, p["start_lat"], p["start_lon"])
        d2 = _dist_km(lat, lon, p["end_lat"], p["end_lon"])
        if d1 is None or d2 is None:
            continue
        ax, ay = (p["start_lon"] - lon) * 102.5, (p["start_lat"] - lat) * 110.9
        bx, by = (p["end_lon"] - lon) * 102.5, (p["end_lat"] - lat) * 110.9
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        t = (-(ax * vx + ay * vy) / L2) if L2 > 0 else None     # 站點在門架連線上的投影參數
        spans = (t is not None and 0.0 <= t <= 1.0)
        # 站點到門架連線的垂直距離(跨過時才有意義)
        perp = None
        if spans:
            px, py = ax + t * vx, ay + t * vy
            perp = (px * px + py * py) ** 0.5
        p.update({"dist_start_km": round(d1, 2), "dist_end_km": round(d2, 2),
                  "spans_site": spans, "offset_km": None if perp is None else round(perp, 2)})
        if min(d1, d2) <= radius_km:
            out.append(p)
    return sorted(out, key=lambda x: (not x["spans_site"], x["offset_km"] if x["offset_km"] is not None else 99,
                                      min(x["dist_start_km"], x["dist_end_km"])))


def discover(center_km: float, road_id: str = None, span_km: float = 6.0) -> list:
    """列出跨過或鄰近指定里程(交流道)的配對,給人挑 TDX_ETAG_PAIRS 用。"""
    out = []
    for p in list_pairs(road_id):
        a, b = p["start_km"], p["end_km"]
        if a is None or b is None:
            continue
        lo, hi = min(a, b), max(a, b)
        p["spans_center"] = lo <= center_km <= hi
        p["dist_km"] = 0.0 if p["spans_center"] else min(abs(center_km - lo), abs(center_km - hi))
        if p["dist_km"] <= span_km:
            out.append(p)
    return sorted(out, key=lambda x: (not x["spans_center"], x["dist_km"]))


def _km(mile) -> Optional[float]:
    """TDX LocationMile 格式如 '7K+500' → 7.5。"""
    if mile is None:
        return None
    try:
        s = str(mile).upper().replace(" ", "")
        if "K" in s:
            k, rest = s.split("K", 1)
            m = rest.replace("+", "") or "0"
            return float(k) + float(m) / 1000.0
        return float(s)
    except Exception:
        return None


# ── 即時:抓站間旅行時間並落庫 ──────────────────────────────────────────
def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS tdx_travel_time (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, pair_id TEXT, direction TEXT,
        travel_time_sec REAL, speed_kmh REAL, vehicle_count INTEGER,
        data_time TEXT, fetched_at TEXT)""")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_tdx_travel_ts ON tdx_travel_time(ts, pair_id)")


def fetch_live(road_id: str = None) -> list:
    road_id = road_id or ROAD_ID
    rows = _get_json("/v2/Road/Traffic/Live/ETagPairLive/Freeway",
                     {"$filter": f"RoadID eq '{road_id}'", "$format": "JSON", "$top": 1000})
    out = []
    for r in rows:
        pid = r.get("ETagPairID")
        if PAIRS and pid not in PAIRS:
            continue
        # Flows 依車種拆,總旅行時間/車速通常在 Flows 內;有的版本在頂層
        flows = r.get("Flows") or []
        tt = r.get("TravelTime")
        sp = r.get("SpaceMeanSpeed")
        cnt = r.get("VehicleCount")
        if tt is None and flows:
            tts = [f.get("TravelTime") for f in flows if f.get("TravelTime")]
            sps = [f.get("SpaceMeanSpeed") for f in flows if f.get("SpaceMeanSpeed")]
            cnt = sum(int(f.get("VehicleCount") or 0) for f in flows)
            tt = (sum(tts) / len(tts)) if tts else None
            sp = (sum(sps) / len(sps)) if sps else None
        out.append({"pair_id": pid, "direction": r.get("Direction"),
                    "travel_time_sec": tt, "speed_kmh": sp, "vehicle_count": cnt,
                    "data_time": r.get("DataCollectTime") or r.get("EndTime")})
    return out


def store(rows: list, db_path: str = None) -> int:
    if not rows:
        return 0
    conn = sqlite3.connect(db_path or DB_PATH, timeout=20)
    _ensure_table(conn)
    now = datetime.now().isoformat(timespec="seconds")
    n = 0
    for r in rows:
        ts = _local_iso(r.get("data_time")) or now
        # 同一配對同一資料時間不重複寫(TDX 5 分鐘更新一次,我們 5 分鐘抓一次會撞到)
        dup = conn.execute("SELECT 1 FROM tdx_travel_time WHERE pair_id=? AND data_time=? LIMIT 1",
                           (r["pair_id"], r.get("data_time"))).fetchone()
        if dup:
            continue
        conn.execute("INSERT INTO tdx_travel_time(ts,pair_id,direction,travel_time_sec,speed_kmh,vehicle_count,data_time,fetched_at)"
                     " VALUES(?,?,?,?,?,?,?,?)",
                     (ts, r["pair_id"], r.get("direction"), r.get("travel_time_sec"), r.get("speed_kmh"),
                      r.get("vehicle_count"), r.get("data_time"), now))
        n += 1
    conn.commit()
    conn.close()
    return n


def _local_iso(s) -> Optional[str]:
    """TDX 時間帶 +08:00,轉成本地 ISO(不帶時區),跟號誌側一致。"""
    if not s:
        return None
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if d.tzinfo is not None:
            d = d.astimezone(tz=None).replace(tzinfo=None)
        return d.isoformat(timespec="seconds")
    except Exception:
        return None


def summarize(since_local: str, until_local: str, db_path: str = None, pairs: Optional[list] = None) -> dict:
    """時段內各配對的平均旅行時間/車速(給 /report 用)。"""
    conn = sqlite3.connect(f"file:{db_path or DB_PATH}?mode=ro", uri=True, timeout=20)
    try:
        q = ("SELECT pair_id, direction, COUNT(*), AVG(travel_time_sec), AVG(speed_kmh), SUM(vehicle_count) "
             "FROM tdx_travel_time WHERE ts>=? AND ts<? ")
        args = [since_local, until_local]
        pl = pairs or PAIRS
        if pl:
            q += "AND pair_id IN (%s) " % ",".join("?" * len(pl))
            args += pl
        q += "GROUP BY pair_id, direction"
        rows = conn.execute(q, args).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    per = [{"pair_id": r[0], "direction": r[1], "n": r[2],
            "travel_time_sec": round(r[3], 1) if r[3] is not None else None,
            "speed_kmh": round(r[4], 1) if r[4] is not None else None,
            "vehicles": r[5]} for r in rows]
    tts = [p["travel_time_sec"] for p in per if p["travel_time_sec"] is not None]
    return {"source": "TDX eTag 站間旅行時間(國道主線,實測)", "method": "measured",
            "pairs": per, "avg_travel_time_sec": round(sum(tts) / len(tts), 1) if tts else None,
            "n": sum(p["n"] for p in per)}


# ── 背景抓取 ───────────────────────────────────────────────────────────
def _loop() -> None:
    while not _stop.is_set():
        try:
            rows = fetch_live()
            n = store(rows)
            _state["rows_total"] += n
            _state["last_fetch"] = datetime.now().isoformat(timespec="seconds")
            _state["last_error"] = None
        except Exception as e:
            _state["last_error"] = f"{type(e).__name__}: {e}"
        _stop.wait(FETCH_SEC)


def start() -> bool:
    """有金鑰才啟動;沒有就閒置,不報錯 —— 現場沒申請 TDX 時系統其他部分照常。"""
    global _thread
    if not enabled():
        return False
    if _thread is not None and _thread.is_alive():
        return False
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="tdx-travel", daemon=True)
    _thread.start()
    return True


def stop() -> None:
    _stop.set()
