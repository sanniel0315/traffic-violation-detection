"""對外報表 API 端點 — VD 車流報表 + 壅塞報表"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Optional

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.models import AggregationJobState, ApiKey, CongestionReportAgg, get_db
from api.utils.api_key_auth import require_scope, resolve_api_key
from api.utils.report_aggregation import (
    BUCKET_SECONDS,
    _LARGE_LABELS,
    INCREMENTAL_JOB_NAME,
    _camera_meta,
    direction_label,
    normalize_direction,
    build_vd_report_rows,
    normalize_bucket_size,
    to_utc_naive,
)

router = APIRouter(prefix="/api/v1/external", tags=["External API"])

TZ_TAIPEI = timezone(timedelta(hours=8))

_BUCKET_INTERVALS = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "1h": timedelta(hours=1)}
_MAX_RANGE = {"1m": timedelta(hours=24), "5m": timedelta(days=7), "1h": timedelta(days=90)}
_MAX_RECORDS = 10000
_DEVICE_ID = "jetson-nx-001"


# 對外規格的回應範例。FastAPI 對「回 dict 又沒宣告 response_model」的端點
# 只會產生空的 200 schema,匯入 Postman / Swagger 後看得到怎麼打、
# 卻看不到會拿到什麼 —— 客戶得先打一次才知道欄位長相。
# 這裡的數字取自現場實際回應(2026-08-10),欄位齊全可直接對照。
_SPEC_EXAMPLES = {
    "/api/v1/external/realtime": {
        "status": "success",
        "data": {
            "mode": "minute",
            "window_sec": None,
            "elapsed_sec": 16,
            "period": {"start": "2026-08-10T09:26:00+08:00",
                       "end": "2026-08-10T09:26:16+08:00"},
            "stats": {
                "record_count": 4, "bucket_count": 1, "detector_count": 4,
                "overall": {
                    "total_flow": 20, "in_flow": 5, "out_flow": 3,
                    "small_vehicle_flow": 17, "large_vehicle_flow": 3,
                    "avg_speed_kmh": 26.2, "avg_occupancy_pct": 29.6,
                    "max_queue_length_m": 11.5,
                },
                "by_detector": [
                    {"detector_id": "台62基隆段隧道口", "road_name": "台62線基隆段",
                     "total_flow": 17, "in_flow": 5, "out_flow": 3,
                     "small_vehicle_flow": 14, "large_vehicle_flow": 3,
                     "avg_speed_kmh": None, "avg_occupancy_pct": 29.6,
                     "max_queue_length_m": 11.5},
                ],
                "peak_bucket": {"time_start": "2026-08-10T09:26:00+08:00",
                                "detector_id": "台62基隆段隧道口", "total_flow": 17},
            },
            "records": [{
                "detector_id": "台62基隆段隧道口",
                "road_name": "台62線基隆段",
                "status": "active",
                "time_start": "2026-08-10T09:26:00+08:00",
                "time_end": "2026-08-10T09:26:16+08:00",
                "direction": "straight", "direction_label": "直行",
                "total_flow": 17, "flow_per_hour": 1020.0,
                "small_vehicle_flow": 14, "large_vehicle_flow": 3,
                "avg_speed_kmh": None, "avg_occupancy_pct": 29.6,
                "direction_counts": {"straight": 12, "INOUT": 5, "IN": 5, "EXIT": 3},
                "avg_queue_length_m": 0.8, "max_queue_length_m": 11.5,
                "queue_duration_sec": 3.0, "max_queue_duration_sec": 1.0,
                "lane_count": 2,
                "lanes": [
                    {"lane_no": 1, "flow": 12, "small_vehicle_flow": 12,
                     "large_vehicle_flow": 0, "avg_speed_kmh": None,
                     "avg_occupancy_pct": 25.4, "avg_queue_length_m": None,
                     "max_queue_length_m": None, "queue_duration_sec": None,
                     "max_queue_duration_sec": None},
                    {"lane_no": 2, "flow": 5, "small_vehicle_flow": 2,
                     "large_vehicle_flow": 3, "avg_speed_kmh": None,
                     "avg_occupancy_pct": 37.4, "avg_queue_length_m": 0.8,
                     "max_queue_length_m": 11.5, "queue_duration_sec": 3.0,
                     "max_queue_duration_sec": 1.0},
                ],
                "in_flow": 5, "out_flow": 3,
            }],
        },
        "meta": {"request_time": "2026-08-10T09:26:16+08:00", "api_version": "1.0",
                 "device_id": "jetson-nx-001", "format": "json"},
    },
    "/api/v1/external/vd-report/latest": {
        "status": "success",
        "data": {
            "interval": "1m",
            "period": {"start": "2026-08-10T09:26:00+08:00",
                       "end": "2026-08-10T09:28:00+08:00"},
            # 資料已聚合到哪個時刻;period.end 不會超過它,可用來判斷新鮮度
            "aggregated_through": "2026-08-10T09:28:00+08:00",
            "stats": {
                "record_count": 8, "bucket_count": 2, "detector_count": 4,
                "overall": {"total_flow": 96, "in_flow": 15, "out_flow": 14,
                            "small_vehicle_flow": 82, "large_vehicle_flow": 14,
                            "avg_speed_kmh": 28.4, "avg_occupancy_pct": 24.1,
                            "max_queue_length_m": 32.0},
                "by_detector": [{"detector_id": "台62基隆段隧道口",
                                 "road_name": "台62線基隆段", "total_flow": 44,
                                 "in_flow": 15, "out_flow": 14,
                                 "small_vehicle_flow": 40, "large_vehicle_flow": 4,
                                 "avg_speed_kmh": None, "avg_occupancy_pct": 30.5,
                                 "max_queue_length_m": 11.5}],
                "peak_bucket": {"time_start": "2026-08-10T09:26:00+08:00",
                                "detector_id": "國8", "total_flow": 29},
            },
            "records": [{
                "detector_id": "台62基隆段隧道口", "road_name": "台62線基隆段",
                "status": "active",
                "time_start": "2026-08-10T09:26:00+08:00",
                "time_end": "2026-08-10T09:27:00+08:00",
                "direction": "straight", "direction_label": "直行",
                "total_flow": 23, "small_vehicle_flow": 20, "large_vehicle_flow": 3,
                "avg_speed_kmh": None, "avg_occupancy_pct": 31.2,
                "direction_counts": {"straight": 15, "INOUT": 8, "IN": 8, "EXIT": 7},
                "avg_queue_length_m": 0.8, "max_queue_length_m": 11.5,
                "queue_duration_sec": 3.0, "max_queue_duration_sec": 1.0,
                "lane_count": 2,
                "lanes": [{"lane_no": 1, "flow": 15}, {"lane_no": 2, "flow": 8}],
                "in_flow": 8, "out_flow": 7,
            }],
        },
    },
    "/api/v1/external/vd-report": {
        "status": "success",
        "data": {
            "interval": "1h",
            "period": {"start": "2026-08-10T08:00:00+08:00",
                       "end": "2026-08-10T09:00:00+08:00"},
            "records": [{
                "detector_id": "台62基隆段隧道口", "road_name": "台62線基隆段",
                "status": "active",
                "time_start": "2026-08-10T09:26:00+08:00",
                "time_end": "2026-08-10T09:27:00+08:00",
                "direction": "straight", "direction_label": "直行",
                "total_flow": 23, "small_vehicle_flow": 20, "large_vehicle_flow": 3,
                "avg_speed_kmh": None, "avg_occupancy_pct": 31.2,
                "direction_counts": {"straight": 15, "INOUT": 8, "IN": 8, "EXIT": 7},
                "avg_queue_length_m": 0.8, "max_queue_length_m": 11.5,
                "queue_duration_sec": 3.0, "max_queue_duration_sec": 1.0,
                "lane_count": 2,
                "lanes": [{"lane_no": 1, "flow": 15}, {"lane_no": 2, "flow": 8}],
                "in_flow": 8, "out_flow": 7,
            }],
        },
    },
    "/api/v1/external/congestion-report/latest": {
        "status": "success",
        "data": {
            "interval": "1m",
            "period": {"start": "2026-08-10T09:26:00+08:00",
                       "end": "2026-08-10T09:28:00+08:00"},
            "stats": {
                "record_count": 6, "bucket_count": 2, "detector_count": 3,
                "overall": {"avg_occupancy_pct": 24.1, "max_occupancy_pct": 62.0,
                            "avg_vehicle_count": 1.9, "avg_stopped_vehicle_count": 0.4,
                            "avg_queue_length_m": 12.5, "max_queue_length_m": 32.0},
                "by_detector": [], "peak_bucket": None,
            },
            "records": [{
                "detector_id": "2", "camera_name": "台62基隆段隧道口",
                "time_start": "2026-08-10T09:26:00+08:00",
                "time_end": "2026-08-10T09:27:00+08:00",
                "zone_name": "車流區 1", "lane_no": 1, "direction": "straight",
                "avg_occupancy_pct": 31.2, "max_occupancy_pct": 62.0,
                "avg_vehicle_count": 1.9, "avg_stopped_vehicle_count": 0.4,
                "avg_queue_length_m": 12.5, "max_queue_length_m": 32.0,
                "queue_active_duration_sec": 8.0, "sample_count": 30,
            }],
        },
    },
    "/api/v1/external/congestion-report": {
        "status": "success",
        "data": {
            "interval": "1h",
            "period": {"start": "2026-08-10T08:00:00+08:00",
                       "end": "2026-08-10T09:00:00+08:00"},
            "records": [{
                "detector_id": "2", "camera_name": "台62基隆段隧道口",
                "time_start": "2026-08-10T08:00:00+08:00",
                "time_end": "2026-08-10T09:00:00+08:00",
                "zone_name": "車流區 1", "lane_no": 1, "direction": "straight",
                "avg_occupancy_pct": 28.7, "max_occupancy_pct": 74.0,
                "avg_vehicle_count": 2.1, "avg_stopped_vehicle_count": 0.3,
                "avg_queue_length_m": 14.8, "max_queue_length_m": 45.5,
                "queue_active_duration_sec": 126.0, "sample_count": 1800,
            }],
        },
    },
    "/api/v1/external/streams": {
        "status": "success",
        "data": {
            "device_id": "jetson-nx-001", "host": "192.168.0.102",
            "ports": {"rtsp": 8554, "http": 1984}, "stream_count": 1,
            "streams": [{
                "stream_id": "cam_2", "camera_id": 2, "name": "台62基隆段隧道口",
                "location": "", "online": True, "codec": "h264", "profile": "Main",
                "resolution": "1920x1080", "fps": 30.0, "bytes_received": 544435123,
                "urls": {
                    "rtsp": "rtsp://192.168.0.102:8554/cam_2",
                    "hls": "http://192.168.0.102:1984/api/stream.m3u8?src=cam_2",
                    "mjpeg": "http://192.168.0.102:1984/api/stream.mjpeg?src=cam_2",
                    "webrtc_signal": "http://192.168.0.102:1984/api/ws?src=cam_2",
                },
            }],
        },
    },
}


def _require_docs_token(request: Request, token: str, db: Session) -> None:
    """文件頁的認證：?token= 或 X-API-Key header 皆可。

    瀏覽器直接開文件網址沒辦法帶 header，所以要接受查詢字串；
    但驗證邏輯與一般 API 呼叫共用 resolve_api_key，不另寫一套。
    """
    value = str(token or "").strip() or str(request.headers.get("X-API-Key") or "").strip()
    if not resolve_api_key(value, db):
        raise HTTPException(
            status_code=401,
            detail="需要有效的 API 金鑰（網址加 ?token=<金鑰> 或帶 X-API-Key header）",
        )


@router.get("/openapi.json", summary="對外 API 規格（只含 /api/v1/external/*）", include_in_schema=False)
def external_openapi(request: Request,
                     token: str = Query("", description="API 金鑰；也可改用 X-API-Key header"),
                     db: Session = Depends(get_db)):
    """只回對外那幾條的 OpenAPI 規格。

    🛑 不要把完整規格給客戶 —— 那裡面有 173 條內部端點
    （/api/auth/users、/api/io/do/{ch}、/api/frigate/restart …），
    客戶只需要 /api/v1/external/* 這幾條。
    瀏覽器開文件頁沒辦法帶 header，所以這裡也接受 ?token=。
    """
    _require_docs_token(request, token, db)
    full = request.app.openapi()
    paths = {p: v for p, v in (full.get("paths") or {}).items() if p.startswith("/api/v1/external")}
    # servers 要帶上本次請求的主機 —— 匯入 Postman / 其他工具時才知道要打哪台。
    # 沒有這段的話匯入後每一條都要手動補 baseUrl。
    base = str(request.base_url).rstrip("/")

    # 🛑 components 不可以整包照抄 —— 完整規格裡有 46 個內部模型
    # (LoginRequest / UserPasswordUpdateRequest / DOCommand …),
    # 全給客戶等於把內部資料結構一起交出去。只留對外真的會用到的:
    # 錯誤模型 + 安全定義。
    full_components = full.get("components") or {}
    keep_schemas = {
        name: schema
        for name, schema in (full_components.get("schemas") or {}).items()
        if name in ("HTTPValidationError", "ValidationError")
    }

    # FastAPI 對「回 dict 又沒宣告 response_model」的端點只會產生空的 200 schema,
    # 匯入 Postman 後看得到怎麼打、卻看不到會拿到什麼。補上實際回應範例。
    for path, item in paths.items():
        example = _SPEC_EXAMPLES.get(path)
        if not example:
            continue
        media = (((item.get("get") or {}).get("responses") or {})
                 .get("200", {}).get("content", {}).get("application/json"))
        if media is not None:
            media["example"] = example

    return {
        "openapi": full.get("openapi", "3.1.0"),
        "info": {
            "title": "交通資料對外 API",
            "version": str(full.get("info", {}).get("version", "1.0")),
            "description": (
                "VD 車流報表 / 壅塞報表 / 串流清單。所有端點需帶 X-API-Key header。\n\n"
                "即時查詢用 /realtime：\n"
                "  mode=minute — 從當前整分累積到現在，跨分歸零\n"
                "  mode=window — 往回推 window_sec 秒的滾動視窗\n"
                "要不重疊的固定統計區間（適合存檔）用 /vd-report/latest。\n\n"
                "判讀注意：total_flow 不可與 in_flow/out_flow 相加（會重複計算）；"
                "沒有畫進出線的偵測器不會有 in_flow/out_flow 這兩個欄位；"
                "null 代表沒有量到，與 0 的意義不同。"
            ),
        },
        "servers": [{"url": base, "description": "本系統"}],
        "security": [{"APIKeyHeader": []}],
        "paths": paths,
        "components": {
            "schemas": keep_schemas,
            "securitySchemes": full_components.get("securitySchemes") or {},
        },
    }


@router.get("/docs", summary="對外 API 文件（Swagger UI）", include_in_schema=False)
def external_docs(request: Request,
                  token: str = Query("", description="API 金鑰"),
                  db: Session = Depends(get_db)):
    _require_docs_token(request, token, db)
    spec_url = f"/api/v1/external/openapi.json?token={quote(token)}" if token else "/api/v1/external/openapi.json"
    return get_swagger_ui_html(openapi_url=spec_url, title="交通資料對外 API")


def _aggregation_watermark(db: Session):
    """背景聚合 job 已經處理到哪個時刻（UTC, tz-aware）；沒有紀錄回 None。

    比這個時刻新的桶,聚合表裡的數字還不是最終值。
    """
    try:
        state = (
            db.query(AggregationJobState)
            .filter(AggregationJobState.job_name == INCREMENTAL_JOB_NAME)
            .first()
        )
    except Exception:
        return None
    value = getattr(state, "last_processed_at", None) if state else None
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _meta(fmt: str = "json") -> dict:
    return {
        "request_time": datetime.now(TZ_TAIPEI).isoformat(),
        "api_version": "1.0",
        "device_id": _DEVICE_ID,
        "format": fmt,
    }


# ── 即時資料查詢 ────────────────────────────────────────────────────
# 報表端點回的是「已結束的統計區間」,最小 1 分鐘桶 → 每 20 秒輪詢會拿到同一份。
# 這支不同:回「現在往回推 N 秒」的滾動視窗,每次呼叫的視窗都不一樣,
# 所以每 20 秒查都是新數字。資料直接讀原始表,不經聚合,沒有聚合延遲。
#
# 🛑 查詢一定要用 INDEXED BY 釘住 created_at 索引。
#    SQLite planner 在有低選擇性條件時會挑錯索引,現場實測:
#      traffic_events     滾動 60 秒  364 ms → 0.3 ms
#      congestion_samples 滾動 60 秒 5663 ms → 0.5 ms
#    差一萬倍。沒釘住的話,20 秒打一次會把系統拖垮。
# 🛑 大車判定一律用 report_aggregation._LARGE_LABELS,不可以在這裡另定一套。
# 曾經自己寫成 ("truck","bus","trailer","tractor") —— 那個 "truck" 用 LIKE 比對
# 會把 light_truck(小貨車)一起吃進大車,同一分鐘即時與報表的 small/large 就對不起來
# (實測 國8 即時 7/5、報表 8/4;total_flow 相同但分法不同)。
# 專案規則:未細分的 truck 視為小車(保守估計,避免誤算為大車)。


@router.get("/realtime", summary="即時 VD 車流報表（預設 mode=minute 分鐘內累積；mode=window 為滾動視窗）")
def external_realtime(
    mode: str = Query(
        # 預設 minute:客戶要的是「這一分鐘到目前為止的累積」。
        # 🛑 不要把預設改回 window —— 匯入 OpenAPI 的工具(Postman 等)query 參數
        # 預設是沒勾選的,呼叫端一旦忘了帶 mode,會靜默拿到另一種語意的數字
        # (滾動視窗會與前一次重疊,拿去存檔就重複計算)。預設值必須是最常用、
        # 且拿錯也最不會出事的那個。
        "minute", pattern="^(window|minute)$",
        description="minute=從當前整分累積到現在(跨分歸零,預設);window=往回推 window_sec 秒的滾動視窗",
    ),
    window_sec: int = Query(60, ge=10, le=600, description="滾動視窗長度(mode=window 時有效)"),
    detector_id: Optional[int] = Query(None, description="攝影機 camera_id;留空=全部"),
    api_key: ApiKey = Depends(require_scope("vd_report")),
    db: Session = Depends(get_db),
):
    """即時 VD 車流報表 —— 兩種取樣方式，記錄格式與 /vd-report 完全相同。

    **mode=minute（分鐘內累積）** — 起點釘在當前整分，終點是「現在」，跨分歸零。
    分鐘內任何時間查，拿到的都是「從整分到當下的累積」：

        18:49:14  起點 18:49:00  經過 14 秒  累積  2
        18:49:54  起點 18:49:00  經過 54 秒  累積 18   ← 同一起點，只增不減
        18:50:15  起點 18:50:00  經過 15 秒  累積  6   ← 跨分歸零

    分鐘結束時的累積值，等於 /vd-report 該分鐘桶的值（已實測，含大車 0 不符）。
    統計單位是一分鐘、又想在分鐘內看到進度時用這個。

    **mode=window（滾動視窗，預設）** — 每次往回看 window_sec 秒，起點終點一起移動。
    適合畫面顯示，數字較平滑。⚠️ 視窗互相重疊，直接加總會重複計算，不可拿來存檔。

    共通：
    - 資料直接讀原始表、不經聚合，沒有聚合延遲。
    - `elapsed_sec` 是本次涵蓋的秒數；`flow_per_hour` 依它換算成每小時車流率。
    - `in_flow` / `out_flow` **只有畫了進出線的攝影機才有這兩個欄位**（不是給 0）。
    - `status`：active = 正常運作，disabled = 該攝影機停用（數值恆 0，不是沒有車）。
    - 沒測速的攝影機 `avg_speed_kmh` 為 null（不是 0 —— 0 代表車速真的是 0）。

    要存檔請改用 /vd-report/latest：那是不重疊的固定分鐘桶，一分鐘只有一個答案。
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if mode == "minute":
        # 分鐘內累積:起點釘在當前整分,終點是「現在」。
        # 同一分鐘內連續查詢,起點不動、終點往前 → 數字只會往上累加;
        # 跨到下一分鐘就自動歸零重算。整分那一刻累積為 0(這一分鐘才剛開始),
        # 而分鐘結束時的累積值,會等於 /vd-report 該分鐘桶的值。
        start = now.replace(second=0)
        span = now - start
    else:
        span = timedelta(seconds=int(window_sec))
        start = now - span
    since = start.replace(tzinfo=None)
    elapsed = max(int(span.total_seconds()), 0)

    camera_by_id, _ = _camera_meta(db)
    large_case = " OR ".join(
        "LOWER(COALESCE(label,'')) LIKE '%%%s%%'" % lbl for lbl in _LARGE_LABELS
    )
    cam_filter = " AND camera_id = :cam" if detector_id is not None else ""
    params = {"since": since.isoformat(sep=" ")}
    if detector_id is not None:
        params["cam"] = int(detector_id)

    flow_rows = db.execute(text(f"""
        SELECT camera_id, COALESCE(lane_no, 0) AS lane_no,
               UPPER(COALESCE(direction, '')) AS direction,
               COUNT(*) AS n,
               AVG(CASE WHEN speed_kmh > 0 THEN speed_kmh END) AS avg_speed,
               AVG(CASE WHEN occupancy >= 0 THEN occupancy END) AS avg_occ,
               SUM(CASE WHEN ({large_case}) THEN 1 ELSE 0 END) AS large_n
        FROM traffic_events INDEXED BY ix_traffic_events_created_at
        WHERE created_at >= :since{cam_filter}
        GROUP BY camera_id, lane_no, direction
    """), params).fetchall()

    cong_rows = db.execute(text(f"""
        SELECT camera_id, COALESCE(lane_no, 0) AS lane_no, is_overall,
               AVG(estimated_queue_length_m) AS avg_q,
               MAX(estimated_queue_length_m) AS max_q,
               SUM(CASE WHEN queue_active = 1 THEN COALESCE(sample_interval_sec, 0) ELSE 0 END) AS q_dur,
               MAX(queue_duration_sec) AS max_q_dur,
               AVG(occupancy) AS avg_occ
        FROM congestion_samples INDEXED BY ix_congestion_samples_created_at
        WHERE created_at >= :since{cam_filter}
        GROUP BY camera_id, lane_no, is_overall
    """), params).fetchall()

    # 組成與 build_vd_report_rows 相同結構的 row,才能共用輸出函式
    rows: dict[int, dict] = {}

    def ensure(cam: int) -> dict:
        if cam not in rows:
            meta = camera_by_id.get(cam, {})
            rows[cam] = {
                "deviceId": str(meta.get("camera_name") or f"cam_{cam}"),
                "roadName": str(meta.get("road_name") or "未知"),
                "timeKey": int(start.timestamp() * 1000),
                "timeText": start.astimezone(TZ_TAIPEI).strftime("%Y-%m-%d %H:%M:%S"),
                "direction": normalize_direction(meta.get("direction")),
                "directionText": direction_label(normalize_direction(meta.get("direction"))),
                "directionCounts": {}, "totalFlow": 0, "smallFlow": 0, "largeFlow": 0,
                "avgSpeed": None, "avgOccupancyPct": None,
                "avgQueueLengthM": None, "maxQueueLengthM": None,
                "queueDurationSec": None, "maxQueueDurationSec": None,
                "laneCount": int(meta.get("lane_count") or 0),
                "inoutEnabled": bool(meta.get("inout_enabled")),
                "status": "active" if meta.get("enabled", True) else "disabled",
                # 設定好的車道先建出來(值 0) —— 沒有車的時段若回空陣列,
                # 呼叫端照 lane_count 跑迴圈讀 lanes[i] 會爆掉。
                "lanes": {int(ln): {
                    "flow": 0, "smallFlow": 0, "largeFlow": 0, "avgSpeed": None,
                    "avgOccupancyPct": None, "avgQueueLengthM": None, "maxQueueLengthM": None,
                    "queueDurationSec": None, "maxQueueDurationSec": None,
                    "_sp_sum": 0.0, "_sp_n": 0, "_oc_sum": 0.0, "_oc_n": 0,
                } for ln in (meta.get("lane_nos") or [])},
                "_sp_sum": 0.0, "_sp_n": 0, "_oc_sum": 0.0, "_oc_n": 0,
            }
        return rows[cam]

    for r in flow_rows:
        if r[0] is None:
            continue
        d = ensure(int(r[0]))
        n, large = int(r[3] or 0), int(r[6] or 0)
        direction = normalize_direction(r[2])
        d["directionCounts"][direction] = d["directionCounts"].get(direction, 0) + n
        if direction in ("IN", "EXIT", "OUT"):
            continue          # 進出場事件只進 directionCounts,不進總流量
        d["totalFlow"] += n
        d["largeFlow"] += large
        d["smallFlow"] += n - large
        if r[4] is not None:
            d["_sp_sum"] += float(r[4]) * n
            d["_sp_n"] += n
        if r[5] is not None:
            d["_oc_sum"] += float(r[5]) * n
            d["_oc_n"] += n
        lane_no = int(r[1] or 0)
        if lane_no > 0:
            lane = d["lanes"].setdefault(lane_no, {
                "flow": 0, "smallFlow": 0, "largeFlow": 0, "avgSpeed": None,
                "avgOccupancyPct": None, "avgQueueLengthM": None, "maxQueueLengthM": None,
                "queueDurationSec": None, "maxQueueDurationSec": None,
                "_sp_sum": 0.0, "_sp_n": 0, "_oc_sum": 0.0, "_oc_n": 0,
            })
            lane["flow"] += n
            lane["largeFlow"] += large
            lane["smallFlow"] += n - large
            # 🛑 SQL 是 GROUP BY (camera, lane, direction),同一條車道會有多組,
            # 直接指派等於被最後一組蓋掉;要用事件數加權累加,和偵測器層一致。
            # 佔用率原本整個漏掉,車道層永遠是 null —— 但原始事件是有值的。
            if r[4] is not None:
                lane["_sp_sum"] += float(r[4]) * n
                lane["_sp_n"] += n
            if r[5] is not None:
                lane["_oc_sum"] += float(r[5]) * n
                lane["_oc_n"] += n
            d["laneCount"] = max(int(d["laneCount"] or 0), lane_no)

    for r in cong_rows:
        if r[0] is None:
            continue
        d = ensure(int(r[0]))
        lane_no, is_overall = int(r[1] or 0), bool(r[2])
        target = d if is_overall else d["lanes"].get(lane_no)
        if target is None:
            continue
        if r[3] is not None:
            target["avgQueueLengthM"] = round(float(r[3]), 1) or None
        if r[4] is not None:
            target["maxQueueLengthM"] = round(float(r[4]), 1) or None
        if r[5] is not None:
            target["queueDurationSec"] = round(float(r[5]), 1) or None
        if r[6] is not None:
            target["maxQueueDurationSec"] = round(float(r[6]), 1) or None
        if is_overall and r[7] is not None and not d["_oc_n"]:
            occ = float(r[7])
            d["avgOccupancyPct"] = round(occ * 100.0 if occ <= 1 else occ, 1)

    # 視窗內完全沒有事件的相機也要出現(值為 0),否則呼叫端會以為那台不見了
    for cam, meta in camera_by_id.items():
        if meta.get("vd_eligible") and (detector_id is None or cam == int(detector_id)):
            ensure(cam)

    out_rows = []
    for cam, d in sorted(rows.items()):
        if d["_sp_n"]:
            d["avgSpeed"] = d["_sp_sum"] / d["_sp_n"]
        if d["_oc_n"]:
            occ = d["_oc_sum"] / d["_oc_n"]
            d["avgOccupancyPct"] = occ * 100.0 if occ <= 1 else occ
        for lane in d["lanes"].values():
            if lane["_sp_n"]:
                lane["avgSpeed"] = lane["_sp_sum"] / lane["_sp_n"]
            if lane["_oc_n"]:
                occ_l = lane["_oc_sum"] / lane["_oc_n"]
                lane["avgOccupancyPct"] = occ_l * 100.0 if occ_l <= 1 else occ_l
            for key in ("_sp_sum", "_sp_n", "_oc_sum", "_oc_n"):
                lane.pop(key, None)
        for key in ("_sp_sum", "_sp_n", "_oc_sum", "_oc_n"):
            d.pop(key, None)
        out_rows.append(d)

    records = _vd_rows_to_records(out_rows, "1m", span=span)
    # 即時特有:換算每小時車流率,呼叫端不必知道視窗多長就能直接顯示
    for rec in records:
        # 用實際經過秒數換算 —— mode=minute 時經過秒數會一直變,
        # 用 window_sec 算會得到錯的車流率。分鐘剛開始(elapsed 很小)時
        # 這個外推值抖動很大,呼叫端要顯示的話建議等 elapsed 夠大再用。
        rec["flow_per_hour"] = (round(rec["total_flow"] * 3600.0 / elapsed, 1)
                                if elapsed > 0 else None)

    return {
        "status": "success",
        "data": {
            "mode": mode,
            "window_sec": int(window_sec) if mode == "window" else None,
            "elapsed_sec": elapsed,
            "period": {"start": start.astimezone(TZ_TAIPEI).isoformat(),
                       "end": now.astimezone(TZ_TAIPEI).isoformat()},
            "stats": _vd_stats(records),
            "records": records,
        },
        "meta": _meta(),
    }


def _validate_time_range(start_time: datetime, end_time: datetime, bucket_size: str) -> None:
    if end_time <= start_time:
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "error": {"code": "INVALID_PARAMETER", "message": "end_time 必須大於 start_time"},
        })
    max_range = _MAX_RANGE.get(bucket_size, timedelta(days=7))
    if (end_time - start_time) > max_range:
        raise HTTPException(status_code=400, detail={
            "status": "error",
            "error": {"code": "RANGE_TOO_LARGE", "message": f"時間範圍超過上限 ({bucket_size} 最多 {max_range})"},
        })


def _num(value, digits: int = 1):
    """有量到才給數字，沒量到給 null。

    🛑 不可以用 `value or 0` —— 那會把「沒有測速」變成「時速 0」。
    對交控中心來說 0 km/h 是「完全停止＝嚴重壅塞」，與「這支沒裝測速」
    是天差地遠的兩件事。實測 cam_2 近 10 分鐘 329 個事件的 speed_kmh
    全是 NULL（=0 的 0 筆），API 卻回 0。占用率同理。
    """
    return None if value is None else round(float(value), digits)


def _vd_rows_to_records(rows: list, bucket: str, span: timedelta | None = None) -> list:
    """把 build_vd_report_rows 的原始列轉成對外 JSON 記錄。

    vd-report / vd-report-latest / realtime 三支共用這裡 —— 客戶只要寫一套解析。
    span:即時查詢用的滾動視窗長度;不給就用 bucket 對應的固定長度。
    """
    bucket_delta = span or _BUCKET_INTERVALS.get(bucket, timedelta(minutes=5))
    records = []
    for row in rows:
        ts = row.get("timeKey")
        time_start = datetime.fromtimestamp(ts / 1000, tz=TZ_TAIPEI).isoformat() if ts else None
        time_end = (datetime.fromtimestamp(ts / 1000, tz=TZ_TAIPEI) + bucket_delta).isoformat() if ts else None

        lanes = []
        for lane_no, ld in (row.get("lanes") or {}).items():
            lanes.append({
                "lane_no": int(lane_no) if str(lane_no).isdigit() else lane_no,
                "flow": ld.get("flow", 0),
                "small_vehicle_flow": ld.get("smallFlow", 0),
                "large_vehicle_flow": ld.get("largeFlow", 0),
                "avg_speed_kmh": _num(ld.get("avgSpeed")),
                "avg_occupancy_pct": _num(ld.get("avgOccupancyPct")),
                "avg_queue_length_m": round(ld.get("avgQueueLengthM") or 0, 1) if ld.get("avgQueueLengthM") else None,
                "max_queue_length_m": round(ld.get("maxQueueLengthM") or 0, 1) if ld.get("maxQueueLengthM") else None,
                "queue_duration_sec": round(ld.get("queueDurationSec") or 0, 1) if ld.get("queueDurationSec") else None,
                "max_queue_duration_sec": round(ld.get("maxQueueDurationSec") or 0, 1) if ld.get("maxQueueDurationSec") else None,
            })

        records.append({
            "detector_id": row.get("deviceId", ""),
            "road_name": row.get("roadName", ""),
            # active = 正常運作;disabled = 該攝影機被停用(數值恆 0,不是沒有車)
            "status": row.get("status", "active"),
            "time_start": time_start,
            "time_end": time_end,
            "direction": row.get("direction", ""),
            "direction_label": row.get("directionText", ""),
            "total_flow": row.get("totalFlow", 0),
            "small_vehicle_flow": row.get("smallFlow", 0),
            "large_vehicle_flow": row.get("largeFlow", 0),
            "avg_speed_kmh": _num(row.get("avgSpeed")),
            "avg_occupancy_pct": _num(row.get("avgOccupancyPct")),
            "direction_counts": row.get("directionCounts") or {},
            # 進出流量只算「轉場事件」:
            #   IN            = 車輛進入 ROI
            #   OUT / EXIT    = 車輛離開 ROI(EXIT 是內部用的名稱)
            # 不可把 INOUT 加進來 —— INOUT 是該 ROI 的「一般流量計數」
            # (見 stream.py 與 report_aggregation.py 的並存設計),
            # 它已經計入 total_flow;再加到 in/out 就是重複計算。
            # 87 實測:directionCounts={straight:67, IN:40, INOUT:40, EXIT:39},
            # 舊公式會得到 in=80/out=79(灌水約 100%),正解是 in=40/out=39。
            "avg_queue_length_m": round(row.get("avgQueueLengthM") or 0, 1) if row.get("avgQueueLengthM") else None,
            "max_queue_length_m": round(row.get("maxQueueLengthM") or 0, 1) if row.get("maxQueueLengthM") else None,
            "queue_duration_sec": round(row.get("queueDurationSec") or 0, 1) if row.get("queueDurationSec") else None,
            "max_queue_duration_sec": round(row.get("maxQueueDurationSec") or 0, 1) if row.get("maxQueueDurationSec") else None,
            "lane_count": row.get("laneCount", 0),
            "lanes": lanes,
        })
        # 🛑 只有畫了進出線的相機才輸出 in_flow / out_flow。
        # 沒畫的給 0,呼叫端分不出「沒有車進出」和「這支根本不做進出計數」;
        # 用「欄位在不在」表達最清楚。
        if row.get("inoutEnabled"):
            dc = row.get("directionCounts") or {}
            records[-1]["in_flow"] = int(dc.get("IN", 0))
            records[-1]["out_flow"] = int(dc.get("OUT", 0)) + int(dc.get("EXIT", 0))
    return records


def _vd_stats(records: list) -> dict:
    """對一批記錄做統計摘要:整體 + 各偵測器 + 尖峰時段(flow 加權平均速度/佔有率)。"""
    def summarize(recs: list) -> dict:
        flow = sum(r["total_flow"] for r in recs)
        small = sum(r["small_vehicle_flow"] for r in recs)
        large = sum(r["large_vehicle_flow"] for r in recs)
        # 只有「該組裡真的有畫進出線的相機」才給進出量。
        # 用 .get(...,0) 一律回 0 的話,records[] 遵守「有畫才給」、
        # by_detector 卻每台都回 0 —— 同一份回應兩套規則,使用者無所適從。
        has_io = any("in_flow" in r for r in recs)
        in_f = sum(r.get("in_flow", 0) for r in recs) if has_io else None
        out_f = sum(r.get("out_flow", 0) for r in recs) if has_io else None
        # 車流加權平均;速度與佔有率各自獨立分母 — 塞車分鐘速度為 None 但佔有率有值,
        # 共用分母會讓佔有率被灌爆超過 100%。
        spd_w = sum(r["total_flow"] for r in recs if r["total_flow"] and r["avg_speed_kmh"])
        spd = sum(r["avg_speed_kmh"] * r["total_flow"] for r in recs if r["total_flow"] and r["avg_speed_kmh"])
        occ_w = sum(r["total_flow"] for r in recs if r["total_flow"] and r["avg_occupancy_pct"])
        occ = sum(r["avg_occupancy_pct"] * r["total_flow"] for r in recs if r["total_flow"] and r["avg_occupancy_pct"])
        qmax = [r["max_queue_length_m"] for r in recs if r["max_queue_length_m"] is not None]
        return {
            "total_flow": flow,
            **({"in_flow": in_f, "out_flow": out_f} if has_io else {}),
            "small_vehicle_flow": small,
            "large_vehicle_flow": large,
            "avg_speed_kmh": round(spd / spd_w, 1) if spd_w else None,
            "avg_occupancy_pct": round(occ / occ_w, 1) if occ_w else None,
            "max_queue_length_m": round(max(qmax), 1) if qmax else None,
        }

    by_detector = {}
    for r in records:
        by_detector.setdefault(r["detector_id"], []).append(r)
    peak = max(records, key=lambda r: r["total_flow"], default=None)
    return {
        # 🛑 這三個要分清楚,以前只有一個 bucket_count 而且數的是 len(records)
        # (= 偵測器 x 時間桶的組合數),名稱與內容對不上:
        #   realtime 4 台 x 1 個視窗 → 顯示 bucket_count 4,但根本沒有「桶」
        #   vd-report 4 台 x 3 桶     → 顯示 bucket_count 12,實際只有 3 個桶
        "record_count": len(records),
        "bucket_count": len({r["time_start"] for r in records}),
        "detector_count": len({r["detector_id"] for r in records}),
        "overall": summarize(records),
        "by_detector": [
            {"detector_id": did, "road_name": recs[0]["road_name"], **summarize(recs)}
            for did, recs in by_detector.items()
        ],
        "peak_bucket": {"time_start": peak["time_start"], "detector_id": peak["detector_id"],
                        "total_flow": peak["total_flow"]} if peak and peak["total_flow"] else None,
    }


# ── VD 車流報表 ──────────────────────────────────────────────

@router.get("/vd-report", summary="VD 車流報表 — 指定時間區間")
async def external_vd_report(
    start_time: datetime = Query(
        ..., description="起始時間 ISO8601,例 2026-07-13T00:00:00+08:00。不帶時區視為 UTC;查台北時間請帶 +08:00"),
    end_time: datetime = Query(
        ..., description="結束時間 ISO8601,例 2026-07-13T23:59:59+08:00。格式同 start_time"),
    detector_id: Optional[int] = Query(None, description="攝影機 camera_id(2/3/6/8,見 /streams);留空=全部"),
    interval: str = Query("5m", description="時間桶大小:1m / 5m / 1h"),
    format: str = Query("json", description="輸出格式:json / csv"),
    api_key: ApiKey = Depends(require_scope("vd_report")),
    db: Session = Depends(get_db),
):
    bucket = normalize_bucket_size(interval)
    _validate_time_range(start_time, end_time, bucket)

    # 只讀聚合表（背景 job 每分鐘增量維護），不在請求當下重建 — 重建的 DELETE
    # 會與即時事件寫入搶鎖，正是報表 0 筆/500 的根因。
    rows = build_vd_report_rows(db, start_time, end_time, bucket, camera_id=detector_id)
    records = _vd_rows_to_records(rows, bucket)

    if len(records) > _MAX_RECORDS:
        raise HTTPException(status_code=413, detail={
            "status": "error",
            "error": {"code": "TOO_MANY_RECORDS", "message": f"結果超過 {_MAX_RECORDS} 筆，請縮小時間範圍"},
        })

    if format == "csv":
        return _vd_csv_response(records, start_time, end_time, bucket)

    return {
        "status": "success",
        "data": {
            "interval": bucket,
            "period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "records": records,
        },
        "meta": _meta("json"),
    }


@router.get("/vd-report/latest", summary="VD 車流報表(快捷) — 最近 N 分鐘 + 統計摘要")
async def external_vd_report_latest(
    minutes: int = Query(5, ge=1, le=360, description="回傳最近幾個完整桶(interval=1m 時即分鐘數)"),
    interval: str = Query("1m"),
    detector_id: Optional[int] = Query(None),
    include_records: bool = Query(True, description="False 只回統計摘要,不回逐桶明細"),
    api_key: ApiKey = Depends(require_scope("vd_report")),
    db: Session = Depends(get_db),
):
    """快捷查詢:免自己算時間/UTC,自動回最近 N 個「已結束」的桶 + 統計摘要。
    上層每分鐘輪詢就打這個(建議 minutes 給 3~5 容忍聚合延遲,以 time_start 去重 upsert)。"""
    bucket = normalize_bucket_size(interval)
    step = BUCKET_SECONDS[bucket]
    now = datetime.now(timezone.utc)
    # 對齊到桶邊界 → 排除當下還在累積的桶(end 為 exclusive 上界)
    epoch = int(now.timestamp())
    end = datetime.fromtimestamp(epoch - (epoch % step), tz=timezone.utc)

    # 🛑 再往回退到「聚合已經算完」的位置。
    # 報表讀的是聚合表,而聚合是背景每 60 秒跑一次 —— 桶雖然在時間上結束了,
    # 聚合還沒輪到它,這時查出來是 0。實測 11:47 那個桶:API 回 0,原始資料
    # 其實有 5 台和 4 台;等下一輪聚合跑過就自己修正成 5 和 4。
    # 對呼叫端來說 0 和「真的沒車」長得一模一樣,分不出來 —— 所以乾脆不要送
    # 還沒算完的桶,寧可最新資料晚一兩分鐘,也不要給一個會變的數字。
    watermark = _aggregation_watermark(db)
    if watermark is not None:
        w_epoch = int(watermark.timestamp())
        end = min(end, datetime.fromtimestamp(w_epoch - (w_epoch % step), tz=timezone.utc))

    start = end - timedelta(seconds=step * minutes)

    rows = build_vd_report_rows(db, start, end, bucket, camera_id=detector_id)
    records = _vd_rows_to_records(rows, bucket)
    stats = _vd_stats(records)

    data = {
        "interval": bucket,
        "period": {"start": start.astimezone(TZ_TAIPEI).isoformat(),
                   "end": end.astimezone(TZ_TAIPEI).isoformat()},
        # 資料已聚合到哪個時刻 —— period.end 不會超過它。呼叫端可用它判斷
        # 新鮮度(正常落後一個聚合週期以內;明顯落後代表背景 job 有問題)。
        "aggregated_through": end.astimezone(TZ_TAIPEI).isoformat(),
        "stats": stats,
    }
    if include_records:
        data["records"] = records
    return {"status": "success", "data": data, "meta": _meta("json")}


def _vd_csv_response(records: list, start_time, end_time, bucket):
    output = io.StringIO()
    writer = csv.writer(output)
    # in_flow / out_flow 是「整框進出」的量,不分車道。
    # 有車道的相機只會輸出逐車道列(沒有偵測器層級那一列),所以把值掛在
    # 「該筆的第一條車道列」,其餘車道列留空 —— 這樣依 detector+time 分組加總
    # 會拿到正確的一份;若複製到每條車道,下游一加總就變成 N 倍灌水。
    header = [
        "detector_id", "road_name", "time_start", "time_end", "direction",
        "lane_no", "flow", "small_vehicle_flow", "large_vehicle_flow",
        "avg_speed_kmh", "avg_occupancy_pct", "avg_queue_length_m", "max_queue_length_m",
        "queue_duration_sec", "max_queue_duration_sec",
        "in_flow", "out_flow",
    ]
    writer.writerow(header)
    for rec in records:
        lanes = rec.get("lanes") or []
        if lanes:
            for idx, lane in enumerate(lanes):
                writer.writerow([
                    rec["detector_id"], rec["road_name"], rec["time_start"], rec["time_end"],
                    rec["direction"], lane["lane_no"], lane["flow"],
                    lane["small_vehicle_flow"], lane["large_vehicle_flow"],
                    lane["avg_speed_kmh"], lane["avg_occupancy_pct"],
                    lane.get("avg_queue_length_m", ""), lane.get("max_queue_length_m", ""),
                    lane.get("queue_duration_sec", ""), lane.get("max_queue_duration_sec", ""),
                    # 只掛第一條車道列,避免下游加總灌水(見 header 註解)
                    rec.get("in_flow", 0) if idx == 0 else "",
                    rec.get("out_flow", 0) if idx == 0 else "",
                ])
        else:
            writer.writerow([
                rec["detector_id"], rec["road_name"], rec["time_start"], rec["time_end"],
                rec["direction"], "", rec["total_flow"],
                rec["small_vehicle_flow"], rec["large_vehicle_flow"],
                rec["avg_speed_kmh"], rec["avg_occupancy_pct"],
                rec.get("avg_queue_length_m", "") or "", rec.get("max_queue_length_m", "") or "",
                rec.get("queue_duration_sec", "") or "", rec.get("max_queue_duration_sec", "") or "",
                rec.get("in_flow", 0), rec.get("out_flow", 0),
            ])

    output.seek(0)
    filename = f"vd_report_{bucket}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── 壅塞報表 ──────────────────────────────────────────────

def _congestion_rows_to_records(rows: list, bucket: str) -> list:
    """把 CongestionReportAgg 列轉成對外 JSON 記錄(congestion-report / latest 共用)。"""
    bucket_delta = _BUCKET_INTERVALS.get(bucket, timedelta(minutes=5))
    records = []
    for r in rows:
        ts = r.bucket_start.replace(tzinfo=timezone.utc).astimezone(TZ_TAIPEI) if r.bucket_start else None
        records.append({
            "detector_id": str(r.camera_id or ""),
            "camera_name": r.camera_name or "",
            "time_start": ts.isoformat() if ts else None,
            "time_end": (ts + bucket_delta).isoformat() if ts else None,
            "zone_name": r.zone_name or "",
            "lane_no": r.lane_no,
            "direction": r.direction or "",
            "avg_occupancy_pct": round((r.avg_occupancy or 0) * 100, 1),
            "max_occupancy_pct": round((r.max_occupancy or 0) * 100, 1),
            "avg_vehicle_count": round(r.avg_vehicle_count or 0, 1),
            "avg_stopped_vehicle_count": round(r.avg_stopped_vehicle_count or 0, 1),
            "avg_queue_length_m": round(r.avg_queue_length_m or 0, 1),
            "max_queue_length_m": round(r.max_queue_length_m or 0, 1),
            "queue_active_duration_sec": round(r.queue_active_duration_sec or 0, 1),
            "sample_count": r.sample_count or 0,
        })
    return records


def _congestion_stats(records: list) -> dict:
    """壅塞統計摘要:整體 + 各偵測器(sample_count 加權平均)+ 最壅塞桶。"""
    def summarize(recs: list) -> dict:
        wsum = sum(r["sample_count"] for r in recs) or 0
        def wavg(key):
            if not wsum:
                return None
            return round(sum(r[key] * r["sample_count"] for r in recs) / wsum, 1)
        return {
            "avg_occupancy_pct": wavg("avg_occupancy_pct"),
            "max_occupancy_pct": round(max((r["max_occupancy_pct"] for r in recs), default=0), 1),
            "avg_vehicle_count": wavg("avg_vehicle_count"),
            "avg_stopped_vehicle_count": wavg("avg_stopped_vehicle_count"),
            "avg_queue_length_m": wavg("avg_queue_length_m"),
            "max_queue_length_m": round(max((r["max_queue_length_m"] for r in recs), default=0), 1),
            "total_queue_active_duration_sec": round(sum(r["queue_active_duration_sec"] for r in recs), 1),
            "sample_count": wsum,
        }

    by_detector = {}
    for r in records:
        by_detector.setdefault(r["detector_id"], []).append(r)
    peak = max(records, key=lambda r: r["max_occupancy_pct"], default=None)
    return {
        # 🛑 這三個要分清楚,以前只有一個 bucket_count 而且數的是 len(records)
        # (= 偵測器 x 時間桶的組合數),名稱與內容對不上:
        #   realtime 4 台 x 1 個視窗 → 顯示 bucket_count 4,但根本沒有「桶」
        #   vd-report 4 台 x 3 桶     → 顯示 bucket_count 12,實際只有 3 個桶
        "record_count": len(records),
        "bucket_count": len({r["time_start"] for r in records}),
        "detector_count": len({r["detector_id"] for r in records}),
        "overall": summarize(records),
        "by_detector": [
            {"detector_id": did, "camera_name": recs[0]["camera_name"], **summarize(recs)}
            for did, recs in by_detector.items()
        ],
        "peak_bucket": {"time_start": peak["time_start"], "detector_id": peak["detector_id"],
                        "camera_name": peak["camera_name"], "max_occupancy_pct": peak["max_occupancy_pct"]}
                       if peak and peak["max_occupancy_pct"] else None,
    }


@router.get("/congestion-report", summary="壅塞報表 — 指定時間區間")
async def external_congestion_report(
    start_time: datetime = Query(
        ..., description="起始時間 ISO8601,例 2026-07-13T00:00:00+08:00。不帶時區視為 UTC;查台北時間請帶 +08:00"),
    end_time: datetime = Query(
        ..., description="結束時間 ISO8601,例 2026-07-13T23:59:59+08:00。格式同 start_time"),
    detector_id: Optional[int] = Query(None, description="攝影機 camera_id(2/3/6/8,見 /streams);留空=全部"),
    interval: str = Query("5m", description="時間桶大小:1m / 5m / 1h"),
    format: str = Query("json", description="輸出格式:json / csv"),
    api_key: ApiKey = Depends(require_scope("congestion_report")),
    db: Session = Depends(get_db),
):
    bucket = normalize_bucket_size(interval)
    _validate_time_range(start_time, end_time, bucket)

    # 🛑 bucket_start 在 DB 是 naive UTC,查詢條件一定要先轉成 naive UTC。
    # 直接綁 tz-aware 的 datetime,SQLite 方言只會把它格式化成字串、把 tzinfo 丟掉
    # → 台北的牆上時間被當成 UTC,整個視窗往後位移 8 小時、查不到任何資料。
    # 實測:同一小時 start_time=2026-08-08T16:00:00(naive) 回 7 筆,
    #       start_time=2026-08-09T00:00:00+08:00(同一時刻) 回 0 筆 ——
    # 而端點自己的說明就是叫人「查台北時間請帶 +08:00」。
    # vd-report 沒這問題是因為 build_vd_report_rows 內部有做 to_utc_naive。
    start_utc = to_utc_naive(start_time)
    end_utc = to_utc_naive(end_time)

    # 同 vd-report：只讀聚合表，重建交給背景 job
    query = db.query(CongestionReportAgg).filter(
        CongestionReportAgg.bucket_size == bucket,
        CongestionReportAgg.bucket_start >= start_utc,
        CongestionReportAgg.bucket_start < end_utc,
    )
    if detector_id:
        query = query.filter(CongestionReportAgg.camera_id == detector_id)

    rows = query.order_by(CongestionReportAgg.bucket_start).all()

    if len(rows) > _MAX_RECORDS:
        raise HTTPException(status_code=413, detail={
            "status": "error",
            "error": {"code": "TOO_MANY_RECORDS", "message": f"結果超過 {_MAX_RECORDS} 筆，請縮小時間範圍"},
        })

    records = _congestion_rows_to_records(rows, bucket)

    if format == "csv":
        return _congestion_csv_response(records, start_time, end_time, bucket)

    return {
        "status": "success",
        "data": {
            "interval": bucket,
            "period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "records": records,
        },
        "meta": _meta("json"),
    }


@router.get("/congestion-report/latest", summary="壅塞統計報表(快捷) — 最近 N 分鐘 + 統計摘要")
async def external_congestion_report_latest(
    minutes: int = Query(5, ge=1, le=360, description="回傳最近幾個完整桶(interval=1m 時即分鐘數)"),
    interval: str = Query("1m"),
    detector_id: Optional[int] = Query(None),
    include_records: bool = Query(True, description="False 只回統計摘要,不回逐桶明細"),
    api_key: ApiKey = Depends(require_scope("congestion_report")),
    db: Session = Depends(get_db),
):
    """壅塞統計報表快捷:免自己算時間/UTC,自動回最近 N 個「已結束」的桶 + 統計摘要。
    上層每分鐘輪詢就打這個(建議 minutes 給 3~5 容忍聚合延遲,以 time_start 去重 upsert)。"""
    bucket = normalize_bucket_size(interval)
    step = BUCKET_SECONDS[bucket]
    now = datetime.now(timezone.utc)
    epoch = int(now.timestamp())
    end = datetime.fromtimestamp(epoch - (epoch % step), tz=timezone.utc)
    start = end - timedelta(seconds=step * minutes)

    query = db.query(CongestionReportAgg).filter(
        CongestionReportAgg.bucket_size == bucket,
        CongestionReportAgg.bucket_start >= start.replace(tzinfo=None),
        CongestionReportAgg.bucket_start < end.replace(tzinfo=None),
    )
    if detector_id:
        query = query.filter(CongestionReportAgg.camera_id == detector_id)
    rows = query.order_by(CongestionReportAgg.bucket_start).all()
    records = _congestion_rows_to_records(rows, bucket)
    stats = _congestion_stats(records)

    data = {
        "interval": bucket,
        "period": {"start": start.astimezone(TZ_TAIPEI).isoformat(),
                   "end": end.astimezone(TZ_TAIPEI).isoformat()},
        "stats": stats,
    }
    if include_records:
        data["records"] = records
    return {"status": "success", "data": data, "meta": _meta("json")}


def _congestion_csv_response(records: list, start_time, end_time, bucket):
    output = io.StringIO()
    writer = csv.writer(output)
    header = [
        "detector_id", "camera_name", "time_start", "time_end",
        "zone_name", "lane_no", "direction",
        "avg_occupancy_pct", "max_occupancy_pct",
        "avg_vehicle_count", "avg_stopped_vehicle_count",
        "avg_queue_length_m", "max_queue_length_m",
        "queue_active_duration_sec", "sample_count",
    ]
    writer.writerow(header)
    for r in records:
        writer.writerow([r.get(h, "") for h in header])

    output.seek(0)
    filename = f"congestion_report_{bucket}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── 即時影像串流清單 ──────────────────────────────────────────────
# 給機關既設數位影像平台統一拿 URL + 規格，後續直接連 RTSP/HLS/MJPEG

def _detect_host_ip() -> str:
    """自動偵測本機對外 IP (UDP socket trick — 不真連線)，env STREAM_HOST 可覆寫"""
    import os, socket
    override = os.getenv("STREAM_HOST", "").strip()
    if override:
        return override
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _parse_go2rtc_sdp(sdp: str) -> dict:
    """從 SDP 抽 codec/profile/resolution/fps"""
    result = {"codec": None, "resolution": None, "fps": None, "profile": None}
    for line in sdp.split("\r\n"):
        if line.startswith("a=rtpmap:") and "H264" in line.upper():
            result["codec"] = "h264"
        elif line.startswith("a=rtpmap:") and ("HEVC" in line.upper() or "H265" in line.upper()):
            result["codec"] = "h265"
        elif line.startswith("a=framesize:"):
            try:
                size = line.split()[-1].replace("-", "x")
                result["resolution"] = size
            except Exception:
                pass
        elif line.startswith("a=framerate:"):
            try:
                result["fps"] = float(line.split(":")[1])
            except Exception:
                pass
        elif "profile-level-id=" in line:
            try:
                pid = line.split("profile-level-id=")[1].split(";")[0][:6]
                # H.264 profile_idc: 42=Baseline 4D=Main 64=High
                profile_idc = pid[:2].upper()
                result["profile"] = {
                    "42": "Baseline", "4D": "Main", "64": "High",
                }.get(profile_idc, profile_idc)
            except Exception:
                pass
    return result


@router.get("/streams", summary="即時影像串流清單 — 取得 cam id 與 RTSP/HLS/MJPEG URL")
def external_streams(
    api_key: ApiKey = Depends(require_scope("streams")),
    db: Session = Depends(get_db),
):
    """回傳本系統可對外提供的所有即時影像串流 URL + 規格。
    供機關既設數位影像平台統一查詢。
    """
    import requests as _req
    from api.models import Camera

    host = _detect_host_ip()
    rtsp_port = 8554
    http_port = 1984

    # 從 go2rtc HTTP API 拿目前活躍 streams 跟 producer SDP
    g2rtc = {}
    try:
        r = _req.get(f"http://127.0.0.1:{http_port}/api/streams", timeout=3)
        if r.ok:
            g2rtc = r.json() or {}
    except Exception:
        pass

    # cameras DB 對應 (id → Camera)
    cam_map = {c.id: c for c in db.query(Camera).all()}

    streams = []
    for stream_id, info in g2rtc.items():
        # 只匯出 cam_X 命名的 (跳過內部 / 暫存 stream)
        if not stream_id.startswith("cam_"):
            continue
        try:
            cam_id = int(stream_id.replace("cam_", ""))
        except ValueError:
            continue
        cam = cam_map.get(cam_id)

        producers = info.get("producers") or []
        online = bool(producers)
        spec = {"codec": None, "resolution": None, "fps": None, "profile": None}
        bytes_recv = 0
        if producers:
            p = producers[0]
            spec = _parse_go2rtc_sdp(p.get("sdp", ""))
            for rcv in (p.get("receivers") or []):
                bytes_recv += int(rcv.get("bytes") or 0)

        streams.append({
            "stream_id": stream_id,
            "camera_id": cam_id,
            "name": (cam.name if cam else None),
            "location": (cam.location if cam else None),
            "online": online,
            "codec": spec["codec"],
            "profile": spec["profile"],
            "resolution": spec["resolution"],
            "fps": spec["fps"],
            "bytes_received": bytes_recv,
            "urls": {
                "rtsp":  f"rtsp://{host}:{rtsp_port}/{stream_id}",
                "hls":   f"http://{host}:{http_port}/api/stream.m3u8?src={stream_id}",
                "mjpeg": f"http://{host}:{http_port}/api/stream.mjpeg?src={stream_id}",
                "webrtc_signal": f"http://{host}:{http_port}/api/ws?src={stream_id}",
            },
        })

    streams.sort(key=lambda s: s["camera_id"])

    return {
        "meta": _meta(),
        "status": "success",
        "data": {
            "device_id": _DEVICE_ID,
            "host": host,
            "ports": {"rtsp": rtsp_port, "http": http_port},
            "stream_count": len(streams),
            "streams": streams,
            "spec_requirement": {
                "codec": "H.264 (passthrough，無重編碼)",
                "min_resolution": "1280x720",
                "min_fps": 15,
                "note": "實際解析度/FPS 視相機本機設定，本系統 go2rtc 多路分發不重編碼",
            },
        },
    }
