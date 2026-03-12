# DB ER Model

資料來源：`api/models.py`（SQLAlchemy ORM）

## ER Diagram

```mermaid
erDiagram
    CAMERAS ||--o{ VIOLATIONS : "camera_id (logical)"
    CAMERAS ||--o{ LPR_RECORDS : "camera_id (logical)"
    CAMERAS ||--o{ TRAFFIC_EVENTS : "camera_id (logical)"

    CAMERAS {
        int id PK
        string name
        string source
        string ip
        string username
        string password
        string port
        string stream_path
        string location
        json detection_config
        json zones
        string status
        bool enabled
        bool detection_enabled
        int total_violations
        int today_violations
        datetime last_seen
        datetime created_at
        datetime updated_at
    }

    VIOLATIONS {
        int id PK
        string violation_type
        string violation_name
        string license_plate
        string vehicle_type
        string location
        int camera_id
        int track_id
        datetime violation_time
        float confidence
        json bbox
        string image_path
        string video_path
        string status
        string reviewed_by
        datetime reviewed_at
        int fine_amount
        int points
        float speed_kmh
        float speed_limit_kmh
        float overspeed_kmh
        bool flow_roi_hit
        bool speed_roi_hit
        datetime created_at
        datetime updated_at
    }

    LPR_RECORDS {
        int id PK
        int camera_id
        string camera_name
        string plate_number
        float confidence
        bool valid
        string vehicle_type
        string snapshot
        string raw
        datetime created_at
    }

    TRAFFIC_EVENTS {
        int id PK
        int camera_id
        string label
        float speed_kmh
        int lane_no
        string direction
        json entered_zones
        json bbox
        string source
        datetime created_at
    }

    SYSTEM_LOGS {
        int id PK
        string level
        string source
        text message
        datetime created_at
    }

    USERS {
        int id PK
        string username UK
        string password_hash
        string role
        bool enabled
        datetime created_at
        datetime updated_at
    }
```

## Tables

## `cameras`
- PK: `id`
- 用途：攝影機主檔、串流參數、ROI/偵測設定
- 關聯：被 `violations.camera_id`、`lpr_records.camera_id`、`traffic_events.camera_id` 參照（邏輯關聯）

## `violations`
- PK: `id`
- 用途：違規事件（含速度與 ROI 命中欄位）
- 重點索引欄位：`violation_type`, `license_plate`, `camera_id`, `violation_time`, `status`

## `lpr_records`
- PK: `id`
- 用途：車牌辨識紀錄
- 重點索引欄位：`camera_id`, `plate_number`, `created_at`

## `traffic_events`
- PK: `id`
- 用途：交通流量事件（車種、速度、車道、方向）
- 重點索引欄位：`camera_id`, `label`, `lane_no`, `direction`, `source`, `created_at`

## `system_logs`
- PK: `id`
- 用途：系統日誌
- 重點索引欄位：`level`, `source`, `created_at`

## `users`
- PK: `id`
- Unique: `username`
- 用途：登入帳號與角色權限

## Notes

- 目前模型未宣告實體 `ForeignKey`；`camera_id` 為應用層維護的邏輯關聯。
- 預設 DB：`sqlite:///./data/violations.db`（可由 `DATABASE_URL` 覆蓋）。
