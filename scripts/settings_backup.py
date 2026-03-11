#!/usr/bin/env python3
"""Export/import deploy settings without backing up recognition/event records."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "violations.db"
DEFAULT_BACKUP_PATH = ROOT / "config" / "settings_backup.json"
SYSTEM_FILES = {
    "feature_state": ROOT / "config" / "system" / "feature_state.json",
    "ntp_settings": ROOT / "config" / "system" / "ntp_settings.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_json_value(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return text


def _dump_json_value(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False, separators=(",", ":"))


def _bool_to_int(v: Any, default: bool = False) -> int:
    if v is None:
        return 1 if default else 0
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        return 1 if int(v) != 0 else 0
    except Exception:
        return 1 if str(v).strip().lower() in ("true", "yes", "on") else 0


def _ensure_db() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"database not found: {DB_PATH}")


def export_settings(path: Path) -> None:
    _ensure_db()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cam_rows = con.execute(
            """
            SELECT id, name, source, ip, username, password, port, stream_path,
                   location, enabled, detection_enabled, detection_config, zones
            FROM cameras
            ORDER BY id ASC
            """
        ).fetchall()
        user_rows = con.execute(
            """
            SELECT username, role, enabled
            FROM users
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        con.close()

    cameras: List[Dict[str, Any]] = []
    for r in cam_rows:
        cameras.append(
            {
                "id": int(r["id"]),
                "name": r["name"] or "",
                "source": r["source"] or "",
                "ip": r["ip"] or "",
                "username": r["username"] or "",
                "password": r["password"] or "",
                "port": r["port"] or "554",
                "stream_path": r["stream_path"] or "",
                "location": r["location"] or "",
                "enabled": bool(r["enabled"]),
                "detection_enabled": bool(r["detection_enabled"]),
                "detection_config": _parse_json_value(r["detection_config"]) or {},
                "zones": _parse_json_value(r["zones"]) or [],
            }
        )

    users: List[Dict[str, Any]] = []
    for r in user_rows:
        users.append(
            {
                "username": r["username"] or "",
                "role": r["role"] or "viewer",
                "enabled": bool(r["enabled"]),
            }
        )

    payload: Dict[str, Any] = {
        "meta": {
            "schema": "traffic-settings-backup/v1",
            "exported_at": _utc_now(),
            "db_path": str(DB_PATH),
        },
        "cameras": cameras,
        "users": users,
        "system_files": {k: _read_json_file(v) for k, v in SYSTEM_FILES.items()},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] exported settings -> {path}")
    print(f"      cameras={len(cameras)} users={len(users)}")


def _upsert_camera(con: sqlite3.Connection, cam: Dict[str, Any]) -> None:
    cam_id = cam.get("id")
    name = str(cam.get("name") or "").strip()
    row = None
    if isinstance(cam_id, int):
        row = con.execute("SELECT id FROM cameras WHERE id = ?", (cam_id,)).fetchone()
    if row is None and name:
        row = con.execute("SELECT id FROM cameras WHERE name = ?", (name,)).fetchone()

    fields = (
        cam.get("name") or "",
        cam.get("source") or "",
        cam.get("ip") or "",
        cam.get("username") or "",
        cam.get("password") or "",
        cam.get("port") or "554",
        cam.get("stream_path") or "",
        cam.get("location") or "",
        _bool_to_int(cam.get("enabled"), True),
        _bool_to_int(cam.get("detection_enabled"), True),
        _dump_json_value(cam.get("detection_config") or {}),
        _dump_json_value(cam.get("zones") or []),
        datetime.utcnow().isoformat(),
    )

    if row is None:
        con.execute(
            """
            INSERT INTO cameras (
                name, source, ip, username, password, port, stream_path, location,
                enabled, detection_enabled, detection_config, zones, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fields,
        )
    else:
        con.execute(
            """
            UPDATE cameras
            SET name = ?, source = ?, ip = ?, username = ?, password = ?, port = ?,
                stream_path = ?, location = ?, enabled = ?, detection_enabled = ?,
                detection_config = ?, zones = ?, updated_at = ?
            WHERE id = ?
            """,
            fields + (int(row["id"]),),
        )


def _upsert_user(con: sqlite3.Connection, user: Dict[str, Any]) -> None:
    username = str(user.get("username") or "").strip()
    if not username:
        return
    exists = con.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if exists is None:
        return
    con.execute(
        "UPDATE users SET role = ?, enabled = ?, updated_at = ? WHERE username = ?",
        (
            user.get("role") or "viewer",
            _bool_to_int(user.get("enabled"), True),
            datetime.utcnow().isoformat(),
            username,
        ),
    )


def _write_system_files(data: Dict[str, Any]) -> None:
    files = data.get("system_files") or {}
    if not isinstance(files, dict):
        return
    for key, path in SYSTEM_FILES.items():
        raw = files.get(key)
        if raw is None:
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[WARN] skip writing {path}: {e}")


def import_settings(path: Path) -> None:
    _ensure_db()
    if not path.exists():
        raise FileNotFoundError(f"backup file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras = data.get("cameras") or []
    users = data.get("users") or []

    con = sqlite3.connect(DB_PATH)
    try:
        for cam in cameras:
            if isinstance(cam, dict):
                _upsert_camera(con, cam)
        for user in users:
            if isinstance(user, dict):
                _upsert_user(con, user)
        con.commit()
    finally:
        con.close()

    _write_system_files(data)
    print(f"[OK] imported settings <- {path}")
    print(f"      cameras={len(cameras)} users={len(users)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backup/restore deploy settings (exclude recognition/event data)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="export settings to JSON")
    p_export.add_argument(
        "--file",
        default=str(DEFAULT_BACKUP_PATH),
        help=f"output file path (default: {DEFAULT_BACKUP_PATH})",
    )

    p_import = sub.add_parser("import", help="import settings from JSON")
    p_import.add_argument(
        "--file",
        default=str(DEFAULT_BACKUP_PATH),
        help=f"input file path (default: {DEFAULT_BACKUP_PATH})",
    )

    args = parser.parse_args()
    path = Path(args.file).resolve()

    if args.cmd == "export":
        export_settings(path)
    elif args.cmd == "import":
        import_settings(path)
    else:
        raise ValueError(f"unsupported cmd: {args.cmd}")


if __name__ == "__main__":
    main()
