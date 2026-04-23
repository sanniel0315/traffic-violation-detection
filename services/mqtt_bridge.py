#!/usr/bin/env python3
"""MQTT bridge：負責連 broker、publish 我方資料、subscribe 監控設備、記錄收發訊息給 UI 看"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETTINGS_PATH = os.getenv(
    "MQTT_SETTINGS_PATH",
    os.path.join(_PROJECT_ROOT, "config", "system", "mqtt_settings.json"),
)
_MAX_LOG = 500  # UI 顯示的訊息 ring buffer 大小


class MqttBridge:
    """單一 MQTT client，支援 embedded / external 兩種模式切換"""

    def __init__(self):
        self._lock = threading.Lock()
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._conn_err: str = ""
        self.settings: Dict[str, Any] = self._default_settings()
        self.log_sent: deque = deque(maxlen=_MAX_LOG)
        self.log_recv: deque = deque(maxlen=_MAX_LOG)
        self.stats = {"sent": 0, "recv": 0, "errors": 0}
        self.load_settings()

    # ---------- settings ----------
    @staticmethod
    def _default_settings() -> Dict[str, Any]:
        return {
            "mode": "embedded",  # embedded | external | off
            "host": "localhost",
            "port": 1883,
            "username": "",
            "password": "",
            "client_id": "traffic-api",
            "base_topic": "traffic",
            "subscribe_patterns": ["traffic/#"],  # 監看自己發出的訊息
            "qos": 0,
            "retain": False,
        }

    def load_settings(self) -> Dict[str, Any]:
        try:
            if os.path.exists(_SETTINGS_PATH):
                with open(_SETTINGS_PATH) as f:
                    data = json.load(f) or {}
                merged = self._default_settings()
                merged.update({k: v for k, v in data.items() if k in merged})
                self.settings = merged
        except Exception as e:
            print(f"[mqtt] load settings 失敗: {e}", flush=True)
        return self.settings

    def save_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._default_settings()
        merged.update(self.settings)
        for k, v in (new_settings or {}).items():
            if k in merged:
                if isinstance(v, str):
                    v = v.strip()
                merged[k] = v
        os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
        with open(_SETTINGS_PATH, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        self.settings = merged
        # 設定變了 → 重連
        self.reconnect()
        return merged

    # ---------- connection ----------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            self._connected = True
            self._conn_err = ""
            for pat in (self.settings.get("subscribe_patterns") or []):
                try:
                    client.subscribe(pat, qos=self.settings.get("qos", 0))
                except Exception as e:
                    print(f"[mqtt] subscribe {pat} 失敗: {e}", flush=True)
            print(f"[mqtt] connected to {self.settings.get('host')}:{self.settings.get('port')}", flush=True)
        else:
            self._connected = False
            self._conn_err = f"reason_code={reason_code}"

    def _on_disconnect(self, client, userdata, reason_code, properties=None, *args):
        self._connected = False

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="replace")
        except Exception:
            payload = str(msg.payload)
        self.log_recv.append({
            "ts": time.time(),
            "topic": msg.topic,
            "payload": payload[:1000],  # 截斷避免 UI 吃太多
            "qos": msg.qos,
        })
        self.stats["recv"] += 1

    def reconnect(self) -> bool:
        with self._lock:
            # 斷開舊連線
            if self._client is not None:
                try:
                    self._client.loop_stop()
                    self._client.disconnect()
                except Exception:
                    pass
                self._client = None
                self._connected = False

            if self.settings.get("mode") == "off":
                return False

            try:
                client_id = self.settings.get("client_id") or f"traffic-api-{int(time.time())}"
                c = mqtt.Client(
                    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                    client_id=client_id,
                    clean_session=True,
                )
                user = self.settings.get("username") or ""
                pwd = self.settings.get("password") or ""
                if user:
                    c.username_pw_set(user, pwd)
                c.on_connect = self._on_connect
                c.on_disconnect = self._on_disconnect
                c.on_message = self._on_message
                host = self.settings.get("host") or "localhost"
                port = int(self.settings.get("port") or 1883)
                # embedded 模式強制 host=localhost
                if self.settings.get("mode") == "embedded":
                    host = "localhost"
                c.connect_async(host, port, keepalive=30)
                c.loop_start()
                self._client = c
                return True
            except Exception as e:
                self._conn_err = str(e)
                self.stats["errors"] += 1
                print(f"[mqtt] reconnect failed: {e}", flush=True)
                return False

    def connected(self) -> bool:
        return self._connected

    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.settings.get("mode"),
            "host": ("localhost" if self.settings.get("mode") == "embedded" else self.settings.get("host")),
            "port": self.settings.get("port"),
            "connected": self._connected,
            "error": self._conn_err,
            "stats": dict(self.stats),
            "settings": {k: v for k, v in self.settings.items() if k != "password"},
        }

    # ---------- publish ----------
    def publish(self, topic: str, payload: Any, qos: Optional[int] = None, retain: Optional[bool] = None) -> bool:
        if not self._client or not self._connected or self.settings.get("mode") == "off":
            return False
        try:
            if isinstance(payload, (dict, list)):
                body = json.dumps(payload, ensure_ascii=False)
            elif isinstance(payload, (bytes, bytearray)):
                body = payload
            else:
                body = str(payload)
            q = int(qos if qos is not None else self.settings.get("qos", 0))
            r = bool(retain if retain is not None else self.settings.get("retain", False))
            self._client.publish(topic, body, qos=q, retain=r)
            self.log_sent.append({
                "ts": time.time(),
                "topic": topic,
                "payload": (body if isinstance(body, str) else "<bytes>")[:1000],
                "qos": q,
                "retain": r,
            })
            self.stats["sent"] += 1
            return True
        except Exception as e:
            self._conn_err = f"publish failed: {e}"
            self.stats["errors"] += 1
            return False

    def recent_sent(self, limit: int = 100):
        lst = list(self.log_sent)
        return lst[-limit:] if limit else lst

    def recent_recv(self, limit: int = 100):
        lst = list(self.log_recv)
        return lst[-limit:] if limit else lst


# 模組級 singleton
bridge = MqttBridge()


def start():
    """啟動 MQTT bridge（在 api/main.py startup 呼叫）"""
    if bridge.settings.get("mode") != "off":
        bridge.reconnect()
