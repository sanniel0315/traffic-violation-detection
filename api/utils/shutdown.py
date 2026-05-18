"""Process-wide shutdown event。

lifespan shutdown 時 set，streaming generator 週期性檢查以乾淨退出 —
不檢查的 sync `while True` generator 會卡到 systemd `TimeoutStopSec` 觸發
SIGKILL，造成 10 秒體感停頓。
"""
import threading

shutdown_event = threading.Event()
