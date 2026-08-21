"""號誌 TC3 常駐 daemon —— 獨立於 traffic-api(web) 的常駐服務。

為什麼獨立:中央電腦連線(:1001 中央中繼)、號誌控制器抄錄(NPort)、下傳控制、frame
持久化,都不該綁在 web service 的生死上。traffic-api 重啟/崩潰/SERVICE 異常時,
這個 daemon 照跑,中央連線不斷。web 只是「開啟介面觀看」—— 透過 traffic-api 反向
代理 /api/signal/* 到這裡。

比照 io_daemon(127.0.0.1:8011) 的拆法。

跑法:
    AUTH_SECRET=... UTC_TC3_PATH=/home/ubuntu/utc-tc3 SIGNAL_TC3_CENTER_RELAY=1 \
    uvicorn services.signal_daemon:app --host 127.0.0.1 --port 8012
systemd unit: traffic-signal.service(見部署)。

🛑 這個 process 才是 :1001 中央中繼與 NPort 抄錄的唯一持有者。
   traffic-api 那邊絕不可再 start_recorder/start_center_relay(否則搶 :1001 / 雙抄錄)。
"""
from fastapi import FastAPI

from api.routes import signal_tc3
from api.routes.auth import get_current_user

app = FastAPI(title="traffic-signal-daemon", docs_url=None, redoc_url=None)


class _InternalUser:
    """daemon 只聽 127.0.0.1(內部),不做登入驗證 —— 由前面的 traffic-api proxy 保留
    web 端的登入檢查。這裡把 get_current_user 覆蓋掉,讓 signal_tc3 的端點直接可用。"""
    username = "signal-daemon"
    role = "admin"


app.dependency_overrides[get_current_user] = lambda: _InternalUser()
app.include_router(signal_tc3.router)      # 提供 /api/signal/*


@app.on_event("startup")
def _startup() -> None:
    # 這裡才啟動連線層:抄錄器 + 中央中繼 + 訊框持久化 writer。
    try:
        signal_tc3.start_recorder()
        signal_tc3.start_center_relay()
        signal_tc3.start_frame_writer()
        print("📶 [signal-daemon] 已啟動 recorder + 中央中繼 + 持久化 writer", flush=True)
    except Exception as exc:
        print(f"⚠️ [signal-daemon] 啟動失敗: {exc}", flush=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "traffic-signal"}
