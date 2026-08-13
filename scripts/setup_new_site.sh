#!/usr/bin/env bash
# 新 Jetson 站點一鍵 bootstrap — 把相同專案在另一台 Jetson 上線。
# 假設：Ubuntu/JetPack、user=ubuntu、專案在 /home/ubuntu/traffic-violation-detection。
# 用法： bash scripts/setup_new_site.sh          # 本機硬碟
#        TRAFFIC_STORAGE_ROOT=/mnt/nvme/traffic bash scripts/setup_new_site.sh   # 放 NVMe
#
# 這支腳本做「可自動化」的部分；模型/engine/攝影機/ROI 等需人工的會在最後列出。
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
echo "==> 專案目錄: $REPO"

# 1) 系統依賴
echo "==> [1/6] 系統套件 (tesseract / docker compose)"
if ! command -v tesseract >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-tra
fi
command -v docker >/dev/null 2>&1 || echo "  !! 未偵測到 docker，請先安裝 docker + compose plugin"

# 2) Python 依賴 (Jetson 用 --break-system-packages；torch 是 Jetson 專屬 wheel,需另裝)
echo "==> [2/6] pip install -r requirements.txt"
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt
echo "  注意: Jetson 的 torch/torchvision 要用 NVIDIA 專屬 wheel 另行安裝 (見 DEPLOY_NEW_SITE.md)"

# 3) 資料/模型/輸出/儲存 目錄
echo "==> [3/6] init_dirs (data/models/output/storage)"
if [ -n "${TRAFFIC_STORAGE_ROOT:-}" ]; then
  TRAFFIC_STORAGE_ROOT="$TRAFFIC_STORAGE_ROOT" bash scripts/init_dirs.sh
else
  bash scripts/init_dirs.sh
fi

# 4) .env
echo "==> [4/6] .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  已從 .env.example 複製成 .env — 請編輯填入 FRIGATE_RTSP_PASSWORD / EXTERNAL_API_KEY 等"
else
  echo "  .env 已存在,略過"
fi

# 5) systemd units — unit 檔以 /home/ubuntu/User=ubuntu 為樣板，安裝時自動換成本機實際 user/home
echo "==> [5/6] 安裝 systemd units (user=$(whoami) home=$HOME)"
U="$(whoami)"; H="$HOME"
for f in deploy/systemd/*.service deploy/systemd/*.timer; do
  sed "s|/home/ubuntu|$H|g; s|^User=ubuntu$|User=$U|" "$f" | sudo tee "/etc/systemd/system/$(basename "$f")" >/dev/null
done
sudo mkdir -p /etc/systemd/system/traffic-api.service.d
sed "s|/home/ubuntu|$H|g" deploy/systemd/traffic-api.service.d/override.conf | sudo tee /etc/systemd/system/traffic-api.service.d/override.conf >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable traffic-ocr traffic-io traffic-frigate traffic-api
sudo systemctl enable --now traffic-cleanup.timer
echo "  已 enable: traffic-ocr / traffic-io / traffic-frigate / traffic-api / traffic-cleanup.timer(每日02:30快照清理,保留30天)"

# 6) journald 半年保存 — 需要大容量碟(20G上限,eMMC裝不下),只在有 TRAFFIC_STORAGE_ROOT 時做
echo "==> [6/7] journald 半年保存"
if [ -n "${TRAFFIC_STORAGE_ROOT:-}" ]; then
  JDIR="$(dirname "$TRAFFIC_STORAGE_ROOT")/journal"   # 例: /mnt/nvme/traffic → /mnt/nvme/journal
  if [ ! -L /var/log/journal ]; then
    sudo mkdir -p "$JDIR"
    [ -d /var/log/journal ] && sudo rsync -a /var/log/journal/ "$JDIR/" && sudo rm -rf /var/log/journal
    sudo ln -s "$JDIR" /var/log/journal
  fi
  sudo mkdir -p /etc/systemd/journald.conf.d
  sudo cp deploy/journald/retention.conf /etc/systemd/journald.conf.d/retention.conf
  sudo systemctl restart systemd-journald
  echo "  journal → $JDIR (20G / 6 個月)"
else
  echo "  !! 未設 TRAFFIC_STORAGE_ROOT，跳過（journal 半年約需 16G+，eMMC 放不下）"
fi

# 7) 校時 — 現場網段內的 NTP。現場常常沒有外網,不設就只能靠 RTC 慢慢漂,報表時戳會歪。
echo "==> [7/7] NTP 校時來源"
if [ -n "${FIELD_NTP:-}" ]; then
  sudo mkdir -p /etc/systemd/timesyncd.conf.d
  # 檔名 zz- 開頭才排在 Jetson 出廠的 nv-fallback-ntp.conf 之後,蓋得掉它的外網清單
  sed "s|__FIELD_NTP__|$FIELD_NTP|" deploy/timesyncd/field-ntp.conf.template \
    | sudo tee /etc/systemd/timesyncd.conf.d/zz-field-ntp.conf >/dev/null
  sudo rm -f /etc/systemd/timesyncd.conf.d/field-ntp.conf   # 舊檔名殘留會混淆排查
  sudo systemctl restart systemd-timesyncd
  echo "  NTP → $FIELD_NTP (現場封閉網段,不留外網 fallback)"
  echo "  驗證: timedatectl show-timesync --property=ServerName --property=ServerAddress"
else
  echo "  !! 未設 FIELD_NTP，沿用系統預設(外網 NTP)。現場若沒外網會校不到時,"
  echo "     請改用: FIELD_NTP=<現場NTP位址> bash scripts/setup_new_site.sh"
fi

cat <<'NEXT'

================ 自動部分完成。以下需人工 ================
1. 放模型: 把 .pt 模型放到 models/ (yolov8n.pt 等;.pt 跨 Jetson 可重用,.engine 不行)
2. 建 TensorRT engine (必須在「這台」板子上跑,GPU 綁定):
     export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib
     yolo export model=models/yolov8n.pt format=engine half=True
   建好後把 .env 的 DETECT_MODEL_ENGINE 填回 (沒 engine 會 fallback .pt,慢但能跑)
3. 編輯 .env: FRIGATE_RTSP_PASSWORD / EXTERNAL_API_KEY (換新隨機值)
4. Frigate 攝影機: 改 config/frigate/config.yml 的 go2rtc streams + cameras 成「本站」攝影機 RTSP
5. 啟動: sudo systemctl start traffic-frigate traffic-ocr traffic-io traffic-api
6. 帶設定(不用手打): 主機匯出 settings_backup.json scp 過來,再
     python3 scripts/settings_backup.py import --file config/settings.json
   (涵蓋攝影機/偵測參數/全部ROI/使用者;不含紀錄/模型/大檔)。匯入後微調本站差異(上游RTSP/角度不同的ROI/車速校正)
7. 驗證: python3 scripts/smoke_check.py
============================================================
NEXT
