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
echo "==> [1/5] 系統套件 (tesseract / docker compose)"
if ! command -v tesseract >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-tra
fi
command -v docker >/dev/null 2>&1 || echo "  !! 未偵測到 docker，請先安裝 docker + compose plugin"

# 2) Python 依賴 (Jetson 用 --break-system-packages；torch 是 Jetson 專屬 wheel,需另裝)
echo "==> [2/5] pip install -r requirements.txt"
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt
echo "  注意: Jetson 的 torch/torchvision 要用 NVIDIA 專屬 wheel 另行安裝 (見 DEPLOY_NEW_SITE.md)"

# 3) 資料/模型/輸出/儲存 目錄
echo "==> [3/5] init_dirs (data/models/output/storage)"
if [ -n "${TRAFFIC_STORAGE_ROOT:-}" ]; then
  TRAFFIC_STORAGE_ROOT="$TRAFFIC_STORAGE_ROOT" bash scripts/init_dirs.sh
else
  bash scripts/init_dirs.sh
fi

# 4) .env
echo "==> [4/5] .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "  已從 .env.example 複製成 .env — 請編輯填入 FRIGATE_RTSP_PASSWORD / EXTERNAL_API_KEY 等"
else
  echo "  .env 已存在,略過"
fi

# 5) systemd units
echo "==> [5/5] 安裝 systemd units"
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo mkdir -p /etc/systemd/system/traffic-api.service.d
sudo cp deploy/systemd/traffic-api.service.d/override.conf /etc/systemd/system/traffic-api.service.d/
sudo systemctl daemon-reload
sudo systemctl enable traffic-ocr traffic-io traffic-frigate traffic-api
echo "  已 enable: traffic-ocr / traffic-io / traffic-frigate / traffic-api"

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
6. 開站設定(網頁 :8000): 攝影機、車道/方向、偵測ROI、車速區、計數線、停車格 ROI 等現場重設
7. 驗證: python3 scripts/smoke_check.py
============================================================
NEXT
