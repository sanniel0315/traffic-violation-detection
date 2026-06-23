# 在另一台 Jetson 開新站 — 部署手冊

把相同專案在另一台 Jetson（Orin Nano/NX/AGX 皆可）上線。**程式與 `.pt` 模型可帶過去；
`.engine` 必須在新板重建（TensorRT 綁 GPU）；攝影機/ROI 為現場新設。**

> 一鍵 bootstrap：`bash scripts/setup_new_site.sh`（做完自動部分後會列出人工項）。
> 本文件是完整版＋背景說明。

---

## 0. 前置（新板 OS）
- JetPack（含 CUDA/cuDNN/TensorRT）、Python 3.10、Docker + compose plugin
- user 慣例 `ubuntu`、專案路徑 `/home/ubuntu/traffic-violation-detection`
  （若不同，systemd unit 內的路徑/User 要一起 `sed` 改）

## 1. 取得程式
```bash
git clone <repo> /home/ubuntu/traffic-violation-detection
cd /home/ubuntu/traffic-violation-detection
```

## 2. 依賴
```bash
pip install -r requirements.txt --break-system-packages
```
**torch / torchvision 例外**：Jetson 要裝 NVIDIA 專屬 wheel（不是 PyPI 版），
版本對應 JetPack（現有 AGX 是 torch 2.5.0+nv24.08）。從 NVIDIA Jetson wheel index 裝。
系統套件：`sudo apt install tesseract-ocr tesseract-ocr-chi-tra`。

## 3. 目錄（每台自管，不在版控）
```bash
bash scripts/init_dirs.sh                                  # 本機硬碟
# 或放外接/NVMe：
TRAFFIC_STORAGE_ROOT=/mnt/nvme/traffic bash scripts/init_dirs.sh
```
建立 `data/ models/ output/ storage/`。

## 4. 模型
- 把 `.pt` 模型放進 `models/`（`yolov8n.pt`、`truck_cls_*.pt`、LPR 模型等）。`.pt` 跨 Jetson 可直接重用。
- **TensorRT engine 必須在這台板子上 build**（不同 GPU 的 engine 不通用）：
  ```bash
  export LD_LIBRARY_PATH=$HOME/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib
  yolo export model=models/yolov8n.pt format=engine half=True
  ```
  沒 engine 也能跑：偵測器會自動 fallback 用 `.pt`（慢但正確），有 engine 後在 `.env` 填回 `DETECT_MODEL_ENGINE`。

## 5. 環境變數
```bash
cp .env.example .env    # 編輯
```
重點：`FRIGATE_RTSP_PASSWORD`、`EXTERNAL_API_KEY`（**換新隨機值**，別跟舊站共用）、`TZ`。

## 6. Frigate / 攝影機（本站專屬）
編輯 `config/frigate/config.yml`：
- `go2rtc.streams.*`：填**本站攝影機**的 RTSP（URL/帳密/IP）
- `cameras.*`：對應的 detect/record 設定
> 🛑 `config/frigate/config.yml` 是 git-tracked，CI deploy 會 `git reset --hard` 覆蓋；
> 改完要 commit 進該站自己的分支/設定，否則會被蓋回。

## 7. systemd 服務
```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo mkdir -p /etc/systemd/system/traffic-api.service.d
sudo cp deploy/systemd/traffic-api.service.d/override.conf /etc/systemd/system/traffic-api.service.d/
sudo systemctl daemon-reload
sudo systemctl enable --now traffic-frigate traffic-ocr traffic-io traffic-api
```
服務角色：
| unit | 用途 |
|------|------|
| traffic-frigate | Frigate NVR（docker compose） |
| traffic-ocr | 車牌字元 YOLO OCR 微服務 |
| traffic-io | RS-485 Modbus IO daemon（127.0.0.1:8011，獨立 process 防 SEGV 互拖） |
| traffic-api | 主 API（uvicorn :8000，偵測/壅塞/LPR/報表） |

> `IO_PORT`（traffic-io）依現場序列埠調整；無 RS-485 硬體可不啟 traffic-io。
> 停車場自動標註（parking-collector/retrain）是特定測試點專屬，新站通常不用。

## 8. 開站設定（網頁 http://<板IP>:8000）
現場重設（每站不同，不會自動帶）：攝影機、車道/方向、偵測 ROI、車速區、計數線、停車格 ROI。
車速區記得設**真實世界尺寸校正**（width_m/length_m）否則速度不準。

## 9. 驗證
```bash
python3 scripts/smoke_check.py
curl -s http://127.0.0.1:8000/api/cameras
```

---

## 自動部署（CI/CD）注意
現有 `.github/workflows/jetson-verify.yml` 綁定 **單一 self-hosted runner，label `jetson-agx-orin`**，
只會部署到原本那台 AGX。第二站要自動部署，二選一：
- **A（建議起步）手動部署**：新站 `git pull` + `sudo systemctl restart traffic-api`（用 `scripts/restart_and_verify.sh`）
- **B 多站 CI**：在新板註冊**自己的 runner + 不同 label**，並把 workflow 的 `runs-on` 參數化成各站 label

> 對外網路不穩時 runner 會接不到 job / `git fetch` 失敗 → 手動 scp 改檔 + 重啟為 fallback。
