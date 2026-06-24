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

## 8. 帶設定過來（主機匯出 → 新站匯入）
**不要手動重打**——用 `settings_backup.py` 從主機帶過來。涵蓋：
攝影機(含帳密/source)、`detection_config`(偵測/壅塞/車速參數)、`zones`(全部 ROI：偵測區/車速區/
計數線/不偵測區/停車格)、使用者角色、system_files(feature_state/ntp)。
**不含**：辨識/違規/事件紀錄、模型、錄影、DB 等大檔案（這些不需備份，各機自管）。

```bash
# 主機(現有站)匯出：
python3 scripts/settings_backup.py export --file /tmp/settings.json
scp /tmp/settings.json <新板>:/home/ubuntu/traffic-violation-detection/config/

# 新站匯入(先確定 data/violations.db 已建好 schema：跑過一次 traffic-api 或 init)：
python3 scripts/settings_backup.py import --file config/settings.json
sudo systemctl restart traffic-api
```

> 匯入後上網頁 :8000 微調**本站差異**：攝影機 source / 上游 RTSP（見 §6 frigate）、
> 以及因鏡頭角度不同需重畫的 ROI。**車速區記得設真實世界尺寸校正**（width_m/length_m）否則速度不準。
> users 匯入只更新「已存在」帳號的角色（不建新帳號/不帶密碼），新站的登入帳密另設。

## 9. 驗證
```bash
python3 scripts/smoke_check.py
curl -s http://127.0.0.1:8000/api/cameras
```

---

## 自動部署（CI/CD）— 多站已參數化
`.github/workflows/jetson-verify.yml` 已改成 **matrix 多站**：每站 = 一台 Jetson + 一個有唯一 label 的 runner。
```yaml
strategy:
  matrix:
    site: [agx-orin]          # ← 新增站點就在這加 label
runs-on: [self-hosted, "${{ matrix.site }}"]
```
push main 時，matrix 會在「每一台」對應 runner 各自 verify + 自我部署。

**把第二站加入自動部署**：
1. 新板註冊 self-hosted runner，給**唯一 label**（例 `site2`）：
   `./config.sh --url <repo> --token <T> --labels self-hosted,jetson,gpu,site2`
2. 把 label 加進 workflow 的 `matrix.site`：`site: [agx-orin, site2]`
3. push → 兩站都會自動部署。

> 🛑 **加自動部署前必解的雷**：deploy 的 `git reset --hard origin/main` 會把 `config/frigate/config.yml`
> 還原成 origin 版（= 第一站的攝影機）。各站攝影機不同，所以第二站直接吃進 matrix 會被洗掉設定。
> **在做「每站獨立 frigate 設定」之前，第二站請維持手動部署**：
> `git pull && sudo systemctl restart traffic-api`（或 `scripts/restart_and_verify.sh`）。

> 對外網路不穩時 runner 會接不到 job / `git fetch` 失敗 → 手動 scp 改檔 + 重啟為 fallback。
