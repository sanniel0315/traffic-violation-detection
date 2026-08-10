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
重點：`FRIGATE_RTSP_PASSWORD`、`EXTERNAL_API_KEY`（**換新隨機值**，別跟舊站共用）、`TZ`、
`STREAM_HOST`（客戶連得到的主機位址，見 §A-2 —— **雙網路的站一定要設**，
不設會讓對外串流網址指向錯的網卡且無錯誤訊息）。

## 6. Frigate / 攝影機（本站專屬）
編輯 `config/frigate/config.yml`：
- `go2rtc.streams.*`：填**本站攝影機**的 RTSP（URL/帳密/IP）
- `cameras.*`：對應的 detect/record 設定
> ✅ `config/frigate/config.yml` 雖然 git-tracked，但 deploy 會在
> `git reset --hard` 前後保留現場那份（`scripts/deploy_keep_runtime_config.sh`），
> **改完不用 commit 也不會被蓋回**。repo 裡的版本只是新站的種子。
> 反過來說：想從 git 把設定派到現場，得先在現場刪掉該檔再部署。

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

> ✅ **這顆雷已拆**（2026-08-08）：deploy 會在 `git reset --hard origin/main` 前後
> 保留現場的執行期設定檔（frigate `config.yml`、`ui_settings.json`、`go2rtc.yaml`、
> 功能開關、IO / NVR 設定），各站攝影機設定不同不再是自動部署的阻礙，
> 第二站可以直接加進 `matrix.site`。
> 機制在 `scripts/deploy_keep_runtime_config.sh`，回歸測試
> `tests/test_deploy_keep_runtime_config.sh`（用真的 git repo 跑一次 reset --hard）。
> 新增「程式會自己寫、又被 git 追蹤」的設定檔時，要加進該腳本的 `RUNTIME_CONFIGS`。

> 對外網路不穩時 runner 會接不到 job / `git fetch` 失敗 → 手動 scp 改檔 + 重啟為 fallback。

---

## 遠端運維、現場網路與遠端連線（現場站必讀）

> 現場主機平時**沒有對外網際網路**，插上網卡才上線。以下配置是**每台主機的系統設定**（含
> tailscaled 憑證、nmcli 連線、systemd watchdog），**不進版控**、機器各自管；本節只記
> 機制、位置與排查方式，方便任何一台上的 Claude Code / 運維人員接手。真正的 auth key 不寫在此。

### A. 現場網路（雙網路並存：4G 上網 + 現場網段）

現場會同時接**兩條網路**，用途不同、互不搶路由：

| 網卡 | 網路 | 用途 |
|---|---|---|
| `enP5p4s0` | 4G / 可上網（DHCP） | 維護通道（Tailscale、CI 部署、對外連線） |
| `enP5p5s0` | 現場網段 `10.42.x` | **客戶連線走這條** |

nmcli 連線 `field-net` 已配好（**不用再下設定命令**，開機自動套用）：

```
ipv4.addresses   10.42.38.35/20          遮罩 255.255.240.0
ipv4.routes      10.0.0.0/8 → 10.42.32.254
ipv4.never-default  yes
connection.autoconnect  yes
```

**🛑 兩個關鍵設計，改動前先看懂：**

1. **`never-default: yes`** —— 現場那條永遠不會變成預設路由，上網固定走 4G。
   若現場網路才是唯一對外出口，才需要拿掉這個設定。

2. **閘道寫在 `ipv4.routes` 的下一跳，不是 `ipv4.gateway`** ——
   同時設 `ipv4.gateway` 與 `never-default=yes` 時，NetworkManager 會把
   gateway **直接丟掉**（`nmcli con show` 顯示 `--`），閘道等於沒設。
   寫成明確路由的下一跳才會生效。

驗證方式（看核心實際決策，不要只看設定檔）：
```bash
ip route get 10.42.32.254   # → dev enP5p5s0
ip route get 10.26.4.123    # → dev enP5p5s0 via 10.42.32.254
ip route get 8.8.8.8        # → dev enP5p4s0 via <4G 閘道>
```

`10.42.32.0/20` 內是直連；範圍外的 `10.x` 靠 `10.0.0.0/8` 那條走現場閘道。
若現場有非 `10.x` 的網段，要另外加路由。

> 未接到現場網路時 `ping 10.42.32.254` 不通是正常的（ARP 顯示 `INCOMPLETE`），
> 實體層 `carrier=1`、協商 1000 Mbps 即代表線與網口沒問題。

### A-2. 對外串流位址（`STREAM_HOST`，一定要設）

`/api/v1/external/streams` 回傳的 RTSP／HLS 網址是用主機 IP 組出來的，
而程式是靠「往外網的路由」推測本機 IP —— 配合上面的 `never-default`，
**推出來的必然是 4G 那張網卡**，客戶從 `10.42.x` 連不到，
而且不會有任何錯誤訊息。

所以 `.env` 必須明確指定：

```env
STREAM_HOST=10.42.38.35
```

`.env` 有 gitignore，`git reset --hard` 不會動它，設一次即可。
驗證：`curl -H "X-API-Key: <key>" http://10.42.38.35:8000/api/v1/external/streams`
回傳的 `host` 應為 `10.42.38.35`。

> 網頁介面**不受影響** —— 它用相對路徑走 traffic-api 代理，
> 瀏覽器用連進來的位址取影像，主機 IP 怎麼變都看得到。

### B. 遠端連線（Tailscale mesh VPN，穿 CGNAT / 隔離網）
- 本機 Tailscale：機器名 `field-jetson`、tailnet IP **`100.92.17.87`**、tailnet 帳號 `sannielshi@`、已開 `--ssh`、`tailscaled` 開機自啟+自動重連。
- **系統登入用戶是 `ubuntu`**；`sannielshi@` 只是 Tailscale 帳號，別拿來當 ssh user。
- 連法（你的電腦先裝 Tailscale 並登入同帳號）：
  - `ssh ubuntu@100.92.17.87`，或 MagicDNS `ssh ubuntu@field-jetson`
  - Cowork / Claude Code 桌面版：SSH host 填 `ubuntu@100.92.17.87` 或 `ubuntu@field-jetson`
- **關鍵特性**：Tailscale 是疊在底層網路上的 overlay，兩端不用同網段、你的電腦也不用能直連現場 10.42.x；只要**現場那條 eth port 能通外網**，主機就自動回 tailnet，`100.92.17.87` 就連得進（不隨底層 IP 變）。純內網不通外網才需改插行動網卡。

### C. 上線監控 watchdog + 手機通知（ntfy）
- `/usr/local/bin/field-link-monitor.sh` + `field-link-monitor.service`（systemd 開機自啟）：
  探測網路（gstatic 204）+ tailscale 狀態，**上線那刻推一則 ntfy 心跳（附 Tailscale IP）**，斷線時補跑 `tailscale up`。狀態機 offline / net_no_ts / online 只在轉態時動作。
- ntfy topic 存在主機 `/etc/default/field-link-monitor`（`NTFY_TOPIC=...`）；手機 ntfy app 訂閱該 topic 即收上線通知。**知道 topic 即可收發，勿外流。**
- **現場上線確認流程**：手機收到 ntfy 上線通知 → `ssh ubuntu@100.92.17.87 hostname`（應回 `nvagxorinafer750a1`）→ 通了代表遠端運維正常。離線時連不上是正常，不是故障。

### D. 電子鎖讀卡硬體切換（USB-485 / THS2，改 env 不改碼）
- 環境變數 `LOCK_SERIAL_PORT`（在 `.env`）決定讀卡通道：
  - **留空 / 註解** → THS2 自動方向轉換器模式：門磁、開關鎖動作可讀，但**讀不到刷卡卡號**（長回應被 turnaround 吃中段）、FC10 長幀寫入失敗（優雅降級）。
  - **設 `/dev/ttyUSB0`**（USB-485，FTDI FT232）→ 全功能：可讀 0xC000 開鎖成功記錄（含卡號+持卡人）、0xD000 失敗記錄（未授權/失效卡告警）。
- 現場換好硬體後只要 uncomment `LOCK_SERIAL_PORT=/dev/ttyUSB0` 再重啟 traffic-api，**不用改程式碼**。另見 `電子鎖_README.md`。

### E. 給 Jetson 本機 Claude Code 的提示
- 本專案程式碼透過 `git pull` 同步；**上面 A–D 的系統配置不在 repo**，實體在主機的 nmcli / systemd / `/etc/default` / `.env`。要查現況直接讀那些檔或 `systemctl status field-link-monitor tailscaled traffic-api traffic-io`。
- 部署機資訊、CI 流程見 `RUNBOOK.md` 與本檔上半部。
