# 交通違規影像分析系統 — Claude 指引

Jetson NX 邊緣運算：車輛偵測、車牌辨識、違規偵測、**壅塞偵測**、NVR 整合。

## 技術棧
- **後端**：FastAPI（:8000）+ SQLAlchemy + SQLite（`data/violations.db`）
- **前端**：Vue 3 SPA（`web/index.html`）+ Element Plus
- **AI**：YOLOv8n + TensorRT（車輛）、YOLO26s-cls（大型車細分）、YOLO 字元偵測微服務 :8010（車牌 OCR，見 `ocr 流程.md`）
- **NVR**：Frigate + MQTT
- **平台**：JetPack 6.0 / CUDA 12.2 / TensorRT 8.6 / Python 3.10

## 關鍵目錄
| 路徑 | 用途 |
|------|------|
| `api/routes/` | FastAPI 路由（stream、lpr、congestion、frigate、violations …） |
| `api/routes/congestion.py` | **壅塞偵測服務**（狀態、啟停、串流） |
| `detection/congestion_detector.py` | **壅塞偵測核心演算法**（佔用率、四級等級） |
| `detection/vehicle_detector.py` | YOLOv8 車輛偵測（整合大型車分類） |
| `detection/violation_detector.py` | 違規規則引擎（闖紅燈/超速/停車/逆向） |
| `recognition/plate_recognizer.py` | 車牌 OCR |
| `config/frigate/config.yml` | Frigate NVR 設定 |

## 重要文件
- `README.md` — 系統總覽、架構圖、安裝部署
- `RUNBOOK.md` — 運維手冊
- `API整合文件.md` — 對外 API 規格
- `DB_ERMODEL.md` — 資料庫 ER Model
- `ocr 流程.md` — 車牌 OCR pipeline
- `ramp_analyzer_README.md` — 匝道分析器
- `MODEL_PATHS_PROPOSAL.md` / `model_paths.py` — 模型路徑管理

## 攝影機命名慣例
格式：`<camera_id>_<lane_id>`，例如 `62_1` 表示 62 號攝影機第 1 車道。

## 開發注意
- 模型檔（`models/*.pt`, `*.engine`）與 `storage/` 不納入版控。
- 即時串流除錯：直接從 live overlay 抓 frame 診斷，不要等使用者截圖。
- 自動行為必須對應使用者設定 toggle，不要 hardcode。

## ⚠️ 修改 web/index.html 前後必跑（強制 SOP）

`web/index.html` 是單檔 11000+ 行 Vue 3 inline template SPA，HTML parser 對
balance 敏感，**任何 `<div>` open/close 不平衡會讓 `<div id="app">` 提早關**，
造成後半段 page 跟 dialog 變 raw `{{ mustache }}` 顯示。歷史教訓：
- `dfea447` (dashboard V3) 殘留 1 個 extra `</div>` → user 看「NVR Motion
  事件預覽」內 `{{ formatNVRTime(...) }}` raw 字串
- `f226929` (NVR overlay 重做) 加 `v-if="x > 60"` HTML parser 把 `>` 當
  end-tag → Vue compile 整頁失敗白屏 → user 無法登入

⚠️ **這條規則適用於「所有 inline-template 頁」，不是只有 index.html。**
`web/` 下同時有 `createApp` 與 `<div id="app">` 的頁面都算：
`index.html`、`roi_editor.html`、`parking_editor.html`、`parking_label.html`、
`nvr_playback.html`、`io_panel.html`、`lock_panel.html`、`lpr_verify.html`、
`source-test.html`。
2026-08-07 教訓：`roi_editor.html` 有 27 個自閉合 `<el-* />`，車流區列表每一列
只剩 `#N` 標籤 —— 名稱框、下拉、點數、**連「儲存」「刪除」按鈕都是隱形的**。
SOP 當時只寫 index.html，所以這個 bug 藏了很久沒被發現。

**改動任何 inline-template 頁後 commit 前必跑這兩步**：

```bash
# 1. 整檔 div balance（open 必須等於 close）
python -c "import re,pathlib; s=pathlib.Path('web/index.html').read_text(encoding='utf-8'); print('open', len(re.findall(r'<div[\s>]', s)), 'close', s.count('</div>'))"
# 預期: open 1089 close 1089 (數字可變但兩個必須相等)

# 2. Vue template lint —— 一次掃過所有 inline-template 頁
python scripts/lint_vue_template.py --all
# 或只掃改到的那幾頁: python scripts/lint_vue_template.py web/roi_editor.html
#
# 檢查兩件事:
#   (a) Vue directive 內 unescaped >/<  → 新加的條件寫 `x > 60` 要改
#       `60 < x` 或 `&gt;` entity
#   (b) [SELF-CLOSING] 不渲染 slot 的元件自閉合會吞掉後面的兄弟節點
#       (el-input / el-input-number / el-switch / el-date-picker …)
#       一律改成成對閉合 <el-input ...></el-input>
#
# 🛑 要批次修自閉合「不要用 regex」——惰性量詞會跨過標籤邊界，
#    把 <el-option/> 關成 </el-select>，整份模板會被改壞（已實際踩過）。
#    用逐字元掃描：遇到 <el- 讀標籤名 → 掃到該開始標籤自己的 `>`（跳過引號內容）
#    → 只在結尾是 `/>` 時改寫。改完驗每種標籤 open/close 數量對稱。
```

**Vue template rules**（避免 HTML parser 截斷）：
- v-if / v-show 內**不可寫**：`x > N`、`x < N`、`a && (b > c)` 等比較式
  - 改用反向 `N < x` (但 `<` 仍要 escape 成 `&lt;`)
  - 或抽 computed: `:isOverThreshold` 然後 `v-if="isOverThreshold"`
  - 或用 `Math.max`/`Math.min` 之類避開比較
- attribute value 內已有 `>` `<` 的 :class / :style 是 HTML5 spec 合法可保留
- v-html 不會解析 mustache — 內容是字串

**Deploy 流程已自動化**（不需要手動 ssh）：
- dev PC `git push origin main`
- GitHub Actions `Jetson Device Verification` 自動觸發
- self-hosted runner (jetson-agx-orin) 跑 5 個 critical step
- verify 過 → 自動 `git pull production` + `sudo systemctl restart traffic-api`
- 2 分鐘內 production 上線 + commit hash 驗證

**Memory bridge** — 已建立的 feedback memory：
- `feedback_vue_template_html_parser.md` — v-if 比較運算子要 escape
- `reference_jetson_deploy.md` — Jetson SSH key auth 已雙向設好

## Clone 後第一次設定
`data/` `models/` `output/` `storage/` 四個目錄**不在版控**，每台機器自己管。clone 後跑：

```bash
# 本機硬碟模式（開發機、無外接碟的 Jetson）
bash scripts/init_dirs.sh

# 想放在 NVMe / 外接硬碟
TRAFFIC_STORAGE_ROOT=/mnt/nvme/traffic bash scripts/init_dirs.sh
```

腳本會自動偵測壞掉的 symlink、既有實體目錄，安全重跑。
