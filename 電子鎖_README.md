# 電子鎖 (E-1507 LEEKA) 功能說明

E-1507 電子鎖透過 Modbus RTU 接 Jetson THS2 RS-485，提供即時狀態顯示與刷卡/開鎖事件記錄。

---

## 1. 硬體 / 通訊

| 項目 | 說明 |
|------|------|
| 鎖型號 | E-1507 (LEEKA)，Modbus RTU，9600 8N1 |
| 接線 | Jetson 40-pin THS2 RS-485（`/dev/ttyTHS2`），與 PD3R3 IO 模組**同一條匯流排** |
| 位址 | PD3R3 IO=**1**、鎖①=**2**、鎖②=**3**（都出廠 1，用 FC06 寫 0x2000 改開）|
| 啟用 | 環境變數 `LOCK_MODBUS_ADDR`（traffic-io 的 `.env`）；單鎖 `=2`，雙鎖 `=2,3`；未設則停用、零影響 |

### 多鎖（同一條 485 掛兩顆）

- 每顆鎖一份獨立狀態（`_LockState`）：連線、門磁、告警旗標、邊沿偵測全部各自算，**一顆斷線不影響另一顆**。
- 事件共用一條 deque，每筆帶 `addr`（DB `lock_events.lock_addr`）；前端事件列表會標 `#2` / `#3`。
- **卡片白名單是每顆鎖各自的** —— 要兩顆都能刷，兩顆都要加卡。卡片庫的 `lock_addr` 記錄該卡加在哪顆。
- 🛑 **新鎖出廠位址是 1，和 PD3R3 撞號**。Modbus 寫入是位址導向的，`old=1` 改位址會**同時寫到 PD3R3 的 0x2000**。所以匯流排上必須只剩那顆要改的鎖（PD3R3 拔掉/未接）才能做，API 預設擋下，要帶 `force=1`。
- 加鎖流程：`GET /api/lock/scan` 看它現在在哪個位址 → `POST /api/lock/set-addr` 改開 → `.env` 加進 `LOCK_MODBUS_ADDR` → 重啟 traffic-io。

### 讀不到的鎖要退避（`LOCK_RETRY_SEC`，預設 5 秒）

serial timeout 是 0.3 秒。一顆沒接的鎖若每個 150ms tick 都去讀，光 timeout 就把迴圈週期拖到 350ms+，
**連帶讓「有接的那顆」漏掉只保持 ~150ms 的刷卡動作** —— 一顆沒接拖垮另一顆。
所以讀失敗的鎖會退避到 `LOCK_RETRY_SEC` 後才重試，期間只走慢輪詢做失聯判定。

### THS2 換向限制與對策（重要）
THS2 板載 RS-485 轉換器 TX→RX 換向慢，會吃掉「多暫存器長回應」的前段（只剩尾端 ~3 byte）。對策：
- **狀態暫存器一律 count=1 逐個讀**（7-byte 短回應能完整收到、CRC 自驗通過）。見 `pd3r3.py read_holding_at`。
- `io_module.read_holding` **繞過 PD3R3 `_ok` gate**：鎖是獨立從機，PD3R3 沒接也能讀。
- 鎖 poll 用**獨立 `_lock_loop` thread**，與 PD3R3 共用 `IOModule._lock` 序列化 RS-485（避免並發 native SEGV）。

---

## 2. 即時狀態

讀 4 個狀態暫存器（每個 count=1）：

| 暫存器 | 功能 | 值 |
|--------|------|-----|
| 0x0020 | 手柄狀態 | 0=在位 / 1=不在位 |
| 0x0021 | 門磁狀態 | **1=閉合(門關) / 0=斷開(門開)** — 門磁為 NO 常開接點，與協議文件寫的相反，現場實測為準 (2026-08-12) |
| 0x0022 | 鑰匙狀態 | 0=在位 / 1=不在位 |
| 0x0023 | 鎖具動作 | 0=無 / 1=刷卡 / 2=密碼 / 3=指紋 / 4=鑰匙轉 / 5=手柄開（idle 實測回 0xfa=250，當作「無」） |

- 門磁/手柄/鑰匙每 ~1 秒慢輪詢；**動作暫存器每 150ms 高頻讀**（刷卡只保持 ~0.15s，慢輪詢會漏）。
- 邏輯在 `io_service.py`：`_lock_loop` / `_poll_lock_states`。

---

## 3. 刷卡 / 開鎖事件

- 動作暫存器 0x0023 做**上升沿偵測**（idle→1~5）→ 觸發一筆事件（`_fire_lock_event`）。
- 記錄欄位：開鎖方式、時間、當下門磁/手柄/鑰匙狀態。
- daemon 端進 deque（`lock_events`，maxlen=100）+ 單調 seq；traffic-api(client) 拉去**寫 DB + 推 WS**。
- **事件類型 `event_type`**:swipe(刷卡)/door(門開關)/handle(手柄)/key(鑰匙)/alarm(警報)/unlock(遠端開鎖)。門磁/手柄/鑰匙狀態變化(邊沿偵測)也記一筆,前端記錄列表按類型分色(刷卡綠/門磁手柄藍/警報紅)。
- **即時警報**:門開超過 `LOCK_DOOR_ALARM_SEC`(預設30秒) → 觸發 alarm 事件(前端整行紅色醒目),門關後重置。`_lock_loop._detect_state_events`。
- **遠端開鎖**:`POST /api/lock/unlock` → 寫 0x2004=0x0033 + 記 unlock 事件。前端綠色「遠端開鎖」按鈕(confirm 後執行)。

### 新增卡片（學習式加卡）
- 前端「**+ 新增卡片**」按鈕 → `POST /api/lock/add-card` → 寫 `0x2005=0x0033` → 鎖進入**加卡模式**（約 10 秒）→ 在鎖上**刷要新增的卡**，鎖自動學習錄入卡號。
- 寫入用**寬容模式**（`pd3r3.write_holding_at` FC06：THS2 換向會吃 echo，寫完 drain 不驗回應，寫入已生效；同改位址經驗）。
- 鏈路：`POST /api/lock/add-card`(lock.py) → daemon `POST /lock/add_card` → `io_service.add_lock_card` → `io_module.write_holding`(繞過 _ok) → `pd3r3.write_holding_at`。
- **加卡建庫**：加卡成功後 daemon 同步監測 0x0044/45 拿新卡號 → 寫進 `lock_cards` 卡片庫(卡號+持有人+部門)，前端卡片庫列表顯示。
- **刪卡(記錄內直接刪,不需現場刷)**：卡片庫列表點「刪除」→ `DELETE /api/lock/cards/{id}` → 用卡號 **FC10 寫刪卡寄存器 0x0047-0x0049**(工號0+卡號) 直接刪鎖內卡 + 庫標記停用。pd3r3 `write_multi_holding_at`(FC10 寬容寫)。
- （學習式刪卡 `0x2005=0x0055`+現場刷 的 API `remove-card` 仍保留備用，但前端已改用上面的直接刪。）
- ⚠️ 卡號讀取限制：每次**刷卡開鎖的卡號**在 0xC000(連讀7,THS2 換向讀不到,四條路實測全堵) → 「刷卡是哪張卡」需 USB-RS485;故只做「加卡建庫」(加卡時 0x0044/45 可讀)。

---

## 4. API

| 端點 | 說明 |
|------|------|
| `GET /api/lock/status` | 即時狀態快照；`locks[]` 每顆一筆，頂層 addr/connected/status 是第一顆（相容單鎖呼叫端） |
| `GET /api/lock/scan?lo&hi` | 掃描 485 上哪些位址會回應（唯讀，不用停 traffic-io） |
| `POST /api/lock/set-addr` | 改鎖位址（FC06 寫 0x2000）；撞號會擋，`old` 撞 PD3R3 需 `force` |
| `GET /api/lock/events?page&page_size` | 刷卡/開鎖歷史（DB，分頁，最新在前） |
| `WS /api/lock/live` | 即時推送刷卡/開鎖事件 |
| `POST /api/lock/add-card` | 加卡(學習式)：鎖進入加卡模式，隨後鎖上刷卡錄入 |
| `POST /api/lock/add-card` 回 card_no | 加卡(學習式)+同步監測新卡號寫卡片庫 |
| `GET /api/lock/cards` | 卡片庫列表(卡號/持有人/部門) |
| `PUT /api/lock/cards/{id}` | 編輯持有人/部門 |
| `DELETE /api/lock/cards/{id}` | 刪卡(FC10 直接刪鎖內卡+庫停用,不需現場刷) |
| `POST /api/lock/remove-card` | (備用)學習式刪卡:進刪卡模式現場刷 |
| `POST /api/lock/unlock` | 遠端開鎖(0x2004=0x0033) |
| daemon `GET /lock_events?since=N` | (內部) traffic-api long-poll 拉事件 |
| daemon `POST /lock/add_card` `/lock/remove_card` | (內部) 寫 0x2005=0x0033 / 0x0055 |

---

## 5. 前端

- 導航選單（admin 區）→「**電子鎖**」分頁，iframe 載入 `web/lock_panel.html`。
- **即時狀態卡**：門磁/手柄/鑰匙/動作 4 項 + 連線 badge（2 秒輪詢 `/api/lock/status`）。
- **刷卡/開鎖記錄卡**：WS 即時跳綠色 banner + 歷史列表（時間 + 方式 badge + 門磁註記）。

---

## 6. 資料庫

`lock_events` 表（`api/models.py`）：

| 欄位 | 說明 |
|------|------|
| action_code / action_label | 開鎖方式 (1~5) |
| door_closed / handle_in_place / key_in_place | 當下狀態 |
| lock_addr / created_at | 位址 / 時間 (UTC) |

---

## 7. 調參 / 環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `LOCK_MODBUS_ADDR` | 0(停用) | 鎖 Modbus 位址；單鎖 `2`，多鎖逗號分隔 `2,3` |
| `LOCK_RETRY_SEC` | 5 | 讀不到的鎖隔多久重試（避免一顆沒接拖垮另一顆，見 §1）|

---

## 8. 已知限制

**工號 / 卡號 / 開鎖成功失敗區分**目前**讀不到**：
- 這些在開鎖成功記錄 0xC000(連讀7暫存器) / 失敗記錄 0xD000(連讀6暫存器)，是**特殊功能暫存器必須連讀多個**。
- THS2 換向吃掉長回應前段 → 只剩尾 3 byte → 讀不到。
- **要取得須換 USB-RS485 轉接器**（換向正常，整幀可讀）。

目前能記：「**何時、用哪種方式**開鎖 + 當下門/手柄/鑰匙狀態」；無法記「**誰**開的(工號/卡號)、成功/失敗」。

---

## 9. 檔案清單

| 檔案 | 內容 |
|------|------|
| `services/pd3r3.py` | `read_holding_at` / `write_holding_at`（Modbus FC03/FC06，count=1 短幀） |
| `services/io_module.py` | `read_holding`（繞過 _ok gate） |
| `services/io_service.py` | `_lock_loop` / `_poll_lock_states` / `_fire_lock_event` / `_start_lock_monitor` / `_client_lock_poller` / `_persist_lock_event` / `lock_events` |
| `services/io_daemon.py` | `/lock_events` 端點 |
| `api/routes/lock.py` | `/api/lock/status` `/events` `/live` |
| `api/models.py` | `LockEvent` 表 |
| `web/lock_panel.html` | 獨立電子鎖頁面 |
| `web/index.html` | 導航項 + iframe |

---

## 10. 故障排查

- **狀態顯示「未連線」**：確認 `LOCK_MODBUS_ADDR=2` 在 `.env`、鎖位址確實是 2（FC06 0x2000）、A/B 接線、`traffic-io` 服務狀態。
- **動作/刷卡抓不到**：動作只保持 ~0.15s，確認 `_lock_loop` 在跑（log `[io_svc] _lock_loop entered`）。
- **讀到亂碼/只剩尾段**：THS2 換向問題，確認用 count=1 逐個讀（不要一次讀多個狀態暫存器）。

備註：相關開發歷程見 memory `project_e1507_lock_ths2_turnaround`。
