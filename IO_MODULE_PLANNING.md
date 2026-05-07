# I/O 模組整合規劃 — AFE-R750 + tM-PD3R3

外部 I/O 規範（A 控制 / B 燈號顯示）對應到 ICP DAS **tM-PD3R3** 的 3-DI/3-DO + AFE-R750 實體電源鍵。本文件涵蓋 BOM、合規對照、雙電源設計、P1 測試腳本與 P2 正式 driver 規劃。

---

## 1. 合規對照（最終版）

| 規範 | 實作 | 腳位 | 狀態 |
|------|------|------|------|
| (A) 電源開/關 | AFE-R750 實體電源鍵 | 24V 純硬體 | ✅ 軟體不介入 |
| (A) 控制模式 | DO2 白燈顯示（**滅=自動(遠端)、亮=手動**） | DO2 | ✅ |
| (A) 遠端下載 | tM-PD3R3 **DI0** + Φ22 momentary 按鍵 → config sync | DI0 | ✅ |
| (A) Reset | tM-PD3R3 **DI1** + Φ22 momentary 按鍵 | DI1 | ✅ |
| (B) 電源 ⚪ | 不處理 | — | — |
| (B) 通訊故障 🔴 | tM-PD3R3 **DO0** → 12V 紅燈 | DO0 | ✅ 恆亮=故障、滅=正常 |
| (B) 運作狀況 🟢 | tM-PD3R3 **DO1** → 12V 綠燈 | DO1 | ✅ 恆亮=正常（含閒置）、滅=故障 |
| (B) 操作模式 ⚪ | tM-PD3R3 **DO2** → 12V 白燈（與控制模式共用） | DO2 | ✅ **恆亮=手動、閃爍 ~2.5Hz=遠端下載中、滅=自動(遠端)+閒置** |

**DO2 雙功能說明**：同一顆白燈，用「點亮樣式」區分「控制模式」vs「下載中」——  
`白燈 恆亮` = 手動模式；`白燈 閃爍 (~2.5Hz)` = 遠端下載中；`白燈 滅` = 自動(遠端)模式且非下載中（且無通訊故障）。  
**設計理念**：自動(遠端)模式為預設運行狀態 → 燈號最少 (只剩綠燈)；切到手動模式時白燈恆亮、下載中時白燈閃爍提醒操作者。  
**設計亮點**：DI2 釋出保留（尚未指派功能），2-DI + 3-DO 實現所有燈號顯示；(B) 電源燈不處理。

---

## 2. BOM 清單

| 項目 | 規格 | 數量 | 狀態 |
|------|------|------|------|
| AFE-R750-X1A1U | AGX Orin 64GB | 1 | ✅ 已有 |
| 24V/230W 變壓器 | 主機電源 | 1 | ✅ 已有 |
| 12V DC 電源供應器 | ≥2A，工業級 | 1 | 🛒 採購（供 tM-PD3R3 + 燈號/按鍵迴路） |
| **tM-PD3R3 CR** | **3 DI + 3 DO，Modbus RTU** | **1** | **✅ 已有** |
| 12V 工業指示燈 | Φ22mm，**白 ×1、綠 ×1、紅 ×1**（操作模式/運作/故障） | 3 | ✅ 現有紅綠白各 1 足夠 |
| **照明型工業按鍵** | Φ22mm，**內建 12V LED**，無鎖 momentary | **2**（DI0 遠端下載 + DI1 Reset） | 🛒 採購（**指定 LED 照明款**，同價位） |
| RS-485 雙絞線 | 屏蔽，2C | 視距離 | 🛒 採購 |
| 120Ω 終端電阻 | 1/4W | 2 | 🛒 採購 |
| 端子台 / 配線槽 / DIN 導軌 | — | — | 🛒 採購 |

---

## 3. 雙電源域設計

```
┌─ 24V/230W ──┬──▶ AFE-R750（主機 + 推論卡）
              │
              └──▶ 24V→12V DC-DC 或獨立 12V PSU
                          │
                          ├──▶ tM-PD3R3（VS+/VS−）
                          ├──▶ 4 顆 12V 工業指示燈（電源/運作/操作/故障）
                          └──▶ 2 顆按鍵迴路上拉源（DI0/DI1）
```

- **24V 域**：AGX 主機 + 推論卡（高耗電）
- **12V 域**：低壓控制 / 燈號 / I/O 模組（隔離高頻干擾）
- 共地：所有 GND 接共同接地排（**先確認 tM-PD3R3 的 ISO 是否支援電源隔離**）

---

## 4. 通訊設計

| 項目 | 設定 |
|------|------|
| 介面 | RS-485 半雙工 |
| 協定 | Modbus RTU |
| 預設 baudrate | 9600 8N1（出廠值，依現場可調） |
| Slave ID | 1（出廠預設） |
| 訊號線 | D+ / D− / GND（共 3 線屏蔽雙絞線） |
| 終端電阻 | 兩端各 120Ω |
| 主機介面 | AFE-R750 內建 COM（`/dev/ttyTHS1`）**或** USB-RS485 轉換器（Plan B）|

### Modbus 暫存器映射（tM-PD3R3 出廠規格）

| 用途 | Modbus 功能碼 | 起始位址 | 數量 |
|------|--------------|---------|------|
| 讀 DI 狀態 | 02 (Read Discrete Input) | 0x00000 | 3 |
| 讀 DO 狀態 | 01 (Read Coils) | 0x00000 | 3 |
| 寫單一 DO | 05 (Write Single Coil) | 0x00000 | 1 |
| 寫多個 DO | 15 (Write Multiple Coils) | 0x00000 | 3 |

---

## 5. 燈號 / 按鍵顯示邏輯

### 5.1 面板 4 顆獨立指示燈（燈號顯示）

| # | 燈號 | 顏色 | 訊號源 | 邏輯 | 閃爍 |
|---|------|------|--------|------|------|
| 1 | 電源 | ⚪ 白 | — | 不處理 | — |
| 2 | 通訊故障 | 🔴 紅 | DO0 | 恆亮=故障 (NTP/link/IP)、滅=正常 | — |
| 3 | 運作狀況 | 🟢 綠 | DO1 | 恆亮=系統正常（含閒置）、滅=故障 | — |
| 4 | 操作模式／下載中 | ⚪ 白 | DO2 | **恆亮=手動模式；閃爍=遠端下載中；滅=自動(遠端)+閒置** | ~2.5 Hz when downloading |

**燈號組合速查：**

| 場景 | 🔴 DO0 | 🟢 DO1 | ⚪ DO2 |
|------|--------|--------|--------|
| **正常 / 閒置（自動／遠端）** | OFF | **ON** | OFF |
| 正常 / 閒置（手動） | OFF | **ON** | **ON (恆亮)** |
| 遠端下載中（自動模式） | OFF | **ON** | **BLINK ~2.5Hz** |
| 遠端下載中（手動模式） | OFF | **ON** | **BLINK ~2.5Hz** (蓋過手動恆亮) |
| 通訊故障 | **ON** | OFF | OFF |
| Sync 失敗 | OFF | **ON** | OFF (回原狀態，**只記 log 不動燈**) |

### 5.2 照明型按鍵內建 LED 邏輯

2 顆 momentary 按鍵都選**內建 12V LED** 款（DI2 釋出保留，不配按鍵）：

| 按鍵 | 對應 DI | 功能 | LED 行為 |
|------|--------|------|---------|
| 遠端下載 | DI0 | 觸發 config sync | 按下發光；可並聯 **DO2** 顯示下載中 |
| Reset | DI1 | 系統 Reset → `systemctl restart traffic-api.service` | 按下發光，放開熄滅 |

→ DI0 按下 → `config_sync.trigger()` → **DO2 白燈閃爍 ~2.5Hz（下載中）** → 完成後白燈回原狀態（自動=滅 / 手動=恆亮）；**失敗只記 log，不動任何燈號**（避免跟 DO0 通訊故障語意混淆）。
→ DI1 按下 → log 提示 → 2 秒後 **主動把 3 顆 DO 全滅**（DO1 綠燈滅 = 系統沒在跑的視覺提示）→ `sudo -n systemctl restart traffic-api.service`（依靠 NOPASSWD sudo）→ 重啟期間 ~15-100 秒燈號維持全暗；新 process 起來後 `start() → _apply_do()` 自動復寫燈號（DO1 綠燈重新亮起代表 ready）。

### ✅ 已決議（2026-05-07）

1. **「通訊故障」DO0 觸發條件** — 改採 **網路層三層檢查**（NTP / 主網卡 link / 主網卡 IP），任一失敗即觸發。實作於 `services/network_health.py`，30 秒輪詢。原本提的 MQTT/相機/health 路線太貼業務層，遇到後端啟動順序或單一 service 短暫掛掉就誤報，網路層較穩定。
2. **中央伺服器 config URL** — 預設 `http://192.168.0.101:8080/api/config`，存於 `config/system/io_settings.json`（網頁可改寫）；`.env CONFIG_SYNC_URL` 為 fallback。實際可達性待 P3 驗收。

---

## 6. 軟體階段規劃

### P1 — 通訊驗證腳本（`scripts/test_modbus_io.py` 已交付，待實機驗收）

3 階段漸進測試，**任何一段失敗就停下來修**，避免後續封裝白做：

1. **通訊驗證** — 讀 DI 確認 Modbus 有回應，失敗時印硬體/BIOS/驅動四層排查清單
2. **DO 跑馬燈** — 3 顆燈輪流亮 3 圈，肉眼驗證接線正確
3. **按鍵監聽** — 20Hz 輪詢 DI，按下/放開印 edge event

### P2 — 正式 Driver + Service（P1 通過後動工）

| 模組 | 路徑 | 職責 |
|------|------|------|
| `TmPD3R3Driver` | `services/io_module.py` | Modbus 讀寫、連線管理、自動重連 |
| `IOService` | `services/io_service.py` | 系統健康度 → DO 對映、DI edge → 內部事件、閃爍 timer |
| API 路由 | `api/routes/io.py` | `GET /api/io/status` / `POST /api/io/do/{id}` / WS 即時 DI 事件 |
| Web UI | `web/index.html` | I/O 監控面板（admin only） |
| MQTT 整合 | 透過 `services/mqtt_bridge` | DI 事件 → publish `io/event/...`；DO 狀態心跳 |

### P3 — 規範驗收

- 場勘對照規範條文 → 點亮每顆燈、按每個按鈕、量電氣特性
- 文件交付：本檔 + 接線圖 + 規範條對應截圖

---

## 7. Jetson RS-485 已知雷點（P1 必經）

### 7.1 `/dev/ttyTHS*` 命名與權限

- Jetson 用 Tegra High-Speed UART (`ttyTHS`)，**不是** `ttyS` 或 `ttyUSB`
- JetPack 6 預設 `ttyTHS0` 被 serial console (`nvgetty.service`) 佔用
- AFE-R750 上實際裝置編號可能是 `ttyTHS1` 或更後面，**先用 `ls -l /dev/tty*` 確認**

```bash
# 釋放被 console 佔用的 UART
sudo systemctl stop nvgetty.service
sudo systemctl disable nvgetty.service

# 給用戶權限（避免每次都 sudo）
sudo usermod -a -G dialout ubuntu
# 登出再登入生效
```

### 7.2 ⚠️ 最大雷點：RS-485 方向控制 (DE/RE)

RS-485 半雙工要切「發送/接收」方向：

| AFE-R750 設計 | 軟體要做的事 |
|--------------|------------|
| 自動方向控制（推薦） | 不用管 ✅ |
| RTS 控制 | 透過 `TIOCSRS485` ioctl 強制切換 |
| 不支援自動切換 | **直接換 USB-RS485** |

**症狀**：能發但讀不回 / 收到自己 echo / Modbus CRC 永遠錯。

### 7.3 BIOS / 硬體模式切換

AFE-R750 COM 可在 **RS-232 / RS-422 / RS-485** 三模切換，要確認：
- BIOS 設定切到 RS-485
- Carrier board 上的 jumper / DIP switch 對應位置

### 7.4 Plan B：USB-RS485 轉換器（強烈建議備一條）

| 比較 | AFE-R750 內建 COM | USB-RS485 轉換器 |
|------|------------------|----------------|
| 成本 | 0（已有） | NT$300~800 |
| 驅動穩定性 | Jetson 驅動可能有雷 | 100% 即插即用 |
| 自動方向控制 | 看硬體設計 | 晶片內建，免煩惱 |
| 裝置路徑 | `/dev/ttyTHS1` | `/dev/ttyUSB0` |

**推薦晶片**：CP2102 / CP2104（Silicon Labs）或 FT232RL + MAX485（FTDI）。
**避開**：CH340 便宜貨（在工業環境不穩）。

---

## 8. P1 測試 SOP

```bash
# 0. 安裝依賴
pip install --user pymodbus pyserial

# 1. 環境診斷（不送資料）
python3 scripts/test_modbus_io.py --diagnose

# 2. 內建 COM 試（Plan A）
sudo chmod 666 /dev/ttyTHS1
python3 scripts/test_modbus_io.py --port /dev/ttyTHS1 --slave 1 --baudrate 9600

# 3. 失敗 → 強制 RS-485 模式
python3 scripts/test_modbus_io.py --port /dev/ttyTHS1 --force-rs485-mode

# 4. 還是失敗 → 換 USB-RS485（Plan B）
python3 scripts/test_modbus_io.py --port /dev/ttyUSB0
```

**預期通過標誌**：
- ✅ 通訊正常，DI 目前狀態: [False, False, False]
- DO 跑馬燈肉眼可見三顆輪流
- 按下實體按鍵 print 出 edge event

P1 跑通 → 可進 P2。

---

## 9. 待辦清單

### ✅ 已完成

- [x] **DO0 通訊故障觸發條件確定** — 改採 NTP / link / IP 網路層三層（`services/network_health.py`，30s 輪詢）
- [x] **CONFIG_SYNC_URL 預設值** — `http://192.168.0.101:8080/api/config`，存於 `config/system/io_settings.json`（網頁可改寫）
- [x] **DI 編號修正** — `services/io_service.py` 最終 `DI_DOWNLOAD=0, DI_RESET=1`（DI2 改釋出保留）
- [x] **WS 事件追蹤改用 monotonic seq** — `deque(maxlen=50)` 飽和後 `len()` 不再變大，舊邏輯第 51 個事件起不送
- [x] **P2 driver / service / API / UI / 規劃文件** — `services/io_module.py` `io_service.py` `network_health.py` `config_sync.py`、`api/routes/io.py`、`web/io_panel.html` 全部交付

### ❌ 待辦

- [ ] **P1 測試腳本實機跑** — `python3 scripts/test_modbus_io.py --diagnose` → 連線 → 跑馬燈 → 按鍵
- [ ] **config_sync 驗收** — 按 DI0 → DO2 白燈亮 → 拉到 JSON → 白燈滅；失敗則 DO0 紅燈閃 3 秒
- [ ] **tM-PD3R3 電源隔離規格確認**（共地策略：所有 GND 接共同接地排是否安全）
- [ ] **採購清單下單**：照明型 Φ22 momentary ×2（DI0 遠端下載 + DI1 Reset）、Φ22 白燈 ×1、12V PSU、RS-485 線材、終端電阻 ×2
- [ ] **AFE-R750 COM RS-485 模式 BIOS 設定值記錄**（pd3r3.py docstring 已記 `COM1_SW1 = 1011`，但 BIOS 設定值待補）
- [ ] **中央伺服器 (192.168.0.101:8080) 可達性 / API 規格驗證**

---

## 10. 文件版本

| 日期 | 變更 |
|------|------|
| 2026-04-25 | 初版（Claude + 使用者協作） |
| 2026-05-06 | 燈號重設計：DO0=🔴紅/故障、DO1=🟢綠/恆亮(正常)、DO2=⚪白/下載中；DI2 觸發 config sync |
| 2026-05-07 | DI/DO 編號 bug 修正、WS 事件 seq tracker、待辦狀態整理；P2 driver 全部交付 |
| 2026-05-07b | DI 重映射：**DI0 = 遠端下載**、**DI1 = Reset**、**DI2 = 保留**（原規劃 DI0 釋出，現改為 DI2 釋出）；面板列出全部 3 個 DI |
| 2026-05-07c | DI1 Reset 接通系統重啟動作：`sudo -n systemctl restart traffic-api.service`；2 秒延遲、防連按 |
| 2026-05-07d | DI1 Reset 重啟前 3 顆 DO 全滅（系統沒在跑的視覺提示）；重啟完成後 `_apply_do()` 自動恢復 |
| 2026-05-07e | **白燈邏輯反向**：滅=自動(遠端)、亮=手動 或 下載中（原本相反）；正常閒置(遠端) 改為只有綠燈亮 |
| 2026-05-07f | 下載中改用**白燈閃爍 ~2.5Hz**（手動恆亮、下載閃爍以樣式區分）；**sync 失敗不再閃紅燈**（避免跟 DO0「通訊故障」語意混淆，只記 log） |
