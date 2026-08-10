# 終端控制器識別碼設定功能驗證暨業主確認文件

> 文件版本：V1.0  
> 文件日期：2026-08-10  
> 驗證項目：規範 (C) 識別碼設定  
> 驗證目的：確認終端控制器具備「個別通訊識別碼（設備編號）」之軟體設定能力，作為 Unit Test 與業主功能確認依據。

---

## 1. 規範依據

### 1.1 規範條文

> **(C) 識別碼設定**  
> 每一個終端控制器須具有個別之通訊識別碼（設備編號）設定，  
> 指撥開關（DIP SWITCH）16 位元或以軟體控制。

### 1.2 本系統符合方式

本系統採用規範允許之 **「軟體控制」** 方式設定設備編號，不採用實體 DIP SWITCH。

設備編號可由系統設定檔 `.env` 之 `DEVICE_ID` 指定；設定完成並重新啟動服務後，系統即以該值作為本終端控制器之通訊識別碼。

因此，本項驗證重點為：

1. 終端控制器可透過軟體設定設備編號。
2. 設定值可由系統 API 查詢並正確回傳。
3. 設定值優先於系統自動產生之預設值。
4. 專案部署時，每一台終端控制器應配置不同的 `DEVICE_ID`，以符合「個別之通訊識別碼」要求。

---

## 2. 識別碼設定機制

### 2.1 識別碼來源與優先順序

| 優先序 | 識別碼來源 | 用途 | API `source` |
|---|---|---|---|
| 1 | `.env` 之 `DEVICE_ID` | 正式部署／業主指定設備編號 | `configured` |
| 2 | 板載實體網卡 MAC 衍生值 | 未設定 `DEVICE_ID` 時之出廠／初始化預設值 | `mac` |

正式部署時，以 **`DEVICE_ID` 軟體設定值** 作為設備識別碼之主要依據。

### 2.2 預設識別碼

若未設定 `DEVICE_ID`，系統會由板載實體網卡 MAC 產生預設識別碼：

- 現場實體網卡 MAC：`74:fe:48:be:6b:2e`
- 取 MAC 末 2 bytes：`6B2E`
- 預設設備編號：`TVD-6B2E`

> **說明：** MAC 衍生識別碼屬於未設定時的備援／初始化機制。最終工程部署仍應由業主或系統整合端為每一台設備配置明確且不重複的 `DEVICE_ID`，避免僅以 MAC 末 16 位元作為跨設備唯一性的唯一依據。

---

## 3. 實機驗證環境

| 項目 | 實機資訊 |
|---|---|
| 終端控制器平台 | Jetson AGX Orin |
| 現場使用網路介面 | `enP5p5s0` |
| NetworkManager Connection | `field-net` |
| 現場 IP | `10.42.38.35/20` |
| 板載實體 MAC 範圍 | `74:fe:48:be:6b:2e` ~ `74:fe:48:be:6b:31` |
| 設備識別碼查詢 API | `GET /api/system/device-id` |
| **本案配置設備編號** | **`R34_動態號誌VD`** |

---

## 4. Unit Test 驗證項目

### UT-ID-001：未指定設備編號時可產生預設識別碼

**測試目的**  
確認系統於未設定 `DEVICE_ID` 時，仍可自動產生可供識別之預設設備編號。

**測試步驟**

1. 確認 `.env` 未指定 `DEVICE_ID`。
2. 啟動／重新啟動終端控制器服務。
3. 執行：

```bash
curl -s localhost:8000/api/system/device-id
```

**實機結果**

```json
{
  "device_id": "TVD-6B2E",
  "source": "mac",
  "source_label": "板載網卡 MAC 自動產生",
  "mac_based_default": "TVD-6B2E",
  "physical_macs": [
    "74:fe:48:be:6b:2e",
    "74:fe:48:be:6b:2f",
    "74:fe:48:be:6b:30",
    "74:fe:48:be:6b:31"
  ]
}
```

**判定：PASS**

---

### UT-ID-002：可透過軟體指定設備編號

**測試目的**  
確認設備編號可透過軟體設定，符合規範「DIP SWITCH 16 位元或以軟體控制」之要求。

**測試步驟**

1. 於 `.env` 設定本案配置之設備編號：

```dotenv
DEVICE_ID=R34_動態號誌VD
```

2. 依系統標準程序重新啟動服務。
3. 執行：

```bash
curl -s localhost:8000/api/system/device-id
```

**實機結果**

```json
{
  "device_id": "R34_動態號誌VD",
  "source": "configured",
  "source_label": "軟體設定（.env DEVICE_ID）",
  "mac_based_default": "TVD-6B2E",
  "physical_macs": [
    "74:fe:48:be:6b:2e",
    "74:fe:48:be:6b:2f",
    "74:fe:48:be:6b:30",
    "74:fe:48:be:6b:31"
  ]
}
```

**驗證結果**

- `device_id` 已由預設值 `TVD-6B2E` 改為本案指定值 `R34_動態號誌VD`。
- `source` 已由 `mac` 改為 `configured`。
- 系統可明確辨識設備編號之來源。
- 軟體設定值優先於 MAC 預設值。

**判定：PASS**

> **本測試為規範 (C)「以軟體控制」之核心達標測試。**

---

### UT-ID-003：對外 API 應帶出相同設備編號

**測試目的**  
確認設備編號不只存在於設定層，而能作為對外通訊資料之終端識別資訊。

**預期結果**

對外 API：

```text
/api/v1/external/*
```

每筆回應之：

```json
meta.device_id
```

應與目前設定之 `DEVICE_ID` 完全一致：

```json
{
  "meta": {
    "device_id": "R34_動態號誌VD"
  }
}
```

**判定方式**

- `meta.device_id = DEVICE_ID`：PASS
- 不一致或未輸出：FAIL

**實機結果**（2026-08-10 14:25，現場主機，已配置正式設備編號）

```bash
curl -s -H "X-API-Key: ****" "localhost:8000/api/v1/external/realtime?mode=minute"
```

```json
{
  "meta": {
    "request_time": "2026-08-10T14:25:56.134350+08:00",
    "api_version": "1.0",
    "device_id": "R34_動態號誌VD",
    "format": "json"
  }
}
```

同一時間 `GET /api/system/device-id` 回傳 `R34_動態號誌VD`，兩者一致。

**判定：PASS**

> 對外 API 之 `meta.device_id` 與系統設定值取自同一來源，非另行寫死之常數。
> 先前於未設定 `DEVICE_ID` 之預設狀態下抽驗時回傳 `TVD-6B2E`，
> 設定後即同步變更為 `R34_動態號誌VD`，可證明兩者連動。

---

### UT-ID-004：不同終端控制器可配置不同設備編號

**測試目的**  
確認「每一個終端控制器須具有個別之通訊識別碼」可在實際部署中落實。

**建議測試方式**

| 終端控制器 | `DEVICE_ID` | 預期 API 回傳 |
|---|---|---|
| Controller-01（本案） | `R34_動態號誌VD` | `R34_動態號誌VD` |
| Controller-02 | `VD-0302` | `VD-0302` |
| Controller-03 | `VD-0303` | `VD-0303` |

**驗收標準**

- 每一台終端控制器均可獨立設定 `DEVICE_ID`。
- 各設備之 `DEVICE_ID` 不重複。
- 查詢 API 與對外 API 均回傳該設備自身識別碼。

> 本項屬於「部署／系統驗收」層級；單機 Unit Test 可先驗證設定能力，多機個別性則建議搭配實際設備編號清冊確認。

---

## 5. 實機證據

![終端控制器設備編號設定實機畫面](screenshots/device-id-mac.png)

### 截圖內容說明

1. `ip -br link`：確認板載網卡及 MAC 位址。
2. `nmcli -f DEVICE,TYPE,STATE,CONNECTION device status`：確認現場使用之網路介面與連線狀態。
3. `GET /api/system/device-id`（未指定 `DEVICE_ID`）：回傳 `TVD-6B2E`，`source=mac`。
4. 設定 `DEVICE_ID=R34_動態號誌VD` 後：回傳 `R34_動態號誌VD`，`source=configured`。
5. 對外 API `/api/v1/external/realtime` 之 `meta.device_id` 同步帶出 `R34_動態號誌VD`。

由上述實機結果可確認：

- 系統具備設備編號自動產生機制。
- 系統具備設備編號軟體設定機制。
- 軟體指定值可正確覆寫預設值。
- API 可回報目前實際採用之設備編號及其來源。

---

## 6. 規範符合性對照

| 規範要求 | 系統實作 | 驗證方式 | 結果 |
|---|---|---|---|
| 每一終端控制器具有通訊識別碼 | 系統提供 `DEVICE_ID` | `GET /api/system/device-id` | PASS |
| 可使用 DIP SWITCH 16 位元**或軟體控制** | 本系統採 `.env DEVICE_ID` 軟體設定 | UT-ID-002 | PASS |
| 設備編號可由人員指定 | 修改 `DEVICE_ID` 後重新啟動服務 | `TVD-6B2E` → `R34_動態號誌VD` | PASS |
| 設定值可被系統識別 | API 回傳 `source=configured` | API 查詢 | PASS |
| 各終端設備之編號應個別化 | 每台設備配置不同 `DEVICE_ID` | 設備編號清冊／多機抽驗 | 部署驗收確認 |
| 對外通訊攜帶設備編號 | `/api/v1/external/*` 之 `meta.device_id` | UT-ID-003 對外 API 抽驗 | PASS |

---

## 7. 驗證結論

依據規範 (C)「指撥開關（DIP SWITCH）16 位元**或以軟體控制**」之要求，本系統採 **軟體控制設備編號** 方式實作。

實機測試已證明：

- 未設定時，系統可由板載實體網卡產生預設識別碼 `TVD-6B2E`。
- 設定 `.env` 之 `DEVICE_ID=R34_動態號誌VD` 後，設備編號可成功改為 `R34_動態號誌VD`。
- API 明確回傳 `source=configured`，證明目前設備編號為軟體設定值。

### Unit Test 判定

**規範 (C)「設備編號可由軟體控制設定」功能：PASS。**

### 最終部署要求

為完整符合「每一個終端控制器須具有個別之通訊識別碼」之要求，正式上線前應建立設備編號清冊，並為每台終端控制器配置不重複之 `DEVICE_ID`。MAC 衍生值僅作為未設定時之預設值，不建議作為專案層級唯一編號管理的唯一依據。

---

## 8. 業主確認欄

| 確認項目 | 確認結果 |
|---|---|
| 終端控制器具備軟體設定設備編號功能 | □ 確認　□ 不確認 |
| 設備編號可由 API 正確查詢 | □ 確認　□ 不確認 |
| 軟體設定值可覆寫預設識別碼 | □ 確認　□ 不確認 |
| 正式部署將依設備編號清冊配置個別 `DEVICE_ID` | □ 確認　□ 不確認 |
| 對外 API `meta.device_id` 現場抽驗 | □ 通過　□ 待補驗 |

**業主代表：** ____________________  
**承商代表：** ____________________  
**確認日期：** ______ 年 ______ 月 ______ 日

---

## 附錄 A：現場查驗指令

```bash
# 1. 查詢實體／虛擬網路介面與 MAC
ip -br link

# 2. 查詢 NetworkManager 網卡狀態
nmcli -f DEVICE,TYPE,STATE,CONNECTION device status

# 3. 查詢目前設備編號
curl -s localhost:8000/api/system/device-id
```

正式指定設備編號（本案配置值）：

```dotenv
DEVICE_ID=R34_動態號誌VD
```

重新啟動服務後，再次執行：

```bash
curl -s localhost:8000/api/system/device-id
```

預期 `device_id` 為 `R34_動態號誌VD`，且 `source` 為 `configured`。

---

## 附錄 B：預設識別碼之 MAC 取用規則

未設定 `DEVICE_ID` 時，系統僅採用**板載實體網卡**之永久 MAC，並排除「本地管理位址」
（MAC 第一個 byte 之 bit 1 = 1，表示該位址由軟體自行產生、非原廠燒錄）：

| 介面 | MAC | 採用 | 說明 |
|---|---|---|---|
| `enP5p3s0` ~ `enP5p6s0` | `74:fe:48:be:6b:2e` ~ `:31` | ✅ | 板載實體網卡，永久位址 |
| `usb0` / `usb1` | `f2:12:58:...` | ❌ | USB gadget，每次開機隨機產生 |
| `docker0` | `52:49:ca:...` | ❌ | 虛擬橋接介面 |
| `l4tbr0` | `3e:c7:e7:...` | ❌ | 虛擬橋接介面 |
| `tailscale0` / `lo` | 無 | ❌ | 無實體位址 |

於合格之 MAC 中取最小值，因此預設識別碼與「哪一個網孔有接線、接於哪一孔」無關，
拔插網線或變更接線位置均不會造成識別碼改變。

主機板送修更換時 MAC 會隨之變動，屆時將原識別碼填入 `.env` 之 `DEVICE_ID` 即可維持一致。
