# OPAC 動態號誌整合 — 現況紀錄

> 國8 東向 新市交流道（N8-E-9）匝道適應性號誌控制整合。
> 建立：2026-09-02（從中心端 k8s 實地挖掘 + 我方 API 實測）。
> 站點兩相：分相1 上匝道（儲車210m）、分相2 下匝道（儲車600m，主線保護優先）。

---

## 一、最重要的事實（推翻先前理解）

**動態號誌決策引擎不是我方要寫的 —— 它已經存在，叫 OPAC，是別人（TCIS）做好的。**

先前記憶寫「Jetson 為動態演算本體、決策引擎待建」是誤解。實際上：

- **決策演算法 = OPAC**（`opac-adaptive-control`），跑在中心端。
- **我方（Jetson 交通分析）只負責提供偵測資料**（排隊/流量），OPAC 主動來拉。
- 整條管道**早就接好了** —— OPAC 的設定檔就指向我方 API，金鑰也配好了。

---

## 二、中心端部署（10.42.38.35）

| 項目 | 內容 |
|---|---|
| 主機 | `10.42.38.35`，Ubuntu x86_64（**非 Jetson**），跑 k8s |
| SSH | `myuser` / `84778623`（OS 帳號；與網頁 Basic `admin`/`!QAZ2wsx#EDC` 不同） |
| k8s 服務 | `r34sig`（號誌管理後端）、`r34sig-redis`、`opac-adaptive-control`（決策引擎）、`ic-agent-service1`（號誌下發代理） |
| OPAC image | `asia-east1-docker.pkg.dev/c24002105003-cicd-tcis/tcis/opac-adaptive-control` |
| OPAC API | `/api/opac`，port 8090，NodePort 30080 |
| **OPAC 現況** | **1/1 Running（2026-09-02 11:2x 中心端開啟測試）**，dataMode ADAPTIVE |

---

## 三、整合合約（OPAC configmap `constant.yaml`）

### 3.1 OPAC 來拉我方（不是我方推）

```yaml
aicctv:
  base-url: "http://10.42.40.21:8000"   # 我方 Jetson
  api-key:  "tvd_hwacom_traffic_2026"
  poll-interval-ms: 5000                 # 每 5 秒拉一次
  http-timeout-ms:  2000
  stale-after-seconds: 30                # 失聯逾 30s → 固定綠燈降級模式
  recover-successes: 2                   # 降級後連 2 輪成功才恢復適應
  meters-per-vehicle: 7.0                # 公尺→輛換算
  detectors:
    - id: CCTV-N8-E-9-L-NE-1-SIG  direction: ENTRY_NE  type: UPSTREAM  metric: OUT_FLOW
    - id: CCTV-N8-E-9-L-NE-2-SIG  direction: ENTRY_NE  type: STOPLINE  metric: OUT_FLOW
        queue-source: MEASURED  lane-no: 1          # 上匝道停等,分相1約束
    - id: CCTV-N8-E-9-L-WN-1-SIG  direction: EXIT_WN  type: UPSTREAM  metric: OUT_FLOW
    - id: CCTV-N8-E-9-L-WN-2-SIG  direction: EXIT_WN  type: STOPLINE  metric: OUT_FLOW
        queue-source: MEASURED  lane-no: 1          # 下匝道停等,分相2約束
```

### 3.2 OPAC 演算法參數

Δt=5s、min-green 15s、max-green 100s、飽和流 1800 vph、
CL=2（clearance-discharge）、EL=3（換相損失，CL<EL 是切換成本模型前提）、
黃 8s、全紅 2s、degraded-green 40s（降級固定綠）、auto-start=false。

### 3.3 下發與發布

- **下發**：`icagent`（`ic-agent-service1:10000`），`enabled:true`（接實體控制器）。
  takeover-strategy = 路側手動 + 分相；phases: ENTRY_NE→分相1、EXIT_WN→分相2。
- **MQTT**：發布 `tcis/opac`（intersection N8-E-9），前端顯示與決策稽核用。

---

## 四、我方要提供的資料（OPAC 實際拉到的）

從 OPAC `aicctv-data.log`（8/28 實證），每偵測器：
`{id, status, total, in, out, queueM}`。`queueM` = 排隊公尺。

### 關鍵改善（2026-09-02 實測）

| 偵測器 | 8/28 OPAC log | 2026-09-02 我方 API |
|---|---|---|
| NE-1（上高速前） | queueM=null | **5.3 / 32.0** |
| NE-2（上匝道停等，分相1約束） | **queueM=null** | **4.6 / 24.5** ✅ |
| WN-1（高速下匝道） | 61.7 | 15.0 / 33.0 |
| WN-2（下匝道停等，分相2約束） | 24.6 | 20.5 / 44.5 |

- 8/28 時 **NE-2 排隊是 null**（分相1 沒資料，OPAC 只能退差分，configmap 註解說差分「長期漂移出假隊伍」）。
- **2026-09-02 NE-2 已有排隊值** —— 分相1 約束偵測器補上了。
  ⚠️ 尚待確認：OPAC 讀的 `queueM` 是否對應我方 `avg_queue_length_m`（欄位名不同），要 OPAC 開起來看它 log 才能 100% 確認。

---

## 五、開啟前的整備狀態（我方這端，2026-09-02）

| 項 | 狀態 |
|---|---|
| API 健康 | ✅ healthy |
| 分析率 | ✅ 1.5–1.9 fps |
| 四台排隊資料 | ✅ 全部有值（含 NE-2） |

**前置事件**：2026-09-02 上午我方 Jetson 曾因 LPR 崩潰循環（`5db7609` 已修）+ 過熱（tj 91°C、風扇曲線壞）導致分析率崩到 0.3。已處理：LPR 修復 + 風扇強制滿轉（停用曲線壞掉的 nvfancontrol）。分析率回到 1.5–1.9。詳見該次事件。

> 🛑 **OPAC 每 5 秒拉我方一次。我方分析率若再掉到過熱狀態，OPAC 會在 30s 內進降級固定綠燈模式。** 熱是這個整合的隱性風險 —— 散熱硬體待現場保養。

---

## 六、開啟 OPAC 的操作與監控點

### 開啟（尚未執行，待指示）

```bash
# 中心端 k8s
kubectl scale deploy opac-adaptive-control --replicas=1
```

⚠️ **開 OPAC = 透過 icagent 接管真實路口號誌**（`icagent.enabled:true`）。
`auto-start:false`，實際接管由接管程序以現場實際綠燈方向啟動。

### 開啟後要監控的中心端指標

1. OPAC 是否正常拉到我方四台排隊（會不會又 null）
2. OPAC decision log 是否進降級模式（`stale-after 30s` — 我方分析率掉就會觸發）
3. icagent 下發是否被拒/逾時（5F03 每秒佔通道，指令撞上會被拒，retry 2 次）
4. 我方 API 被 OPAC 拉的回應時間/成功率
5. MQTT `tcis/opac` 發布是否正常

---

## 六之一、開啟後首次監控（2026-09-02 中心端已開）

中心端（.35）已 `kubectl scale --replicas=1` 開啟 OPAC，測試中。首次監控：

**OPAC 健康**：`running:true, dataMode:ADAPTIVE, dataStaleSeconds:1, greenDirection:ENTRY_NE, controlState:GREEN` —— 適應模式運轉、正在控制真實號誌。

**拿到我方排隊**：NE-2 `queueM=25.5`、WN-2 `queueM=9.8`（8/28 為 null，現有值）。

**⚠️ 觀察到的異常訊號**：`queueM` 一陣一陣回 null（11:44:02 四台全 null → 11:44:07 只 NE-2 有值 → …）。對應我方分析率剛從過熱壓下、還在 1.5–1.9，偵測有空窗時排隊即回 null。目前 `dataStaleSeconds` 仍低（未達 30s 降級門檻），但需持續盯 null 頻率，過高會讓 OPAC 退差分或進降級。

**監控指標對應**：`queueM` null 頻率 ↔ 我方分析率穩定度（根因：散熱）。

---

## 七、待確認/待辦

- [ ] OPAC `queueM` ↔ 我方 `avg_queue_length_m`/`max_queue_length_m` 欄位對應（開起來驗）
- [ ] OPAC 8/28 為何被停到 0/0（查 deploy 歷史/事件）
- [ ] 我方散熱硬體保養（風扇曲線壞、tj 91°C 壓不下）
- [ ] LPR 外層 watchdog 逾時是否需放寬（過熱時偶爾誤重啟）

---

## 附：相關檔案

- 我方對外 API：`api/routes/external.py`（OPAC 拉的 `/api/v1/external/realtime` 等）
- 我方排隊來源：`detection/congestion_detector.py`（`estimated_queue_length_m`）
- 官方時制基準：`detection/signal_timing_lookup.py` + `config/system/ramp_timing_baseline.json`
- 我方 TC3 抄錄/監看：`api/routes/signal_tc3.py`
- 號誌管理系統手冊：`docs/號誌管理系統_操作手冊.md`
- 缺口分析（部分已被本紀錄修正）：`docs/號誌整合_功能缺口分析.md`
