# 壅塞判定邏輯 — 現況與改善計畫

> 重點：壅塞判級的**誤報**（把「車多但在動」「一台大車擋鏡頭」判成嚴重壅塞）。
> 2026-08-31 在 87 實測：20 分鐘內喊了 **95 次「嚴重壅塞」，全部是誤報** ——
> 當下流量 20~31 輛/分，車一直在通過。
>
> 更新日期：2026-09-01 ｜ 主要檔案：`detection/congestion_detector.py`、`api/routes/congestion.py`

---

## 1. 現況：完整判定鏈

```
影格
 └─ YOLO 偵測 (conf 0.12，fallback 0.05)
     └─ ROI 遮罩過濾（只留中心落在車流區內的車）
         └─ VehicleTracker（IoU 0.15，max_age=window）
             └─ 固定物抑制  ← 地上箭頭/標線被誤判成車
                 └─ 三個訊號
                     ├─ occupancy      面積佔用率
                     ├─ count_density  車輛密度
                     └─ stopped_ratio  靜止車比例
                         └─ congestion_score
                             └─ 平滑（window=10）
                                 └─ 門檻階梯 → level
                                     └─ 四道封頂 → 最終 level
```

### 1.1 三個訊號怎麼算

| 訊號 | 算法 | 備註 |
|------|------|------|
| `occupancy` | 車輛 bbox **聯集 ∩ ROI ／ ROI 面積** | 原本用 bbox 面積**加總**，重疊與超出 ROI 的部分重複計入，近鏡頭大車 2 台就灌到 100%（實測 100%→45%） |
| `count_density` | `車輛數 ／ (ROI面積 ÷ (平均車面積 × 2.2))` | 平均車面積下限 2500 px，避免小框把容量估爆 |
| `stopped_ratio` | 靜止 track 數 ／ 總 track 數 | 靜止＝連續 `stop_min_frames`(3) 幀位移 < `stop_distance_px`(45px) |

### 1.2 合成分數

```python
occ_for_level = occupancy if stopped_ratio >= flowing_stopped_ratio else 0.0
queue_score   = min(1.0, count_density * (0.45 + 0.55 * stopped_ratio))   # 需 >= 2 台車
congestion_score = max(occ_for_level, queue_score)
smoothed = 最近 window(10) 筆的平均
```

`flowing_stopped_ratio`(0.3) 這道閘門的意思是：**車在流動時佔用率完全不納入判級**，只看排隊分數。
匝道近鏡頭幾台流動車就吃掉 ROI 40%，但無排隊 ≠ 壅塞。

### 1.3 排隊判定（與判級分開）

```python
queue_active = (停止車數 >= 2)
           and (occupancy >= queue_min_occupancy 0.05)
           and (queue_score >= 門檻 or occupancy >= medium_t or stopped_ratio >= 0.5)
```

排隊長度 = 各車的**等效車長**相加 + 車間距(`safety_gap_m` 1.5m)，依 bbox `y2` 由近而遠排序。

### 1.4 門檻階梯

| 等級 | 條件 | 預設門檻 |
|------|------|---------|
| 暢通 low | smoothed < 0.2 | — |
| 中等 medium | ≥ `medium_threshold` | 0.2 |
| 擁擠 high | ≥ `high_threshold` | 0.4 |
| 嚴重壅塞 critical | ≥ `critical_threshold` | 0.6 |

外加兩個既有規則：車輛數 < 2 一律 low（例外：單台停著且達 medium 仍可升級）；
連續 2 幀 0 車 → 立刻強制 low（不等平滑窗）。

### 1.5 四道封頂（2026-08-31 新增，只往下修不往上升）

| `level_capped_by` | 觸發條件 | 封到 | 分車種 |
|---|---|---|---|
| `no_queue` | `queue_active = False` | 擁擠 | 否 |
| `vehicle_count` | 車數 < 2 ／ < 3 | 中等 ／ 擁擠 | **是**（僅大貨車、大客車） |
| `free_flow` | `flow_vpm >= free_flow_vpm` | 中等 | 否 |
| — | 多條同時成立 | 取最嚴格者 | |

`flow_vpm`＝最近 `flow_window_sec`(60s) 內「出現過又消失的 track 數」換算成輛/分。
停著不走的車 track 一直在，不會被計入 —— 流量衡量的是「車走掉的速率」。

**整體層與車道（zone）層套用同一組封頂**，用各自的車輛數、大型車數、流量、排隊狀態。

---

## 2. 誤報來源與對應防呆

| # | 誤報樣態 | 根因 | 對策 | 狀態 |
|---|---------|------|------|------|
| 1 | 近鏡頭 2 台大車 → 佔用率 100% | bbox 面積加總 | 改成聯集 ∩ ROI | ✅ 已修 |
| 2 | 地上白色箭頭被判成車，佔用率長灌 | 低信心誤判且靜止 | 固定物抑制（記在「位置」不記在 track，因為誤判會閃爍換 id） | ✅ 已修 |
| 3 | 43% 佔用 / 0m 排隊卻判壅塞 | 車在流動也算佔用率 | `flowing_stopped_ratio` 閘門 | ✅ 已修 |
| 4 | 1.5% 佔用卻宣告排隊 11.5m | 排隊無佔用率下限 | `queue_min_occupancy` | ✅ 已修 |
| 5 | **一台大貨車 → 嚴重壅塞** | 佔用率是面積比，大車一台就過門檻 | `min_vehicles_high/critical`（限大型車） | ✅ 已修 |
| 6 | **車順暢通過但佔用率高 → 嚴重壅塞** | 判級完全不看流量 | `free_flow_vpm` | ✅ 已修 |
| 7 | **排隊 0m 卻嚴重壅塞** | 判級與排隊脫鉤 | `critical_requires_queue` | ✅ 已修 |
| 8 | **整體「中等」但車道「嚴重壅塞」** | 封頂只套整體層 | 車道層套同一組 | ✅ 已修 |
| 9 | 車輛 0 但畫面顯示佔用率 1% | 顯示用平滑值、車輛數用瞬時值 | 顯示改用 `raw_occupancy` | ✅ 已修 |

---

## 3. 尚未解決的問題

### 3.1 分析率過低造成 `stopped_ratio` 系統性虛高（**最重要**）

尖峰時 `analysis_fps` 只有 **0.55**（兩次取樣間隔 1.8 秒）。
靜止判定是「連續 3 幀位移 < 45px」，在 1.8 秒的間隔下，**紅燈前正常停等的車必然滿足**，
於是 `stopped_ratio` 長期是 1.0。實測那 95 筆誤報，停止比中位數全部是 **1.0**。

這讓 §1.2 的 `flowing_stopped_ratio` 閘門形同虛設 —— 它本來要擋的就是這件事。

> 目前是靠 §1.5 的封頂在下游補救，**根因沒有解決**。
> 分析率的瓶頸是 GPU 飽和（util 0.99），見 §5。

### 3.2 「佔用率」的定義與交通工程慣例不同

本系統的 occupancy 是**空間面積佔有率**（ROI 被車覆蓋的比例）。
交通工程的 occupancy 是**時間佔有率**（偵測區被車佔據的時間比例）。

兩者在自由車流下差很多，而且面積佔有率**受鏡頭透視嚴重影響** ——
同一台車在近端與遠端的面積可差 10 倍以上。這是 §2 第 1、5 項誤報的共同根源。

對外 API 兩種佔有率同名不同義（VD 51.0 vs 壅塞 55.0），已在
`即時車流查詢說明.md` 第八節註明，但**根本上沒有統一**。

### 3.3 門檻是在「沒有真壅塞」的資料上訂的

2026-08-31 兩輪各 20 分鐘取樣，**沒有觀察到任何一次真正的壅塞**。
`free_flow_vpm` 的四個值（cam2/3/4/5 = 20/20/25/12）取自各站平時流量的 p25，
是保守估計，但**沒有「塞住時流量長什麼樣」的對照資料**。

門檻設太低會壓掉真壅塞 —— 這是比誤報更嚴重的方向。

### 3.4 平滑窗與即時性的取捨未經檢討

`smoothing_window=10` 搭配 `analyze_interval_sec=1.0`，代表佔用率有約 10 秒的尾巴。
車走光之後平滑值還有 24.9%（實測）。顯示端已改用即時值，但**判級仍吃平滑值**，
所以「車走了但等級還沒降」的延遲仍在。

---

## 4. 怎麼證明有效（評測）

沒有這一節，上面所有調整都只是憑感覺。

### 4.1 已建立的量測

- `level_capped_by` 欄位：直接看判級被哪一條壓下來（`no_queue` / `vehicle_count` / `free_flow`）
- `flow_vpm`、`large_vehicle_count`：整體層與車道層都有
- 取樣腳本：每 5 秒打 `/api/congestion/{id}/status`，記整體 + 車道層，跑 20 分鐘

### 4.2 已有的基準數字（2026-08-31，87）

| 指標 | 改前（尖峰 20 分） | 改後（離峰 20 分） |
|---|---|---|
| `critical` 次數 | **95（全部誤報）** | **8（全部為真）** |
| 誤報當下流量 | 20~31 輛/分 | — |
| 改後 critical 的排隊長度 | — | 11.5~38 公尺 |
| 整體/車道判級落差 | 未量 | cam4、cam5 為 0 |

> ⚠️ **兩輪的車流條件不同**（尖峰流量中位 16~30，離峰 8~17），
> 所以「95 → 8」不能全記在修正上。**尚欠一次日間尖峰對照。**

### 4.3 還需要建立的

1. **尖峰對照組** —— 同時段、同流量條件下重收一輪
2. **真壅塞樣本** —— 錄一段真的塞住的影片，用來驗「不會漏報」
3. **人工標註基準** —— 抽樣若干分鐘，人眼標「這分鐘算不算壅塞」，算 Precision / Recall
4. **漏報監測** —— 目前只監測誤報。`critical_requires_queue` 等封頂有壓掉真壅塞的風險，需要反向檢查

---

## 5. 分階段

| 階段 | 內容 | 狀態 |
|---|---|---|
| P0 | 四道封頂（no_queue / vehicle_count / free_flow）＋ 車道層對齊 | ✅ 已上線 |
| P1 | 現場門檻校準（`free_flow_vpm` per-camera） | ✅ 已套用，待尖峰驗證 |
| P2 | **尖峰驗收** —— 日間同時段對照，含漏報檢查 | ⬜ 待做 |
| P3 | **提高分析率**（根治 §3.1）：`congestion` 共用 `detection` 的偵測結果，不另跑推論 | ⬜ 待做，預估省 18~25% GPU |
| P4 | LPR 降頻或 ROI 觸發 | ⬜ 待評估，佔 25~30% GPU |
| P5 | 檢討面積佔有率 vs 時間佔有率（§3.2） | ⬜ 待評估，影響對外 API 相容性 |

### P3 補充：為什麼這是根治

GPU 目前三個消費者：`detection` 48%、`lpr` 25~30%、`congestion` 19~26%。
壅塞偵測**自己跑一次 YOLO**，但同一台相機的 `detection` 執行緒已經算過同一張畫面。
共用之後分析率會直接上升，§3.1 的 `stopped_ratio` 虛高才會真正改善。

代價：兩者的 `conf_threshold` 不同（壅塞用 0.12 求高召回、detection 用較高值），
共用前要確認低信心框對兩邊都安全，或改成 detection 跑低 conf、下游各自過濾。

---

## 6. 實作落點

| 檔案 | 負責 |
|------|------|
| `detection/congestion_detector.py` | 核心演算法：訊號、分數、判級、封頂、流量、排隊 |
| `detection/congestion_detector.py` `_cap_level()` | 四道封頂（整體層與車道層共用） |
| `detection/congestion_detector.py` `_update_flow_vpm()` | 流量估計（track 消失計為通過） |
| `api/routes/congestion.py` | 參數預設值、Pydantic 驗證、per-camera 存取、樣本落庫 |
| `web/index.html`「壅塞參數」面板 | 門檻、封頂參數、`critical_requires_queue` 開關 |
| `api/routes/external.py` | 對外壅塞報表（`/congestion-report/*`） |

**per-camera 參數存在 DB 的 `detection_config.congestion`**，不在版控；
87 目前的值記在專案記憶 `project_87_congestion_overlay_tuning.md`。

---

## 7. 風險與備註

- **封頂只能往下修** —— 這是刻意的。流量會低估（見 §3.1），
  低估的流量看起來像壅塞，若允許往上升級就會製造新的誤報。
- **`critical_requires_queue` 是最強的一道**，它不分車種、直接封掉所有「無排隊的最高級」。
  如果現場出現「確實塞住但排隊估不出來」的情況，這條會造成漏報 —— 前端有開關可關。
- **`free_flow_vpm` 預設 0（不啟用）**，必須依現場實測填。
  沒有量過就開，等於用猜的門檻壓判級。
- **大型車判定含 `truck`**，不只 `heavy_truck`/`bus`：細分類沒跑或判不出來時類別會停在
  `truck`，前端顯示就是「大貨車」。細分類降頻後停在 `truck` 的比例會上升，漏掉它
  這條規則會大半時間失效。
- 相關文件：`README.md`（系統總覽）、`即時車流查詢說明.md`（對外 API）、
  `DB_ERMODEL.md`（`congestion_samples` / `congestion_report_aggs`）。
