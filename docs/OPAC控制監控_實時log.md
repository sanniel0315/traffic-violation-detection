# OPAC 控制監控 — 實時 log

> 中心端 OPAC 適應性控制的持續監控紀錄，供我方控制比對。
> **存於本 repo（開發機），現場不留檔。** 資料經 Jetson 中轉自 `.35` 拉取。
> 監控起始：2026-09-02。

## 決策 log 欄位說明

`green`=綠燈方向｜`greenElapsed`=已亮綠秒｜`action`=KEEP/SWITCH｜
`swlGreen/Red`=綠/紅側排隊(來自我方 queueM÷7.0)｜`fa/sa`=第一/第二階段到達｜
`pn1/pn2`=保持vs切換的效益指標(OPAC 內算)｜`el=3 cl=2`=切換成本模型｜
`forcedByMaxGreen`=是否撞 max-green 100s 上限｜`mode`=ADAPTIVE/DEGRADED。

**判讀**：`SWITCH` 發生在 `pn1 > pn2`（切換效益較高）。
`DEGRADED` = 失聯逾 30s，兩相走固定綠 40s 輪替。

---

## 監控事件紀錄（只記異常/特殊控制行為 + 定期快照）

### 2026-09-02 起始快照

OPAC 狀態：`running, ADAPTIVE, dataStale 2s` —— 正常運轉、正在控制真實號誌。

決策樣本（可見適應性行為）：
```
03:49:00 EXIT_WN 綠25s swlG=2 pn1=0 pn2=1 → KEEP
03:49:15 EXIT_WN 綠40s swlG=5 pn2=3       → KEEP   (綠燈側排隊增,續綠)
03:49:20 EXIT_WN 綠45s swlR=1 saR=1 pn1=2 → SWITCH (對向累積,切換) ★撐到45s
03:49:45 ENTRY_NE綠15s pn1=2 pn2=1        → SWITCH
03:50:10 EXIT_WN 綠15s pn1=3 pn2=0        → SWITCH
03:50:35 ENTRY_NE綠15s pn1=2 pn2=0        → SWITCH
```

**特別行為**：03:49 EXIT_WN 綠燈撐到 45s（綠燈側 swl 持續有車、對向無累積時，OPAC 不切、讓下匝道continue 消化）—— 符合主線保護傾向（分相2=下匝道優先）。守 min-green 15s、未撞 max-green。

**我方對照**：此時我方分析率 1.5–1.9、四台排隊有值，OPAC dataMode 維持 ADAPTIVE（未降級）。

---

<!-- 後續事件往下追加：進降級/恢復、撞 max-green、下發被拒、我方分析率過低致 null、綠燈方向長時間不變等 -->

### 2026-09-02 12:02 ⚠ 異常:icagent 下發失敗(5F1C, errorCode 2)

icagent 對控制器送 5F1C(分相步階秒數)命令,連續兩次 Wait ACK/NAK timeout →
resend 到上限(2次)→ 控制器回 `commandFailRsp cmdId f81, commandId 24348,
errorCode 2`。該次配時下發**失敗**。
```
12:02:03 Wait ACK/NAK timeout, resend
12:02:06 Wait ACK/NAK timeout, resend
12:02:09 Send request message failed, resend times: 2
12:02:10 commandFailRsp 5F1C errorCode=2
```
研判:5F03 每秒佔用通訊通道 ~0.1s,指令撞上會被拒(config 已知,retry-times=2)。
此次退避重試仍撞滿 2 次 → 耗盡重試。屬間歇性通道競爭,非持續故障。
影響:單筆配時未下達,下一決策週期(5s後)會重下。需觀察是否頻繁。

同時觀察:
- OPAC 仍 ADAPTIVE(未降級)。controlState 當下 ALL_RED(相位間全紅,正常)。
- dataStaleSeconds=11(較起始 1~2 升高,趨向 30 降級門檻但未達)。
- 我方分析率掉到 ~1.0(起始 1.5~1.9),tj 91°C —— 熱又在爬,是 stale 升高主因。
- 決策仍正常:04:02:00 swlG faGreen=4 pn2=4 → KEEP(綠燈側到達多,續綠)。

### 演算法邏輯反推(從 ~25 筆 decision.log 樣本)

**非任意調整,是確定性規則:**

1. 切換規則(25 筆無例外):`SWITCH ⟺ pn1 > pn2`,否則(含平手)KEEP。
2. pn1(切換壓力)公式完全對出、12 筆 100% 命中:
   `pn1 = swlRed + faRed + saRed`(紅燈側排隊+第一/第二階段到達)。
   → 紅燈側累積越多等待車,切換效益越高。
3. pn2(保持壓力)= 綠燈側可消化需求,方向明確(綠側 swl/fa/sa 越多越續綠),
   但確切係數未鎖定 —— 隨 collectionStage(FIRST_STAGE_FA/SECOND_STAGE_SA)
   與清空門檻 cl=2 變化,單純加總對不齊。

**整體語意**:最小延滯相位切換 —— 紅燈側累積需求 > 綠燈側可消化需求就切,
cl=2/el=3 切換成本模型防抖動。解釋了 EXIT_WN 綠燈撐 45s(綠側持續有車消化、
紅側無累積 → pn1 長期 < pn2)。

待辦:多蒐不同 collectionStage 樣本,鎖定 pn2 確切公式 → 演算法完全白盒。

### 2026-09-02 12:08 敏感期異常已解除 + pn2 反推進展

**12:02 的異常自行解除**:icagent 下發失敗/5min 3→0、dataStale 11→2、
分析率回 1.45、dataMode 續 ADAPTIVE。確認是短暫通道競爭尖峰,非持續問題。
tj 91.5°C 熱牆仍在但未再惡化。

**pn2 是「有狀態」量(關鍵發現)**:
```
04:06:40 swlG=2 faG=0 saG=0 → pn2=1
04:07:30 swlG=2 faG=0 saG=0 → pn2=0   ← 綠側瞬時輸入相同,pn2 不同
```
→ pn2 不是單行算出,是綠燈相位期間的殘餘需求(隨放行以飽和流遞減)。
所以無法從單行 log 讀出確切公式,需放行歷史 —— 這是本質不是查不到。

**演算法完整語意**:
- pn1 = 紅燈側當下累積等待需求(切換效益) = swlR+faR+saR [公式確定]
- pn2 = 綠燈側殘餘可消化需求(保持效益),隨綠燈放行遞減 [有狀態]
- 規則:pn1 > pn2 → 切相位。最小延滯策略,cl/el 防抖動。
pn1 公式本輪 9 筆再次 100% 命中。

### 2026-09-02 12:13 fps 讀數雜訊(非真異常)

analysis_fps 讀到 0.57~0.70(較 12:08 的 1.45 低),初判熱惡化,但深查:
- LPR 近3分鐘重啟 0 次(無 thrash)
- GPU infer/s=19.1、hold 50ms(健康,遠優於過熱時 2~4/s、280ms)
- 風扇 215 滿轉、load 12.3、tj 91.8°C
GPU 吞吐 19/s 與「0.6 fps/台」矛盾 → 判定 analysis_fps 為瞬時雜訊/滯後平均,
非真的掉。**OPAC 不受影響**(dataStale 2、ADAPTIVE、下發失敗 0)。
結論:非異常。tj 91.8°C 熱牆仍在但 GPU 跑得動。續盯。

### 2026-09-02 12:19 分析率真降(非雜訊)—— 熱受限常態

用上輪判準交叉驗證:GPU infer/s 也從 19.1 掉到 4.9 → 分析率 0.31~0.34 是真降,
非雜訊。上輪 19/s 是突發峰值,這台熱牆下常態=GPU 4~5/s、分析率 0.3~0.4。
- OPAC 仍撐住:dataStale 3、ADAPTIVE、下發失敗 1/5min
- ENTRY_NE swl=6(真有排隊),分析率 0.3 → 排隊量測約 3s 更新一次,勉強夠
- tj 91.75°C,風扇滿轉壓不下
判斷:非紅色警報(OPAC 未降級、路口正常),但是真實限制 —— 散熱不改善,
我方分析率卡 0.3~0.4,OPAC 排隊資料解析度持續受限、處於邊際。
根治:現場散熱保養(風扇曲線壞+積熱)。

### 2026-09-02 12:29 措施:LPR YOLO 推論節流(LPR_INFER_SKIP=2)—— 效果顯著

熱受限期主動把 GPU 讓給排隊偵測(OPAC 要的),對 LPR(與 OPAC 無關)降頻。
部署 d515a7b + 87 設 LPR_INFER_SKIP=2(LPR 每2幀跑1次 YOLO)。

| 指標 | 節流前 | 節流後 |
|---|---|---|
| LPR GPU 佔比 | 39% | 30% |
| GPU util | 0.99(飽和) | 0.73(有餘裕) |
| 分析率 | 0.3~0.8 | 1.18~3.88 |
| tj | 92.25°C | 89.9°C(降到90以下) |
| OPAC dataStale | 3~4 | 1 |

三重效益:分析率翻倍+GPU 解飽和+LPR 少跑 YOLO 連帶降溫。
OPAC dataStale 降到 1、queueM 即時。LPR 仍運作(車牌辨識降到 5fps,足夠)。
🛑 這是熱受限期的權宜措施,散熱硬體修好後可移除 zz-lpr-throttle.conf(設回 SKIP=1)。
根治仍是散熱保養(使用者將處理)。

### 2026-09-02 12:58 穩定(監控放寬10分鐘)
OPAC ADAPTIVE/dataStale1/下發失敗1、queueM有值(12:58單筆四台None=分鐘邊界暫態,
最近幾筆該偵測器排隊 4.2→9.0 遞增正常)、我方fps 2.25(LPR節流後穩)、tj92.4°C。
OPAC 跨近1小時多次震盪持續穩定,dataStale從未逼近30。監控放寬至10分鐘。

### 演算法邏輯 — 完整確認(41 筆連續樣本, 2026-09-02 13:0x)

**① 切換規則(41筆零例外,硬規則)**
```
SWITCH ⟺ pn1 > pn2    ;  pn1 ≤ pn2 → KEEP(含平手)
```

**② 修正先前說法:pn1 不是 swlR+faR+saR 的單純加總**
先前 12 筆剛好吻合,本批出現反例:
```
05:00:10 swlR=6 faR=3 saR=0 → 加總9 但 pn1=10  ✗(多1)
05:00:15 swlR=7 faR=0 saR=1 → 加總8    pn1=8   ✓
```
→ pn1 主體是紅燈側累積需求,但帶跨週期殘量,非純瞬時加總。

**③ pn2 = 綠燈側尚未消化完的放行需求(決定性證據)**
```
04:59:15 swlG=3 faG=2 → pn2=3
04:59:20 swlG=4 faG=0 → pn2=3   ← swlG 增加,pn2 不變
04:59:25 swlG=4 faG=1 → pn2=3   ← 又到達,仍不變
04:59:30 swlG=4 faG=0 → pn2=2   ← 遞減
05:04:00 saG=5        → pn2=7   ← 來一批車,跳高
05:04:10 saG=0        → pn2=2   ← 消化,回落
05:04:20 swlG=0       → pn2=0 → SWITCH(消化完就切)
```
pn2 隨綠燈以飽和流遞減,與瞬時 swlG 無直接對應 → 是殘量不是瞬時量。

**④ 完整語意 = 標準 OPAC 最小延滯**
- pn1 = 紅燈側累積等待成本(含殘量)
- pn2 = 綠燈側未消化放行需求(隨綠燈遞減)
- 「繼續紅燈的代價 > 繼續綠燈的效益」就切相位
- cl=2/el=3 = 換相損失懲罰,防抖動

**⑤ 綠燈長度完全由車流決定**:實測 15s(消化完即切) ~ 55s(持續有車續綠),
全落在 min15/max100 內,41 筆**無一次 forcedByMaxGreen** → 車流未逼上限,
演算法有充分裕度。

### 2026-09-02 13:10 ⚠ 真異常:我方 API 每分鐘偶發卡頓 >2s,觸發 OPAC 輪詢 timeout

**現象**:OPAC 拉到的 queueM 連續 2 筆四台全 null(非分鐘邊界暫態)。

**排除**:我方 API 直查有值(NE-1 12.5/NE-2 14.8/WN-1 19.7/WN-2 11.2)、
壅塞服務四台 running 且有 result → 不是我方沒資料。

**根因**:OPAC pod log 顯示
```
13:06:19 WARN aicctv 輪詢失敗:Read timed out (GET /api/v1/external/realtime)
13:07:19 WARN aicctv 輪詢失敗:Read timed out
```
我方 API 平常回應 0.026~0.16s(從 .35 拉僅 0.047s),卻偶發 >2s 觸發 OPAC
`http-timeout-ms: 2000`。**每分鐘規律各一次** → 指向週期性阻塞(疑報表聚合/
DB 寫入/統計 flush 卡住 API 執行緒)。該次輪詢失敗 → queueM 記 null。

**影響**:目前 OPAC 仍 ADAPTIVE(連續失敗未達 stale-after 30s),但每分鐘掉一筆
資料。若阻塞加長或頻率上升,會推向 DEGRADED。

**另**:LPR 近5分鐘又 thrash 4 次(節流未完全止住),fps 回落 0.55。

**待辦**:①找出每分鐘卡住 API 的週期性工作 ②評估是否加大 OPAC http-timeout
或我方把該工作移出請求執行緒。

### 2026-09-02 13:33 API timeout 異常已自行解除

近10分鐘 OPAC 輪詢失敗 **0 次**(13:06~13:07 曾每分鐘一次)、queueM 恢復有值
(WN-1 16.1/WN-2 27.4/NE-2 3.6;NE-1 None 是該偵測器當下真無排隊)、
OPAC ADAPTIVE/dataStale1/下發失敗0、我方 fps 回升 1.36~1.58、tj 92.7°C。

**修正 13:10 的研判**:先前推測「每分鐘規律的週期性工作阻塞」不成立 ——
若是固定週期工作不會自行消失。實際應為當時的短暫負載尖峰所致
(該時段 LPR thrash 4 次),與 LPR 不穩相關,非獨立的週期性阻塞。
→ 待辦「找每分鐘卡住 API 的週期性工作」取消;改為續盯 LPR thrash 與
  API timeout 是否同時再現(兩者相關性已有跡象)。

## 控制權接管與恢復機制(2026-09-02 從 OPAC 啟動 log 實證)

### ① 接管(啟動時) —— 不跳燈
```
開始接管:5F10 切時相控制
5F10 切時相控制(effectTime=5分) → success
接管完成:控制器目前 EXIT_WN 綠燈(分相2 步階1 剩餘30s),以此方向啟動決策
```
先讀控制器**當下實際綠燈方向**,接著那個方向繼續決策 → 接管瞬間不會跳燈。

### ② 恢復 = 死人開關(dead-man switch)★最重要
```
5F10 effectTime = 5 分鐘,每 60 秒 renew 一次(11:28:50、11:29:50、11:30:50…)
```
接管指令**只有 5 分鐘有效**,靠每分鐘續約維持。因此:
> **OPAC 掛掉/網路斷/pod 被砍 → 沒人續約 → 5 分鐘後控制器自動回原定時時制。**

**不需要任何人做任何事,號誌會自己回去。** 這是 fail-safe,也是「最後恢復」的答案。
手動要停:中心端 `kubectl scale deploy opac-adaptive-control --replicas=0`,
5 分鐘內控制器自動歸還定時控制。

### ③ 資料失聯降級(撐住,非恢復)
`資料新鮮度看門狗:失聯 ≥30s 進入降級、連續成功 2 輪恢復`
- 我方資料斷 → 不亂控,改兩方向固定綠 40s 輪替(degraded-green)
- 我方恢復 → 連 2 輪成功自動回 ADAPTIVE

### ④ 綠燈上限保護
`keepGreen=100s` — 即使決策卡住,單方向綠燈不超過 100 秒。

### 三層安全網總結
| 失效情境 | 保護機制 | 結果 |
|---|---|---|
| 我方資料斷 | 新鮮度看門狗 30s | 降級固定綠 40s 輪替 |
| OPAC 整個掛掉 | 5F10 effectTime 5分不續約 | 自動歸還定時時制 |
| 決策卡住 | keepGreen 100s | 綠燈不會無限長 |

### 2026-09-02 13:45 🔴 重大:icagent↔控制器通訊逾時,配時下發全失敗

**現象**
```
icagent 失敗: 175 次/5分鐘(基準 0~1)
icagent log:  Wait ACK/NAK timeout 持續, tcCommStatus:false 出現 16 次(先前恆 true)
OPAC 下發:    5F1C 分相2 → packageStatus=fail 連續 3 次,重試耗盡
              「icAgent 指令回覆失敗(第1/3、2/3、3/3次)」
```
**判定:非我方問題** —— 是 icagent 與號誌控制器之間的通訊逾時(控制器沒回 ACK)。
我方 API 僅 1 次輪詢失敗,可忽略。

**路口安全性:安全**。三層安全網作用中:
1. 下發失敗 → 控制器維持既有配時,不跳燈
2. 5F10 死人開關 → 若 5F10 續約也送不到,5 分鐘後自動回定時時制
3. OPAC 本身仍活(ADAPTIVE、持續決策),只是送不出去

**我方次要狀況**:LPR thrash 10 次/5分鐘(節流未完全止住)、fps 1.07、
queueM 連2筆全 null。但因下發已送不到控制器,我方資料暫不影響控制。

**建議**:中心端檢查 icagent→控制器實體通訊(RS-232／MiiNePort 10.42.40.222／
控制器是否被其他連線佔用)。tcCommStatus:false + ACK timeout 指向實體層或
連線競爭(MaxConnect=1,中心搶線?),非軟體邏輯。

### 2026-09-02 13:50 🔴 真相:控制權被反覆奪回,OPAC 30分鐘內重新接管 3 次

**不是單純通訊故障,是控制權爭奪。** OPAC 自己的 ERROR 訊息:
```
13:47:54 ERROR 控制器已離開時相控制模式(斷線逾時或現場介入):
         ControlStrategy[fixTime=1, dynamic=0, roadSideManual=1, phase=0, ...]
         停止決策並重新接管
13:47:54 開始接管:5F10 → success
13:48:06 接管完成:EXIT_WN 綠燈(分相2 剩餘34s)
13:48:50 ERROR 又離開時相控制模式 ← 不到1分鐘再掉
13:48:50 重新接管 → success → 接管完成:ENTRY_NE 綠燈
近30分鐘「開始接管」3 次
```

**關鍵**:控制器 ControlStrategy 被打回 `fixTime=1, roadSideManual=1, phase=0`
—— **精確等於 OPAC 接管前的原始設定**(對照 SIG-01 現況 fixTime/roadSideManual)。
OPAC 接管成功後不到一分鐘又被打回。

**兩種可能(OPAC 訊息已列出)**:
1. 斷線逾時 —— 5F10 五分鐘 effectTime 沒續上(但續約 log 有 success,較不像)
2. **現場介入** —— 有人/系統把策略改回定時+路側手動(roadSideManual=現場控制箱)
傾向 2:打回的值精確等於原始設定,且規律重複。
疑點:現場有人操作控制箱?或 r34sig 號誌管理系統有排程/自動機制寫回策略?

**伴隨數據**:icagent 失敗 156/5min(略降)、下發成功率僅 12%(success 8/fail 58)、
tcCommStatus false 12/true 107。下發大量失敗與「控制權不在 OPAC 手上」一致 ——
不在時相控制模式時,5F1C 配時命令自然被拒。

**我方**:fps 1.58~1.74 回升、LPR thrash 降到 4。我方非本次瓶頸。

**路口安全**:控制器落在定時+路側手動 = 正常定時號誌運作,安全。

### 2026-09-02 14:08 ✅中心端通訊恢復 / ⚠️我方 LPR watchdog 誤判 thrash

**中心端:完全恢復**
| 指標 | 異常時 | 現在 |
|---|---|---|
| icagent 失敗/5min | 175 | 6 |
| tcCommStatus | false12/true107 | **true 159, false 0** |
| OPAC 下發成功率 | 12% | **78%**(14success/4fail) |
| 重新接管/10min | 3 | **0** |
| 控制器策略 | 被打回 fixTime | phase=true(OPAC 控制中) |
→ 控制權爭奪停止,那波是暫時性擾動。

**我方:新問題 —— LPR watchdog 誤判造成 thrash**
```
FPS 0.17~0.19(基準1.4~1.7)  GPU util 1.0 但 GR3D_FREQ 僅 4%
wait_p95 2066ms  LPR 佔 50% GPU、hold 490ms  load 16.55
```
GPU 硬體幾乎沒在算、鎖卻飽和 → 有東西「拿著鎖沒在用 GPU」= 重啟中的 LPR。
確認**無「執行緒異常結束」**→ 5db7609 修的崩潰已好,這次是外層 watchdog 誤判:
```
GPU忙(等鎖2秒)→LPR產不出新幀→watchdog 20秒門檻誤判卡住→重啟
→重啟載模型/建CUDA context拿著鎖不放→GPU更塞→再誤判→循環
```
**修正 6060504**:門檻 20→90 秒(env LPR_STALL_RESTART_SEC),
順帶修正原註解錯誤(寫死20秒卻註解寫60秒)。

### 2026-09-02 14:2x LPR watchdog 修正有效,但發現更深層瓶頸

**6060504 修正有效**:LPR thrash 停止(2分鐘 0 次,先前每5分鐘 4~10 次)。

**但分析率仍低(0.33,基準1.4)**,且關鍵矛盾持續:
```
SKIP=2: GPU util 0.996  infer/s 3.80  wait_p95 1437ms  LPR 50% hold 319ms  GR3D 12%
SKIP=4: GPU util 0.996  infer/s 7.97  wait_p95  929ms  LPR 47% hold 167ms  GR3D 16%
        ↑吞吐翻倍、等待減半,但分析率沒上來(0.37→0.33)
```
**GPU 鎖飽和(0.996)但 GR3D 硬體僅 12~16%** → 有東西「拿著 GPU 鎖卻沒在用 GPU」。

**研判:LPR 拿著 GPU 鎖跑 CPU 的 Tesseract OCR**(LPR status: engine=Tesseract,
gpu=false)。所以減少 YOLO 次數(SKIP 2→4)幫不了 —— 佔住鎖的是 OCR 那段不是 YOLO。
證據:SKIP 加倍後 LPR 佔比僅 50%→47%,幾乎沒降。

**這是設計層面問題**:GPU 鎖的範圍涵蓋了不需要 GPU 的 OCR,把偵測擋在外面。
要改須縮小鎖範圍(只包 YOLO 推論,OCR 移出鎖外),風險中等,需評估併發安全。
**未擅自更動**,列為待決策項。

現況:LPR_INFER_SKIP=4、tj 91.4°C、OPAC 側正常。

### 2026-09-02 14:24 ✅仍在控制 + ★首次出現 forcedByMaxGreen(車流變大)

**控制狀態:正常控制中**
```
控制器 phase=true(控制權在 OPAC)、OPAC running/ADAPTIVE/dataStale2
下發 14success/4fail(78%)、決策每5秒持續、重新接管 0 次/10分鐘
```

**★ 首次撞 max-green 強制切換**(監控以來第一次,先前41筆全 False):
```
07:22:37 EXIT_WN 100s pn1=5 pn2=6 → SWITCH forcedByMaxGreen=True
```
隨後 EXIT_WN 又連續 KEEP 到 85 秒仍在 KEEP。

**判定:非異常,是真的塞車。** 演算法決策正確:
```
07:24:42 swlG=11(下匝道排隊11台) vs swlR=2(上匝道2台) pn1=10<pn2=11 → KEEP
```
下匝道排隊 10~11 台、上匝道僅 2~3 台 → 持續給下匝道綠燈是對的,
且 phase2=下匝道=主線保護優先,符合設計意圖(避免回堵國道主線)。

**⚠️ 帶出的參數問題**:
- OPAC `max-green-seconds: 100`
- 官方時制表 baseline `max_green: 210`
OPAC 上限比官方時制緊一倍以上。現在車流讓下匝道需要 >100s 綠燈,
卻被 OPAC 自己的 100s 上限強制切走 → 可能導致下匝道消化不完、排隊累積。
建議與中心端確認 max-green 是否應向官方時制表(210s)靠攏。

### ⚠️ 時區陷阱(2026-09-02 發現,重要)

**OPAC 兩個 log 檔用不同時區**:
```
decision.log     ts: 2026-09-02T07:27:52.149Z        ← 尾巴 Z = UTC
aicctv-data.log  ts: 2026-09-02T15:28:12.226+08:00   ← +08:00 = 台北
系統/容器時間: 15:28 CST(TZ=Asia/Taipei)
```
→ **decision.log 是 UTC,比台北慢 8 小時**。

本文件先前所有引用 decision.log 的時間(如「07:22:37 forcedByMaxGreen」、
「03:49 EXIT_WN 撐45秒」、「04:59/05:00 系列」)**都是 UTC**,
換算台北需 +8(即 15:22:37、11:49、12:59/13:00)。事件內容與判讀不受影響,
僅時間標示需換算。

日後判讀:看 ts 尾巴是 `Z` 還是 `+08:00`,不可直接取字串當本地時間。

### 5F10 來源攔截判別法(2026-09-02 建立)

**問題**:icagent log 的 5F10 寫入只記內容不記發起方,無法直接看出誰寫的。
**解法**:用特徵比對。OPAC 續約有固定指紋:
```
值:   fixTime:0, dynamic:0, roadSideManual:1, centerManual:0,
      phase:1, realtime:0, trigger:0, specialRoute:0, effectTime:5
節奏: 每分鐘 :50 秒整,間隔精準 60 秒(reconcile-interval-ms=60000)
```
→ **任何偏離此指紋的 5F10 寫入 = 非 OPAC 來源**(中央電腦/現場/其他系統)。

**基準驗證(15:18~15:36, 30分鐘)**:icagent 端 19 筆 5F10 寫入,
**全部符合 OPAC 指紋、零筆外來**。時間精準落在每分鐘 :50。
(OPAC 自報 30 次 > icagent 19 次,差額是 OPAC 端把重試也計入,不影響判別。)

**另**:192.168.1.100 每 20 秒連 icagent:1968(Liveness 保活,連上即斷),
疑為中央電腦探測,但**不下 5F10、不寫策略**。

**先前 14:47-48 控制權被打回 fixTime=1 的最可能解釋**:當時 icagent↔控制器
通訊異常(失敗175次/5min),5F10 續約送不到 → effectTime 5分鐘到期 →
控制器自動回退原策略。**即死人開關正常運作,非中央搶控**。通訊恢復後未再發生,
支持此解釋。

### max-green 的兩個層級(2026-09-02 查證,重要澄清)

**號誌控制器上的 maxGreen 在哪**:
時制計畫 → **基本參數(TC3 5F14)** → 每分相各一組。
管理系統路徑:`/timing-plan` 頁面 → 選計畫編號 → 基本參數區。

**SIG-01 計畫1 實際值**(實查 `/timing-plan/params/SIG-01?planId=1`):
| 分相 | minGreen | maxGreen | yellow | allRed | 行人綠閃 |
|---|---|---|---|---|---|
| 分相1(上匝道 ENTRY_NE) | 10 | **210** | 3 | 2 | 5 |
| 分相2(下匝道 EXIT_WN) | 20 | **210** | 3 | 2 | 5 |
與官方時制表 baseline `max_green:210` 一致。

**🛑 兩個 max-green 是不同層級,不會互相覆蓋**:
| | 位置 | 值 | 作用 |
|---|---|---|---|
| 控制器 maxGreen | 時制計畫基本參數 5F14 | 210s | 控制器**自主**跑感應控制時的上限 |
| OPAC max-green-seconds | OPAC configmap | 100s | OPAC **演算法內部**決策上限 |

OPAC 以「路側手動+時相控制」接管時是**逐步階下 5F1C 指令**指揮控制器,
控制器只照做 → **實際限制綠燈的是 OPAC 的 100s**,控制器的 210s 在此模式下不作用。
→ 要放寬綠燈上限必須改 OPAC configmap;改控制器的 210 無效。

**目前執行中計畫**:計畫1(週期85s、分相1綠35s、分相2綠40s) ——
即 OPAC 接管前的定時基準,也是死人開關到期後會回去的那組。

### 現場基本參數核對:✅ 正確,與官方時制表完全一致(2026-09-02)

| 參數 | 官方時制表(計畫1) | 現場控制器實查 | |
|---|---|---|---|
| minGreen 分相1 | 10 | 10 | ✅ |
| minGreen 分相2 | 20 | 20 | ✅ |
| maxGreen | 210 | 210(兩分相) | ✅ |
| yellow | 3 | 3 | ✅ |
| allRed | 2 | 2 | ✅ |
| cycle | 85 | 85(執行中) | ✅ |
| 分相1綠 | 35 | 35 | ✅ |
| 分相2綠 | 40 | 40 | ✅ |

**逐項吻合,現場沒有設錯。**
→ 先前撞 max-green 上限**不是現場參數問題**,純粹是 OPAC 自己的
  `max-green-seconds:100` 比官方基準(210)保守。
→ 要放寬只需改 OPAC configmap,不用動現場控制器。

**另注意**:官方時制表部分計畫的 max_green 是 **999**(計畫9、11),
代表官方對某些時段本就允許很長綠燈;而 OPAC 一律用 100s,不分計畫。

## 「路側手動」來源調查(2026-09-02 完整實證)

### 結論:roadSideManual=1 是 OPAC 寫的,不是現場控制箱

**① 現場控制箱沒有停在手動** —— 異動紀錄 FIELD_OPERATION(5F08)18 筆:
```
2026-08-28 14:15:51  128 = 回復自動   ← 最後一筆
2026-08-28 14:15:50    1 = 手動
... 全部 18 筆成對出現(手動→回復自動,間隔 1~14 秒)
最近一次現場操作:8/28(五天前),今天無任何現場操作
```
→ **現場是自動狀態**,沒人把它留在手動。
(對照我方 signal_tc3.py:FIELD_OPERATE={0x01:手動, 0x02:全紅, 0x40:閃光, 0x80:回復自動})

**② roadSideManual 來自 OPAC 的 takeover-strategy**:
- configmap 明寫 `road-side-manual: 1`
- 攔截的 19 筆 5F10 續約,值全是 `roadSideManual:1, phase:1`
- 回退後 roadSideManual 仍是 1 → 是 OPAC 最後寫入的殘留:
  5F10 設定整組策略,effectTime 到期只讓 phase 失效回 fixTime,
  **roadSideManual 這個 bit 不會被清掉**

### 停止測試後的預期狀態
```
fixTime=1, phase=0, roadSideManual=1(殘留), dynamic=0
→ 跑定時計畫1(週期85s/分相1綠35s/分相2綠40s),與官方時制表 08:30-16:30 時段一致
→ 16:30 後控制器依自己時段表切計畫37
```
若要連 roadSideManual 也清掉,需**主動送一次 5F10** 設 `roadSideManual:0, fixTime:1`
(從管理系統控制策略頁面),但須先停 OPAC,否則 60 秒內會被續約覆蓋。

### 待與中心端確認
1. takeover-strategy 用 `road-side-manual:1` 而非 `dynamic:1` —— 語意上
   roadSideManual 是「現場人員手動」,OPAC 是遠端演算法,用此旗標會造成
   管理系統顯示誤導。需確認控制器在 dynamic 模式下是否接受外部 5F1C 指令。
2. max-green-seconds:100 vs 官方時制表 210(部分計畫甚至 999)。

## 現場號誌比對驗證(2026-09-02 16:05)—— 三層一致,控制確實落地

| 層級 | 實測內容 | |
|---|---|---|
| A. OPAC 決策 | greenDirection=EXIT_WN, controlState=YELLOW | |
| B. 控制器分相 | 5FCC 回報:分相 2, 步階 1, 剩餘 23 秒 | ✅ 對上 |
| C. 實際燈態 | 5F03 綠燈在 signal 索引 [2,3,5] | ✅ 對上 |

**分相映射驗證**(OPAC config phases):
```
ENTRY_NE → sub-phase-id 1     EXIT_WN → sub-phase-id 2
```
OPAC 說 EXIT_WN、控制器回報分相 2 → 吻合。

**方向對位驗證**(signalMap: 北1 東北1 東1 東南1 南1 西南0 西1 西北0,
啟用 6 個方向依序索引 0~5):
| 索引 | 方向 | 燈態 |
|---|---|---|
| 2 | 東 | 綠 ✅ |
| 3 | 東南 | 綠 ✅ |
| 5 | 西 | 綠 ✅ |

對照 baseline 分相定義:`分相2(下匝道 EXIT_WN) green_directions=["東","西"]`
→ 綠燈落在東、西,**與分相2 定義相符**。東、西正是下匝道方向。

**結論:OPAC 決策 → 控制器分相 → 實際燈態,三層完全一致,
控制指令確實在驅動真實號誌,無錯位。**

### 2026-09-02 18:35 影子模式啟動受阻:我方號誌抄錄器未啟用

影子模式(ea469c7/a339684)已部署且啟動訊息正常,但**執行緒空轉、DB 未建表** ——
`_live_phase()` 拿不到燈態就 continue,永遠走不到寫入。

根因:**我方 SIGNAL_TC3_ENABLED 未設**,號誌抄錄器沒跑(執行緒數 0),
`_by_addr` 恆空。`SIGNAL_TC3_HOST=10.42.40.222` 有設、MiiNePort 也連得到
(port 1001 測試可連線),但 enabled 沒開。

🛑 **未擅自開啟**。原因:控制器 MaxConnect=1,signal_tc3.py 註解明寫
「交控中心一旦接上,我們會被拒絕或踢掉…讓中心優先」。而現在中心端 icagent
正用那條連線控制號誌(OPAC 每 5 秒下發、控制器每秒回報 5F03)。
我方抄錄器若去搶連線,可能干擾正在運作的控制。

兩條路待決策:
(a) 開我方抄錄器(SIGNAL_TC3_ENABLED=1) —— 需先確認 MiiNePort 是否支援
    多連線、或中心端走的是否同一 port。若會搶線則有干擾現行控制的風險。
(b) 影子改為不碰號誌連線:決策輸入只用我方排隊資料,「實際動作」改從
    中心端 OPAC decision.log 取(已有 SSH 管道)。依賴中心端但零風險。

**另發現(安全相關)**:現場 `SIGNAL_TC3_CONTROL=1` 且
`SIGNAL_TC3_CONTROL_QUERY_ONLY=0` —— 我方號控下傳是**開啟且無 query-only
保護**的狀態,具備直接改變真實號誌運轉的能力。與程式碼預設(關閉)不同,
值得確認是否為預期設定。
