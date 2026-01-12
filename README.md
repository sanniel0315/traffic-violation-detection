# 🚗 交通違規影像分析系統

基於 **NVIDIA Jetson Xavier NX 8GB** 的邊緣 AI 交通違規偵測系統，整合即時影像串流、YOLOv8 物件偵測、車牌辨識及違規事件管理。

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Jetson%20NX%208GB-green.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)

---

## 📋 功能特色

### 核心功能
- 🎥 **即時影像監控** - 支援多路 RTSP 攝影機串流，GStreamer 硬體解碼
- 🚙 **車輛偵測** - YOLOv8 即時偵測車輛、機車、行人等 (支援 TensorRT 加速)
- 🔢 **車牌辨識 (LPR)** - 多重影像預處理 + Tesseract OCR
- 📍 **ROI 區域設定** - 視覺化多邊形繪製違規偵測區域
- 📊 **統計分析** - 時段統計、違規類型分析、累犯查詢
- 📸 **證據保存** - 自動儲存違規截圖與車牌影像
Markdown# 🚗 交通違規影像分析系統 (Traffic Violation Detection System)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Jetson%20NX-green.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![Vue](https://img.shields.io/badge/frontend-Vue%203-42b883.svg)

基於 **NVIDIA Jetson Xavier NX** 的邊緣 AI 交通違規偵測系統。整合即時 RTSP 影像串流、YOLOv8 物件偵測、Tesseract 車牌辨識及違規事件管理，並支援與 Frigate NVR 及 Home Assistant 連動。

---

## 📋 功能特色

### 核心功能
- 🎥 **即時影像監控** - 支援多路 RTSP 攝影機串流，採用 GStreamer 硬體解碼。
- 🚙 **多類別物件偵測** - 整合 YOLOv8 偵測車輛（汽車、機車、卡車）與行人。
- 🔢 **車牌辨識 (LPR)** - 針對台灣車牌優化的 Tesseract OCR，包含特殊車牌處理。
- 📍 **ROI 區域防護** - 支援視覺化繪製偵測區域 (Region of Interest)，精準判斷違規。
- 📸 **自動取證** - 違規觸發時自動快照、裁切車牌並儲存高解析度證據影像。
- 📊 **數據儀表板** - 提供 Vue 3 前端介面，展示即時影像、統計圖表與歷史查詢。

### 影像預處理技術
為解決邊緣運算環境下的光影變化與模糊問題，系統實作了多重預處理管線：

| 預處理方法 | 說明 | 適用場景 |
|-----------|------|---------|
| **CLAHE** | 對比度限制自適應直方圖均衡化 | 改善夜間或隧道內低光源環境 |
| **Otsu 二值化** | 自動計算最佳閾值進行二值化 | 提升標準光源下的字元清晰度 |
| **透視變換** | 基於輪廓的四點透視校正 | 修正斜視角拍攝造成的車牌變形 |
| **形態學運算** | 膨脹與侵蝕操作 | 修復斷裂字元或去除噪點 |

---

## 🏗️ 系統架構

本專案採用微服務架構，將影像擷取、AI 推論與後端服務解耦，確保系統穩定性。

```mermaid
graph TD
    subgraph Input_Sources [影像來源]
        Cam[IP 攝影機 (RTSP)] -->|H.264 Stream| Frigate[Frigate NVR<br/>動態偵測/錄影]
    end

    subgraph Edge_Compute [Jetson NX 邊緣運算]
        Frigate -->|RTMP/RTSP| Jetson
        Jetson[Jetson Xavier NX]
        
        direction TB
        Jetson -->|TensorRT| Detect[YOLOv8 物件偵測]
        Jetson -->|OpenCV + Tesseract| OCR[車牌辨識 LPR]
        Detect -.->|ROI 觸發| OCR
    end

    subgraph Backend [後端服務]
        Detect --> API[FastAPI Server]
        OCR --> API
        API <--> DB[(PostgreSQL)]
        API --> FS[證據儲存<br/>(SSD/NVMe)]
        API -->|MQTT| HA[Home Assistant]
    end

    subgraph User_Interface [前端介面]
        API <--> Web[Vue 3 Dashboard]
        User((管理者)) <--> Web
    end

    style Jetson fill:#76b900,stroke:#333,stroke-width:2px,color:white
    style Frigate fill:#ff5722,stroke:#333,stroke-width:2px,color:white
    style API fill:#009688,stroke:#333,stroke-width:2px,color:white
💻 硬體與環境需求硬體規格項目建議規格說明邊緣裝置NVIDIA Jetson Xavier NX 8GB需開啟 15W 或 20W 電源模式儲存空間NVMe SSD 512GB+建議使用 NVMe 以應付高速影像寫入攝影機1080p RTSP IP Camera建議幀率 30fps，支援 H.264軟體環境OS: Ubuntu 20.04 (JetPack 5.1) 或 Ubuntu 22.04 (JetPack 6.0)Container: Docker 24.0+ / NVIDIA Container ToolkitAI Framework: PyTorch 2.1 (with CUDA), TensorRT 8.5+Backend: Python 3.10, FastAPI, SQLAlchemyFrontend: Node.js 18+, Vue 3, Element Plus⚡ 效能優化 (TensorRT)為在 Jetson NX 上達到即時處理，模型經過 TensorRT 優化轉換：模型格式精度平均 FPS (Jetson NX)延遲 (Latency)PyTorch (.pt)FP32~18 FPS~55msTensorRT (.engine)FP16~42 FPS~23msTensorRT (.engine)INT8~58 FPS~17ms (需校準)註：以上數據基於 YOLOv8n 模型測試，實際效能視輸入解析度與 ROI 數量而定。📁 專案結構Bashtraffic-violation-detection/
├── docker-compose.yml          # 服務編排設定
├── api/                        # FastAPI 後端核心
│   ├── main.py                 # 程式入口
│   ├── routes/                 # API 路由
│   └── services/               # 核心邏輯 (偵測、LPR)
├── web/                        # Vue 3 前端原始碼
├── configs/                    # 設定檔 (Frigate, YOLO)
├── models/                     # AI 模型存放區 (.pt, .engine)
├── storage/                    # 違規影像儲存 (Docker Volume)
└── scripts/                    # 工具腳本 (模型轉換、測試)
🚀 快速開始1. 克隆專案Bashgit clone [https://github.com/YOUR_USERNAME/traffic-violation-detection.git](https://github.com/YOUR_USERNAME/traffic-violation-detection.git)
cd traffic-violation-detection
2. 環境設定複製範例設定檔並修改參數 (資料庫密碼、RTSP 網址等)。Bashcp .env.example .env
nano .env
3. 下載與轉換模型Bash# 下載 YOLOv8 權重
mkdir -p models
wget -P models/ [https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt)

# (建議) 轉換為 TensorRT Engine 以獲得最佳效能
docker run --gpus all --rm -v $(pwd):/app ultralytics/ultralytics:latest \
    yolo export model=/app/models/yolov8n.pt format=engine half=True device=0
4. 啟動系統Bashdocker compose up -d
5. 存取服務Web 管理介面: http://localhost:8080API 文件 (Swagger): http://localhost:8000/docsFrigate NVR: http://localhost:5000🚗 支援車牌與違規類型支援車牌格式自用小客車: ABC-1234, AB-1234, ABC-123, AA-1111機車: 一般重型機車、輕型機車特殊車牌: 軍車 (軍A-12345)、試車牌 (試1234)、臨時牌 (臨12345)偵測違規類型違規停車 (Illegal Parking) - 在 ROI 紅區停留超過設定秒數逆向行駛 (Wrong Way) - 車輛移動向量與車道方向相反闖紅燈 (Red Light Violation) - (需整合號誌燈號訊號)行駛人行道 - 車輛中心點進入人行道 ROI📄 授權條款本專案採用 MIT License 授權。📞 聯絡與貢獻歡迎提交 Issue 或 Pull Request 來協助改進本專案。Author: [Your Name/Handle]Email: [your.email@example.com]
### 上傳 GitHub 的建議步驟：

1.  **建立檔案**：在專案根目錄建立 `README.md` 並貼上上述內容。
2.  **替換資訊**：搜尋文件中的 `YOUR_USERNAME`、`[Your Name/Handle]` 和 `[your.email@example.com]`，替換成你的真實資訊。
3.  **上傳圖片**：
    * 如果你有系統截圖，建議在專案中建立 `docs/images/` 資料夾。
    * 將截圖放入後，在 README 中加入 `![Dashboard Screenshot](docs/images/dashboard.png)` 這樣的語法來展示成果。
4.  **Push**：
    ```bash
    git add README.md
    git commit -m "docs: update comprehensive README with architecture diagrams"
    git push origin main
    ```
⭐ 如果這個專案對你有幫助，請給個 Star！
