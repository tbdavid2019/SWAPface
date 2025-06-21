# 🎭 AI 頭像工作室 (Gradio 版)

這是一個使用 Gradio 和 InsightFace 技術建立的 AI 換臉應用程式。專案經過簡化，移除了前後端分離的複雜架構，改用單一的 Gradio 應用程式，使其更容易在本地端執行、打包成 Docker 映像檔，或部署到 Hugging Face Spaces。

## ✨ 功能

- **互動式介面**: 使用 Gradio 建立，操作直觀，無需前端開發知識。
- **自訂換臉**: 上傳您的照片和目標風格照片，一鍵生成換臉結果。
- **預設模板**: 提供多個預設模板，方便快速體驗。
- **多人臉支援**: 支援對圖片中的多張人臉進行選擇性替換。
- **自動模型下載**: 首次執行時，應用程式會自動下載所需的 AI 模型。

## 📂 專案結構

專案已簡化為適合 Gradio 應用的結構：

```
swapFace/
├── Dockerfile          # 用於建立 Docker 映像檔
├── app.py              # 主要的 Gradio 應用程式
├── requirements.txt    # Python 依賴
├── README.md           # 就是您正在看的這個檔案
├── core/               # 核心 AI 處理邏輯
│   ├── __init__.py
│   ├── config.py
│   └── face_processor.py
└── models/             # 存放 AI 模型和圖片模板
    └── templates/
        ├── step01.jpg
        └── ...
```

---

## 🚀 如何執行

您可以選擇兩種方式來執行此應用程式：**1. 在本地端直接執行** 或 **2. 使用 Docker**。

### 方案一：在本地端直接執行 (推薦給開發者)

1.  **環境準備**:
    建議在 Python 虛擬環境中進行操作。
    ```bash
    # 建立虛擬環境
    python3 -m venv venv

    # 啟用虛擬環境 (macOS/Linux)
    source venv/bin/activate
    # (Windows)
    # venv\Scripts\activate
    ```

2.  **安裝依賴**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **啟動應用**:
    ```bash
    python app.py
    ```

    應用程式啟動後，會顯示一個本地 URL (例如 `http://127.0.0.1:7860`)。在瀏覽器中打開此連結即可開始使用。

    **⚠️ 注意**: 首次啟動時，程式會自動從網路下載約 530MB 的 AI 模型 (`inswapper_128.onnx`) 到 `models` 目錄下，請耐心等待。這只需要執行一次。

### 方案二：使用 Docker 執行 (推薦用於部署或隔離環境)

確保您的電腦上已經安裝了 [Docker](https://www.docker.com/products/docker-desktop/)。

1.  **建立 Docker 映像檔**:
    在專案根目錄下（與 `Dockerfile` 同層級），執行以下指令：
    ```bash
    # -t 後面是您為映像檔取的名稱，例如 'ai-avatar-studio'
    docker build -t ai-avatar-studio .
    ```

2.  **執行 Docker 容器**:
    映像檔建立成功後，使用以下指令來啟動容器：
    ```bash
    # -p 7860:7860 將您本機的 7860 連接埠映射到容器的 7860 連接埠
    # --rm 會在容器停止後自動刪除，保持系統乾淨
    docker run --rm -p 7860:7860 ai-avatar-studio
    ```

3.  **訪問應用**:
    容器啟動後，同樣會先下載模型（如果映像檔中沒有的話）。完成後，在瀏覽器中打開 `http://localhost:7860` 即可使用。

---

## 🤗 部署到 Hugging Face Spaces

這個專案結構也非常適合直接部署到 Hugging Face Spaces。

1.  在 Hugging Face 上建立一個新的 Space，並選擇 **Gradio** 作為 SDK。
2.  將專案中的所有檔案（**除了 `Dockerfile`**）上傳到 Space 的 Git 儲存庫。
3.  Hugging Face 會自動處理環境安裝和應用啟動。大功告成！
