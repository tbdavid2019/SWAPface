# 使用官方 Python 3.11 slim 版本作為基礎映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 依賴
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 創建模型目錄（模型檔案由 docker-compose.yml 中的 model-downloader 服務下載）
RUN mkdir -p /app/{models,results,uploads}

# 由於使用 volume 掛載，不需要 COPY 後端代碼
# 後端代碼會通過 docker-compose.yml 中的 volume 掛載

# 設定環境變數
ENV PYTHONPATH=/app
ENV ENVIRONMENT=development

# 暴露端口
EXPOSE 3001

# 啟動命令（會被 docker-compose.yml 中的 command 覆蓋）
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]
