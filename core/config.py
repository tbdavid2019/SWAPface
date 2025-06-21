"""
應用配置檔案
"""
import os
from pathlib import Path

# 基礎路徑 (現在是專案根目錄)
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" # Gradio 可能不太需要這個了

# AI 模型配置
MODEL_CONFIG = {
    "FACE_ANALYSIS_MODEL": "buffalo_l",
    "FACE_SWAP_MODEL": "inswapper_128.onnx",
    "DETECTION_SIZE": (640, 640),
    "CTX_ID": 0,  # CPU: -1, GPU: 0
}

# 模板配置
TEMPLATE_CONFIG = {
    "TEMPLATES": {
        "1": {
            "id": "1",
            "name": "模板 1",
            "description": "預設模板",
            "path": "./models/templates/step01.jpg"
        },
        "2": {
            "id": "2", 
            "name": "模板 2",
            "description": "預設模板",
            "path": "./models/templates/step02.jpg"
        },
        "3": {
            "id": "3",
            "name": "模板 3", 
            "description": "預設模板",
            "path": "./models/templates/step03.jpg"
        },
        "4": {
            "id": "4",
            "name": "模板 4",
            "description": "預設模板", 
            "path": "./models/templates/step04.jpg"
        },
        "5": {
            "id": "5",
            "name": "模板 5",
            "description": "預設模板",
            "path": "./models/templates/step05.jpg"
        },
        "6": {
            "id": "6",
            "name": "模板 6",
            "description": "預設模板",
            "path": "./models/templates/step06.jpg"
        }
    }
}

def get_model_path(model_name: str) -> Path:
    """獲取模型檔案路徑"""
    # Hugging Face Spaces 會將模型快取在 /data，但我們的邏輯是下載到 models/
    local_model_path = MODELS_DIR / model_name
    if local_model_path.exists():
        return local_model_path
    # 如果本地沒有，face_processor 會嘗試下載
    return local_model_path

def get_template_path(template_id: str) -> Path:
    """獲取模板圖片路徑"""
    template = TEMPLATE_CONFIG["TEMPLATES"].get(template_id)
    if not template:
        raise ValueError(f"Template {template_id} not found")
    
    template_path = template["path"]
    # 將相對於專案根目錄的路徑轉換為絕對路徑
    if template_path.startswith("./"):
        return BASE_DIR / template_path[2:]
    return Path(template_path)

def ensure_directories():
    """確保必要的目錄存在"""
    # 確保模型和模板目錄存在，以便下載和存取
    (MODELS_DIR / "templates").mkdir(parents=True, exist_ok=True)

