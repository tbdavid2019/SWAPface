"""
臉部處理核心模組
"""
import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import onnxruntime
from pathlib import Path
import urllib.request
import os

class FaceProcessor:
    """臉部處理器，封裝了所有 AI 模型"""
    def __init__(self):
        self.swapper = None
        self.face_analyzer = None
        self._initialize_models()

    def _ensure_model_downloaded(self, root_dir: Path, model_name: str, url: str):
        """
        確保指定的模型檔案存在於 models 資料夾中。
        如果不存在，則從提供的 URL 下載。
        """
        model_dir = root_dir / 'models'
        model_path = model_dir / model_name

        model_dir.mkdir(parents=True, exist_ok=True)

        if model_path.exists():
            print(f"INFO:core.face_processor:找到本地模型: {model_path}，將跳過下載。")
            return

        print(f"INFO:core.face_processor:在本地找不到模型，將從以下 URL 下載：")
        print(f"   -> URL: {url}")
        print(f"   -> 目標路徑: {model_path}")

        try:
            response = urllib.request.urlopen(url)
            total_length = response.getheader('content-length')
            
            with open(model_path, 'wb') as f:
                if total_length is None:
                    f.write(response.read())
                    print("INFO:core.face_processor:模型下載完成。")
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response:
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        print(f"\r   [ {'=' * done}{' ' * (50-done)} ] {dl * 100 / total_length:.2f}%", end='')
            print("\nINFO:core.face_processor:模型下載完成。")

        except Exception as e:
            print(f"\nERROR:core.face_processor:下載模型時發生錯誤。")
            if model_path.exists():
                os.remove(model_path)
            print(f"   請檢查您的網路連線，或嘗試手動從以下 URL 下載模型：")
            print(f"   {url}")
            print(f"   並將其放置在 '{model_path.parent}' 資料夾中。")
            raise RuntimeError(f"模型下載失敗: {url}") from e

    def _initialize_models(self):
        """初始化所有需要的模型"""
        try:
            print("INFO:core.face_processor:正在初始化 AI 模型...")
            onnxruntime.set_default_logger_severity(3)
            
            root_dir = Path(__file__).parent.parent
            models_dir = root_dir / 'models'
            models_dir.mkdir(exist_ok=True) # 確保 models 資料夾存在

            providers = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                print("INFO:core.face_processor:檢測到 CUDA，將使用 GPU。")
            else:
                print("INFO:core.face_processor:未檢測到 CUDA，將使用 CPU。")
                if 'CoreMLExecutionProvider' in providers:
                    providers.insert(0, 'CoreMLExecutionProvider')

            # --- 載入臉部分析模型 (buffalo_l) ---
            print(f"INFO:core.face_processor:正在檢查臉部分析模型 'buffalo_l'...")
            buffalo_l_path = models_dir / 'buffalo_l'
            print(f"   -> 預期路徑: {buffalo_l_path}")

            if not buffalo_l_path.is_dir() or not any(buffalo_l_path.iterdir()):
                 print("WARNING:core.face_processor:未在預期路徑找到 'buffalo_l' 模型資料夾，或該資料夾為空。")
                 print("   -> 程式將嘗試從網路下載。若要手動放置，請將解壓縮後的 'buffalo_l' 資料夾完整放入 'models' 目錄中。")
            else:
                 print("INFO:core.face_processor:成功找到本地 'buffalo_l' 模型資料夾。")

            # insightface 會在 root 目錄下尋找 'models' 資料夾，所以 root 應設為專案根目錄
            self.face_analyzer = FaceAnalysis(name='buffalo_l', root=str(root_dir), providers=providers)
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

            # --- 載入換臉模型 (inswapper_128.onnx) ---
            model_name = "inswapper_128.onnx"
            model_url = "https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/insightface/inswapper_128.onnx?download=true"
            
            self._ensure_model_downloaded(root_dir, model_name, model_url)

            model_path = root_dir / 'models' / model_name
            print(f"INFO:core.face_processor:準備從本地路徑載入模型: {model_path}")
            
            self.swapper = insightface.model_zoo.get_model(
                str(model_path),
                download=False,
                download_zip=False
            )
            print("INFO:core.face_processor:AI 模型載入完成！")

        except Exception as e:
            print(f"ERROR:core.face_processor:模型初始化失敗：{e}")
            raise RuntimeError(f"無法初始化 AI 模型：{e}") from e

    def get_faces(self, image):
        """從圖片中偵測所有臉部"""
        try:
            faces = self.face_analyzer.get(image)
            if faces:
                print(f"INFO:core.face_processor:偵測到 {len(faces)} 張臉部")
            else:
                print("WARNING:core.face_processor:在來源圖片中未偵測到任何臉部。")
            return faces
        except Exception as e:
            print(f"ERROR:core.face_processor:臉部偵測失敗：{e}")
            return []

    def swap_face(self, source_img, target_img, face_index=0):
        """
        在來源和目標圖片之間交換臉部。
        :param source_img: 來源圖片 (包含要使用的臉部)
        :param target_img: 目標圖片 (要將臉部換到這張圖上)
        :param face_index: 要從來源圖片中使用的臉部索引 (預設為第 0 張)
        :return: 換臉後的圖片
        """
        try:
            source_faces = self.get_faces(source_img)
            if not source_faces:
                raise ValueError("在來源圖片中找不到任何臉部，無法進行換臉。")
            
            if face_index >= len(source_faces):
                raise ValueError(f"指定的臉部索引 {face_index} 超出範圍，來源圖片只有 {len(source_faces)} 張臉。")

            target_faces = self.get_faces(target_img)
            if not target_faces:
                print("WARNING:core.face_processor:在目標圖片中未偵測到臉部，將直接在原圖上操作。")
                # 如果目標沒有臉，某些模型可能允許直接返回，或我們可以選擇返回原圖
                # 在此案例中，我們讓 swapper 決定如何處理

            source_face = source_faces[face_index]
            
            # 執行換臉
            # INSwapper.get() 的正確用法：get(target_img, target_face, source_face, paste_back=True)
            # 如果目標圖片沒有臉部，我們使用 source_face 作為 target_face
            target_face = target_faces[0] if target_faces else source_face
            result_img = self.swapper.get(target_img, target_face, source_face, paste_back=True)
            print("INFO:core.face_processor:換臉處理完成")
            return result_img

        except ValueError as ve:
            print(f"ERROR:core.face_processor:{ve}")
            raise
        except Exception as e:
            print(f"ERROR:core.face_processor:換臉處理失敗：{e}")
            raise RuntimeError("換臉過程中發生未預期的錯誤") from e
