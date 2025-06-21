import gradio as gr
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# 導入臉部處理器
from core.face_processor import FaceProcessor

# --- 初始化模型 ---
face_processor = None
try:
    print("🚀 正在初始化 AI 模型，請稍候...")
    face_processor = FaceProcessor()
    print("✅ AI 模型載入成功！")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    print("   如果看到下載錯誤，請參考 README.md 中的手動下載指南。")
    face_processor = None

# --- Gradio 介面 ---
def create_examples():
    """檢查範例圖片是否存在，如果存在則建立範例列表"""
    example_list = []
    template_dir = Path(__file__).parent / 'models' / 'templates'
    
    # 檢查範例圖片 step01.jpg 和 step02.jpg 是否存在
    source_example_path = template_dir / 'step01.jpg'
    target_example_path = template_dir / 'step02.jpg'

    if source_example_path.exists() and target_example_path.exists():
        print("✅ 找到範例圖片，正在建立 Gradio 範例...")
        example_list.append([
            str(source_example_path),
            str(target_example_path),
            "0", # 臉部索引
            None # 模板
        ])
    else:
        print("⚠️  未找到範例圖片（例如 models/templates/step01.jpg），Gradio 將不會顯示範例。")
        print(f"   - 檢查路徑: {source_example_path}")
        print(f"   - 檢查路徑: {target_example_path}")

    return example_list

# 建立範例
examples = create_examples()

# 取得模板圖片列表
def get_template_files():
    template_dir = Path(__file__).parent / "models" / "templates"
    try:
        templates = [str(p) for p in template_dir.glob("*.jpg")]
        print(f"✅ 成功載入 {len(templates)} 個模板。")
        return templates
    except FileNotFoundError:
        print("⚠️  找不到模板資料夾 'models/templates'，模板選擇功能將無法使用。")
        return []
    except Exception as e:
        print(f"❌ 載入模板時發生錯誤: {e}")
        return []

def swap_face_gradio(source_image, target_image, face_index_str, template_choice):
    """Gradio 的主處理函數"""
    if face_processor is None:
        raise gr.Error("AI 模型未能成功載入，無法處理請求。請檢查後台日誌。")

    print(f"ℹ️ 開始處理... 來源臉孔索引: {face_index_str}")

    try:
        # --- 1. 參數驗證和準備 ---
        if source_image is None:
            raise gr.Error("請務必上傳來源圖片。")

        # 如果選擇了模板，目標圖片變成模板
        if template_choice and template_choice != "無":
            print(f"   -> 使用模板: {template_choice}")
            target_image = np.array(Image.open(template_choice))
        elif target_image is None:
            raise gr.Error("請上傳目標圖片，或選擇一個模板。")

        try:
            face_index = int(face_index_str)
            if face_index < 0:
                raise ValueError
        except (ValueError, TypeError):
            raise gr.Error("臉部索引必須是一個大於等於 0 的整數。")

        # --- 2. 圖片格式轉換 ---
        # Gradio 提供 RGB 格式，insightface 需要 BGR 格式
        source_bgr = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
        target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)

        # --- 3. 執行換臉 --- 
        result_bgr = face_processor.swap_face(
            source_bgr,
            target_bgr,
            face_index
        )

        # --- 4. 處理結果 ---
        if result_bgr is None:
            raise gr.Error("換臉失敗，可能是因為無法在圖片中偵測到足夠的臉部。")

        # 轉回 RGB 格式供 Gradio 顯示
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        print("✅ 處理完成！")
        return result_rgb

    except ValueError as ve:
        print(f"❌ 處理時發生錯誤: {ve}")
        raise gr.Error(str(ve))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ 發生未預期的錯誤: {e}")
        print(f"詳細錯誤堆疊:\n{error_details}")
        raise gr.Error(f"錯誤: {str(e)}")

# --- 建立 Gradio 介面 ---
with gr.Blocks(
    title="換臉 by DAVID888",
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 1000px !important}"
) as demo:
    
    gr.HTML("""
        <div style="text-align: center; font-family: 'Arial', sans-serif; color: #333;">
            <h1 style="color: #2c3e50;">換臉工具</h1>
            <p style="font-size: 1.1em;">上傳一張包含臉部的<b>來源圖片</b>和一張<b>目標圖片</b>，或從下方選擇一個<b>模板</b>作為目標，即可進行換臉。</p>
            <p style="font-size: 0.9em; color: #7f8c8d;">如果來源圖片中有多張臉，請在「來源臉部索引」中指定要替換的臉部（從 0 開始）。</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            source_image = gr.Image(
                label="來源圖片 (您的臉部)",
                type="numpy",
                height=300
            )
            face_index_input = gr.Textbox(
                label="來源臉部索引",
                value="0",
                placeholder="如果有多張臉，指定要使用的臉部（從 0 開始）",
                info="0=第一張臉, 1=第二張臉, 以此類推"
            )

        with gr.Column(scale=1):
            target_image = gr.Image(
                label="目標圖片 (要換到的目標)",
                type="numpy", 
                height=300
            )
            
            # 模板選擇 (如果有可用模板)
            template_files = get_template_files()
            if template_files:
                template_choice = gr.Dropdown(
                    choices=["無"] + template_files,
                    value="無",
                    label="或選擇一個模板",
                    info="選擇模板會覆蓋上方的目標圖片"
                )
            else:
                template_choice = gr.Dropdown(
                    choices=["無"],
                    value="無",
                    label="模板 (無可用模板)",
                    interactive=False
                )

    # 處理按鈕
    swap_button = gr.Button(
        "🚀 開始換臉",
        variant="primary",
        size="lg"
    )

    # 結果顯示
    result_image = gr.Image(
        label="換臉結果",
        type="numpy",
        height=400
    )

    # 範例 (如果有的話)
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[source_image, target_image, face_index_input, template_choice],
            label="點擊範例快速開始",
            examples_per_page=3
        )

    # 綁定處理函數
    swap_button.click(
        fn=swap_face_gradio,
        inputs=[
            source_image,
            target_image, 
            face_index_input,
            template_choice
        ],
        outputs=result_image,
        api_name="face_swap"
    )

# --- 啟動應用 ---
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=True
    )
