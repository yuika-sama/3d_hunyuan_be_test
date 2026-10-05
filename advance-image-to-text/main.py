# main.py (Final Solution: Ollama Brain + Google Translator)
import uvicorn
import logging
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import ollama

# Thư viện dịch (để chuyển lời giải tiếng Anh sang tiếng Việt)
# Nếu chưa có: pip install deep-translator
from deep_translator import GoogleTranslator 

app = FastAPI(title="Local Vision API (Hybrid)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo công cụ dịch
translator = GoogleTranslator(source='auto', target='vi')

@app.post("/caption")
async def caption(
    image: UploadFile = File(...),
    prompt: str = Form("Describe this image")
):
    try:
        image_bytes = await image.read()
        
        # 1. Ép Ollama tư duy bằng Tiếng Anh (để nó thông minh nhất)
        # Chúng ta sẽ bỏ qua prompt tiếng Việt của người dùng, 
        # thay vào đó dùng prompt tiếng Anh cố định để lấy nội dung chuẩn.
        english_prompt = "Describe this image in detail."
        
        logger.info(f"Ollama đang 'nhìn' ảnh (bằng tiếng Anh)...")

        response = ollama.chat(
            model='llava', 
            messages=[{
                'role': 'user',
                'content': english_prompt,
                'images': [image_bytes]
            }]
        )
        
        english_caption = response['message']['content']
        logger.info(f"English Result: {english_caption[:50]}...")

        # 2. Dịch kết quả sang Tiếng Việt
        logger.info("Đang dịch sang Tiếng Việt...")
        vietnamese_caption = translator.translate(english_caption)

        # 3. Trả về kết quả
        return {
            "caption": vietnamese_caption,
            "original_english": english_caption # Trả kèm tiếng Anh để bạn so sánh
        }

    except Exception as e:
        logger.error(f"Lỗi: {e}")
        return {
            "caption": f"Lỗi hệ thống: {str(e)}", 
            "error": True
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)