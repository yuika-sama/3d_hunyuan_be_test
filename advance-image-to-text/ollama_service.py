# Tên file: ollama_service.py (Đã sửa lỗi không giải phóng VRAM)
import os
import uuid
import ollama
from flask import Flask, request, jsonify
from deep_translator import GoogleTranslator
import httpx # Dùng để gọi API Ollama Force Unload

app = Flask(__name__)

# Thư mục lưu ảnh tạm
UPLOAD_FOLDER = 'temp_ollama_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Địa chỉ API nội bộ của Ollama Daemon
OLLAMA_API_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_NAME = 'llava' # Thay đổi nếu bạn dùng 'llava-phi3'

@app.route('/custom_describe', methods=['POST'])
def custom_describe():
    image_path = None
    try:
        # 1. Nhận dữ liệu và lưu ảnh
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        user_prompt_vi = request.form.get('prompt', 'Mô tả ảnh này')

        filename = f"{uuid.uuid4()}_{file.filename}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        print(f"--- [Port 5004] Nhận request: {user_prompt_vi} ---")

        # 2. Dịch Prompt: Việt -> Anh
        translator_vi_to_en = GoogleTranslator(source='vi', target='en')
        user_prompt_en = translator_vi_to_en.translate(user_prompt_vi)
        full_prompt = f"Describe this image and fulfill this request: {user_prompt_en}"
        
        # 3. Gọi Ollama (LLaVA)
        # Note: Ollama sẽ tự động tải model vào VRAM nếu nó chưa có
        response = ollama.chat(
            model=OLLAMA_MODEL_NAME, 
            messages=[{
                'role': 'user',
                'content': full_prompt,
                'images': [image_path]
            }]
        )
        english_result = response['message']['content']

        # 4. Dịch Kết quả: Anh -> Việt
        translator_en_to_vi = GoogleTranslator(source='en', target='vi')
        vietnamese_result = translator_en_to_vi.translate(english_result)

        print("--- [Port 5004] Hoàn tất ---")
        
        return jsonify({'result': vietnamese_result})

    except Exception as e:
        print(f"Lỗi service 5004: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # VRAM HACK: GỌI API ĐỂ GIẢI PHÓNG MODEL NGAY LẬP TỨC
        try:
            # Gửi lệnh UNLOAD (Tải lại) với thời gian chờ 1s để giải phóng
            httpx.post(f"{OLLAMA_API_URL}/api/generate", json={
                "model": OLLAMA_MODEL_NAME,
                "prompt": "Unload",
                "keep_alive": "1s" # Đặt thời gian chờ 1s để giải phóng ngay
            }, timeout=5)
            print(f"--- [VRAM CLEAN] Đã gửi lệnh giải phóng VRAM cho model {OLLAMA_MODEL_NAME} ---")
        except:
            print("--- [VRAM CLEAN] Không thể gọi API giải phóng Ollama (Daemon có thể chưa chạy) ---")
            pass

        # Dọn dẹp file ảnh tạm
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

if __name__ == '__main__':
    print("Microservice Ollama đang chạy tại http://127.0.0.1:5004")
    app.run(host='0.0.0.0', port=5004)