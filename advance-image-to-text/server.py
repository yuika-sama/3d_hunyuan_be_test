import os
import ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)  # Cho phép HTML gọi API không bị lỗi chặn

# Thư mục lưu ảnh tạm
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/custom_describe', methods=['POST'])
def custom_describe():
    try:
        # 1. Nhận dữ liệu từ HTML
        if 'image' not in request.files:
            return jsonify({'error': 'Không có file ảnh'}), 400
        
        file = request.files['image']
        user_prompt_vi = request.form.get('prompt', 'Mô tả bức ảnh này')

        # Lưu ảnh tạm thời
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        print(f"--- Đã nhận yêu cầu: '{user_prompt_vi}' ---")

        # 2. Dịch Prompt từ Tiếng Việt -> Tiếng Anh (Để AI hiểu tốt nhất)
        print("1. Đang dịch yêu cầu sang tiếng Anh...")
        translator_vi_to_en = GoogleTranslator(source='vi', target='en')
        user_prompt_en = translator_vi_to_en.translate(user_prompt_vi)
        
        # Thêm ngữ cảnh cho AI để nó biết nó đang làm gì
        full_prompt = f"Describe this image and fulfill this request: {user_prompt_en}"
        print(f"   (Prompt tiếng Anh: {full_prompt})")

        # 3. Gửi cho Ollama (Model LLaVA)
        print("2. Đang gửi cho AI (Llava)...")
        response = ollama.chat(
            model='llava', # Hoặc 'llava-phi3' nếu bạn đã tải
            messages=[{
                'role': 'user',
                'content': full_prompt,
                'images': [image_path]
            }]
        )
        english_result = response['message']['content']

        # 4. Dịch kết quả từ Tiếng Anh -> Tiếng Việt
        print("3. Đang dịch kết quả về tiếng Việt...")
        translator_en_to_vi = GoogleTranslator(source='en', target='vi')
        vietnamese_result = translator_en_to_vi.translate(english_result)

        print("--- Hoàn tất! ---")
        
        # Xóa ảnh tạm để nhẹ máy
        os.remove(image_path)

        # Trả kết quả JSON về HTML
        return jsonify({'result': vietnamese_result})

    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Chạy server ở port 5000
    print("Server đang chạy tại http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)