import ollama
from deep_translator import GoogleTranslator

# Đường dẫn ảnh
image_path = 'C:/Users/TomaszPC/Desktop/CODING/doan PTIT/crawl/Crawled_Data_V5/Images/0df4f24a-5f80-44de-aae9-8939f4f222d3.png'

def generate_and_translate():
    try:
        print("1. Đang nhìn ảnh (Thinking in English)...")
        # Bước 1: Hỏi bằng tiếng Anh để lấy nội dung chính xác nhất
        response = ollama.chat(
            model='llava', # Hoặc 'llava-phi3' nếu bạn đã tải
            messages=[{
                'role': 'user',
                'content': 'Describe this image in detail and write a very short poem about it.',
                'images': [image_path]
            }]
        )
        
        english_text = response['message']['content']
        if not english_text:
            print("Lỗi: AI không trả lời.")
            return

        print(f"\n[AI Original]: {english_text[:100]}...") # In nháp đoạn đầu tiếng Anh

        print("2. Đang dịch sang tiếng Việt...")
        # Bước 2: Dịch sang tiếng Việt
        # Dùng thư viện deep_translator để dịch tự động
        vietnamese_text = GoogleTranslator(source='auto', target='vi').translate(english_text)

        print("\n--- KẾT QUẢ CUỐI CÙNG ---")
        print(vietnamese_text)

    except Exception as e:
        print(f"Có lỗi: {e}")

if __name__ == "__main__":
    generate_and_translate()