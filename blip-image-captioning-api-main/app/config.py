# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    blip_model_name: str = "Salesforce/blip-image-captioning-large"
    max_new_tokens: int = 60
    
    # THÊM DÒNG NÀY: Để hứng key từ file .env
    openai_api_key: str = "Hahahoho_tudiendi" 

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
