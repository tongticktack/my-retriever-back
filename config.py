from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8", extra="ignore")

    # Firebase
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    FIREBASE_CREDENTIALS_JSON_STRING: Optional[str] = None

    # GCP / Vertex AI
    GCP_PROJECT_ID: Optional[str] = None
    VERTEX_LOCATION: str = "asia-northeast3-a"  # 기본 리전
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"  # 기본 모델
    VERTEX_SEARCH_DATA_STORE_ID: Optional[str] = None  # Vertex AI Search Data Store ID

    # 기능 플래그
    ENABLE_VERTEX_SEARCH: bool = True
    ENABLE_GEMINI: bool = True

settings = Settings()
