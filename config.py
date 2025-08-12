from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8", extra="ignore")

    # 로컬 개발 시 .env.local 파일에서 경로를 읽어옵니다. (Docker 환경에서는 사용되지 않을 수 있음)
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    # Docker 환경에서 환경 변수로 직접 주입될 JSON 문자열
    FIREBASE_CREDENTIALS_JSON_STRING: Optional[str] = None

settings = Settings()
