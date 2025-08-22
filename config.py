from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8", extra="ignore")

    # Firebase
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    FIREBASE_CREDENTIALS_JSON_STRING: Optional[str] = None
    FIREBASE_STORAGE_BUCKET: Optional[str] = None  # firebase storage bucket name (e.g. my-app.appspot.com)

    # LLM 선택 및 설정
    LLM_PROVIDER: str = "auto"
    LLM_AUTO_PRIORITY: str = "openai,echo"
    LLM_MAX_HISTORY_MESSAGES: int = 20

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    # (텍스트 임베딩 제거됨)
    
    # Image embedding (OpenAI 단일 사용; 미지원 시 hash fallback)
    EMBEDDING_PROVIDER: str = "openai"  # openai (기본) | hash (fallback)
    EMBEDDING_DIM_IMAGE: int = 512
    EMBEDDING_VERSION: str = "v1"

    # Multi-image distinct object detection threshold (cosine similarity)
    MULTI_IMAGE_MIN_INTERNAL_SIMILARITY: float = 0.45  # env override 가능
    
    # prompt
    # Prompt bundle version (intent/extraction/guard). Set via env PROMPT_VERSION.
    PROMPT_VERSION: int = 3
    # Default system prompt now upgraded to v3 (richer self-intro & structured style guidelines)
    SYSTEM_PROMPT_FILE: str = "app/prompts/system_prompt_v3.txt"


settings = Settings()
