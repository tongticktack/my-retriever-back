from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8", extra="ignore")

    # Firebase
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    FIREBASE_CREDENTIALS_JSON_STRING: Optional[str] = None
    FIREBASE_STORAGE_BUCKET: Optional[str] = None  # firebase storage bucket name (e.g. my-app.appspot.com)

    # Gemini (AI Studio)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    # (Gemini 사용 여부는 GEMINI_API_KEY 존재로 판단)

    # LLM 선택 및 설정
    LLM_PROVIDER: str = "auto"
    LLM_AUTO_PRIORITY: str = "gemini,openai,echo"
    LLM_MAX_HISTORY_MESSAGES: int = 20

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"  # 텍스트 임베딩 (OpenAI 사용 시)
    
    # Embeddings (이미지/텍스트 공통)
    EMBEDDING_PROVIDER: str = "hash"  # hash | gemini | openai
    EMBEDDING_IMAGE_MODEL: str = "gemini-1.5-flash"  # (추후 멀티모달 임베딩 모델명 교체 가능)
    EMBEDDING_TEXT_MODEL: str = "text-embedding-004"  # AI Studio 텍스트 임베딩
    EMBEDDING_DIM_IMAGE: int = 512  # 해시 기본값 (실제 모델 사용 시 런타임 감지 가능)
    EMBEDDING_DIM_TEXT: int = 768
    EMBEDDING_VERSION: str = "v1"

    # Multi-image distinct object detection threshold (cosine similarity)
    MULTI_IMAGE_MIN_INTERNAL_SIMILARITY: float = 0.45  # env override 가능
    
    # prompt
    # Prompt bundle version (intent/extraction/guard). Set via env PROMPT_VERSION.
    PROMPT_VERSION: int = 3
    # Default system prompt now upgraded to v3 (richer self-intro & structured style guidelines)
    SYSTEM_PROMPT_FILE: str = "app/prompts/system_prompt_v3.txt"


settings = Settings()
