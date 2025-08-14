"""Embedding service abstraction.

Supports providers:
  - hash (deterministic local hashing; default / fallback)
  - openai (text only for now)
  - gemini (TODO: implement real API calls)

The FAISS index dimensions must match the vectors produced here. For dynamic
model dims, either persist dims or detect at runtime before index creation.
"""
from __future__ import annotations
import hashlib
from typing import Optional
import numpy as np

from config import settings

PROVIDER = settings.EMBEDDING_PROVIDER.lower()

IMAGE_DIM = settings.EMBEDDING_DIM_IMAGE
TEXT_DIM = settings.EMBEDDING_DIM_TEXT

_openai_client = None
_gemini_ready = False
if PROVIDER == "openai":
    try:
        import openai  # type: ignore
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            _openai_client = openai
    except Exception as e:  # pragma: no cover
        print("[embeddings] OpenAI init failed, fallback to hash:", e)
        PROVIDER = "hash"
elif PROVIDER == "gemini":
    try:
        import google.generativeai as genai  # type: ignore
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            _gemini_ready = True
    except Exception as e:  # pragma: no cover
        print("[embeddings] Gemini init failed, fallback to hash:", e)
        PROVIDER = "hash"


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-9)


def _hash_to_vec(data: bytes, dim: int) -> np.ndarray:
    h = hashlib.sha256(data).digest()
    raw = (h * ((dim // len(h)) + 1))[:dim]
    arr = np.frombuffer(bytes(raw), dtype=np.uint8).astype("float32")
    return _l2_normalize(arr)


def embed_image(image_bytes: bytes) -> np.ndarray:
    if PROVIDER == "hash":
        return _hash_to_vec(image_bytes, IMAGE_DIM)
    if PROVIDER == "gemini":
        # AI Studio 현재 공식 멀티모달 embedding 별도 모델 미제공 → 해시 대체 (추후 교체)
        return _hash_to_vec(image_bytes, IMAGE_DIM)
    # OpenAI: 현재 공개 이미지 임베딩 모델 미사용 → 해시 fallback
    return _hash_to_vec(image_bytes, IMAGE_DIM)


def embed_text(text: str) -> np.ndarray:
    if PROVIDER == "hash":
        return _hash_to_vec(text.encode("utf-8"), TEXT_DIM)
    if PROVIDER == "gemini" and _gemini_ready:
        try:
            import google.generativeai as genai  # type: ignore
            resp = genai.embed_content(model=settings.EMBEDDING_TEXT_MODEL, content=text)
            vec = np.array(resp["embedding"], dtype="float32")  # type: ignore
            return _l2_normalize(vec)
        except Exception as e:  # pragma: no cover
            print("[embeddings] Gemini text embed error, fallback hash:", e)
            return _hash_to_vec(text.encode("utf-8"), TEXT_DIM)
    if PROVIDER == "openai" and _openai_client is not None:
        try:
            resp = _openai_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text,
                timeout=10,
            )  # type: ignore
            vec = np.array(resp.data[0].embedding, dtype="float32")  # type: ignore
            return _l2_normalize(vec)
        except Exception as e:  # pragma: no cover
            print("[embeddings] OpenAI embed error, fallback hash:", e)
            return _hash_to_vec(text.encode("utf-8"), TEXT_DIM)
    return _hash_to_vec(text.encode("utf-8"), TEXT_DIM)


def current_provider() -> str:
    return PROVIDER

def dims() -> tuple[int, int]:
    return IMAGE_DIM, TEXT_DIM
