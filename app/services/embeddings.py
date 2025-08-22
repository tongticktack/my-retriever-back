"""Image embedding service abstraction (text embedding 제거).

지원 provider:
    - hash (deterministic local hashing 기본)
    - openai / gemini: 현재 실제 이미지 임베딩 미사용 시 hash fallback

FAISS index 는 이제 이미지 벡터만 유지.
"""
from __future__ import annotations
import hashlib
from typing import Optional
import numpy as np

from config import settings

PROVIDER = settings.EMBEDDING_PROVIDER.lower()

IMAGE_DIM = settings.EMBEDDING_DIM_IMAGE

_openai_client = None  # (보류: 이미지 전용 실제 모델 도입 시 확장)
_gemini_ready = False


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


def current_provider() -> str:
    """Return the effective provider actually generating vectors (after fallback)."""
    return PROVIDER

def dims() -> tuple[int, int]:
    return IMAGE_DIM, 0  # 텍스트 차원 제거
