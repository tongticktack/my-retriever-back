"""Image embedding service abstraction (OpenAI 단일 경로 + hash fallback).

현재 실제 OpenAI 이미지 embedding 모델 미사용 → hash 기반 대체.
나중에 교체 시 PROVIDER == "openai" 분기 확장.
"""
from __future__ import annotations
import hashlib
from typing import Optional
import numpy as np

from config import settings

PROVIDER = settings.EMBEDDING_PROVIDER.lower()
IMAGE_DIM = settings.EMBEDDING_DIM_IMAGE
_openai_client = None  # reserved


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-9)


def _hash_to_vec(data: bytes, dim: int) -> np.ndarray:
    h = hashlib.sha256(data).digest()
    raw = (h * ((dim // len(h)) + 1))[:dim]
    arr = np.frombuffer(bytes(raw), dtype=np.uint8).astype("float32")
    return _l2_normalize(arr)


def embed_image(image_bytes: bytes) -> np.ndarray:
    # TODO: OpenAI 이미지 embedding API 도입 시 구현
    return _hash_to_vec(image_bytes, IMAGE_DIM)


def current_provider() -> str:
    """Return the effective provider actually generating vectors (after fallback)."""
    return PROVIDER

def dims() -> tuple[int, int]:
    return IMAGE_DIM, 0  # 텍스트 차원 제거
