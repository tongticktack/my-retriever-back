"""Image embedding service abstraction.

우선순위:
1. sentence-transformers 기반 로컬/캐시 CLIP(SigLIP) 모델 사용 (설정: EMBEDDING_IMAGE_MODEL)
2. 실패 시 hash fallback (deterministic)

NOTE:
- EMBEDDING_DIM_IMAGE 이 실제 모델 차원과 다르면 강제 project(잘라내거나 zero-pad)하여 고정 길이 유지.
- 추후 OpenAI vision embedding API 추가 시 PROVIDER 분기로 확장 가능.
"""
from __future__ import annotations
import hashlib
from typing import Optional
import numpy as np
import io
from PIL import Image

_model = None
_model_dim: Optional[int] = None
_model_name = getattr(settings, "EMBEDDING_IMAGE_MODEL", "clip-vit-b32")

def _load_model():  # lazy
    global _model, _model_dim
    if _model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        # Mapping convenience: allow short aliases
        alias_map = {
            "clip-vit-b32": "clip-ViT-B-32",
            "clip-vit-b16": "clip-ViT-B-16",
            "clip-vit-l14": "clip-ViT-L-14",
        }
        real_name = alias_map.get(_model_name.lower(), _model_name)
        _model = SentenceTransformer(real_name)
        # Infer dim
        test_vec = _model.encode(["dim"], convert_to_numpy=True)
        if isinstance(test_vec, np.ndarray):
            _model_dim = test_vec.shape[-1]
        else:
            _model_dim = test_vec[0].shape[-1]
    except Exception as e:  # pragma: no cover
        print(f"[embeddings] model load failed ({_model_name}) -> hash fallback: {e}")
        _model = None
        _model_dim = None

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


def _project(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[-1] == target_dim:
        return vec
    if vec.shape[-1] > target_dim:
        return vec[:target_dim]
    # pad
    pad = np.zeros(target_dim, dtype=vec.dtype)
    pad[:vec.shape[-1]] = vec
    return pad


def embed_image(image_bytes: bytes) -> np.ndarray:
    # Try model path
    _load_model()
    if _model is not None:
        try:
            # sentence-transformers 이미지 입력: PIL Image 리스트 or path
            with Image.open(io.BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                vec = _model.encode([im], convert_to_numpy=True, normalize_embeddings=True)
            if isinstance(vec, np.ndarray):
                emb = vec[0]
            else:
                emb = vec[0]
            if _model_dim is not None and IMAGE_DIM != _model_dim:
                emb = _project(emb, IMAGE_DIM)
            return _l2_normalize(emb.astype("float32"))
        except Exception as e:  # pragma: no cover
            print(f"[embeddings] encode error -> hash fallback: {e}")
    # Fallback
    return _hash_to_vec(image_bytes, IMAGE_DIM)


def current_provider() -> str:
    """Return the effective provider actually generating vectors (after fallback)."""
    return PROVIDER

def dims() -> tuple[int, int]:
    return IMAGE_DIM, 0  # 텍스트 차원 제거
