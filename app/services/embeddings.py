from __future__ import annotations

import io
import hashlib
import threading
from typing import Optional

import numpy as np
from PIL import Image

from config import settings

# ------------------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------------------
_MODEL_NAME_CFG = getattr(settings, "EMBEDDING_IMAGE_MODEL", "clip-vit-b32")
_DEVICE_CFG = getattr(settings, "EMBEDDING_DEVICE", None)  # "cpu", "cuda", "mps" 등
_PROVIDER_CFG = getattr(settings, "EMBEDDING_PROVIDER", "unknown")

# ✅ 차원은 512로 고정
_EMBEDDING_DIM_IMAGE = getattr(settings, "EMBEDDING_DIM_IMAGE", 512)

# ------------------------------------------------------------------------------
# 내부 상태
# ------------------------------------------------------------------------------
_model = None
_provider_effective: Optional[str] = None
_load_lock = threading.RLock()

# 짧은 별칭 → sentence-transformers 실제 모델명
_ALIAS = {
    "clip-vit-b32": "clip-ViT-B-32",
    "clip-vit-b16": "clip-ViT-B-16",
    "clip-vit-l14": "clip-ViT-L-14",
    "siglip-base-patch16-224": "sentence-transformers/siglip-base-patch16-224",
    "siglip-large-patch16-384": "sentence-transformers/siglip-large-patch16-384",
}


# ------------------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------------------
def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-9)


def _hash_to_vec(data: bytes, dim: int) -> np.ndarray:
    h = hashlib.sha256(data).digest()
    raw = (h * ((dim // len(h)) + 1))[:dim]
    arr = np.frombuffer(bytes(raw), dtype=np.uint8).astype("float32")
    return _l2_normalize(arr)


def _project(vec: np.ndarray, target_dim: int) -> np.ndarray:
    d = vec.shape[-1]
    if d == target_dim:
        return vec
    if d > target_dim:
        return vec[:target_dim]
    out = np.zeros(target_dim, dtype=vec.dtype)
    out[:d] = vec
    return out


def _resolve_model_name(name: str) -> str:
    return _ALIAS.get(name.lower(), name)


def _load_model() -> None:
    """SentenceTransformer 모델 lazy 로딩."""
    global _model, _provider_effective

    if _model is not None:
        return

    with _load_lock:
        if _model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            real_name = _resolve_model_name(_MODEL_NAME_CFG)
            kw = {}
            if _DEVICE_CFG:
                kw["device"] = _DEVICE_CFG

            _model = SentenceTransformer(real_name, **kw)
            _provider_effective = f"model:{real_name}"
        except Exception as e:
            print(f"[embeddings] model load failed ({_MODEL_NAME_CFG}) -> hash fallback: {e}")
            _model = None
            _provider_effective = "hash"


# ------------------------------------------------------------------------------
# 공개 API
# ------------------------------------------------------------------------------
def embed_image(image_bytes: bytes) -> np.ndarray:
    """
    입력: 원본 이미지 바이트
    출력: float32 L2-normalized 벡터 (길이 = 512)
    """
    global _provider_effective

    _load_model()

    # 1) 모델 경로
    if _model is not None:
        try:
            with Image.open(io.BytesIO(image_bytes)) as im:
                im = im.convert("RGB")
                vec = _model.encode([im], convert_to_numpy=True, normalize_embeddings=True)

            emb = vec[0] if isinstance(vec, np.ndarray) else vec[0]
            if emb.shape[-1] != _EMBEDDING_DIM_IMAGE:
                emb = _project(emb, _EMBEDDING_DIM_IMAGE)
            _provider_effective = _provider_effective or "model"
            return _l2_normalize(emb.astype("float32"))
        except Exception as e:
            print(f"[embeddings] encode error -> hash fallback: {e}")

    # 2) 해시 경로
    _provider_effective = "hash"
    return _hash_to_vec(image_bytes, _EMBEDDING_DIM_IMAGE)


def current_provider() -> str:
    """실제 사용된 provider 반환."""
    return _provider_effective or _PROVIDER_CFG


def dims() -> int:
    """항상 512 반환."""
    return _EMBEDDING_DIM_IMAGE