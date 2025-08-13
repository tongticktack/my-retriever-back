# Placeholder embedding functions
# 실제 구현 시 Vertex AI 멀티모달/텍스트 임베딩 호출로 교체
import hashlib
import numpy as np

IMAGE_DIM = 512
TEXT_DIM = 768


def _hash_to_vec(data: bytes, dim: int):
    h = hashlib.sha256(data).digest()
    # 반복시켜 dim 채우기
    raw = (h * ((dim // len(h)) + 1))[:dim]
    arr = np.frombuffer(bytes(raw), dtype=np.uint8).astype('float32')
    # 단순 정규화
    arr /= np.linalg.norm(arr) + 1e-9
    return arr


def embed_image(image_bytes: bytes):
    return _hash_to_vec(image_bytes, IMAGE_DIM)


def embed_text(text: str):
    return _hash_to_vec(text.encode('utf-8'), TEXT_DIM)
