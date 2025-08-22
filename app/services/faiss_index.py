import os
import json
import threading
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # placeholder if not installed yet

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

IMAGE_INDEX_PATH = DATA_DIR / "image.index"
META_PATH = DATA_DIR / "meta.json"
IDMAP_IMAGE_PATH = DATA_DIR / "idmap_image.json"

INDEX_LOCK = threading.RLock()
META: Dict[str, dict] = {}
IDMAP_IMAGE: List[str] = []  # position -> item_id

from config import settings
from . import embeddings

EMBED_DIM_IMAGE = settings.EMBEDDING_DIM_IMAGE
EMBEDDING_VERSION = settings.EMBEDDING_VERSION


def _create_flat_index(dim: int):
    if faiss is None:
        raise RuntimeError("faiss 라이브러리가 설치되지 않았습니다.")
    index = faiss.IndexFlatIP(dim)  # cosine 유사도 형태 사용 (사전 정규화 전제)
    return index


def _save_index(index, path: Path):
    faiss.write_index(index, str(path))


def _load_index(path: Path, dim: int):
    if not path.exists():
        return _create_flat_index(dim)
    return faiss.read_index(str(path))


def load_all():
    global IMAGE_INDEX, META, IDMAP_IMAGE, IDMAP_TEXT
    with INDEX_LOCK:
        IMAGE_INDEX = _load_index(IMAGE_INDEX_PATH, EMBED_DIM_IMAGE)
    META = json.loads(META_PATH.read_text("utf-8")) if META_PATH.exists() else {}
    IDMAP_IMAGE = json.loads(IDMAP_IMAGE_PATH.read_text("utf-8")) if IDMAP_IMAGE_PATH.exists() else []
    IDMAP_TEXT = []  # 텍스트 인덱스 제거


def save_all():
    with INDEX_LOCK:
        _save_index(IMAGE_INDEX, IMAGE_INDEX_PATH)
        META_PATH.write_text(json.dumps(META, ensure_ascii=False, indent=2), "utf-8")
        IDMAP_IMAGE_PATH.write_text(json.dumps(IDMAP_IMAGE, ensure_ascii=False, indent=2), "utf-8")


def add_item(item_id: str, image_vec, text_vec, meta: dict):  # text_vec deprecated
    with INDEX_LOCK:
        # FAISS는 numpy float32 배열 필요
        import numpy as np
        if image_vec is not None:
            image_arr = np.array([image_vec], dtype='float32')
            IMAGE_INDEX.add(image_arr)
            IDMAP_IMAGE.append(item_id)
        META[item_id] = meta


def search_image(query_vec, k=5) -> List[Tuple[str, float, dict]]:
    import numpy as np
    with INDEX_LOCK:
        if IMAGE_INDEX.ntotal == 0:
            return []
        q = np.array([query_vec], dtype='float32')
        scores, idxs = IMAGE_INDEX.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if idx >= len(IDMAP_IMAGE):
                continue
            item_id = IDMAP_IMAGE[idx]
            results.append((item_id, float(score), META[item_id]))
        return results


def search_text(*_args, **_kwargs):  # 유지: 기존 API 호출 시 빈 결과 반환
    return []


def needs_reindex(current_version: str, provider: str) -> bool:
    # Compare against effective provider actually in use (may fallback from configured one)
    return current_version != settings.EMBEDDING_VERSION or provider != embeddings.current_provider()


def reindex_all(force: bool = False):
    """Rebuild indices from META if version/provider mismatch or force.
    WARNING: Assumes original image/text not stored; only caption used for text, image skipped if path not available.
    """
    import numpy as np
    if not force and not any(
        needs_reindex(m.get('embedding_version',''), m.get('embedding_provider','hash'))
        for m in META.values()
    ):
        return False
    # Recreate indices
    global IMAGE_INDEX, IDMAP_IMAGE, IDMAP_TEXT
    IMAGE_INDEX = _create_flat_index(EMBED_DIM_IMAGE)
    IDMAP_IMAGE = []
    IDMAP_TEXT = []
    for item_id, meta in META.items():
        # 텍스트 인덱스 제거: caption 기반 재생성 스킵, 메타만 업데이트
        add_item(
            item_id,
            image_vec=None,
            text_vec=None,
            meta={
                **meta,
                'embedding_version': settings.EMBEDDING_VERSION,
                'embedding_provider': embeddings.current_provider(),
            }
        )
    save_all()
    return True

# 모듈 import 시 자동 로드 시도
try:
    load_all()
except Exception as e:
    print("[faiss_index] 초기화 실패:", e)
