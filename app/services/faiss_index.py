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
TEXT_INDEX_PATH = DATA_DIR / "text.index"
META_PATH = DATA_DIR / "meta.json"

INDEX_LOCK = threading.RLock()
META: Dict[str, dict] = {}

EMBED_DIM_IMAGE = 512  # placeholder dimension
EMBED_DIM_TEXT = 768   # placeholder dimension
EMBEDDING_VERSION = "v0"


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
    global IMAGE_INDEX, TEXT_INDEX, META
    with INDEX_LOCK:
        IMAGE_INDEX = _load_index(IMAGE_INDEX_PATH, EMBED_DIM_IMAGE)
        TEXT_INDEX = _load_index(TEXT_INDEX_PATH, EMBED_DIM_TEXT)
        if META_PATH.exists():
            META = json.loads(META_PATH.read_text("utf-8"))
        else:
            META = {}


def save_all():
    with INDEX_LOCK:
        _save_index(IMAGE_INDEX, IMAGE_INDEX_PATH)
        _save_index(TEXT_INDEX, TEXT_INDEX_PATH)
        META_PATH.write_text(json.dumps(META, ensure_ascii=False, indent=2), "utf-8")


def add_item(item_id: str, image_vec, text_vec, meta: dict):
    with INDEX_LOCK:
        # FAISS는 numpy float32 배열 필요
        import numpy as np
        if image_vec is not None:
            image_arr = np.array([image_vec], dtype='float32')
            IMAGE_INDEX.add(image_arr)
        if text_vec is not None:
            text_arr = np.array([text_vec], dtype='float32')
            TEXT_INDEX.add(text_arr)
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
            # FAISS FlatIP 는 id 매핑 없으므로 META 저장 시 순서 기반 별도 매핑 필요 (추후 개선)
            # 지금은 단순히 META keys 순서로 매핑 (MVP) → 안정 위해 별도 ID->position 맵 권장
            item_id = list(META.keys())[idx]
            results.append((item_id, float(score), META[item_id]))
        return results

# 모듈 import 시 자동 로드 시도
try:
    load_all()
except Exception as e:
    print("[faiss_index] 초기화 실패:", e)
