from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # placeholder if not installed yet

import numpy as np

from config import settings
from . import embeddings

# ------------------------------------------------------------------------------------
# 경로/전역 상태
# ------------------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

IMAGE_INDEX_PATH = DATA_DIR / "image.index"
META_PATH = DATA_DIR / "meta.json"
IDMAP_IMAGE_PATH = DATA_DIR / "idmap_image.json"

INDEX_LOCK = threading.RLock()
IMAGE_INDEX = None  # type: ignore
META: Dict[str, dict] = {}
IDMAP_IMAGE: List[str] = []  # position -> item_id

# 구성값
EMBED_DIM_IMAGE: int = int(getattr(settings, "EMBEDDING_DIM_IMAGE", embeddings.dims()))
EMBEDDING_VERSION: str = getattr(settings, "EMBEDDING_VERSION", "v1")


# ------------------------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------------------------
def _l2_normalize(v: np.ndarray) -> np.ndarray:
    # (n, d) 또는 (d,)
    if v.ndim == 1:
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype("float32")
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return (v / n).astype("float32")


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding)
    os.replace(tmp, path)


def _atomic_write_index(index, path: Path) -> None:
    tmp = str(path) + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, str(path))


def _create_flat_index(dim: int):
    if faiss is None:
        raise RuntimeError("faiss 라이브러리가 설치되지 않았습니다.")
    # Cosine 유사도 = Inner Product + 사전 정규화
    index = faiss.IndexFlatIP(dim)
    return index


def _load_index(path: Path, dim: int):
    if faiss is None:
        raise RuntimeError("faiss 미설치")
    if not path.exists():
        return _create_flat_index(dim)
    return faiss.read_index(str(path))


def _ensure_dim_consistency() -> None:
    # 설정/임베딩 실제 차원 불일치 경고
    try:
        emb_dim = int(embeddings.dims())
    except Exception:
        emb_dim = EMBED_DIM_IMAGE
    if emb_dim != EMBED_DIM_IMAGE:
        # 일단 경고만: 인덱스는 설정값으로 생성되며, add_item에서 shape 불일치 시 예외
        print(f"[faiss_index] 경고: settings.EMBEDDING_DIM_IMAGE({EMBED_DIM_IMAGE}) != embeddings.dims()({emb_dim})")


# ------------------------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------------------------
def load_all() -> None:
    """인덱스/메타/ID 맵 로드. ID 수와 인덱스 엔트리 수가 다르면 보정."""
    global IMAGE_INDEX, META, IDMAP_IMAGE

    _ensure_dim_consistency()
    with INDEX_LOCK:
        IMAGE_INDEX = _load_index(IMAGE_INDEX_PATH, EMBED_DIM_IMAGE)

    META = json.loads(META_PATH.read_text("utf-8")) if META_PATH.exists() else {}
    IDMAP_IMAGE = json.loads(IDMAP_IMAGE_PATH.read_text("utf-8")) if IDMAP_IMAGE_PATH.exists() else []

    # 일관성 보정
    with INDEX_LOCK:
        ntotal = getattr(IMAGE_INDEX, "ntotal", 0)
    if ntotal != len(IDMAP_IMAGE):
        print(f"[faiss_index] 경고: 인덱스(ntotal={ntotal}) != IDMAP_IMAGE(len={len(IDMAP_IMAGE)}). "
                f"불일치 항목은 잘라냅니다.")
        # 더 짧은 쪽 기준으로 맞춤
        new_len = min(ntotal, len(IDMAP_IMAGE))
        if faiss is not None and ntotal > new_len:
            # FAISS는 중간 삭제가 까다로워서, 간단히 앞부분만 유지하는 방법을 사용하려면
            # 인덱스를 재구축해야 합니다. 여기서는 IDMAP만 잘라 동기화합니다.
            pass
        IDMAP_IMAGE[:] = IDMAP_IMAGE[:new_len]


def save_all() -> None:
    """인덱스/메타/ID 맵 저장(원자적 저장)."""
    with INDEX_LOCK:
        if IMAGE_INDEX is None:
            return
        _atomic_write_index(IMAGE_INDEX, IMAGE_INDEX_PATH)
        _atomic_write_text(META_PATH, json.dumps(META, ensure_ascii=False, indent=2))
        _atomic_write_text(IDMAP_IMAGE_PATH, json.dumps(IDMAP_IMAGE, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------------------------
# Core API
# ------------------------------------------------------------------------------------
def add_item(item_id: str, image_vec, text_vec, meta: dict) -> None:
    """이미지 벡터를 인덱스에 추가하고 메타를 갱신합니다. (text_vec은 폐기 경로)"""
    # 주: embeddings는 보통 L2 정규화되어 오지만, 안전하게 한 번 더 보장합니다.
    with INDEX_LOCK:
        if image_vec is not None:
            arr = np.asarray(image_vec, dtype="float32")
            if arr.ndim == 1:
                d = arr.shape[0]
                arr = arr.reshape(1, d)
            else:
                if arr.shape[0] != 1:
                    raise ValueError("add_item: image_vec는 (d,) 또는 (1, d) 이어야 합니다.")
                d = arr.shape[1]

            if d != EMBED_DIM_IMAGE:
                raise ValueError(f"add_item: 벡터 차원 불일치 d={d} != EMBED_DIM_IMAGE={EMBED_DIM_IMAGE}")

            arr = _l2_normalize(arr)  # Cosine 목적

            if IMAGE_INDEX is None:
                raise RuntimeError("IMAGE_INDEX가 초기화되지 않았습니다. load_all()을 호출했나요?")
            IMAGE_INDEX.add(arr)
            IDMAP_IMAGE.append(item_id)

        META[item_id] = meta


def search_image(query_vec, k: int = 5) -> List[Tuple[str, float, dict]]:
    """쿼리 벡터로 유사 이미지 검색."""
    with INDEX_LOCK:
        if IMAGE_INDEX is None or IMAGE_INDEX.ntotal == 0:
            return []
        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != EMBED_DIM_IMAGE:
            raise ValueError(f"search_image: 벡터 차원 불일치 qdim={q.shape[1]} != EMBED_DIM_IMAGE={EMBED_DIM_IMAGE}")
        q = _l2_normalize(q)

        ntotal = IMAGE_INDEX.ntotal
        k = max(1, min(int(k), ntotal))

        scores, idxs = IMAGE_INDEX.search(q, k)
        out: List[Tuple[str, float, dict]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if idx >= len(IDMAP_IMAGE):
                # 불일치 방어
                continue
            item_id = IDMAP_IMAGE[idx]
            meta = META.get(item_id, {})
            out.append((item_id, float(score), meta))
        return out


def search_text(*_args, **_kwargs):
    """유지: 기존 API 호환(항상 빈 결과)."""
    return []


def needs_reindex(current_version: str, provider: str) -> bool:
    """현재 메타에 기록된 버전/프로바이더와 실행중 실제 설정이 다른지 확인."""
    return (current_version != settings.EMBEDDING_VERSION) or (provider != embeddings.current_provider())


def reindex_all(force: bool = False) -> bool:
    """
    버전/프로바이더 상이 시 재색인.
    - 원본 이미지/벡터 없이 재생성 불가하면, 기본은 **메타만 업그레이드**하고 인덱스는 보존.
    - force=True 이면서 벡터/이미지 부재 시 → 명시적 예외로 실패 처리(데이터 손실 방지).
    반환값: 실제로 무언가 갱신되었으면 True, 없으면 False.
    """
    # 재색인 필요성 검사(메타를 한 바퀴 스캔)
    mismatches = []
    for m in META.values():
        ver = m.get("embedding_version", "")
        prov = m.get("embedding_provider", "hash")
        if needs_reindex(ver, prov):
            mismatches.append(True)
    if not mismatches:
        return False  # 아무것도 안 함

    # 현재 구조상 원본 이미지/벡터를 보관하지 않으므로 '완전 재색인'은 불가
    # → 메타만 현재 버전/프로바이더로 업데이트하고 저장
    if not force:
        for k in list(META.keys()):
            META[k]["embedding_version"] = settings.EMBEDDING_VERSION
            META[k]["embedding_provider"] = embeddings.current_provider()
        save_all()
        return True

    # force=True: 벡터 재생성 필요하지만 소스가 없으므로 실패 처리
    raise RuntimeError(
        "reindex_all(force=True): 원본 이미지/벡터가 없어서 인덱스를 재구축할 수 없습니다. "
        "데이터 소스를 제공하거나 meta-only 업그레이드를 사용하세요."
    )


# 모듈 import 시 자동 로드
try:
    load_all()
except Exception as e:
    print("[faiss_index] 초기화 실패:", e)