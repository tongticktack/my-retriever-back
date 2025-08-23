# image_indexer.py
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict, Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import firebase_admin
import numpy as np
from aiohttp import ClientResponse
from firebase_admin import credentials

from app.services import chat_store, faiss_index, embeddings
from .logging_config import indexing_logger, log_indexing_event

# Pillow (검증용)
import io
from PIL import Image, UnidentifiedImageError, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
MIN_SIDE = 8

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# 설정 상수
# ------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
USER_AGENT = "Mozilla/5.0 (compatible; LostItemBot/1.1)"
REQ_TIMEOUT_TOTAL = 30
REQ_TIMEOUT_CONNECT = 10
HEAD_TIMEOUT = 8
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15MB
MAX_CONCURRENCY = 8
RETRY_ATTEMPTS = 3
RETRY_BASE_SLEEP = 0.6
EMBED_CACHE_MAX = 5000
FLUSH_EVERY = 100

# ------------------------------------------------------------
# 데이터 모델
# ------------------------------------------------------------
@dataclass
class ItemRow:
    atcId: str
    collection: str
    imageUrl: str
    itemCategory: str = ""
    itemName: str = ""
    foundDate: str = ""
    addr: str = ""


# ------------------------------------------------------------
# 간단한 LRU 캐시 (URL → (vec, meta))
# ------------------------------------------------------------
class LRUCache(OrderedDict):
    def __init__(self, maxsize: int = 1024):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key, last=True)
            return super().get(key)
        return default

    def set(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key, last=True)
        if len(self) > self.maxsize:
            self.popitem(last=False)


# ------------------------------------------------------------
# 처리 상태 (에러/스킵 상세 집계용)
# ------------------------------------------------------------
class ProcStatus(Enum):
    OK = "ok"
    SKIP_INDEXED = "skip_indexed"         # 이미 인덱싱된 atcId
    FAIL_HTTP_404 = "fail_http_404"       # 404 (썸네일 변형 시도 후에도 실패)
    FAIL_DOWNLOAD = "fail_download"       # 그 외 다운로드 실패 (타임아웃 등)
    FAIL_CONTENT_TYPE = "fail_content_type"
    FAIL_TOO_BIG = "fail_too_big"
    FAIL_OPEN = "fail_open"
    FAIL_EMBED = "fail_embed"
    FAIL_FAISS = "fail_faiss"
    FAIL_OTHER = "fail_other"


# ------------------------------------------------------------
# 본 클래스
# ------------------------------------------------------------
class EfficientImageIndexer:
    """효율적인 이미지 인덱싱 시스템"""

    def __init__(self, max_concurrency: int = MAX_CONCURRENCY):
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = indexing_logger
        self.embedding_cache = LRUCache(EMBED_CACHE_MAX)  # URL -> (vec, meta)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._content_hash_seen: set[str] = set()  # 중복 이미지 방지

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=REQ_TIMEOUT_TOTAL, connect=REQ_TIMEOUT_CONNECT)
        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY * 2, ssl=False)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector, headers={"User-Agent": USER_AGENT})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    # --------------------------- 네트워크/유틸 ---------------------------

    def _is_supported_image_url(self, url: str) -> bool:
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower()
        # 확장자 없더라도 통과(후속 MIME 확인)
        return (ext in SUPPORTED_EXTENSIONS) or (ext == "")

    def _looks_like_image(self, content_type: str) -> bool:
        return (content_type or "").lower().strip().startswith("image/")

    def _quick_open_ok(self, data: bytes) -> bool:
        """Pillow로 '열 수 있냐'만 본다. verify()는 쓰지 않음."""
        try:
            with Image.open(io.BytesIO(data)) as im:
                im = ImageOps.exif_transpose(im)
                if getattr(im, "is_animated", False):
                    im.seek(0)
                w, h = im.size
                return (w >= MIN_SIDE and h >= MIN_SIDE)
        except UnidentifiedImageError:
            return False
        except Exception:
            return False

    async def _read_limited(self, resp: ClientResponse, limit_bytes: int = MAX_IMAGE_BYTES) -> Optional[bytes]:
        """응답 바디를 바이트 제한 내에서 읽기."""
        total = 0
        chunks = []
        async for chunk in resp.content.iter_chunked(64 * 1024):
            total += len(chunk)
            if total > limit_bytes:
                self.logger.warning(f"이미지 용량 초과({total} > {limit_bytes})")
                return None
            chunks.append(chunk)
        return b"".join(chunks)

    async def _download_once(self, url: str) -> Tuple[Optional[bytes], Optional[ProcStatus], Optional[int]]:
        """단일 URL 다운로드 시도 → (data, fail_status, http_status)"""
        assert self.session is not None
        try:
            async with self.session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    if resp.status == 404:
                        return None, ProcStatus.FAIL_HTTP_404, 404
                    return None, ProcStatus.FAIL_DOWNLOAD, resp.status
                ctype = resp.headers.get("content-type", "")
                if ctype and not self._looks_like_image(ctype):
                    return None, ProcStatus.FAIL_CONTENT_TYPE, 200
                data = await self._read_limited(resp, MAX_IMAGE_BYTES)
                if not data:
                    return None, ProcStatus.FAIL_TOO_BIG, 200
                if not self._quick_open_ok(data):
                    return None, ProcStatus.FAIL_OPEN, 200
                return data, None, 200
        except Exception:
            return None, ProcStatus.FAIL_DOWNLOAD, None

    async def _download_image_for_embedding(self, url: str, atc_id: Optional[str] = None) -> Tuple[Optional[bytes], Optional[ProcStatus]]:
        """
        임베딩 생성을 위한 이미지 다운로드 (저장하지 않음) + 재시도/백오프/MIME 검증.
        ✅ lost112 404 대응: '/uploadImg/' → '/uploadImg/thumbnail/' 변형 재시도 (+ 시도 로그)
        """
        if not self.session:
            raise RuntimeError("세션이 초기화되지 않았습니다.")

        # URL 1차 필터
        if not self._is_supported_image_url(url):
            return None, ProcStatus.FAIL_DOWNLOAD

        is_lost112 = "lost112.go.kr" in url
        thumb_variant = None
        if is_lost112 and "/uploadImg/" in url:
            thumb_variant = url.replace("/uploadImg/", "/uploadImg/thumbnail/")

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            # 기본 URL 먼저
            data, status, code = await self._download_once(url)
            if data is not None:
                return data, None

            # 404 이고 썸네일 변형 가능하면 1회 바로 시도 (로그 남김)
            if status == ProcStatus.FAIL_HTTP_404 and thumb_variant:
                self.logger.info(f"[404-RETRY] atcId={atc_id}, url={url} → thumb_variant={thumb_variant}")
                data2, status2, code2 = await self._download_once(thumb_variant)
                if data2 is not None:
                    logger.debug(f"404→thumbnail 변형 성공: {thumb_variant}")
                    return data2, None
                # 변형도 실패: status2 / code2를 최종 후보로 사용
                status, code = status2, code2

            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_BASE_SLEEP * (2 ** (attempt - 1)))
            else:
                # 최종 실패
                return None, (status or ProcStatus.FAIL_DOWNLOAD)

        return None, ProcStatus.FAIL_DOWNLOAD  # 논리상 도달하지 않음

    # --------------------------- 임베딩/메타 ---------------------------

    async def _create_embedding(self, url: str, image_data: bytes) -> Tuple[Optional[Tuple[np.ndarray, Dict]], Optional[ProcStatus]]:
        """이미지에서 임베딩 생성 및 메타데이터 반환 (중복 이미지 해시 방지 포함)"""
        try:
            # 콘텐츠 해시(중복 방지)
            content_hash = hashlib.sha256(image_data).hexdigest()
            if content_hash in self._content_hash_seen:
                # 중복 → (요구상 별도 skip 코드가 없으니 기타로 기록)
                return None, ProcStatus.FAIL_OTHER
            self._content_hash_seen.add(content_hash)

            # 임베딩 생성
            vec = embeddings.embed_image(image_data)
            emb_dim = int(vec.shape[-1]) if isinstance(vec, np.ndarray) else len(vec)

            meta = {
                "url": url,
                "image_size": len(image_data),
                "embedding_dim": emb_dim,
                "embedding_version": faiss_index.EMBEDDING_VERSION,
                "embedding_provider": embeddings.current_provider(),
                "created_at": datetime.utcnow().isoformat(),
                "content_sha256": content_hash,
            }
            return (vec, meta), None
        except Exception as e:
            logger.error(f"임베딩 생성 실패: url={url} - {e}")
            return None, ProcStatus.FAIL_EMBED

    # --------------------------- DB 로딩 ---------------------------

    async def get_db_items_with_images(self, limit: Optional[int] = None) -> List[ItemRow]:
        """Firestore에서 이미지 URL이 있는 아이템들을 가져옴"""
        db_start_time = time.time()
        log_indexing_event(self.logger, "db_query_start", {"collections": ["PoliceLostItem", "PortalLostItem"], "start_time": time.time()})

        db = chat_store.get_db()
        all_items: List[ItemRow] = []
        filtered_count = 0
        total_count = 0
        collections = ["PoliceLostItem", "PortalLostItem"]

        for collection_name in collections:
            try:
                col = db.collection(collection_name)
                # 'imageUrl' 존재/비어있지 않음
                docs = col.where("imageUrl", "!=", "").stream()
                collection_items = 0
                collection_filtered = 0

                for doc in docs:
                    total_count += 1
                    data = doc.to_dict() or {}
                    image_url = data.get("imageUrl")
                    if not image_url:
                        continue

                    # URL 기초 필터
                    if not self._is_supported_image_url(image_url):
                        filtered_count += 1
                        collection_filtered += 1
                        continue

                    item = ItemRow(
                        atcId=doc.id,
                        collection=collection_name,
                        imageUrl=image_url,
                        itemCategory=data.get("itemCategory", ""),
                        itemName=data.get("itemName", ""),
                        foundDate=data.get("foundDate", ""),
                        addr=data.get("addr", ""),
                    )
                    all_items.append(item)
                    collection_items += 1

                    if limit and len(all_items) >= limit:
                        break

                logger.info(f"컬렉션 {collection_name}: {collection_items}개 항목 (필터링됨: {collection_filtered}개)")
                if limit and len(all_items) >= limit:
                    break
            except Exception as e:
                logger.error(f"컬렉션 {collection_name} 조회 중 오류: {e}")

        db_duration = time.time() - db_start_time
        log_indexing_event(
            self.logger,
            "db_query_complete",
            {
                "total_processed": total_count,
                "filtered_count": filtered_count,
                "valid_items": len(all_items),
                "duration": db_duration,
                "collections": collections,
            },
        )
        logger.info(f"DB 이미지 조회 완료: 총 {len(all_items)}개 (필터링됨: {filtered_count}개, 소요시간: {db_duration:.2f}초)")

        stats = self.get_index_statistics()
        logger.info(
            "인덱스 통계: 총 %s개, 메타 %s개, idmap %s개, 임베딩 dim %s, 버전 %s, provider %s, 캐시 %s개",
            stats["total_items"],
            stats["meta_count"],
            stats["idmap_count"],
            stats["embedding_dim"],
            stats["embedding_version"],
            stats["embedding_provider"],
            stats["embedding_cache_size"],
        )
        return all_items

    # --------------------------- 인덱싱 파이프라인 ---------------------------

    async def _process_one(self, item: ItemRow) -> Tuple[ProcStatus, Optional[Tuple[str, np.ndarray, Dict]]]:
        """한 건 처리: 다운로드→임베딩"""
        async with self._sem:
            atc_id = item.atcId
            url = item.imageUrl
            try:
                # 이미 인덱싱된 경우 스킵
                if atc_id in faiss_index.META:
                    return ProcStatus.SKIP_INDEXED, None

                # URL 캐시 확인
                cached = self.embedding_cache.get(url)
                if cached:
                    image_vec, meta = cached
                else:
                    # 다운로드 (404면 thumbnail 변형 자동 시도)
                    image_data, dl_status = await self._download_image_for_embedding(url, atc_id=atc_id)
                    if not image_data:
                        self.logger.warning(f"이미지 다운로드 실패: atcId={atc_id}, url={url}, reason={dl_status}")
                        return dl_status or ProcStatus.FAIL_DOWNLOAD, None

                    # 임베딩
                    created, emb_status = await self._create_embedding(url, image_data)
                    if not created:
                        self.logger.warning(f"임베딩 생성 실패: atcId={atc_id}, url={url}, reason={emb_status}")
                        return emb_status or ProcStatus.FAIL_EMBED, None

                    image_vec, meta = created
                    self.embedding_cache.set(url, (image_vec, meta))

                # 메타 확장
                full_meta = {
                    **meta,
                    "atcId": atc_id,
                    "category": item.itemCategory,
                    "found_place": item.addr,
                    "found_time": item.foundDate,
                    "notes": item.itemName,
                    "collection": item.collection,
                }
                return ProcStatus.OK, (atc_id, image_vec, full_meta)

            except Exception as e:
                logger.error(f"아이템 처리 실패: atcId={atc_id}, url={url} - {e}")
                return ProcStatus.FAIL_OTHER, None

    async def index_items_efficiently(self, limit: Optional[int] = None) -> Dict:
        start_time = time.time()
        log_indexing_event(self.logger, "efficient_batch_start", {"limit": limit, "start_time": datetime.now().isoformat()})

        # 성공 기준 limit을 맞추기 위해 넉넉히 가져오기
        fetch_limit = None
        if limit:
            fetch_limit = max(limit * 8, 500)
        items = await self.get_db_items_with_images(fetch_limit)

        if not items:
            logger.warning("인덱싱할 아이템이 없습니다.")
            return {"processed": 0, "success": 0, "error": 0, "skip": 0, "fail_breakdown": {}, "avg_time_per_item": 0.0}

        logger.info(f"효율적 인덱싱 시작: {len(items)}개 아이템")

        tasks = [asyncio.create_task(self._process_one(item)) for item in items]
        processed = 0
        success = 0
        error = 0
        skip = 0
        fail_breakdown: Counter[str] = Counter()
        status_counter: Counter[str] = Counter()  # ✅ ProcStatus 전체 집계

        pending_tasks = set(tasks)
        try:
            for coro in asyncio.as_completed(tasks):
                processed += 1
                status, payload = await coro
                pending_tasks.discard(coro)

                # 상태 카운트 (OK/스킵/실패 전부)
                status_counter[status.value] += 1

                if status == ProcStatus.OK and payload:
                    atc_id, vec, meta = payload
                    try:
                        faiss_index.add_item(atc_id, vec, None, meta)
                        success += 1

                        if success % FLUSH_EVERY == 0:
                            faiss_index.save_all()
                            logger.info(f"[FLUSH] success={success}")

                        if limit and success >= limit:
                            logger.info(f"성공 {success}개로 limit={limit} 도달 → 조기 종료 및 잔여 작업 취소")
                            for t in list(pending_tasks):
                                if not t.done():
                                    t.cancel()
                            if pending_tasks:
                                await asyncio.gather(*pending_tasks, return_exceptions=True)
                            break
                    except Exception as e:
                        error += 1
                        fail_breakdown[ProcStatus.FAIL_FAISS.value] += 1
                        logger.error(f"FAISS 추가 실패: atcId={atc_id} - {e}")
                else:
                    # 스킵/실패 분리 집계
                    if status == ProcStatus.SKIP_INDEXED:
                        skip += 1
                    else:
                        error += 1
                        fail_breakdown[status.value] += 1

                if processed % 50 == 0:
                    logger.info(f"진행률: {processed}/{len(items)} (성공: {success}, 실패: {error}, 스킵: {skip})")
        finally:
            for t in list(pending_tasks):
                if not t.done():
                    t.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

        # flush 보장
        try:
            faiss_index.save_all()
        except Exception as e:
            logger.warning(f"마지막 flush 실패: {e}")

        duration = time.time() - start_time
        denom = max(success + error + skip, 1)
        result = {
            "processed": processed,
            "success": success,
            "error": error,
            "skip": skip,
            "fail_breakdown": dict(fail_breakdown),
            "avg_time_per_item": duration / denom,
        }
        log_indexing_event(self.logger, "efficient_batch_complete", {"result": result, "end_time": datetime.now().isoformat()})

        # ✅ ProcStatus 전체 분포 로그
        logger.info("[STATUS BREAKDOWN] %s", dict(status_counter))

        logger.info(f"효율적 인덱싱 완료: {result}")
        return result

    # --------------------------- 검색/통계 ---------------------------

    async def search_similar_items(self, query_image_data: bytes, top_k: int = 10, min_score: float = 0.5) -> List[Dict]:
        try:
            vec = embeddings.embed_image(query_image_data)
            results = faiss_index.search_image(vec, k=top_k)
            out: List[Dict] = []
            for item_id, score, meta in results:
                if float(score) >= float(min_score):
                    out.append(
                        {
                            "atcId": item_id,
                            "score": float(score),
                            "category": meta.get("category", ""),
                            "itemName": meta.get("notes", ""),
                            "found_place": meta.get("found_place", ""),
                            "found_time": meta.get("found_time", ""),
                            "image_url": meta.get("url", ""),
                            "collection": meta.get("collection", ""),
                        }
                    )
            return out
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            return []

    async def get_image_on_demand(self, image_url: str) -> Optional[bytes]:
        """필요시에만 이미지 다운로드"""
        data, _ = await self._download_image_for_embedding(image_url)
        return data

    def get_index_statistics(self) -> Dict:
        """인덱스 통계 반환 (안전 가드 포함)"""
        image_index = getattr(faiss_index, "IMAGE_INDEX", None)
        total = int(getattr(image_index, "ntotal", 0) or 0)
        stats = {
            "total_items": total,
            "meta_count": len(getattr(faiss_index, "META", {})),
            "idmap_count": len(getattr(faiss_index, "IDMAP_IMAGE", [])),
            "embedding_dim": getattr(faiss_index, "EMBED_DIM_IMAGE", embeddings.dims()),
            "embedding_version": getattr(faiss_index, "EMBEDDING_VERSION", "unknown"),
            "embedding_provider": embeddings.current_provider(),
            "embedding_cache_size": len(self.embedding_cache),
        }
        log_indexing_event(self.logger, "stats_request", stats)
        return stats


# ------------------------------------------------------------
# 편의 함수
# ------------------------------------------------------------
async def index_all_db_images_efficiently(limit: Optional[int] = None) -> Dict:
    async with EfficientImageIndexer() as indexer:
        return await indexer.index_items_efficiently(limit)


async def search_similar_items_efficiently(query_image_data: bytes, top_k: int = 10, min_score: float = 0.5) -> List[Dict]:
    async with EfficientImageIndexer() as indexer:
        return await indexer.search_similar_items(query_image_data, top_k, min_score)


def get_efficient_index_statistics() -> Dict:
    indexer = EfficientImageIndexer()
    return indexer.get_index_statistics()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase-credentials.json")
        firebase_admin.initialize_app(cred)

    parser = argparse.ArgumentParser(description="DB 이미지 인덱싱 즉시 실행")
    parser.add_argument("--limit", type=int, default=None, help="인덱싱할 최대 아이템 수 (기본: 전체)")
    args = parser.parse_args()

    print("[INFO] DB 이미지 인덱싱 즉시 실행 시작...")
    result = asyncio.run(index_all_db_images_efficiently(limit=args.limit))
    try:
        faiss_index.save_all()
    except Exception as e:
        logger.warning(f"종료 flush 실패: {e}")
    print(f"[RESULT] {result}")
    stats = get_efficient_index_statistics()
    print(f"[STATS] {stats}")