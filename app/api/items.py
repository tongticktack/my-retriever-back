from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from datetime import datetime
import uuid
from app.services import faiss_index
from app.services import embeddings
from app.models.items import LostItemIngest, LostItemMeta, SearchResponse, SearchResult

router = APIRouter(prefix="/items", tags=["items"])

@router.post("/ingest")
async def ingest_item(
    category: str = Form(...),
    found_place: str | None = Form(None),
    found_time: str | None = Form(None),
    notes: str | None = Form(None),
    image: UploadFile = File(...),
):
    # 이미지 읽기
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "빈 이미지")

    # 더미 캡션/임베딩 (후에 Gemini + 실제 임베딩 교체)
    caption = f"{category} - 임시 캡션"
    image_vec = embeddings.embed_image(image_bytes)
    text_vec = embeddings.embed_text(caption)

    item_id = str(uuid.uuid4())
    meta = LostItemMeta(
        id=item_id,
        category=category,
        caption=caption,
        found_place=found_place,
        found_time=found_time,
        notes=notes,
        image_path=f"not-stored:{image.filename}",
        embedding_version=faiss_index.EMBEDDING_VERSION,
        embedding_provider=embeddings.current_provider(),
        created_at=datetime.utcnow(),
    )

    faiss_index.add_item(item_id, image_vec, text_vec, meta.model_dump())
    faiss_index.save_all()

    return {"id": item_id, "meta": meta}

@router.post("/search/image", response_model=SearchResponse)
async def search_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "빈 이미지")
    q_vec = embeddings.embed_image(image_bytes)
    raw_results = faiss_index.search_image(q_vec, k=5)
    results = [
        SearchResult(id=item_id, score=score, meta=LostItemMeta(**meta))
        for item_id, score, meta in raw_results
    ]
    return SearchResponse(query_type="image", results=results)


@router.get("/search/text", response_model=SearchResponse)
async def search_text(q: str, k: int = 5):
    if not q.strip():
        raise HTTPException(400, "빈 질의")
    q_vec = embeddings.embed_text(q)
    raw_results = faiss_index.search_text(q_vec, k=k)
    results = [
        SearchResult(id=item_id, score=score, meta=LostItemMeta(**meta))
        for item_id, score, meta in raw_results
    ]
    return SearchResponse(query_type="text", results=results)


@router.get("/search/hybrid", response_model=SearchResponse)
async def search_hybrid(q: str, k: int = 5, alpha: float = 0.5):
    """Simple late-fusion hybrid: alpha * text_score + (1-alpha) * image_score.
    현재는 caption 을 텍스트 embedding 한 것이므로 동일 item ordering 가능성이 높지만
    이후 실제 이미지 임베딩과 결합 시 효과.
    """
    if not (0 <= alpha <= 1):
        raise HTTPException(400, "alpha 0~1 필요")
    if not q.strip():
        raise HTTPException(400, "빈 질의")
    # 텍스트 임베딩
    text_vec = embeddings.embed_text(q)
    text_results = faiss_index.search_text(text_vec, k=k)
    # 이미지 질의 생성: (미디엄) 텍스트 쿼리를 해시 기반 이미지 벡터로 근사 (실제 모델시 cross-modal 지원 필요)
    pseudo_image_vec = embeddings.embed_text(q)  # 임시: 동일 함수 사용
    image_results = faiss_index.search_image(pseudo_image_vec, k=k)

    # 점수 통합
    from collections import defaultdict
    score_map = defaultdict(lambda: {'score': 0.0, 'meta': None})
    for item_id, score, meta in text_results:
        score_map[item_id]['score'] += alpha * score
        score_map[item_id]['meta'] = meta
    for item_id, score, meta in image_results:
        score_map[item_id]['score'] += (1 - alpha) * score
        score_map[item_id]['meta'] = meta
    fused = sorted(score_map.items(), key=lambda x: x[1]['score'], reverse=True)[:k]
    results = [
        SearchResult(id=item_id, score=info['score'], meta=LostItemMeta(**info['meta']))
        for item_id, info in fused
    ]
    return SearchResponse(query_type="hybrid", results=results)
