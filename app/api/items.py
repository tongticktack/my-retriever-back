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
    actId: str | None = Form(None),  # external lost-item reference id
    image: UploadFile = File(...),
):
    # 이미지 읽기
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "빈 이미지")

    image_vec = embeddings.embed_image(image_bytes)

    item_id = str(uuid.uuid4())
    meta = LostItemMeta(
        id=item_id,
        actId=actId,
        category=category,
        found_place=found_place,
        found_time=found_time,
        notes=notes,
        image_path=f"not-stored:{image.filename}",
        embedding_version=faiss_index.EMBEDDING_VERSION,
        embedding_provider=embeddings.current_provider(),
        created_at=datetime.utcnow(),
    )

    faiss_index.add_item(item_id, image_vec, None, meta.model_dump())
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


    # 텍스트/하이브리드 검색 엔드포인트 제거됨
