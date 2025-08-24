from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from app.services import lost_item_store

from app.services import lost_item_extractor as extractor

router = APIRouter(prefix="/lost", tags=["lost-item"])

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    extracted: Dict[str, str]
    missing: list[str]
    next_prompt: str

class UserLostItem(BaseModel):
    id: str
    session_id: str
    item_index: int
    stage: str
    extracted: Dict[str, str]
    missing: List[str]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ListUserItemsResponse(BaseModel):
    user_id: str
    items: List[UserLostItem]

class MarkFoundRequest(BaseModel):
    user_id: str
    item_index: int
    found_match_id: Optional[str] = None  # 매칭된 공개 습득물 ID(optional)
    note: Optional[str] = None

class MarkFoundResponse(BaseModel):
    user_id: str
    item_index: int
    is_found: bool
    found_at: str

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    extracted = extractor.analyze_text(req.text)
    missing = extractor.compute_missing(extracted)
    next_prompt = extractor.build_missing_field_prompt(extracted, missing)
    return AnalyzeResponse(extracted=extracted, missing=missing, next_prompt=next_prompt)


@router.get("/user-items", response_model=ListUserItemsResponse)
def list_user_items(user_id: str, limit: int = 50, stage: Optional[str] = None):
    norm = (user_id or '').strip() or 'guest'
    items_raw = lost_item_store.list_user_items(norm, limit=limit, status=stage)
    items = [UserLostItem(
        id=i.get('id'),
        session_id=i.get('session_id'),
        item_index=i.get('item_index'),
        stage=i.get('stage'),
        extracted=i.get('extracted') or {},
        missing=i.get('missing') or [],
        created_at=i.get('created_at'),
        updated_at=i.get('updated_at'),
    ) for i in items_raw]
    return ListUserItemsResponse(user_id=norm, items=items)


@router.post("/mark-found", response_model=MarkFoundResponse)
def mark_found(req: MarkFoundRequest):
    norm = (req.user_id or '').strip() or 'guest'
    updated = lost_item_store.mark_item_found(norm, req.item_index, match_id=req.found_match_id, note=req.note)
    if not updated:
        raise HTTPException(status_code=404, detail="not_found")
    return MarkFoundResponse(
        user_id=norm,
        item_index=req.item_index,
        is_found=True,
        found_at=updated.get('found_at'),
    )
