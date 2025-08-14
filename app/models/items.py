from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class LostItemIngest(BaseModel):
    category: str
    found_place: Optional[str] = None
    found_time: Optional[datetime] = None
    notes: Optional[str] = None

class LostItemMeta(BaseModel):
    id: str
    category: str
    caption: str
    found_place: Optional[str] = None
    found_time: Optional[str] = None
    notes: Optional[str] = None
    image_path: Optional[str] = None
    embedding_version: Optional[str] = None
    embedding_provider: Optional[str] = None
    created_at: datetime

class SearchResult(BaseModel):
    id: str
    score: float
    meta: LostItemMeta

class SearchResponse(BaseModel):
    query_type: str  # image | text | hybrid
    results: List[SearchResult]
