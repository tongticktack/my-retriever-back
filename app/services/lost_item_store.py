from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timezone

from . import chat_store  # reuse Firestore client

def _collection():
    return chat_store.get_db().collection("lost_items")


def upsert_lost_item(user_id: str, session_id: str, item_index: int, item: Dict):
    """Create or update a lost item document.
    Doc id pattern: sessionId-index (stable & deterministic).
    Fields stored: user_id, session_id, item_index, stage, extracted, missing, created_at, updated_at.
    """
    if user_id is None:
        user_id = "guest"
    doc_id = f"{session_id}-{item_index}"
    col = _collection()
    ref = col.document(doc_id)
    snap = ref.get()
    now = datetime.now(timezone.utc)
    base = {
        "user_id": user_id,
        "session_id": session_id,
        "item_index": item_index,
        "stage": item.get("stage"),
        "extracted": item.get("extracted") or {},
        "missing": item.get("missing") or [],
    "media_ids": item.get("media_ids") or [],
    "sources": item.get("sources") or {},
        "updated_at": now,
    }
    if not snap.exists:
        base["created_at"] = now
        ref.set(base)
    else:
        ref.update(base)


def bulk_upsert(user_id: str, session_id: str, items: List[Dict]):
    for idx, it in enumerate(items):
        try:
            upsert_lost_item(user_id, session_id, idx, it)
        except Exception as e:
            print(f"[lost_item_store] upsert failed idx={idx}: {e}")


def list_user_items(user_id: str, limit: int = 50, status: Optional[str] = None) -> List[Dict]:
    db = chat_store.get_db()
    col = db.collection("lost_items")
    q = col.where("user_id", "==", user_id).order_by("updated_at", direction=chat_store.firestore.Query.DESCENDING)  # type: ignore
    if status:
        # Firestore requires composite index for (user_id, status, updated_at) if added
        q = q.where("stage", "==", status)
    q = q.limit(limit)
    results: List[Dict] = []
    for doc in q.stream():
        d = doc.to_dict() or {}
        # normalize timestamps to iso strings
        for f in ["created_at", "updated_at"]:
            v = d.get(f)
            if hasattr(v, 'isoformat'):
                d[f] = v.isoformat()
            elif v is None:
                d[f] = None
            else:
                d[f] = str(v)
        d["id"] = doc.id
        results.append(d)
    return results
