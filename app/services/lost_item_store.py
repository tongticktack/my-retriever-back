from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timezone

from . import chat_store  # reuse Firestore client


def _collection():
    return chat_store.get_db().collection("lost_items")



def upsert_lost_item(user_id: str, item_index: int, item: Dict):
    """Update items array in user_id doc; create if missing."""
    if user_id is None:
        user_id = "guest"
    col = _collection()
    ref = col.document(user_id)
    snap = ref.get()
    now = datetime.now(timezone.utc)
    # Prepare item payload
    item_payload = {
        "item_index": item_index,
        "stage": item.get("stage"),
        "extracted": item.get("extracted") or {},
        "missing": item.get("missing") or [],
        "media_ids": item.get("media_ids") or [],
        "sources": item.get("sources") or {},
        "updated_at": now,
    }
    if not snap.exists:
        ref.set({
            "user_id": user_id,
            "items": [item_payload],
            "created_at": now,
            "updated_at": now,
        })
    else:
        doc = snap.to_dict() or {}
        items = doc.get("items") or []
        # Replace or append item by item_index
        replaced = False
        for idx, it in enumerate(items):
            if it.get("item_index") == item_index:
                items[idx] = item_payload
                replaced = True
                break
        if not replaced:
            items.append(item_payload)
        ref.update({"items": items, "updated_at": now})



def bulk_upsert(user_id: str, items: List[Dict]):
    for idx, it in enumerate(items):
        try:
            upsert_lost_item(user_id, idx, it)
        except Exception as e:
            print(f"[lost_item_store] upsert failed idx={idx}: {e}")



def list_user_items(user_id: str, limit: int = 50, status: Optional[str] = None) -> List[Dict]:
    db = chat_store.get_db()
    col = db.collection("lost_items")
    ref = col.document(user_id)
    snap = ref.get()
    results: List[Dict] = []
    if not snap.exists:
        return results
    doc = snap.to_dict() or {}
    items = doc.get("items") or []
    # Optionally filter by status
    if status:
        items = [it for it in items if it.get("stage") == status]
    # Sort by updated_at desc
    items.sort(key=lambda it: it.get("updated_at", datetime.min), reverse=True)
    for it in items[:limit]:
        # normalize timestamps to iso strings
        for f in ["updated_at"]:
            v = it.get(f)
            if hasattr(v, 'isoformat'):
                it[f] = v.isoformat()
            elif v is None:
                it[f] = None
            else:
                it[f] = str(v)
        results.append(it)
    return results
