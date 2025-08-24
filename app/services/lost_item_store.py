from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timezone
import uuid
from . import chat_store  # reuse Firestore client
from app.scripts.logging_config import get_logger

logger = get_logger("lost_item_store")


def _collection():
    return chat_store.get_db().collection("lost_items")



def append_item(user_id: str, item: Dict) -> Dict:
    """Always append a new lost item. Ignores any incoming item_index; assigns next sequential index."""
    if user_id is None:
        user_id = "guest"
    col = _collection()
    ref = col.document(user_id)
    snap = ref.get()
    now = datetime.now(timezone.utc)
    # Prepare item payload
    # Default is_found False if not present
    is_found = item.get("is_found")
    if is_found is None:
        is_found = False
    existing_id = item.get("id")
    item_payload = {
        "item_index": -1,  # placeholder until assigned
        "id": existing_id or uuid.uuid4().hex,
        "stage": item.get("stage"),
        "extracted": item.get("extracted") or {},
        "missing": item.get("missing") or [],
        "media_ids": item.get("media_ids") or [],
        "sources": item.get("sources") or {},
        "is_found": is_found,
        "updated_at": now,
    }
    if not snap.exists:
        item_payload["item_index"] = 0
        ref.set({
            "user_id": user_id,
            "items": [item_payload],
            "created_at": now,
            "updated_at": now,
        })
        try:
            logger.info("firestore.write op=set doc=lost_items/%s items_count=%d appended=1", user_id, 1)
        except Exception:
            pass
        return item_payload
    doc = snap.to_dict() or {}
    items = doc.get("items") or []
    # Assign sequential next index
    try:
        next_index = max(it.get("item_index", -1) for it in items) + 1 if items else 0
    except Exception:
        next_index = len(items)
    item_payload["item_index"] = next_index
    items.append(item_payload)
    ref.update({"items": items, "updated_at": now})
    try:
        logger.info("firestore.write op=update doc=lost_items/%s items_count=%d appended=1", user_id, len(items))
        try:
            summary = [f"{it.get('item_index')}:{it.get('stage')}:{(it.get('id') or '')[:6]}" for it in items]
            logger.debug("lost_item_store.items_summary user=%s items=%s", user_id, summary)
        except Exception:
            pass
    except Exception:
        pass
    return item_payload



def append_many(user_id: str, items: List[Dict]):
    for it in items:
        try:
            append_item(user_id, it)
        except Exception as e:
            print(f"[lost_item_store] append failed: {e}")



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


def mark_item_found(user_id: str, item_index: int, match_id: Optional[str] = None, note: Optional[str] = None) -> Optional[Dict]:
    db = chat_store.get_db()
    ref = db.collection("lost_items").document(user_id)
    snap = ref.get()
    if not snap.exists:
        return None
    doc = snap.to_dict() or {}
    items = doc.get("items") or []
    target = None
    for it in items:
        if it.get("item_index") == item_index:
            target = it
            break
    if not target:
        return None
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    target["is_found"] = True
    target["found_at"] = now
    if match_id:
        target["found_match_id"] = match_id
    if note:
        target["found_note"] = note
    ref.update({"items": items, "updated_at": now})
    # normalize before return
    result = target.copy()
    if hasattr(result.get("found_at"), 'isoformat'):
        result["found_at"] = result["found_at"].isoformat()
    return result
