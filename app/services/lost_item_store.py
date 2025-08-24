from __future__ import annotations
from typing import Dict, List, Optional
from datetime import datetime, timezone
import random
from . import chat_store  # reuse Firestore client
from app.scripts.logging_config import get_logger

logger = get_logger("lost_item_store")


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
    # Default is_found False if not present
    is_found = item.get("is_found")
    if is_found is None:
        is_found = False
    item_payload = {
        "item_index": item_index,
        "id": random.uuid4().hex,
        "stage": item.get("stage"),
        "extracted": item.get("extracted") or {},
        "missing": item.get("missing") or [],
        "media_ids": item.get("media_ids") or [],
        "sources": item.get("sources") or {},
        "is_found": is_found,
        "updated_at": now,
    }
    if not snap.exists:
        ref.set({
            "user_id": user_id,
            "items": [item_payload],
            "created_at": now,
            "updated_at": now,
        })
        try:
            logger.info("firestore.write op=set doc=lost_items/%s items_count=%d", user_id, 1)
        except Exception:
            pass
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
        try:
            logger.info(
                "firestore.write op=update doc=lost_items/%s items_count=%d replaced=%s",
                user_id, len(items), replaced
            )
        except Exception:
            pass



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
