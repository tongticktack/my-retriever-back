from typing import List, Dict
from datetime import datetime, timezone
import uuid

from firebase_admin import firestore

_db = None

def get_db():
    global _db
    if _db is None:
        _db = firestore.client()
    return _db

# Firestore 구조
# chat_sessions/{session_id}  { created_at, last_active_at }
# chat_sessions/{session_id}/messages/{message_id}  { role, content, created_at }


def create_session() -> str:
    db = get_db()
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.collection("chat_sessions").document(session_id).set({
        "created_at": now.isoformat(),
        "last_active_at": now.isoformat(),
    })
    return session_id


def add_message(session_id: str, role: str, content: str) -> str:
    """
    Add a message to a chat session with:
    - Content validation (1..1000 chars after trim)
    - Single Firestore transaction for existence check + write + last_active update
    - Server-side timestamp (prevents client/server clock drift)

    Raises ValueError with codes:
      session_not_found | empty_content | content_too_long
    """
    db = get_db()
    trimmed = content.strip() if content is not None else ""
    if not trimmed:
        raise ValueError("empty_content")
    if len(trimmed) > 1000:
        raise ValueError("content_too_long")

    msg_id = str(uuid.uuid4())
    session_ref = db.collection("chat_sessions").document(session_id)
    msg_ref = session_ref.collection("messages").document(msg_id)

    def txn_op(transaction):
        snap = transaction.get(session_ref)
        if not snap.exists:
            raise ValueError("session_not_found")
        # Prepare server timestamp sentinel
        server_ts = firestore.SERVER_TIMESTAMP
        transaction.set(msg_ref, {
            "role": role,
            "content": trimmed,
            "created_at": server_ts,  # Firestore Timestamp
        })
        transaction.update(session_ref, {"last_active_at": server_ts})

    db.transaction()(txn_op)  # Execute transaction
    return msg_id


def fetch_messages(session_id: str, limit: int = 50) -> List[Dict]:
    db = get_db()
    session_ref = db.collection("chat_sessions").document(session_id)
    if not session_ref.get().exists:
        raise ValueError("session_not_found")
    docs = (session_ref.collection("messages")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit).stream())
    messages = []
    for d in docs:
        data = d.to_dict() or {}
        created_raw = data.get("created_at")
        # created_at may be Firestore Timestamp or already string
        if isinstance(created_raw, datetime):
            created_str = created_raw.isoformat()
        else:
            created_str = str(created_raw) if created_raw is not None else ""
        messages.append({
            "id": d.id,
            "role": data.get("role"),
            "content": data.get("content"),
            "created_at": created_str,
        })
    return list(reversed(messages))
