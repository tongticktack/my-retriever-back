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
    db = get_db()
    msg_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    session_ref = db.collection("chat_sessions").document(session_id)
    if not session_ref.get().exists:
        raise ValueError("session_not_found")
    session_ref.collection("messages").document(msg_id).set({
        "role": role,
        "content": content,
        "created_at": now.isoformat(),
    })
    session_ref.update({"last_active_at": now.isoformat()})
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
        data = d.to_dict()
        messages.append({
            "id": d.id,
            "role": data.get("role"),
            "content": data.get("content"),
            "created_at": data.get("created_at"),
        })
    return list(reversed(messages))
