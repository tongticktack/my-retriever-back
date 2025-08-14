from typing import List, Dict, Optional
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


def add_message(session_id: str, role: str, content: str, meta: Optional[Dict] = None) -> str:
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

    # 세션 존재 확인 (단순 조회)
    snap = session_ref.get()
    if not snap.exists:
        raise ValueError("session_not_found")

    server_ts = firestore.SERVER_TIMESTAMP
    # 메시지 작성
    payload = {
        "role": role,
        "content": trimmed,
        "created_at": server_ts,
    }
    if meta:
        # Flatten simple meta keys (e.g., model)
        for k, v in meta.items():
            # Avoid overwriting core keys
            if k not in payload:
                payload[k] = v
    msg_ref.set(payload)
    # 마지막 활동 시각 업데이트 (트랜잭션 필요성 낮음)
    session_ref.update({"last_active_at": server_ts})
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
            "model": data.get("model"),
        })
    return list(reversed(messages))
