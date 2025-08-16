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
# chat_sessions/{session_id}  { created_at, last_active_at, user_id, title }
# chat_sessions/{session_id}/messages/{message_id}  { role, content, created_at }


def create_session(user_id: Optional[str] = None) -> str:
    db = get_db()
    # guest 표준화: None/빈문자열 모두 'guest' 저장
    norm_user_id = (user_id or '').strip() or 'guest'
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.collection("chat_sessions").document(session_id).set({
        "created_at": now.isoformat(),
        "last_active_at": now.isoformat(),
        "user_id": norm_user_id,
    "title": None,  # 첫 user 메시지 들어올 때 생성
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
        for k, v in meta.items():
            if k not in payload:
                payload[k] = v
    msg_ref.set(payload)
    # 마지막 활동 시각 + (필요 시) 타이틀 업데이트
    updates = {"last_active_at": server_ts}
    # 첫 user 메시지일 때 title 설정
    if role == "user":
        try:
            snap_after = session_ref.get()
            data_after = snap_after.to_dict() or {}
            if not data_after.get("title"):
                # LLM 기반 타이틀 생성 시도 (실패하면 fallback)
                from .title_generator import generate_session_title  # 지연 import
                raw = trimmed.replace("\n", " ").strip()
                title = generate_session_title(raw)
                updates["title"] = title
        except Exception:
            # 실패 시 title 생략 (None 유지)
            pass
    session_ref.update(updates)
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


def list_sessions(user_id: str, limit: int = 50) -> List[Dict]:
    """사용자별 세션 목록 반환.

    반환 필드: session_id, title (없으면 "(제목 없음)"), created_at, last_active_at
    최근(last_active_at DESC) 순.
    """
    db = get_db()
    # where + order_by 조합은 Firestore 인덱스 필요할 수 있음 (배포 시 콘솔에서 제안 수락)
    q = (db.collection("chat_sessions")
        .where("user_id", "==", user_id)
        .order_by("last_active_at", direction=firestore.Query.DESCENDING)
        .limit(limit))
    sessions: List[Dict] = []
    for doc in q.stream():
        d = doc.to_dict() or {}
        sessions.append({
            "session_id": doc.id,
            "title": d.get("title") or "(제목 없음)",
            "created_at": d.get("created_at"),
            "last_active_at": d.get("last_active_at"),
        })
    return sessions
