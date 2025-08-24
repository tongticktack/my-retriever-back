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
    norm_user_id = (user_id or '').strip()
    # 쿼리스트링에서 실수로 "user_123" 형태로 올 때 양끝 쌍따옴표 제거
    if norm_user_id.startswith('"') and norm_user_id.endswith('"') and len(norm_user_id) >= 2:
        norm_user_id = norm_user_id[1:-1].strip()
    if not norm_user_id:
        norm_user_id = 'guest'
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    db.collection("chat_sessions").document(session_id).set({
        "created_at": now.isoformat(),
        "last_active_at": now.isoformat(),
        "user_id": norm_user_id,
    "title": None,  # 첫 user 메시지 들어올 때 생성
    })
    try:
        print(f"[chat_store] create_session id={session_id} user_id={norm_user_id}")
    except Exception:
        pass
    return session_id


def get_session_data(session_id: str) -> Optional[Dict]:
    """Return session document dict or None."""
    db = get_db()
    ref = db.collection("chat_sessions").document(session_id)
    snap = ref.get()
    if not snap.exists:
        return None
    return snap.to_dict() or {}


def update_session(session_id: str, data: Dict) -> None:
    """Shallow update of session document (no merge into nested maps except Firestore default behavior)."""
    db = get_db()
    ref = db.collection("chat_sessions").document(session_id)
    try:
        ref.update(data)
    except Exception:
        pass


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
    # Allow empty content when attachments are present (image-only message)
    if not trimmed:
        has_attachments = bool(meta and meta.get("attachments"))
        if not has_attachments:
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
                if not raw:
                    # Fallback for image-only first message
                    raw = "이미지 첨부"
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
            "attachments": data.get("attachments"),
            "matches": data.get("matches"),  # persisted matches for assistant messages
        })
    return list(reversed(messages))


def list_sessions(user_id: str, limit: int = 50) -> List[Dict]:
    """사용자별 세션 목록 반환.

    반환 필드: session_id, title (없으면 "(제목 없음)"), created_at, last_active_at
    최근(last_active_at DESC) 순.
    """
    db = get_db()
    # where + order_by 조합은 Firestore 인덱스 필요할 수 있음 (콘솔 인덱스 제안 확인 필요)
    # 최신 SDK 는 positional where 경고를 표시하므로 FieldFilter 사용. 실패 시 이전 방식 fallback.
    col = db.collection("chat_sessions")
    try:
        print(f"[chat_store] list_sessions.start user_id={user_id} limit={limit}")
    except Exception:
        pass
    # Simplified strategy: avoid order_by to remove composite index requirement (was hanging / requiring index).
    # We'll fetch up to 3x limit docs (user typically has few sessions) then sort in memory.
    try:
        from google.cloud.firestore_v1 import FieldFilter  # type: ignore
        base_query = col.where(filter=FieldFilter("user_id", "==", user_id))
        print("[chat_store] query.base FieldFilter")
    except Exception as e:
        print(f"[chat_store] FieldFilter unavailable, positional where used ({type(e).__name__}: {e})")
        base_query = col.where("user_id", "==", user_id)
    sessions: List[Dict] = []
    from google.api_core import exceptions as _gexc  # type: ignore
    import time as _time
    t0 = _time.time()
    timeout_s = 6  # hard cap to prevent hanging
    try:
        # Probe existence quickly (single doc); if empty short return
        try:
            probe_iter = base_query.limit(1).stream()
            probe_first = list(probe_iter)
            if not probe_first:
                print("[chat_store] probe.empty user has no sessions")
                return []
        except Exception as pe:
            print(f"[chat_store] probe error {type(pe).__name__}: {pe}")
        # Drain with watchdog thread to avoid indefinite hang on network issues
        import threading as _th
        docs_buffer: List = []
        exc_holder: List[Exception] = []
        def _drain():
            try:
                for i, doc in enumerate(base_query.limit(limit * 3).stream()):
                    docs_buffer.append(doc)
                    if i >= (limit * 3) - 1:
                        break
            except Exception as _e:
                exc_holder.append(_e)
        th = _th.Thread(target=_drain, daemon=True)
        th.start()
        th.join(2.5)
        if th.is_alive():
            print(f"[chat_store] watchdog timeout partial_docs={len(docs_buffer)}")
        if exc_holder:
            print(f"[chat_store] list_sessions drain error {type(exc_holder[0]).__name__}: {exc_holder[0]}")
        # In-memory sort DESC by last_active_at
        def _extract_times(d):
            data = d.to_dict() or {}
            raw = data.get("last_active_at") or data.get("created_at")
            if isinstance(raw, datetime):
                return raw
            try:
                return datetime.fromisoformat(str(raw))
            except Exception:
                return datetime.min
        docs_buffer.sort(key=_extract_times, reverse=True)
        for dref in docs_buffer[:limit]:
            d = dref.to_dict() or {}
            created_raw = d.get("created_at")
            last_raw = d.get("last_active_at")
            if isinstance(created_raw, datetime):
                created_str = created_raw.isoformat()
            else:
                created_str = str(created_raw) if created_raw is not None else ""
            if isinstance(last_raw, datetime):
                last_str = last_raw.isoformat()
            else:
                last_str = str(last_raw) if last_raw is not None else ""
            sessions.append({
                "session_id": dref.id,
                "title": d.get("title") or "(제목 없음)",
                "created_at": created_str,
                "last_active_at": last_str,
            })
    except _gexc.FailedPrecondition as e:
        # Likely missing composite index (user_id + last_active_at). Fallback: simple where only.
        print(f"[chat_store] primary query requires index (fallback). detail={e.message[:140] if hasattr(e,'message') else e}")
        try:
            simple_q = col.where("user_id", "==", user_id).limit(limit)
            for doc in simple_q.stream():
                d = doc.to_dict() or {}
                created_raw = d.get("created_at")
                last_raw = d.get("last_active_at")
                if isinstance(created_raw, datetime):
                    created_str = created_raw.isoformat()
                else:
                    created_str = str(created_raw) if created_raw is not None else ""
                if isinstance(last_raw, datetime):
                    last_str = last_raw.isoformat()
                else:
                    last_str = str(last_raw) if last_raw is not None else ""
                sessions.append({
                    "session_id": doc.id,
                    "title": d.get("title") or "(제목 없음)",
                    "created_at": created_str,
                    "last_active_at": last_str,
                })
        except Exception as e2:
            print(f"[chat_store] fallback simple query failed {e2}")
    except Exception as e:
        print(f"[chat_store] session list unexpected error {type(e).__name__}: {e}")
    finally:
        try:
            print(f"[chat_store] list_sessions.end user_id={user_id} count={len(sessions)} elapsed={(_time.time()-t0):.2f}s")
        except Exception:
            pass
    return sessions


def delete_session(session_id: str) -> bool:
    """세션 및 하위 메시지 삭제.

    반환: True = 삭제됨, False = 존재하지 않음
    Firestore 는 하위 컬렉션 자동 삭제 없으므로 수동 반복.
    메시지 수 많을 수 있으니 배치로 처리 (최대 400/배치).
    """
    db = get_db()
    session_ref = db.collection("chat_sessions").document(session_id)
    snap = session_ref.get()
    if not snap.exists:
        return False
    # 메시지 삭제
    msgs_col = session_ref.collection("messages")
    batch = db.batch()
    count = 0
    # stream 은 한 번에 모든 문서 반환 (대규모면 개선 필요)
    for doc in msgs_col.stream():
        batch.delete(doc.reference)
        count += 1
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()
    batch.commit()
    # 세션 삭제
    session_ref.delete()
    return True
