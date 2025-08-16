from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services import chat_store
from app.services.llm_providers import get_llm
from app.services.prompt_builder import build_messages
from config import settings

router = APIRouter(prefix="/chat", tags=["chat"])

class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None

class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None

class SendMessageRequest(BaseModel):
    session_id: str
    content: str
    user_id: Optional[str] = None  # (미인증 상태라 사용 안하지만 필드 예약)

class Message(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    model: str | None = None

class SendMessageResponse(BaseModel):
    user_message: Message
    assistant_message: Message
    session_id: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Message]

class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: Optional[str] = None
    last_active_at: Optional[str] = None

class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]

@router.post("/session", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    session_id = chat_store.create_session(user_id=req.user_id)
    return CreateSessionResponse(session_id=session_id, user_id=req.user_id)

@router.post("/send", response_model=SendMessageResponse)
def send_message(req: SendMessageRequest):
    try:
        user_msg_id = chat_store.add_message(req.session_id, "user", req.content)
    except ValueError as e:
        code = str(e)
        if code == "session_not_found":
            raise HTTPException(status_code=404, detail=code)
        if code == "empty_content":
            raise HTTPException(status_code=400, detail=code)
        if code == "content_too_long":
            raise HTTPException(status_code=413, detail=code)
        raise HTTPException(status_code=400, detail={"error": "invalid_request", "raw": code})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # LLM 호출
    llm = get_llm()
    provider_messages = build_messages(req.session_id, req.content, retrieval_items=None, law_summary=None)
    try:
        assistant_reply = llm.generate(provider_messages)
    except Exception as e:
        assistant_reply = f"(llm-error) {str(e)[:120]}"
    chosen_model = getattr(llm, "last_model_name", None) or getattr(llm, "model", None) or getattr(llm, "name", "unknown")
    assistant_msg_id = chat_store.add_message(
        req.session_id,
        "assistant",
        assistant_reply,
        meta={"model": chosen_model}
    )

    # 최신 히스토리 일부 가져와 user 메시지 timestamp 포함 (limit=2 충분)
    try:
        recent = chat_store.fetch_messages(req.session_id, limit=2)
    except ValueError:
        recent = []
    created_at = ""
    for m in reversed(recent):  # 보존 순서
        if m.get("id") == user_msg_id:
            created_at = m.get("created_at", "")
            break

    # Get assistant created_at (may need refresh, keep simple by refetching last 2)
    assistant_created_at = ""
    try:
        recent2 = chat_store.fetch_messages(req.session_id, limit=2)
        for m in reversed(recent2):
            if m.get("id") == assistant_msg_id:
                assistant_created_at = m.get("created_at", "")
                break
    except Exception:
        pass
    return SendMessageResponse(
        session_id=req.session_id,
    user_message=Message(id=user_msg_id, role="user", content=req.content, created_at=created_at, model=None),
    assistant_message=Message(id=assistant_msg_id, role="assistant", content=assistant_reply, created_at=assistant_created_at, model=chosen_model),
    )

@router.get("/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str, limit: int = 50):
    try:
        msgs = chat_store.fetch_messages(session_id, limit=limit)
    except ValueError:
        raise HTTPException(404, "session_not_found")
    return HistoryResponse(
        session_id=session_id,
        messages=[Message(**m) for m in msgs]
    )


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(user_id: str, limit: int = 50):
    if not user_id:
        raise HTTPException(400, "missing_user_id")
    sessions_raw = chat_store.list_sessions(user_id=user_id, limit=limit)
    summaries = [SessionSummary(**s) for s in sessions_raw]
    return SessionListResponse(sessions=summaries)
