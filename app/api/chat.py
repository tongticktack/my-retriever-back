from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services import chat_store
from app.services.llm_providers import get_llm
from app.services.prompt_builder import build_messages
from config import settings

router = APIRouter(prefix="/chat", tags=["chat"])

class CreateSessionResponse(BaseModel):
    session_id: str

class SendMessageRequest(BaseModel):
    session_id: str
    content: str

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

@router.post("/session", response_model=CreateSessionResponse)
def create_session():
    session_id = chat_store.create_session()
    return CreateSessionResponse(session_id=session_id)

@router.post("/send", response_model=SendMessageResponse)
def send_message(req: SendMessageRequest):
    # 사용자 메시지 저장 (유효성/트랜잭션 내부 처리)
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
        # 미확인 코드 → 디버깅 도움 위해 raw 포함
        raise HTTPException(status_code=400, detail={"error": "invalid_request", "raw": code})
    except RuntimeError as e:
        # 내부 Firestore 트랜잭션 실패
        raise HTTPException(status_code=500, detail=str(e))

    # LLM 호출
    llm = get_llm()
    # Build full prompt messages (retrieval context placeholders currently None)
    provider_messages = build_messages(req.session_id, req.content, retrieval_items=None, law_summary=None)
    try:
        assistant_reply = llm.generate(provider_messages)
    except Exception as e:
        assistant_reply = f"(llm-error) {str(e)[:120]}"
    # Determine model meta (auto provider may have last_model_name)
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
