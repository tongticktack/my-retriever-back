from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services import chat_store

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

class SendMessageResponse(BaseModel):
    message: Message
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
        raise HTTPException(status_code=400, detail="invalid_request")

    # TODO: Gemini 호출로 교체 (현재는 단순 에코)
    assistant_reply = f"(임시 응답) {req.content[:100]}"
    chat_store.add_message(req.session_id, "assistant", assistant_reply)

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

    return SendMessageResponse(
        session_id=req.session_id,
        message=Message(id=user_msg_id, role="user", content=req.content, created_at=created_at),
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
