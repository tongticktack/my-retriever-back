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
    # 사용자 메시지 저장
    try:
        msg_id = chat_store.add_message(req.session_id, "user", req.content)
    except ValueError:
        raise HTTPException(404, "session_not_found")

    # TODO: Gemini 호출 후 assistant 메시지 생성 (현재 에코)
    assistant_reply = f"(임시 응답) {req.content[:100]}"
    chat_store.add_message(req.session_id, "assistant", assistant_reply)

    return SendMessageResponse(
        session_id=req.session_id,
        message=Message(id=msg_id, role="user", content=req.content, created_at=""),
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
