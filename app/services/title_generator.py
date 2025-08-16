"""Session title generation using available LLM providers.

Strategy:
1. Prefer Gemini if GEMINI_API_KEY set.
2. Else prefer OpenAI if OPENAI_API_KEY set.
3. Fallback: heuristic truncate first user message.
Constraints:
- Keep synchronous & fast (single short prompt)
- Limit length ~30 chars (Korean safe truncation)
"""
from __future__ import annotations
from typing import Optional
from config import settings

MAX_LEN = 30

# Lazy provider imports inside functions to avoid heavy init during module import

def _truncate(text: str) -> str:
    t = text.strip().replace("\n", " ")
    if len(t) <= MAX_LEN:
        return t
    return t[:MAX_LEN].rstrip() + "…"


def _gemini_title(user_text: str) -> Optional[str]:  # pragma: no cover (network)
    try:
        if not settings.GEMINI_API_KEY:
            return None
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=settings.GEMINI_API_KEY)
        prompt = (
            "다음 첫 사용자 메시지를 1줄 대화 제목으로 20자 이내 한국어로 요약:")
        full = f"{prompt}\n---\n{user_text}\n---\n제목:"
        resp = genai.GenerativeModel(settings.GEMINI_MODEL_NAME).generate_content(full)
        text = getattr(resp, "text", "") or ""
        if not text and getattr(resp, "candidates", None):
            cand = resp.candidates[0]
            parts = getattr(cand, "content", getattr(cand, "parts", []))
            if parts:
                maybe = getattr(parts[0], "text", str(parts[0]))
                text = str(maybe)
        if text:
            return _truncate(text.splitlines()[0])
    except Exception:
        return None
    return None


def _openai_title(user_text: str) -> Optional[str]:  # pragma: no cover (network)
    try:
        if not settings.OPENAI_API_KEY:
            return None
        import openai  # type: ignore
        openai.api_key = settings.OPENAI_API_KEY
        system = {"role": "system", "content": "You create a concise Korean chat title (<=30 chars) summarizing the first user message only."}
        user = {"role": "user", "content": user_text}
        resp = openai.chat.completions.create(model=settings.OPENAI_MODEL_NAME, messages=[system, user], timeout=10)  # type: ignore
        title = resp.choices[0].message.content.strip()  # type: ignore
        return _truncate(title.splitlines()[0])
    except Exception:
        return None


def generate_session_title(first_user_message: str) -> str:
    # Try providers in priority (Gemini then OpenAI) mirroring typical config
    user_text = first_user_message.strip()
    if not user_text:
        return "새 대화"
    title = _gemini_title(user_text)
    if not title:
        title = _openai_title(user_text)
    if not title:
        title = _truncate(user_text)
    # fallback guard
    return title or "새 대화"
