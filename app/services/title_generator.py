"""Session title generation (OpenAI only, then heuristic).

Strategy:
1. OpenAI (if OPENAI_API_KEY set)
2. Fallback: truncate first user message.
Constraints:
- Single short prompt, <=30 chars (Korean safe truncation)
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
    user_text = first_user_message.strip()
    if not user_text:
        return "새 대화"
    title = _openai_title(user_text)
    if not title:
        title = _truncate(user_text)
    return title or "새 대화"
