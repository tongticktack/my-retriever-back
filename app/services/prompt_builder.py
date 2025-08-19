"""Prompt builder for assembling LLM conversation messages.

build_messages orchestrates:
  - system prompt loading
  - optional retrieval context injection (placeholder now)
  - history loading & optional summarization
  - trimming to history size limit
"""
from __future__ import annotations
from typing import List, Dict, Optional
from functools import lru_cache
from pathlib import Path

from config import settings
from pathlib import Path as _P
from . import chat_store
from .summarizer import summarize_pairs

MessageDict = Dict[str, str]


@lru_cache(maxsize=4)
def _load_system_prompt(path: str) -> str:
    try:
        return Path(path).read_text("utf-8").strip()
    except Exception:
        return "역할: 분실물 상담 도우미"


def _inject_retrieval_context(items: Optional[List[Dict]] = None, law_summary: Optional[str] = None) -> List[MessageDict]:
    messages: List[MessageDict] = []
    if items:
        lines = ["[검색결과]"]
        for i, it in enumerate(items[:3], start=1):
            lines.append(f"{i}) id={it.get('id')} score={it.get('score')} 요약={it.get('summary','')}")
        messages.append({"role": "system", "content": "\n".join(lines)})
    if law_summary:
        messages.append({"role": "system", "content": f"[법령요약]\n{law_summary}"})
    return messages


def build_messages(session_id: str, user_input: str, retrieval_items: Optional[List[Dict]] = None, law_summary: Optional[str] = None) -> List[MessageDict]:
    # dynamic system prompt resolution: try explicit file, else version-based fallback
    sys_path = settings.SYSTEM_PROMPT_FILE
    if not _P(sys_path).exists():
        ver = getattr(settings, 'PROMPT_VERSION', 2)
        candidate = f"app/prompts/system_prompt_v{ver}.txt"
        if _P(candidate).exists():
            sys_path = candidate
    system_prompt = _load_system_prompt(sys_path)
    msgs: List[MessageDict] = [
        {"role": "system", "content": system_prompt}
    ]
    # Retrieval context
    msgs.extend(_inject_retrieval_context(retrieval_items, law_summary))

    # History
    try:
        history = chat_store.fetch_messages(session_id, limit=settings.LLM_MAX_HISTORY_MESSAGES * 2)
    except Exception:
        history = []

    # Optional summarization if history too long
    if len(history) > settings.LLM_MAX_HISTORY_MESSAGES:
        summary = summarize_pairs(history[:-settings.LLM_MAX_HISTORY_MESSAGES])
        msgs.append({"role": "system", "content": f"[이전요약] {summary}"})
    # Keep tail
    introduced = False
    for h in history[-settings.LLM_MAX_HISTORY_MESSAGES:]:
        role = h.get("role")
        if role in {"user", "assistant"}:
            content = h.get("content", "")
            if not introduced and role == "assistant" and "루시" in content and ("안녕하세요" in content or "도와드리는" in content):
                introduced = True
            msgs.append({"role": role, "content": content})
    # If no prior introduction detected, prepend a lightweight system hint to force intro sentence.
    if not introduced:
        msgs.append({"role": "system", "content": "[SELF_INTRO_REQUIRED] 다음 assistant 응답 첫 문장은 자기소개 포함 (1회만)."})

    # Current user input (avoid duplicating if last history message already equals this user turn)
    last_non_system = None
    for m in reversed(msgs):
        if m["role"] != "system":
            last_non_system = m
            break
    if not (last_non_system and last_non_system.get("role") == "user" and last_non_system.get("content") == user_input):
        msgs.append({"role": "user", "content": user_input})
    return msgs
