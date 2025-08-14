"""History summarization utilities.
Currently a lightweight heuristic; can be replaced with LLM summarization later.
"""
from __future__ import annotations
from typing import List, Dict

MAX_SUMMARY_CHARS = 220


def summarize_pairs(messages: List[Dict]) -> str:
    """Create a compact summary of alternating user/assistant messages.
    Strategy: collect key sentences (first sentence of each user turn + verbs from assistant) until limit.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            first_sentence = content.split(".")[0][:80]
            parts.append(f"U:{first_sentence}")
        elif role == "assistant":
            first_sentence = content.split(".")[0][:80]
            parts.append(f"A:{first_sentence}")
        if sum(len(p) for p in parts) > MAX_SUMMARY_CHARS:
            break
    summary = " | ".join(parts)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS-3] + "..."
    return summary
