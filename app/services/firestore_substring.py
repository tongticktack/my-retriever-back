"""N-gram helper for approximate substring querying in Firestore.

To enable substring (부분 문자열) filtering:
 1. Backfill each target text field with bigram array field, e.g. storagePlace -> storagePlace_ngrams2.
 2. Query with array_contains_any on a subset of the needle's bigrams, then client-filter true substring.

Firestore limits: array_contains_any <= 30 values (we keep <=10). Only one such operator per query.
"""
from __future__ import annotations
from typing import List, Dict

def build_ngrams(text: str, n: int) -> List[str]:
    if not text or n <= 0:
        return []
    s = text.strip()
    if len(s) < n:
        return [s]
    grams = [s[i:i+n] for i in range(len(s)-n+1)]
    # preserve order & dedupe
    out = []
    seen = set()
    for g in grams:
        if g not in seen:
            seen.add(g); out.append(g)
    return out[:120]

def choose_query_ngrams(needle: str, n: int = 2, max_take: int = 8) -> List[str]:
    grams = build_ngrams(needle, n)
    if len(grams) <= max_take:
        return grams
    step = max(1, len(grams)//max_take)
    return [grams[i] for i in range(0, len(grams), step)][:max_take]

def ensure_index_fields(doc: Dict, source_field: str, bigrams_field: str):
    txt = doc.get(source_field)
    if isinstance(txt, str):
        doc[bigrams_field] = build_ngrams(txt, 2)
    return doc
