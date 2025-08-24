"""Approximate external lost-item search (PoliceLostItem / PortalLostItem) without embeddings.

Goal:
    Use structured extracted fields (category, subcategory, lost_date, place_query) to retrieve
  likely matches from two Firestore collections that share the same schema:
    - PoliceLostItem
    - PortalLostItem

Assumptions (per user description):
    Document ID == atcId field value.
    Fields: atcId, foundDate (YYYY-MM-DD), imageUrl, itemCategory ("대분류 > 중분류"),
                    itemName (comma separated items possible), location (string or coord), addr (was storagePlace).

Firestore constraints:
    - Keep Firestore query simple: category prefix + exact foundDate equality only.
    - All partial (substring) matching for 장소는 Python 메모리에서 storagePlace 대상으로만 수행.

Scoring heuristic (max ~80):
    +40 category exact match
    +10 subcategory exact match (소분류 신뢰도 낮음 → 낮은 가중치 유지)
    +15 date closeness (within ±3 days: 15 - 5*|Δdays|, floor 0)
    +10 storagePlace 부분 문자열 매치 (place_query 포함)

Returned structure per match:
  {
      'atcId': str,
      'collection': 'PoliceLostItem'|'PortalLostItem',
      'itemCategory': str,
      'itemName': str,
      'foundDate': str,
      'addr': str|None,            # unified; populated from Firestore 'addr' or legacy 'storagePlace'
      'imageUrl': str|None,
      'score': float,
    'components': {<component>: value}
  }

Usage:
  from app.services import external_search
  matches = external_search.approximate_external_matches(extracted_dict)
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from datetime import datetime
import re

from app.domain import lost_item_schema as schema
from . import chat_store  # reuse firestore client
from app.scripts.logging_config import get_logger

logger = get_logger("external_search")

COLLECTIONS = ["PoliceLostItem", "PortalLostItem"]

## 색상 관련 요소 제거 (COLOR_SET 등)

MAX_DOCS_PER_COLLECTION = 350  # category prefix + exact date 조회 최대 개수

_DATE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")

def _canon_color(token: str) -> str | None:
    return None  # 색상 비활성화

def _split_item_category(cat_field: str | None) -> Tuple[str | None, str | None]:
    if not cat_field:
        return None, None
    parts = [p.strip() for p in cat_field.split('>') if p.strip()]
    if len(parts) == 1:
        return parts[0], None
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None, None

def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[\s,;:/]+", text.lower()) if t]

def _score(doc: Dict, extracted: Dict, place_query: str | None) -> Tuple[float, Dict[str, float]]:
    comp: Dict[str, float] = {}
    score = 0.0
    want_cat = extracted.get('category')
    want_sub = extracted.get('subcategory')
    # 색상 사용 안 함
    want_date = extracted.get('lost_date')

    doc_cat, doc_sub = _split_item_category(doc.get('itemCategory'))
    if want_cat and doc_cat == want_cat:
        comp['category'] = 40; score += 40
    if want_sub and doc_sub == want_sub:
        comp['subcategory'] = 10; score += 10  # lowered weight due to possible subcategory extraction errors

    # date closeness within ±3 days
    fd = doc.get('foundDate')
    if want_date and fd and _DATE_RE.match(fd):
        try:
            d_want = datetime.strptime(want_date, '%Y-%m-%d').date()
            d_found = datetime.strptime(fd, '%Y-%m-%d').date()
            dd = abs((d_found - d_want).days)
            if dd <= 3:
                date_score = max(0, 15 - 5 * dd)
                if date_score:
                    comp['date'] = date_score; score += date_score
        except Exception:
            pass

    # storagePlace substring (place_query)
    if place_query:
        sp = doc.get('storagePlace') or doc.get('addr') or ''
        if isinstance(sp, str) and place_query in sp.lower():
            comp['place'] = 10; score += 10

    return score, comp

def _query_candidates(collection: str, extracted: Dict) -> List[Dict]:
    """Fetch candidates by (category prefix + exact lost_date)."""
    try:
        db = chat_store.get_db()
    except Exception:
        return []
    col = db.collection(collection)
    want_cat = extracted.get('category')
    want_date = extracted.get('lost_date')
    if not want_cat or not want_date or not _DATE_RE.match(want_date):
        return []
    docs: Dict[str, Dict] = {}
    try:
        try:
            logger.info(
                "firestore.query op=find collection=%s cat_prefix=%s date=%s range=[%s,%s) limit=%d exact_date=1",
                collection, want_cat, want_date, f"{want_cat}", f"{want_cat}~", MAX_DOCS_PER_COLLECTION
            )
        except Exception:
            pass
        q = (col.where('itemCategory', '>=', f"{want_cat}")
                .where('itemCategory', '<', f"{want_cat}~")
                .where('foundDate', '==', want_date)
                .limit(MAX_DOCS_PER_COLLECTION))
        for snap in q.stream():
            if snap.id not in docs:
                d = snap.to_dict() or {}
                d['atcId'] = snap.id
                d['collection'] = collection
                docs[snap.id] = d
    except Exception as e:
        logger.error("category_date_query_error collection=%s date=%s err=%s", collection, want_date, e)
    return list(docs.values())

def approximate_external_matches(extracted: Dict, place_query: str | None = None, max_results: int = 10) -> List[Dict]:
    # 다시 날짜 필수 (exact date 조회) - lost_date 없으면 검색 안 함
    if not extracted or not extracted.get('category') or not extracted.get('lost_date'):
        return []
    # If caller omitted place_query explicitly, fall back to extracted region field
    if place_query is None:
        place_query = extracted.get('region')
    # normalize place_query (search target is storagePlace only)
    if place_query:
        pq_norm = str(place_query).strip().lower()
        place_query = pq_norm if len(pq_norm) >= 2 else None
    logger.info("start category=%s sub=%s date=%s place_query=%s max=%s exact_date=1", extracted.get('category'), extracted.get('subcategory'), extracted.get('lost_date'), place_query, max_results)
    all_docs: List[Dict] = []
    for coll in COLLECTIONS:
        before = len(all_docs)
        try:
            cand = _query_candidates(coll, extracted)
        except Exception as e:
            logger.error("collection_error collection=%s err=%s", coll, e)
            cand = []
        all_docs.extend(cand)
        added = len(all_docs) - before
    logger.debug("collected collection=%s added=%d total=%d", coll, added, len(all_docs))
    # place_query filtering (hard filter first, fallback if empty)
    filtered = all_docs
    place_filtered = False
    fallback_place = False
    if place_query:
        tmp = [d for d in all_docs if isinstance(d.get('storagePlace'), str) and place_query in d.get('storagePlace').lower()]
        if tmp:
            filtered = tmp
            place_filtered = True
        else:
            fallback_place = True  # keep original set
    scored: List[Tuple[float, Dict]] = []
    for d in filtered:
        s, comp = _score(d, extracted, place_query)
        if s <= 0:
            continue
        d_copy = {k: v for k, v in d.items()}
        d_copy['score'] = round(s, 2)
        d_copy['components'] = comp
        scored.append((s, d_copy))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        logger.info("no_scored_matches category=%s sub=%s date=%s place_query=%s", extracted.get('category'), extracted.get('subcategory'), extracted.get('lost_date'), place_query)
    else:
        top_ids = [d['atcId'] for _, d in scored[:max_results]]
        logger.info("scored_matches count=%d top=%s place_filtered=%s fallback_place=%s", len(scored), top_ids, place_filtered, fallback_place)
        for _, dd in scored[: min(5, len(scored))]:
            logger.debug("detail atcId=%s score=%.2f comp=%s cat=%s foundDate=%s place_hit=%s", dd.get('atcId'), dd.get('score'), dd.get('components'), dd.get('itemCategory'), dd.get('foundDate'), 'place' in (dd.get('components') or {}))
    return [d for _, d in scored[:max_results]]

def summarize_matches(matches: List[Dict], limit: int = 3) -> str:
    if not matches:
        return "외부 공개 습득물 후보 없음"
    lines = []
    for m in matches[:limit]:
        cat = m.get('itemCategory') or ''
        nm = m.get('itemName') or ''
        fd = m.get('foundDate') or ''
        place = m.get('storagePlace') or m.get('addr') or ''
        lines.append(f"- {cat} | {nm[:30]} | {fd} | {place[:20]} | 점수 {m.get('score')}")
    return "유사 공개 습득물 후보:\n" + "\n".join(lines)
