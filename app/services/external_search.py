"""Approximate external lost-item search (PoliceLostItem / PortalLostItem) without embeddings.

Goal:
    Use structured extracted fields (category, subcategory, lost_date, region) to retrieve
  likely matches from two Firestore collections that share the same schema:
    - PoliceLostItem
    - PortalLostItem

Assumptions (per user description):
  Document ID == atcId field value.
  Fields: atcId, createdAt, foundDate (YYYY-MM-DD), imageUrl, itemCategory ("대분류 > 중분류"),
          itemName (comma separated items possible), location (string or coord), storagePlace.

Firestore constraints:
  - Limited composite querying; we keep queries simple (category + foundDate equality) and
    issue multiple small queries for a date window. Post-filter & score client side.

Scoring heuristic (max ~100):
  +40 category exact match
  +15 subcategory exact match
  +15 color token match inside itemName (or itemCategory tail)
  +15 date closeness (15 - 5*abs(day_delta); floor at 0)
  +10 region substring in storagePlace or itemName
  +5  bonus if multiple (>=2) color synonyms matched (rare)

Returned structure per match:
  {
     'atcId': str,
     'collection': 'PoliceLostItem'|'PortalLostItem',
     'itemCategory': str,
     'itemName': str,
     'foundDate': str,
     'storagePlace': str|None,
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
from datetime import datetime, timedelta
import re

from app.domain import lost_item_schema as schema
from . import chat_store  # reuse firestore client

COLLECTIONS = ["PoliceLostItem", "PortalLostItem"]

## 색상 관련 요소 제거 (COLOR_SET 등)

# Window in days around lost_date to probe (inclusive)
DATE_WINDOW = 2  # ±2 days
MAX_DOCS_PER_DAY_PER_COLLECTION = 15  # cap to limit cost
FALLBACK_RECENT_LIMIT = 20

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

def _score(doc: Dict, extracted: Dict) -> Tuple[float, Dict[str, float]]:
    comp: Dict[str, float] = {}
    score = 0.0
    want_cat = extracted.get('category')
    want_sub = extracted.get('subcategory')
    # 색상 사용 안 함
    want_region = extracted.get('region')
    want_date = extracted.get('lost_date')

    doc_cat, doc_sub = _split_item_category(doc.get('itemCategory'))
    if want_cat and doc_cat == want_cat:
        comp['category'] = 40; score += 40
    if want_sub and doc_sub == want_sub:
        comp['subcategory'] = 15; score += 15

    # 색상 스코어링 제거

    # date closeness
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

    # region substring in storagePlace or itemName
    if want_region:
        for field in ['storagePlace', 'itemName']:
            val = doc.get(field)
            if isinstance(val, str) and want_region in val:
                comp['region'] = 10; score += 10
                break

    return score, comp

def _query_candidates(collection: str, extracted: Dict) -> List[Dict]:
    """Issue a handful of cheap Firestore queries and aggregate snapshots.
    We purposely keep queries narrow (category + foundDate equality) to leverage indexes.
    """
    try:
        db = chat_store.get_db()
    except Exception:
        return []
    col = db.collection(collection)
    want_cat = extracted.get('category')
    want_date = extracted.get('lost_date')
    docs: Dict[str, Dict] = {}
    if want_cat and want_date and _DATE_RE.match(want_date):
        try:
            base_date = datetime.strptime(want_date, '%Y-%m-%d').date()
        except Exception:
            base_date = None
        if base_date:
            for offset in range(-DATE_WINDOW, DATE_WINDOW + 1):
                day = (base_date + timedelta(days=offset)).isoformat()
                try:
                    q = col.where('itemCategory', '>=', f"{want_cat}") \
                           .where('itemCategory', '<', f"{want_cat}~")  # prefix hack (~ beyond unicode)
                    q = q.where('foundDate', '==', day).limit(MAX_DOCS_PER_DAY_PER_COLLECTION)
                    for snap in q.stream():
                        if snap.id not in docs:
                            d = snap.to_dict() or {}
                            d['atcId'] = snap.id
                            d['collection'] = collection
                            docs[snap.id] = d
                except Exception:
                    continue
    # Fallback: recent items if not enough docs
    if len(docs) < 5:
        try:
            q2 = col.order_by('createdAt', direction=chat_store.firestore.Query.DESCENDING).limit(FALLBACK_RECENT_LIMIT)  # type: ignore
            for snap in q2.stream():
                if snap.id not in docs:
                    d = snap.to_dict() or {}
                    d['atcId'] = snap.id
                    d['collection'] = collection
                    docs[snap.id] = d
        except Exception:
            pass
    return list(docs.values())

def approximate_external_matches(extracted: Dict, max_results: int = 10) -> List[Dict]:
    if not extracted or not extracted.get('category'):
        return []
    all_docs: List[Dict] = []
    for coll in COLLECTIONS:
        all_docs.extend(_query_candidates(coll, extracted))
    scored: List[Tuple[float, Dict]] = []
    for d in all_docs:
        s, comp = _score(d, extracted)
        if s <= 0:
            continue
        d_copy = {k: v for k, v in d.items()}
        d_copy['score'] = round(s, 2)
        d_copy['components'] = comp
        scored.append((s, d_copy))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:max_results]]

def summarize_matches(matches: List[Dict], limit: int = 3) -> str:
    if not matches:
        return "외부 공개 습득물 후보 없음"
    lines = []
    for m in matches[:limit]:
        cat = m.get('itemCategory') or ''
        nm = m.get('itemName') or ''
        fd = m.get('foundDate') or ''
        place = m.get('storagePlace') or ''
        lines.append(f"- {cat} | {nm[:30]} | {fd} | {place[:20]} | 점수 {m.get('score')}")
    return "유사 공개 습득물 후보:\n" + "\n".join(lines)
