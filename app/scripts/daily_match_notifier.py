"""Daily match notifier.

1. Iterate all users in lost_items collection.
2. For each item where is_found == False, build a simple text embedding query or rely on existing extracted fields.
3. Use existing internal text + (optional) image similarity logic to fetch matches (reuse faiss_index + synonym mapper heuristics similar to chat.send subset).
4. If any candidate score >= NOTIFY_THRESHOLD (e.g., 0.80 or 80 depending on scale) send an email notification.

This is a lightweight version; adapt scoring mapping to your internal scale.
"""
from __future__ import annotations
import os
from typing import List, Dict, Tuple
from firebase_admin import firestore
from app.services import chat_store, faiss_index
from firebase_admin import auth
from app.scripts.logging_config import get_logger, setup_logging
import smtplib
from email.message import EmailMessage
from app.services.synonym_mapper import (
    expand_search_terms_with_subcategory, is_category_match, calculate_category_subcategory_score
)
from app.api.chat import _parse_iso_date, _within_window, PRIMARY_DATE_WINDOW_DAYS, EXPANDED_DATE_WINDOW_DAYS, MIN_TEXT_SEARCH_SCORE
from datetime import datetime, timezone, timedelta

NOTIFY_SCORE_THRESHOLD = float(os.getenv("NOTIFY_SCORE_THRESHOLD", "0.80"))
STRICT_DATE_MATCH_DAYS = 3
NOTIFY_COOLDOWN_HOURS = int(os.getenv("NOTIFY_COOLDOWN_HOURS", "24"))

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_SENDER = os.getenv("SMTP_SENDER") or (SMTP_USER or "no-reply@example.com")

logger = get_logger("notifier")


def resolve_user_email(user_id: str) -> str | None:
    try:
        rec = auth.get_user(user_id)
        if not rec.email:
            logger.warning("user_email_missing user=%s", user_id)
        return rec.email
    except Exception as e:
        logger.warning("user_email_lookup_failed user=%s err=%s", user_id, e)
        return None

def send_email(to_address: str, subject: str, body: str):
    if not to_address:
    logger.info("email_skip_no_address")
    return False
    if not SMTP_HOST:
    logger.warning("email_skip_no_smtp to=%s", to_address)
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_SENDER
        msg["To"] = to_address
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    logger.info("email_sent to=%s subject=%s size=%d", to_address, subject, len(body))
        return True
    except Exception as e:
    logger.error("email_error to=%s err=%s", to_address, e)
        return False


def iter_user_lost_items():
    setup_logging(json_fmt=False)
    db = chat_store.get_db()
    docs = db.collection("lost_items").stream()
    for doc in docs:
        data = doc.to_dict() or {}
        user_id = data.get("user_id") or doc.id
        items = data.get("items") or []
        yield user_id, items


def score_internal(item: Dict) -> List[Dict]:
    extracted = item.get("extracted") or {}
    if not extracted:
        return []
    cat_q = (extracted.get("category") or "").lower().strip()
    sub_q = (extracted.get("subcategory") or "").lower().strip()
    region_q = (extracted.get("region") or "").lower().strip()
    lost_date = _parse_iso_date(extracted.get("lost_date")) if extracted.get("lost_date") else None
    expanded_categories, target_subcats = expand_search_terms_with_subcategory(cat_q, sub_q)

    candidates = []
    primary = []
    expanded = []
    for iid, meta in faiss_index.META.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("type") in {"media","lost_item"}:
            continue
        mcat = str(meta.get("category") or "").lower()
        # category + subcategory match
        category_match = True
        if cat_q or sub_q:
            category_match = is_category_match(mcat, expanded_categories if sub_q and not cat_q else [cat_q] if cat_q else expanded_categories)
        if not category_match:
            continue
        region_match = True
        if region_q:
            mplace = str(meta.get("found_place") or "").lower()
            region_match = region_q in mplace
        if not region_match:
            continue
        found_raw = meta.get('found_time') or meta.get('foundDate') or meta.get('found_date')
        found_d = _parse_iso_date(found_raw) if isinstance(found_raw, str) else None
        in_primary = False
        in_expanded = False
        if lost_date and found_d:
            if _within_window(found_d, lost_date, PRIMARY_DATE_WINDOW_DAYS):
                in_primary = True
            elif _within_window(found_d, lost_date, EXPANDED_DATE_WINDOW_DAYS):
                in_expanded = True
        score = 0.0
        if cat_q or sub_q:
            cat_sub_score = calculate_category_subcategory_score(meta.get('category') or '', cat_q, sub_q)
            score += cat_sub_score * 0.65
        if region_q:
            if region_q in (meta.get('found_place') or '').lower():
                score += 0.20
        if lost_date and found_d:
            delta_days = abs((found_d - lost_date).days)
            max_window = PRIMARY_DATE_WINDOW_DAYS if delta_days <= PRIMARY_DATE_WINDOW_DAYS else EXPANDED_DATE_WINDOW_DAYS
            date_score = max(0.0, 1 - delta_days / max_window)
            score += date_score * 0.15
        elif lost_date and not found_d:
            score += 0.05
        elif not lost_date:
            score += 0.10
        if score < MIN_TEXT_SEARCH_SCORE:
            continue
        entry = (iid, score, meta, in_primary, in_expanded)
        candidates.append(entry)
        if in_primary:
            primary.append(entry)
        elif in_expanded:
            expanded.append(entry)
    ranked = []
    if lost_date:
        if primary:
            ranked.extend(primary)
            if len(primary) < 10:
                ranked.extend([e for e in expanded if e[1] >= 0.7][:5])
        elif expanded:
            ranked.extend(expanded)
        else:
            ranked.extend([e for e in candidates if e[1] >= 0.8])
    else:
        ranked = candidates
    def _ts(meta):
        return meta.get('created_at') or meta.get('createdAt') or ''
    ranked.sort(key=lambda x: (x[1], _ts(x[2])), reverse=True)
    out = []
    for iid, score, meta, *_ in ranked[:30]:
        out.append({
            'atcId': meta.get('atcId') or meta.get('id') or iid,
            'score': round(score, 4),
            'category': meta.get('category'),
            'foundDate': meta.get('found_time'),
            'foundPlace': meta.get('found_place'),
            'name': meta.get('caption') or meta.get('notes') or meta.get('category'),
        })
    return out


def main():
    notified = 0
    checked = 0
    db = chat_store.get_db()
    for user_id, items in iter_user_lost_items():
        email = resolve_user_email(user_id)
        doc_ref = db.collection("lost_items").document(user_id)
        doc_changed = False
        for idx, it in enumerate(items):
            if it.get('is_found') is True:
                continue
            extracted = it.get('extracted') or {}
            if not extracted:
                continue
            matches = score_internal(it)
            if not matches:
                continue
            top = matches[0]
            # 중복/쿨다운 체크
            last_top_id = it.get('last_notified_top_id')
            last_notified_at_raw = it.get('last_notified_at')
            cooldown_ok = True
            if last_notified_at_raw:
                try:
                    if isinstance(last_notified_at_raw, str):
                        last_dt = datetime.fromisoformat(last_notified_at_raw.replace('Z','+00:00'))
                    else:
                        last_dt = last_notified_at_raw
                    if datetime.now(timezone.utc) - last_dt < timedelta(hours=NOTIFY_COOLDOWN_HOURS):
                        cooldown_ok = False
                except Exception:
                    pass
            is_new_top = (top.get('atcId') != last_top_id)
            if top['score'] >= NOTIFY_SCORE_THRESHOLD and (is_new_top or cooldown_ok):
                subject = "분실물 후보 발견 안내"
                body = (
                    f"추적 중인 분실물과 유사한 후보가 발견되었어요.\n"
                    f"유사도: {top['score']}\n"
                    f"카테고리: {top.get('category')}\n"
                    f"습득일: {top.get('foundDate')} 장소: {top.get('foundPlace')}\n"
                    f"대상: {top.get('name')} (ID: {top.get('atcId')})\n"
                    "서비스에 접속해 상세 확인 후 회수 절차를 진행해 주세요."
                )
                if send_email(email, subject, body):
                    it['last_notified_at'] = datetime.now(timezone.utc).isoformat()
                    it['last_notified_top_id'] = top.get('atcId')
                    it['last_notified_score'] = top.get('score')
                    doc_changed = True
                    notified += 1
            checked += 1
        if doc_changed:
            try:
                doc_ref.update({"items": items, "updated_at": datetime.now(timezone.utc)})
            except Exception as e:
                print(f"[notifier] failed to persist notification metadata user={user_id} err={e}")
    logger.info("notifier_done checked_items=%d notified=%d", checked, notified)

if __name__ == "__main__":
    main()
