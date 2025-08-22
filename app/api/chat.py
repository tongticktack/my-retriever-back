from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date

from app.services import chat_store
from app.services.llm_providers import get_llm
from app.services.prompt_builder import build_messages
from app.services import extraction_controller as extc
from app.domain import lost_item_schema as schema
from app.services import lost_item_store
from app.services import media_store, faiss_index, embeddings
from config import settings

router = APIRouter(prefix="/chat", tags=["chat"])

# Image similarity & attachment configuration
IMAGE_SIMILARITY_TOP_K = 5  # max number of similar items to show
IMAGE_SIMILARITY_THRESHOLD = 0.70  # minimum cosine similarity (dot since embeddings normalized)
MAX_MEDIA_PER_MESSAGE = 3  # allow up to 3 images per user message
# Multi-image 내 최소 쌍 유사도 임계 (환경 변수로 조정 가능)
MULTI_IMAGE_MIN_INTERNAL_SIMILARITY = getattr(settings, 'MULTI_IMAGE_MIN_INTERNAL_SIMILARITY', 0.45)
# Date filtering configuration
PRIMARY_DATE_WINDOW_DAYS = getattr(settings, 'PRIMARY_DATE_WINDOW_DAYS', 14)  # 1차 윈도우 (±N일)
EXPANDED_DATE_WINDOW_DAYS = getattr(settings, 'EXPANDED_DATE_WINDOW_DAYS', 30)  # 확장 윈도우
STRICT_DATE_MATCH_DAYS = getattr(settings, 'STRICT_DATE_MATCH_DAYS', 3)  # 최종 matches 에 허용되는 ±일 수

def _parse_iso_date(s: str) -> Optional[date]:
    if not s:
        return None
    s = s.strip().split(' ')[0]  # if accidental datetime string
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _within_window(found: Optional[date], target: Optional[date], window_days: int) -> bool:
    if not found or not target:
        return False
    return abs((found - target).days) <= window_days

class CreateSessionRequest(BaseModel):
    user_id: str

class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None

class SendMessageRequest(BaseModel):
    session_id: str
    content: str
    user_id: Optional[str] = None  # (미인증 상태라 사용 안하지만 필드 예약)
    media_ids: Optional[List[str]] = None  # 업로드된 이미지 참조 ID 목록

class Message(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    model: str | None = None
    attachments: Optional[List[dict]] = None

class SendMessageResponse(BaseModel):
    user_message: Message
    assistant_message: Message
    session_id: str
    # Optional debug state snippet for active lost item (not full state to keep payload small)
    active_lost_item: Optional[dict] = None
    matches: Optional[List[dict]] = None  # unified list: external or internal (image/text) candidates

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Message]

class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: Optional[str] = None
    last_active_at: Optional[str] = None

class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]

class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool

@router.post("/session", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    session_id = chat_store.create_session(user_id=req.user_id)
    return CreateSessionResponse(session_id=session_id, user_id=req.user_id)

@router.post("/send", response_model=SendMessageResponse)
def send_message(req: SendMessageRequest):
    # Validate attachment count (max 3 images)
    if req.media_ids and len(req.media_ids) > MAX_MEDIA_PER_MESSAGE:
        raise HTTPException(status_code=400, detail="too_many_media")
    # 첨부 메타 (사용자 메시지에 이미지 참조 저장)
    user_meta = None
    if req.media_ids:
        try:
            from . import media as media_api
            medias = media_api.get_media_batch(req.media_ids)
        except Exception:
            medias = []
        # Validate existence
        if not medias or len(medias) != len(req.media_ids):
            print(f"[chat.send] invalid_media_id debug - requested={req.media_ids} found={len(medias)} metas={medias}")
            raise HTTPException(status_code=400, detail="invalid_media_id")
        if medias:
            user_meta = {"attachments": medias}
    try:
        user_msg_id = chat_store.add_message(req.session_id, "user", req.content, meta=user_meta)
    except ValueError as e:
        code = str(e)
        if code == "session_not_found":
            raise HTTPException(status_code=404, detail=code)
        if code == "empty_content":
            raise HTTPException(status_code=400, detail=code)
        if code == "content_too_long":
            raise HTTPException(status_code=413, detail=code)
        raise HTTPException(status_code=400, detail={"error": "invalid_request", "raw": code})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Lost-item orchestrated flow via extraction_controller
    session_doc = chat_store.get_session_data(req.session_id) or {}
    lost_state = session_doc.get("lost_items")
    if lost_state is None:
        lost_state = {"items": [], "active_index": None}
    # initialize cache container (per session) for image similarity to avoid duplicate FAISS searches
    if "_img_cache" not in lost_state:
        lost_state["_img_cache"] = {}
    image_search_results = None
    # 색상 추출 비활성화: 이미지 기반 색상 추론 제거
    majority_image_color = None  # retained placeholder
    multi_image_conflict = False
    if user_meta and user_meta.get("attachments"):
        atts = user_meta.get("attachments")
        # Distinct-object heuristic first (same as before)
        if len(atts) > 1:
            embs = []
            try:
                for a in atts[:MAX_MEDIA_PER_MESSAGE]:
                    mid = a.get("media_id")
                    if not mid:
                        continue
                    emb = media_store.get_embedding(mid)
                    if emb is not None:
                        embs.append((mid, emb))
                def _dot(u,v):
                    return sum(x*y for x,y in zip(u,v)) / ( (sum(x*x for x in u)**0.5) * (sum(y*y for y in v)**0.5) + 1e-9 )
                min_sim = 1.0
                for i in range(len(embs)):
                    for j in range(i+1, len(embs)):
                        s = _dot(embs[i][1], embs[j][1])
                        if s < min_sim:
                            min_sim = s
                if len(embs) >= 2 and min_sim < MULTI_IMAGE_MIN_INTERNAL_SIMILARITY:
                    multi_image_conflict = True
            except Exception as e:
                print(f"[chat.multi_image_conflict_check] error: {e}")
    # merge media ids only (색상 세팅 제거)
    if user_meta and user_meta.get("attachments") and lost_state.get("active_index") is not None:
        try:
            active = lost_state["items"][lost_state["active_index"]]
        except Exception:
            active = None
        if active:
            mid_list = [m.get("media_id") for m in user_meta.get("attachments") if m.get("media_id")]
            if mid_list:
                existing = set(active.get("media_ids") or [])
                active["media_ids"] = list(existing.union(mid_list))
    user_lower = req.content.strip().lower()
    start_new_trigger = any(kw in user_lower for kw in ["또 잃어버렸", "다른 물건", "새 물건", "추가 물건"]) or user_lower.startswith("새 물건")

    # Vision LLM: 이미지 URL 목록 (공개 URL 기준) 추출하여 controller 전달
    image_urls = None
    if user_meta and user_meta.get("attachments"):
        image_urls = [a.get("url") for a in user_meta.get("attachments") if a.get("url")]
    if multi_image_conflict:
        assistant_reply = (
            "여러 이미지가 서로 다른 물건으로 보입니다. 한 번에 하나의 분실물 이미지(최대 3장: 같은 물건 다른 각도)만 첨부해주세요. "
            "각 물건은 별도 메시지 또는 '새 물건' 입력 후 이미지를 올려주세요."
        )
        chosen_model = "multi-image-validation"
        active_item_snapshot = None
    else:
        from time import time as _t
        _t0 = _t()
        assistant_reply, lost_state, chosen_model, active_item_snapshot = extc.process_message(
            req.content, lost_state, start_new_trigger, image_urls=image_urls
        )
        print(f"[chat.send] extraction_controller latency={( _t() - _t0):.2f}s model={chosen_model} reply_len={len(assistant_reply) if assistant_reply else 0}")
    # After extraction: run metadata-based candidate filtering + image similarity (lazy) if user supplied images
    if (assistant_reply is not None) and user_meta and user_meta.get("attachments") and not multi_image_conflict:
        try:
            from app.services import faiss_index as _fi
            # Build candidate id set based on extracted info (category, region)
            active_idx = lost_state.get("active_index")
            extracted = None
            if active_idx is not None:
                try:
                    extracted = (lost_state.get("items") or [])[active_idx].get("extracted") or {}
                except Exception:
                    extracted = {}
            cat = (extracted.get("category") or "").strip().lower() if extracted else ""
            region = (extracted.get("region") or "").strip().lower() if extracted else ""
            lost_date = _parse_iso_date(extracted.get("lost_date") if extracted else None)
            candidates = []
            # 1차: category/region + 날짜 윈도우(있으면)
            primary_window_ids = []
            expanded_needed = False
            for iid, meta in _fi.META.items():
                # Skip ephemeral session-added entries (we tagged with type)
                if isinstance(meta, dict) and meta.get("type") in {"media","lost_item"}:
                    continue
                if not isinstance(meta, dict):
                    continue
                mcat = str(meta.get("category","")) .lower()
                if cat and cat not in mcat:
                    continue
                if region:
                    place = str(meta.get("found_place","")) .lower()
                    if region not in place:
                        continue
                # Date filtering
                found_raw = meta.get('found_time') or meta.get('foundDate') or meta.get('found_date')
                found_d = _parse_iso_date(found_raw) if isinstance(found_raw, str) else None
                if lost_date and found_d:
                    if _within_window(found_d, lost_date, PRIMARY_DATE_WINDOW_DAYS):
                        primary_window_ids.append(iid)
                    elif _within_window(found_d, lost_date, EXPANDED_DATE_WINDOW_DAYS):
                        expanded_needed = True
                        candidates.append(iid)  # store for possible expansion
                else:
                    # If no date info, keep as fallback candidate list
                    candidates.append(iid)
            if lost_date:
                if primary_window_ids:
                    candidates = primary_window_ids
                elif expanded_needed:
                    # filter candidates to only those within expanded window already collected
                    pass  # candidates already holds expanded set
            # If no candidates found, fall back to whole index (will be filtered by threshold later)
            candidate_set = set(candidates) if candidates else None
            cache = lost_state.get("_img_cache") or {}
            best = None
            best_score = -1.0
            atts = user_meta.get("attachments")
            for idx, att in enumerate(atts[:MAX_MEDIA_PER_MESSAGE]):
                mid = att.get("media_id")
                if not mid:
                    continue
                emb = media_store.get_embedding(mid)
                if emb is None:
                    continue
                # Query broader then filter
                raw = _fi.search_image(emb, k= max(IMAGE_SIMILARITY_TOP_K*4, IMAGE_SIMILARITY_TOP_K+10))
                filtered = []
                for iid, score, meta in raw:
                    if candidate_set and iid not in candidate_set:
                        continue
                    if score < IMAGE_SIMILARITY_THRESHOLD:
                        continue
                    filtered.append((iid, score, meta))
                    if len(filtered) >= IMAGE_SIMILARITY_TOP_K:
                        break
                if filtered and filtered[0][1] > best_score:
                    best_score = filtered[0][1]
                    best = filtered
            image_search_results = best
        except Exception as e:
            print(f"[chat.image_search.meta] error: {e}")
    # After extraction reconcile image-derived color vs extracted color if mismatch
    # 색상 reconcile 제거
    # If controller produced reply and we have similarity results, append textual list
    if assistant_reply and image_search_results:
        if image_search_results:
            lines = []
            rank = 1
            for iid, score, meta in image_search_results:
                label = meta.get('media_id') if isinstance(meta, dict) else None
                if not label:
                    label = iid
                extra_parts = []
                if isinstance(meta, dict):
                    if meta.get('category'):
                        extra_parts.append(meta['category'])
                    if meta.get('color'):
                        extra_parts.append(meta['color'])
                suffix = ("; "+", ".join(extra_parts)) if extra_parts else ""
                lines.append(f"{rank}. {label} (score {score:.3f}{suffix})")
                rank += 1
            assistant_reply += f"\n\n(이미지 유사도 후보 ≥ {IMAGE_SIMILARITY_THRESHOLD:.2f})\n" + "\n".join(lines)
    # Acknowledge image receipt
    if user_meta and user_meta.get("attachments"):
        attachments_list = user_meta.get("attachments")
        count_imgs = len(attachments_list)
        ack_line = f"이미지 {count_imgs}장 확인했습니다."
        lines_enum = []
        for i, a in enumerate(attachments_list, start=1):
            mid = a.get("media_id") or "-"
            lines_enum.append(f"{i}. {mid}")
        if lines_enum:
            ack_line += "\n" + "\n".join(lines_enum)
        if assistant_reply:
            assistant_reply = ack_line + "\n" + assistant_reply
        else:
            assistant_reply = ack_line
    chat_store.update_session(req.session_id, {"lost_items": lost_state})
    try:
        session_user = session_doc.get("user_id") if session_doc else None
        user_for_items = (req.user_id or session_user or "guest")
        lost_item_store.bulk_upsert(user_for_items, req.session_id, lost_state.get("items", []))
    except Exception as e:
        print(f"[lost_item_store] bulk_upsert error: {e}")

    if assistant_reply is None:
        llm = get_llm()
        provider_messages = build_messages(req.session_id, req.content, retrieval_items=None, law_summary=None)
        try:
            print(f"[chat.send] general_llm_call provider={getattr(llm,'name',None)} model={getattr(llm,'model',None)}")
            assistant_reply = llm.generate(provider_messages)
        except Exception as e:
            assistant_reply = f"(llm-error) {str(e)[:120]}"
        chosen_model = getattr(llm, "last_model_name", None) or getattr(llm, "model", None) or getattr(llm, "name", "unknown")
        print(f"[chat.send] general_llm_done model={chosen_model} len={len(assistant_reply)}")

    assistant_meta = {"model": chosen_model}
    assistant_msg_id = chat_store.add_message(
        req.session_id,
        "assistant",
        assistant_reply,
        meta=assistant_meta
    )

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

    # Get assistant created_at (may need refresh, keep simple by refetching last 2)
    assistant_created_at = ""
    try:
        recent2 = chat_store.fetch_messages(req.session_id, limit=2)
        for m in reversed(recent2):
            if m.get("id") == assistant_msg_id:
                assistant_created_at = m.get("created_at", "")
                break
    except Exception:
        pass
    # Fetch attachments if any (only need from recent user message fetch) - simple re-fetch one message if attachments expected
    user_attachments = None
    if user_meta and user_meta.get("attachments"):
        user_attachments = user_meta.get("attachments")
    # Build matches list (external, then internal text-fallback)
    matches = None
    try:
        active_idx = lost_state.get("active_index")
        if active_idx is not None:
            active_item = (lost_state.get("items") or [])[active_idx]
            raw_matches = active_item.get("external_matches") if active_item else None
            if raw_matches:
                trimmed = []
                for m in raw_matches[:10]:
                    trimmed.append({
                        'atcId': m.get('atcId'),
                        'collection': m.get('collection'),
                        'itemCategory': m.get('itemCategory'),
                        'itemName': m.get('itemName'),
                        'foundDate': m.get('foundDate'),
                        'storagePlace': m.get('storagePlace'),
                        'imageUrl': m.get('imageUrl'),
                        'score': m.get('score'),
                    })
                matches = trimmed
        if matches is None and not (user_meta and user_meta.get("attachments")):
            extracted = active_item.get("extracted") if active_item else None
            if extracted and (extracted.get("category") or extracted.get("region")):
                from app.services import faiss_index as _fi
                cat_q = (extracted.get("category") or "").lower().strip()
                region_q = (extracted.get("region") or "").lower().strip()
                lost_date = _parse_iso_date(extracted.get("lost_date")) if extracted.get("lost_date") else None
                candidates = []
                primary = []
                expanded = []
                for iid, meta in _fi.META.items():
                    if not isinstance(meta, dict):
                        continue
                    if meta.get("type") in {"media","lost_item"}:
                        continue
                    mcat = str(meta.get("category") or "").lower()
                    if cat_q and cat_q not in mcat:
                        continue
                    mplace = str(meta.get("found_place") or "").lower()
                    region_ok = True
                    if region_q:
                        region_ok = (region_q in mplace)
                    if not region_ok:
                        continue
                    # Date handling
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
                    if cat_q:
                        score += 0.55
                    if region_q and region_ok:
                        score += 0.25
                    # Date score (only when both dates present)
                    if lost_date and found_d:
                        delta_days = abs((found_d - lost_date).days)
                        max_window = PRIMARY_DATE_WINDOW_DAYS if delta_days <= PRIMARY_DATE_WINDOW_DAYS else EXPANDED_DATE_WINDOW_DAYS
                        date_score = max(0.0, 1 - delta_days / max_window)
                        score += date_score * 0.20
                    elif lost_date and not found_d:
                        score += 0.08  # partial credit for unknown date
                    entry = (iid, score, meta, in_primary, in_expanded)
                    candidates.append(entry)
                    if in_primary:
                        primary.append(entry)
                    elif in_expanded:
                        expanded.append(entry)
                def _ts(meta):
                    return meta.get('created_at') or meta.get('createdAt') or ''
                ranked_source = []
                if lost_date:
                    if primary:
                        ranked_source.extend(primary)
                    elif expanded:
                        ranked_source.extend(expanded)
                    else:
                        ranked_source.extend(candidates)
                else:
                    ranked_source = candidates
                ranked_source.sort(key=lambda x: (x[1], _ts(x[2])), reverse=True)
                trimmed = []
                for iid, score, meta, *_rest in ranked_source[:10]:
                    trimmed.append({
                        'atcId': meta.get('atcId') or meta.get('id') or iid,
                        'collection': 'internal-text',
                        'itemCategory': meta.get('category'),
                        'itemName': meta.get('caption') or meta.get('notes') or meta.get('category'),
                        'foundDate': meta.get('found_time'),
                        'storagePlace': meta.get('found_place'),
                        'imageUrl': None,
                        'score': round(score, 3),
                    })
                if trimmed:
                    matches = trimmed
    except Exception as e:
        print(f"[chat.send] matches build error: {e}")

    # Do NOT merge image similarity into matches (rollback behavior)

    # Removed: proxy link markdown injection (요청에 따라 링크 제공 X)

    # Merge image similarity candidates into matches with source tag
    try:
        if image_search_results:
            img_items = []
            for iid, score, meta in image_search_results:
                if not isinstance(meta, dict):
                    continue
                atc_id = meta.get('atcId') or meta.get('atc_id') or meta.get('id') or iid
                img_items.append({
                    'atcId': atc_id,
                    'collection': 'internal-image',
                    'itemCategory': meta.get('category'),
                    'itemName': meta.get('caption') or meta.get('notes') or meta.get('category'),
                    'foundDate': meta.get('found_time') or meta.get('foundDate') or meta.get('found_date'),
                    'storagePlace': meta.get('found_place'),
                    'imageUrl': meta.get('url') or meta.get('imageUrl') or meta.get('image_url'),
                    'score': round(float(score), 4),
                    'source': 'image'
                })
            if img_items:
                if matches:
                    seen = set()
                    merged = []
                    for it in img_items + matches:
                        aid = it.get('atcId')
                        if aid in seen:
                            continue
                        seen.add(aid)
                        merged.append(it)
                    matches = merged
                else:
                    matches = img_items
    except Exception as e:
        print(f"[chat.send] image->matches merge error: {e}")

    # Strict date filtering for matches (±STRICT_DATE_MATCH_DAYS) if user provided lost_date
    try:
        active_idx = lost_state.get("active_index")
        active_item = None
        if active_idx is not None:
            active_item = (lost_state.get("items") or [])[active_idx]
        extracted = (active_item or {}).get("extracted") or {}
        user_lost_date = _parse_iso_date(extracted.get("lost_date")) if extracted.get("lost_date") else None
        if user_lost_date and matches:
            filtered = []
            for m in matches:
                f_raw = m.get('foundDate') or m.get('found_time') or m.get('found_date')
                f_date = _parse_iso_date(f_raw) if isinstance(f_raw, str) else None
                if f_date and _within_window(f_date, user_lost_date, STRICT_DATE_MATCH_DAYS):
                    filtered.append(m)
            matches = filtered if filtered else []
    except Exception as e:
        print(f"[chat.send] strict date filter error: {e}")

    return SendMessageResponse(
        session_id=req.session_id,
        user_message=Message(id=user_msg_id, role="user", content=req.content, created_at=created_at, model=None, attachments=user_attachments),
        assistant_message=Message(id=assistant_msg_id, role="assistant", content=assistant_reply, created_at=assistant_created_at, model=chosen_model, attachments=None),
        active_lost_item=active_item_snapshot,
    matches=matches,
    )

@router.get("/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str, limit: int = 50):
    try:
        msgs = chat_store.fetch_messages(session_id, limit=limit)
    except ValueError:
        raise HTTPException(404, "session_not_found")
    # pass through attachments if present
    return HistoryResponse(
        session_id=session_id,
        messages=[Message(**m) for m in msgs]
    )


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(user_id: Optional[str] = None, limit: int = 50):
    # 없거나 공백이면 guest 로 간주 + 양끝 쌍따옴표 제거
    norm_user_id = (user_id or '').strip()
    if norm_user_id.startswith('"') and norm_user_id.endswith('"') and len(norm_user_id) >= 2:
        norm_user_id = norm_user_id[1:-1].strip()
    if not norm_user_id:
        norm_user_id = 'guest'
    sessions_raw = chat_store.list_sessions(user_id=norm_user_id, limit=limit)
    summaries = [SessionSummary(**s) for s in sessions_raw]
    return SessionListResponse(sessions=summaries)


@router.delete("/session/{session_id}", response_model=DeleteSessionResponse)
def delete_session(session_id: str):
    deleted = chat_store.delete_session(session_id)
    if not deleted:
        # 404 로 표현
        raise HTTPException(404, "session_not_found")
    return DeleteSessionResponse(session_id=session_id, deleted=True)
