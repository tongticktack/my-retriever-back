from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date

from app.services import chat_store
from app.services.llm_providers import get_llm
from app.services.prompt_builder import build_messages
from app.services import extraction_controller as extc
from app.services import lost_item_store
from app.services import lost_item_extractor as li_ext
from app.services import media_store, faiss_index
from config import settings
from app.scripts.logging_config import get_logger

logger = get_logger("chat")

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
# Text search configuration
MIN_TEXT_SEARCH_SCORE = getattr(settings, 'MIN_TEXT_SEARCH_SCORE', 0.3)  # 최소 텍스트 검색 점수 임계값
MAX_PERSISTED_MATCHES = 50 

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
    content: Optional[str] = ""  # 이제 이미지-only 메시지 허용 (media_ids 있으면 빈 문자열 허용)
    user_id: Optional[str] = None  # (미인증 상태라 사용 안하지만 필드 예약)
    media_ids: Optional[List[str]] = None  # 업로드된 이미지 참조 ID 목록

class Message(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    model: str | None = None
    attachments: Optional[List[dict]] = None
    matches: Optional[List[dict]] = None  # persisted candidate matches (assistant only)

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
            logger.warning("invalid_media_id requested=%s found=%s", req.media_ids, len(medias))
            raise HTTPException(status_code=400, detail="invalid_media_id")
        if medias:
            user_meta = {"attachments": medias}
    # 이미지 전용 메시지 허용: content 비어있고 media_ids 존재하면 내부적으로 빈 문자열 저장
    safe_content = (req.content or "")
    try:
        user_msg_id = chat_store.add_message(req.session_id, "user", safe_content, meta=user_meta)
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
    # (색상 추출 기능 비활성화 상태)
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
                logger.error("multi_image_conflict_check error=%s", e)
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
    # Start-new trigger keywords retained but no longer required after a confirmed search; user can just describe a new item.
    start_new_trigger = any(kw in user_lower for kw in ["또 잃어버렸", "다른 물건", "새 물건", "추가 물건"]) or user_lower.startswith("새 물건")

    # 최초 이미지-only 메시지인 경우 (텍스트 없음 & 아직 active item 없음) 선행 아이템 생성하여 media_ids 보존
    if user_meta and user_meta.get("attachments") and not safe_content.strip() and (lost_state.get("active_index") is None):
        if not lost_state.get("items"):
            lost_state["items"] = []
        lost_state["items"].append({
            "extracted": {},
            "missing": li_ext.compute_missing({}),
            "stage": "collecting",
            "media_ids": [m.get("media_id") for m in user_meta.get("attachments") if m.get("media_id")]
        })
        lost_state["active_index"] = len(lost_state["items"]) - 1

    # Vision LLM: 이미지 URL 목록 (공개 URL 기준) 추출하여 controller 전달
    image_urls = None
    if user_meta and user_meta.get("attachments"):
        image_urls = [a.get("url") for a in user_meta.get("attachments") if a.get("url")]
    if multi_image_conflict:
        assistant_reply = (
            "여러 이미지가 서로 다른 물건으로 보입니다. 한 번에 하나의 분실물 이미지(최대 3장: 같은 물건 다른 각도)만 첨부해주세요. "
            "새로운 분실물이라면 바로 새 물건 설명 또는 사진을 바로 보내주셔도 됩니다."
        )
        chosen_model = "multi-image-validation"
        active_item_snapshot = None
    else:
        from time import time as _t
        _t0 = _t()
        assistant_reply, lost_state, chosen_model, active_item_snapshot = extc.process_message(
            req.content, lost_state, start_new_trigger, image_urls=image_urls
        )
    logger.info("extraction_controller latency=%.2fs model=%s reply_len=%d", (_t() - _t0), chosen_model, len(assistant_reply) if assistant_reply else 0)
    # After extraction: run metadata-based candidate filtering + image similarity (lazy) if user supplied images
    # 이미지 검색은 confirmed 상태에서만 실행 (확인 후에만)
    should_run_image_search = False
    if user_meta and user_meta.get("attachments") and not multi_image_conflict:
        active_idx = lost_state.get("active_index")
        if active_idx is not None:
            try:
                current_item = (lost_state.get("items") or [])[active_idx]
                current_stage = current_item.get("stage")
                # confirmed 상태에서만 이미지 검색 실행
                if current_stage == "confirmed":
                    should_run_image_search = True
            except Exception:
                pass
    
    if should_run_image_search and (assistant_reply is not None):
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
            # simple in-memory per-session cache placeholder (현재 미사용)
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
            logger.error("image_search.meta error=%s", e)
    
    # 이미지 유사도 검색 결과는 matches 필드로만 전달; 사용자 메시지에는 내부 media_id 노출 금지
    if user_meta and user_meta.get("attachments"):
        attachments_list = user_meta.get("attachments")
        count_imgs = len(attachments_list)
        ack_line = f"이미지 {count_imgs}장 확인했습니다."  # media_id 나열 제거
        # 이미지 전용(텍스트 미비) 안내는 extraction_controller 가 처리하므로 여기서는 단순 확인만
        if assistant_reply:
            assistant_reply = ack_line + "\n" + assistant_reply
        else:
            assistant_reply = ack_line
    chat_store.update_session(req.session_id, {"lost_items": lost_state})
    try:
        session_user = session_doc.get("user_id") if session_doc else None
        user_for_items = (req.user_id or session_user or "guest")
        # Ensure is_found flag exists (default False)
        norm_items = []
        for it in lost_state.get("items", []):
            if "is_found" not in it:
                it["is_found"] = False
            norm_items.append(it)
        lost_item_store.bulk_upsert(user_for_items, norm_items)
    except Exception as e:
        logger.error("bulk_upsert error=%s", e)

    if assistant_reply is None:
        llm = get_llm()
        provider_messages = build_messages(req.session_id, req.content, retrieval_items=None, law_summary=None)
        try:
            logger.info("general_llm_call provider=%s model=%s", getattr(llm,'name',None), getattr(llm,'model',None))
            assistant_reply = llm.generate(provider_messages)
        except Exception as e:
            assistant_reply = f"(llm-error) {str(e)[:120]}"
        chosen_model = getattr(llm, "last_model_name", None) or getattr(llm, "model", None) or getattr(llm, "name", "unknown")
        logger.info("general_llm_done model=%s len=%d", chosen_model, len(assistant_reply))

    # (assistant message 저장은 matches 생성 이후로 이동 - 완전 저장)
    assistant_msg_id = None  # placeholder; 실제 저장은 matches 계산 후

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
    # Build matches list (external, then internal text-fallback with image re-ranking)
    matches = None  # will hold full list we persist
    try:
        active_idx = lost_state.get("active_index")
        if active_idx is not None:
            active_item = (lost_state.get("items") or [])[active_idx]
            # 검색은 confirmed 단계에서만 수행 (사용자 확인 전에는 matches 비움)
            if active_item and active_item.get('stage') == 'confirmed':
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
                # Internal text search (confirmed 단계에서만)
                if matches is None:
                    extracted = active_item.get("extracted") if active_item else None
                    has_images = user_meta and user_meta.get("attachments")
                    has_meaningful_text = extracted and (extracted.get("category") or extracted.get("region"))
                    if extracted and (has_meaningful_text or has_images):
                        from app.services import faiss_index as _fi
                        from app.services.synonym_mapper import (
                            expand_search_terms_with_subcategory, is_category_match, calculate_category_subcategory_score
                        )
                        cat_q = (extracted.get("category") or "").lower().strip()
                        sub_q = (extracted.get("subcategory") or "").lower().strip()
                        region_q = (extracted.get("region") or "").lower().strip()
                        lost_date = _parse_iso_date(extracted.get("lost_date")) if extracted.get("lost_date") else None
                        expanded_categories, _target_subcats = expand_search_terms_with_subcategory(cat_q, sub_q)
                        candidates = []
                        primary = []
                        expanded = []
                        for iid, meta in _fi.META.items():
                            if not isinstance(meta, dict):
                                continue
                            if meta.get("type") in {"media","lost_item"}:
                                continue
                            mcat = str(meta.get("category") or "").lower()
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
                            if region_q and region_q in (meta.get('found_place') or '').lower():
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
                        def _ts(meta):
                            return meta.get('created_at') or meta.get('createdAt') or ''
                        ranked_source = []
                        if lost_date:
                            if primary:
                                ranked_source.extend(primary)
                                if len(primary) < 10:
                                    ranked_source.extend([e for e in expanded if e[1] >= 0.7][:5])
                            elif expanded:
                                ranked_source.extend(expanded)
                            else:
                                ranked_source.extend([e for e in candidates if e[1] >= 0.8])
                        else:
                            ranked_source = candidates
                        ranked_source.sort(key=lambda x: (x[1], _ts(x[2])), reverse=True)
                        trimmed = []
                        for iid, score, meta, *_rest in ranked_source[:20]:
                            trimmed.append({
                                'atcId': meta.get('atcId') or meta.get('id') or iid,
                                'collection': 'internal-text',
                                'itemCategory': meta.get('category'),
                                'itemName': meta.get('caption') or meta.get('notes') or meta.get('category'),
                                'foundDate': meta.get('found_time'),
                                'storagePlace': meta.get('found_place'),
                                'imageUrl': meta.get('url') or meta.get('imageUrl') or meta.get('image_url'),
                                'score': round(score, 3),
                                'meta_ref': meta
                            })
                        if trimmed:
                            matches = trimmed
            else:
                matches = []  # not confirmed yet
    except Exception as e:
        logger.error("matches_build error=%s", e)

    # 이미지 기반 재랭킹 (confirmed 상태에서 이미지가 있는 경우)
    try:
        if image_search_results and matches and user_meta and user_meta.get("attachments"):
            # 이미지 유사도 스코어 맵 생성
            image_scores = {}
            for iid, score, meta in image_search_results:
                atc_id = meta.get('atcId') or meta.get('atc_id') or meta.get('id') or iid
                image_scores[atc_id] = score

            # 텍스트 기반 matches에 이미지 유사도 스코어 적용하여 재랭킹
            for match in matches:
                atc_id = match.get('atcId')
                if atc_id in image_scores:
                    text_score = match.get('score', 0)
                    image_score = image_scores[atc_id]
                    # 가중치: 텍스트 0.4, 이미지 0.6으로 재랭킹
                    combined_score = (text_score * 0.4) + (image_score * 0.6)
                    match['score'] = round(combined_score, 4)
                    match['source'] = 'text+image'
                    match['image_score'] = round(image_score, 4)
                else:
                    # 이미지 유사도가 없는 경우 패널티 적용
                    match['score'] = match.get('score', 0) * 0.3
                    match['source'] = 'text-only'
            
            # 재랭킹된 결과 정렬 후 상위 10개만 유지
            matches.sort(key=lambda x: x.get('score', 0), reverse=True)
            matches = matches[:10]

            # meta_ref 제거 (응답에서 불필요)
            for match in matches:
                match.pop('meta_ref', None)

    except Exception as e:
        logger.error("image_rerank error=%s", e)

    # Do NOT merge raw image similarity into matches (이제 재랭킹으로 처리)

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
        logger.error("strict_date_filter error=%s", e)

    # 최종 matches 정리 및 크기 제한 (완전 저장 모드: 최대 MAX_PERSISTED_MATCHES)
    if matches is None:
        matches = []
    if len(matches) > MAX_PERSISTED_MATCHES:
        matches = matches[:MAX_PERSISTED_MATCHES]

    # assistant 메시지 Firestore 저장 (이 시점에 matches 결정됨)
    assistant_meta = {
        "model": chosen_model,
        "matches": matches,  # 완전 저장
    }
    from datetime import datetime as _dt, timezone as _tz
    assistant_meta["matches_generated_at"] = _dt.now(_tz.utc).isoformat()
    try:
        assistant_msg_id = chat_store.add_message(
            req.session_id,
            "assistant",
            assistant_reply,
            meta=assistant_meta
        )
    except Exception as e:
        # 저장 실패 시 로그만 남기고 진행 (응답에는 matches 포함)
        logger.error("assistant_persist error=%s", e)
        if not assistant_msg_id:
            assistant_msg_id = "assistant-save-failed"

    return SendMessageResponse(
        session_id=req.session_id,
        user_message=Message(id=user_msg_id, role="user", content=req.content, created_at=created_at, model=None, attachments=user_attachments),
        assistant_message=Message(id=assistant_msg_id, role="assistant", content=assistant_reply, created_at=assistant_created_at, model=chosen_model, attachments=None, matches=matches),
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
    enriched = []
    for m in msgs:
        # Firestore meta fields already flattened in fetch_messages; ensure matches included if stored
        if 'matches' in m and m.get('matches'):
            enriched.append(Message(**m))
        else:
            enriched.append(Message(**m))
    return HistoryResponse(session_id=session_id, messages=enriched)


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
