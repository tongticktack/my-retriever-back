from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

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

class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None

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
    # Soft-fill color from image palette if any attachments and active collecting item missing color
    majority_image_color = None
    multi_image_conflict = False
    if user_meta and user_meta.get("attachments"):
        atts = user_meta.get("attachments")
        # Distinct-object heuristic: pairwise embedding similarity check
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
                # pairwise min similarity
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
        # 수집된 팔레트들을 기반으로 대표 hex 후보 집계
        hex_candidates = []
        for a in atts[:MAX_MEDIA_PER_MESSAGE]:
            pal = a.get("palette") or []
            if pal:
                hex_candidates.append(pal[0])
        # hex -> schema.COLORS 매핑 함수
        def _hex_to_color(hx: str) -> str | None:
            import re
            m = re.match(r"#([0-9a-fA-F]{6})", hx or "")
            if not m:
                return None
            rgbhex = m.group(1)
            r = int(rgbhex[0:2],16); g = int(rgbhex[2:4],16); b = int(rgbhex[4:6],16)
            if max(r,g,b) < 40:
                return "검정"
            if min(r,g,b) > 220:
                return "흰색"
            # dominant channel
            if r > g and r > b:
                return "빨강"
            if g > r and g > b:
                return "초록"
            if b > r and b > g:
                return "파랑"
            # grayscale-ish
            if abs(r-g) < 25 and abs(r-b) < 25:
                return "회색"
            return "갈색"
        color_votes = {}
        for hx in hex_candidates:
            c = _hex_to_color(hx)
            if c and c in schema.COLORS:
                color_votes[c] = color_votes.get(c,0)+1
        if color_votes:
            # majority pick (stable tie by sorted name)
            majority_image_color = sorted(color_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        hex_color = hex_candidates[0] if hex_candidates else None
        # run similarity search for each attachment (최대 3), best top-score set 채택
        try:
            from app.services import faiss_index as _fi
            cache = lost_state.get("_img_cache") or {}
            best = None
            best_score = -1.0
            active_index = lost_state.get("active_index")
            for idx, att in enumerate(atts[:MAX_MEDIA_PER_MESSAGE]):
                mid = att.get("media_id")
                if not mid:
                    continue
                emb = media_store.get_embedding(mid)
                if emb is None:
                    continue
                if active_index is not None:
                    item_key = f"lost_{req.session_id}_{active_index}_{idx}"
                    meta_kind = {"type":"lost_item","session_id":req.session_id,"active_index":active_index,"media_id":mid,"img_slot":idx}
                else:
                    item_key = f"media_{mid}_{idx}"
                    meta_kind = {"type":"media","session_id":req.session_id,"media_id":mid,"img_slot":idx}
                if item_key not in _fi.META:
                    _fi.add_item(item_key, emb, None, meta_kind)
                    try:
                        _fi.save_all()
                    except Exception as se:
                        print(f"[faiss.save_all] warn: {se}")
                cached = cache.get(mid)
                if cached is not None:
                    cand = cached
                else:
                    raw = _fi.search_image(emb, k=IMAGE_SIMILARITY_TOP_K + 1)
                    filtered = []
                    for iid, score, meta in raw:
                        if iid == item_key:
                            continue
                        if score < IMAGE_SIMILARITY_THRESHOLD:
                            continue
                        filtered.append((iid, score, meta))
                        if len(filtered) >= IMAGE_SIMILARITY_TOP_K:
                            break
                    cand = filtered if filtered else None
                    cache[mid] = cand
                if cand and cand[0][1] > best_score:
                    best_score = cand[0][1]
                    best = cand
            image_search_results = best
            lost_state["_img_cache"] = cache
        except Exception as e:
            print(f"[chat.image_search] error: {e}")
    if majority_image_color and lost_state.get("active_index") is not None:
        try:
            active = lost_state["items"][lost_state["active_index"]]
        except Exception:
            active = None
        if active:
            # merge media ids
            mid_list = [m.get("media_id") for m in atts if m.get("media_id")]
            if mid_list:
                existing = set(active.get("media_ids") or [])
                active["media_ids"] = list(existing.union(mid_list))
            extracted = active.get("extracted", {})
            if "color" not in extracted:
                extracted["color"] = majority_image_color
                sources = active.get("sources") or {}
                sources["color"] = "image-soft-fill-majority"
                active["sources"] = sources
                active["extracted"] = extracted
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
        assistant_reply, lost_state, chosen_model, active_item_snapshot = extc.process_message(
            req.content, lost_state, start_new_trigger, image_urls=image_urls
        )
    # After extraction reconcile image-derived color vs extracted color if mismatch
    try:
        if majority_image_color and lost_state.get("active_index") is not None:
            active = lost_state["items"][lost_state["active_index"]]
            extracted = active.get("extracted") or {}
            cur_color = extracted.get("color")
            if cur_color and majority_image_color and cur_color != majority_image_color:
                # Prefer image majority; keep old as alt_color
                extracted.setdefault("alt_color_llm", cur_color)
                extracted["color"] = majority_image_color
                sources = active.get("sources") or {}
                sources["color"] = "image-majority-override"
                active["sources"] = sources
                active["extracted"] = extracted
                # notify user in reply (prepend note)
                note = f"(이미지 다수결 색상이 LLM 추출 색상 '{cur_color}' 와 달라 '{majority_image_color}' 로 적용했습니다. 확인 부탁드립니다.)\n"
                assistant_reply = note + (assistant_reply or "")
    except Exception as e:
        print(f"[chat.color_reconcile] error: {e}")
    # If controller produced reply and we have similarity results, augment
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
        # enumerate each image id 1,2,3 ...
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
    # Persist each lost item to dedicated collection for MyPage listing
    try:
        # user_id 가 세션에 있을 수도 있고 요청에 있을 수도 있음
        session_user = session_doc.get("user_id") if session_doc else None
        user_for_items = (req.user_id or session_user or "guest")
        lost_item_store.bulk_upsert(user_for_items, req.session_id, lost_state.get("items", []))
    except Exception as e:
        print(f"[lost_item_store] bulk_upsert error: {e}")

    # lost-item 로직에서 assistant_reply 생성되면 LLM 생략
    if assistant_reply is None:
        llm = get_llm()
        provider_messages = build_messages(req.session_id, req.content, retrieval_items=None, law_summary=None)
        try:
            assistant_reply = llm.generate(provider_messages)
        except Exception as e:
            assistant_reply = f"(llm-error) {str(e)[:120]}"
        chosen_model = getattr(llm, "last_model_name", None) or getattr(llm, "model", None) or getattr(llm, "name", "unknown")
    # else chosen_model already set by controller

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
    return SendMessageResponse(
        session_id=req.session_id,
        user_message=Message(id=user_msg_id, role="user", content=req.content, created_at=created_at, model=None, attachments=user_attachments),
        assistant_message=Message(id=assistant_msg_id, role="assistant", content=assistant_reply, created_at=assistant_created_at, model=chosen_model, attachments=None),
        active_lost_item=active_item_snapshot,
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
