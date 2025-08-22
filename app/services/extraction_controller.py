"""Extraction controller orchestrating friendly persona vs structured JSON extraction.

Responsibilities:
- Decide whether to run structured extraction LLM (JSON only) based on new user input & state.
- Run fast rule extraction first, then (conditionally) strict JSON LLM extraction and merge.
- Validate & normalize fields (delegates to lost_item_extractor._validate where possible).
- Determine next conversational action (ask missing field, disambiguate, confirm, or pass-through).
- Maintain per-item stages: collecting -> ready -> confirmed.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import re
import json
from datetime import datetime, timedelta
from pathlib import Path

from app.services import lost_item_extractor as li_ext
from app.domain import lost_item_schema as schema
from app.services.llm_providers import get_llm
from app.services import external_search
from config import settings

DATE_ISO_RE = re.compile(r"20\d{2}-\d{2}-\d{2}$")

PERSONA_PREFIXES: List[str] = []  # deprecated; kept for backward compatibility

# Intent keyword heuristics
SHORT_GREETING_TOKENS = ["안녕", "안녕하세요", "hi", "hello", "ㅎㅇ"]
# Identity / capability inquiry patterns considered friendly opening, not off-topic
INTRO_PATTERNS = [
    r"너.*누구", r"너.*뭐하는", r"넌 누구", r"누구세요", r"정체성", r"역할이 뭐", r"무슨 역할", r"무엇을 할 수", r"뭐 할 수", r"무슨 일을", r"할 수 있어\??"
]
INTENT_LABELS = ["greeting", "policy", "item", "confirm", "cancel", "other"]
_INTENT_CACHE: Dict[str, str] = {}
_INTENT_CACHE_MAX = 500

# Toggle for final response guard LLM (lightweight style/safety polish)
ENABLE_RESPONSE_GUARD = True
# Optional stylistic enrichment (makes baseline functional message warmer & empathetic)
ENABLE_STYLE_ENRICH = True
# Allow light cute persona tone (dog helper) in enrichment
ALLOW_CUTE_PERSONA = True
# 1=은은, 2=표준 귀여움, 3=강조 (과잉 방지 로직 유지)
CUTE_TONE_LEVEL = 2
# PROMPT_VERSION now driven by environment via config.Settings (PROMPT_VERSION env var)
PROMPT_VERSION = getattr(settings, 'PROMPT_VERSION', 1)

# Vision confidence threshold for optional fields (brand/material/pattern ...)
VISION_OPTIONAL_CONF_THRESHOLD = 0.55
VISION_AUTO_OVERRIDE_THRESHOLD = 0.85  # high-confidence vision override threshold for critical fields

def _persona_wrap(prompt: str, kind: str) -> str:
    # Neutral passthrough; style handled by higher-level system prompt / guard.
    return prompt


def _classify_intent_rule(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return "other"
    # short pure greeting only OR identity/intro queries
    if any(t == g.lower() or (t.startswith(g.lower()) and len(t) <= len(g)+2) for g in SHORT_GREETING_TOKENS):
        return "greeting"
    for pat in INTRO_PATTERNS:
        try:
            if re.search(pat, t):
                return "greeting"
        except Exception:
            continue
    # Lightweight item heuristics: single/short tokens that look like category/subcategory/relative date/region supplement
    hangul_only = re.sub(r"[^가-힣]", "", t)
    token_count = len(re.split(r"\s+", t))
    # relative date keywords
    rel_date = any(kw in t for kw in ["어제","그제","그저께","3일 전","2일 전","지난주","일주일 전"])
    # direct category/subcategory membership or typical endings (카드, 역, 대역)
    if token_count <= 3:
        from app.domain import lost_item_schema as _schema  # local import to avoid cycle
        if any(cat == t for cat in _schema.PRIMARY_CATEGORIES):
            return "item"
        # flatten subcategories
        if any(t == sub for subs in _schema.SUBCATEGORIES.values() for sub in subs):
            return "item"
        if rel_date:
            return "item"
        # endings suggesting region (역) or item type (카드, 폰, 지갑, 노트북)
        if t.endswith("역") and len(t) >= 3:
            return "item"
        if any(t.endswith(suf) for suf in ["카드","폰","지갑","노트북"]):
            return "item"
    return "other"


def _intent_cache_key(text: str) -> str:
    return text.strip().lower()[:200]


def _evict_if_needed():
    if len(_INTENT_CACHE) > _INTENT_CACHE_MAX:
        # naive eviction: drop first 50
        for i, k in enumerate(list(_INTENT_CACHE.keys())):
            del _INTENT_CACHE[k]
            if i >= 49:
                break


def _load_intent_prompt() -> str:
    fname = f"app/prompts/intent_v{PROMPT_VERSION}.txt"
    path = Path(fname)
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _classify_intent_llm(text: str) -> str:
    llm = get_llm()
    system = _load_intent_prompt() or (
        "너는 인텐트 분류기. greeting|policy|item|confirm|cancel|other 중 하나만 출력."
    )
    user = f"메시지: {text.strip()}\n잘 생각하고 가장 알맞은 하나의 레이블만 출력."  # thinking nudge
    try:
        out = llm.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]).strip().lower()
    except Exception:
        return "other"
    token = re.split(r"[^a-z가-힣]+", out)[0] if out else "other"
    if token not in INTENT_LABELS:
        # simple containment fallback
        for lbl in INTENT_LABELS:
            if lbl in out:
                token = lbl
                break
    if token not in INTENT_LABELS:
        token = "other"
    return token


def classify_intent(text: str) -> Tuple[str, str]:
    """Return (intent, source) using rule fast-path + LLM fallback + cache."""
    key = _intent_cache_key(text)
    if key in _INTENT_CACHE:
        return _INTENT_CACHE[key], "cache"
    # confirm/cancel tokens quick
    low = key
    # Removed hard token shortcut for confirm/cancel to let LLM handle nuanced affirmation/negation.
    rule = _classify_intent_rule(text)
    if rule in {"greeting"}:
        _INTENT_CACHE[key] = rule; return rule, "rule"
    # LLM fallback for item vs other nuance
    llm_intent = _classify_intent_llm(text)
    _INTENT_CACHE[key] = llm_intent
    _evict_if_needed()
    return llm_intent, "llm"


def _rule_extract(text: str) -> Dict[str, str]:
    # Directly use internal rule analyzer (safe for controlled import)
    try:
        data = li_ext._analyze_rule(text)  # type: ignore[attr-defined]
    except Exception:
        data = {}
    return data


def _load_extraction_prompt() -> str:
    fname = f"app/prompts/extraction_v{PROMPT_VERSION}.txt"
    path = Path(fname)
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def _load_multimodal_prompt() -> str:
    # Prefer newest v4 taxonomy-aware prompt; fallback to v3.
    for fname in ["app/prompts/multimodal_extraction_v4.txt", "app/prompts/multimodal_extraction_v3.txt"]:
        p = Path(fname)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                continue
    return ""


def _strict_llm_extract(user_text: str, current: Dict[str, str], image_urls: Optional[List[str]] = None) -> Dict[str, Any]:  # pragma: no cover (LLM)
    """LLM JSON-only structured extraction using external prompt file with retry & light repair."""
    llm = get_llm()
    today = datetime.now().date().isoformat()
    # dynamic few-shot date replacements
    try:
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()
        d3 = (datetime.now().date() - timedelta(days=3)).isoformat()
    except Exception:
        yesterday = today
        d3 = today
    # Choose prompt: multimodal if images present else text-only
    if image_urls:
        prompt_tmpl = _load_multimodal_prompt() or _load_extraction_prompt()
    else:
        prompt_tmpl = _load_extraction_prompt()
    current_json = json.dumps(current, ensure_ascii=False) if current else "{}"
    if prompt_tmpl:
        placeholders = {
            'TODAY': today,
            'YESTERDAY': yesterday,
            'D3': d3,
            'D2': (datetime.now().date() - timedelta(days=2)).isoformat(),
            'D7': (datetime.now().date() - timedelta(days=7)).isoformat(),
            'CATEGORIES': ", ".join(schema.PRIMARY_CATEGORIES),
            'CURRENT_JSON': current_json,
            'USER_TEXT': user_text.strip(),
            'IMAGE_COUNT': str(len(image_urls) if image_urls else 0),
        }
        system = prompt_tmpl
        for k, v in placeholders.items():
            system = system.replace('{'+k+'}', v)
    else:
        system = "단일 JSON {category, subcategory, lost_date, region}; 없는 값 키 생략; today=" + today
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system}
    ]
    # Vision 지원 (OpenAI 4o-mini 등) - provider 가 openai 이고 이미지 URL 이 있으면 멀티모달 parts 구성
    llm_name = getattr(llm, 'name', '')
    if image_urls and llm_name == 'openai':
        parts: List[Dict[str, Any]] = []
        if user_text.strip():
            parts.append({"type": "text", "text": user_text.strip()})
        for url in image_urls[:3]:  # safety cap
            parts.append({"type": "image_url", "image_url": {"url": url}})
        # 이미지만 있는 경우를 위해 최소 한개의 text guidance 추가 (추출 지시)
        if not user_text.strip():
            parts.insert(0, {"type": "text", "text": "이미지에 보이는 분실물 정보를 category, subcategory, lost_date(모르면 빈칸), region(모르면 빈칸) JSON 추출"})
        messages.append({"role": "user", "content": parts})
    else:
        messages.append({"role": "user", "content": user_text.strip()})

    def _attempt_parse(raw: str) -> Dict[str, str] | None:
        snippet = raw.strip()
        if '```' in snippet:
            parts = snippet.split('```')
            # choose middle part if possible
            for p in parts:
                if '{' in p and '}' in p:
                    snippet = p; break
        snippet = snippet.strip()
        snippet = re.sub(r'^json\n', '', snippet, flags=re.IGNORECASE).strip()
        # direct parse
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # salvage braces
        m = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return None

    def _clean(parsed: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        scalar_keys = ["category", "subcategory", "lost_date", "region", "brand", "material", "pattern"]
        for k in scalar_keys:
            v = parsed.get(k)
            if isinstance(v, str):
                t = v.strip()
                if t:
                    cleaned[k] = t[:50]
        # Arrays
        def _norm_list(vals, limit):
            out = []
            if isinstance(vals, list):
                for x in vals:
                    if not isinstance(x, str):
                        continue
                    s = x.strip()
                    if not s:
                        continue
                    if len(s) > 25:
                        s = s[:25]
                    if s not in out:
                        out.append(s)
                    if len(out) >= limit:
                        break
            return out
        nf = _norm_list(parsed.get('notable_features'), 5)
        if nf:
            cleaned['notable_features'] = nf
        ts = _norm_list(parsed.get('text_snippets'), 4)
        # rudimentary PII filter: remove email-like / long digit sequences
        filtered_ts = []
        for s in ts:
            if '@' in s:
                continue
            if sum(c.isdigit() for c in s) >= 6:
                continue
            filtered_ts.append(s)
        if filtered_ts:
            cleaned['text_snippets'] = filtered_ts
        # confidences
        confidences = parsed.get('confidences')
        kept_conf: Dict[str, float] = {}
        if isinstance(confidences, dict):
            for k, v in confidences.items():
                try:
                    f = float(v)
                except Exception:
                    continue
                if f < 0 or f > 1:
                    continue
                kept_conf[k] = round(f, 2)
        # Apply threshold removal for optional fields
        for opt_key in ["brand", "material", "pattern"]:
            if opt_key in cleaned:
                conf_v = kept_conf.get(opt_key)
                if conf_v is not None and conf_v < VISION_OPTIONAL_CONF_THRESHOLD:
                    # drop field & its confidence entry
                    cleaned.pop(opt_key, None)
                    kept_conf.pop(opt_key, None)
        if kept_conf:
            cleaned['confidences'] = kept_conf
        return cleaned

    raw = ""
    for attempt in range(2):  # one retry if first malformed
        try:
            raw = llm.generate(messages).strip()
        except Exception:
            return {}
        parsed = _attempt_parse(raw)
        if parsed is not None:
            return _clean(parsed)
        # repair hint for retry: force user to output ONLY JSON
        messages.append({"role": "user", "content": "JSON 형식 오류. 순수 JSON 객체만 다시 출력."})
    return {}


def _merge_extracted(base: Dict[str, str], new: Dict[str, str], *, allow_override: bool = False, override_keys: Optional[List[str]] = None) -> Dict[str, str]:
    """Merge newly extracted fields into base.

    Default: non-destructive (first value wins) to avoid oscillation.
    If allow_override=True (e.g. explicit user correction at confirmation stage),
    then keys in override_keys (or all keys if None) can be overwritten when new has a non-empty differing value.
    """
    if not new:
        return base
    if allow_override:
        if override_keys is None:
            override_keys = list(new.keys())
        for k, v in new.items():
            if not isinstance(v, str):
                continue
            if k in override_keys and v and v.strip():
                # overwrite only if different
                if base.get(k) != v:
                    base[k] = v
        # Ensure we still add missing keys (if override_keys limited)
        for k, v in new.items():
            if k not in base and isinstance(v, str):
                base[k] = v
    else:
        for k, v in new.items():
            if k not in base and isinstance(v, str):
                base[k] = v
    return base


# _has_ambiguity: removed (was unused after logic refactor)


CONFIRM_TOKENS: set[str] = set()  # deprecated
CANCEL_TOKENS: set[str] = set()
OFF_TOPIC_KEYWORDS = [
    "날씨", "주식", "투자", "코인", "암호화폐", "건강", "병원", "진료", "식단", "아침에 뭐", "요리", "레시피",
    "운동", "다이어트", "공부", "시험", "여행", "호텔"
]
SOFT_LOCK_THRESHOLD = 3


def should_run_extraction(item: Dict[str, Any], user_text: str) -> bool:
    """Always run extraction in collecting/ready unless the message is a pure confirm/cancel token.
    This favors accuracy over minimal LLM calls.
    """
    stage = item.get("stage")
    if stage not in {"collecting", "ready"}:
        return False
    # Always run extraction; confirmation now inferred at higher layer (LLM intent), not by raw tokens.
    _log_metric('extraction.trigger', reason='always')
    return True


def process_message(user_text: str, lost_state: Dict[str, Any], start_new: bool, image_urls: Optional[List[str]] = None) -> Tuple[str | None, Dict[str, Any], str, Dict[str, Any] | None]:
    """Process user message relative to lost item flow.
    Returns (assistant_reply_or_none, updated_lost_state, model_label, active_item_snapshot).
    If assistant_reply_or_none is None the caller may proceed with general LLM chat.
    """
    if lost_state is None:
        lost_state = {"items": [], "active_index": None}

    if start_new or not lost_state.get("items"):
        lost_state["items"].append({"extracted": {}, "missing": li_ext.compute_missing({}), "stage": "collecting"})
        lost_state["active_index"] = len(lost_state["items"]) - 1

    idx = lost_state.get("active_index")
    if idx is None:
        return None, lost_state, "lost-item-flow.v2", None
    current = lost_state["items"][idx]
    user_lower = user_text.strip().lower()

    # Intent detection BEFORE extraction (greeting/policy may bypass item flow)
    intent, intent_source = classify_intent(user_text)
    _log_metric('intent.classified', intent=intent, source=intent_source)
    # Soft-lock handling: if previously locked and user still off-topic, short circuit
    if lost_state.get("soft_lock") and intent not in {"item", "greeting"}:
        reply = _persona_wrap(
            "서비스가 분실물 회수 지원 모드로 잠시 잠겼어요. 분실한 물건의 종류, 장소, 날짜 중 하나라도 알려주시면 다시 도와드릴 수 있어요.",
            'ask'
        )
        return reply, lost_state, "intent:locked", _snapshot(idx, current)
    # Allow greeting to release soft lock naturally
    if lost_state.get("soft_lock") and intent == 'greeting':
        lost_state.pop("soft_lock", None)

    if intent in {"greeting", "policy", "other"}:
        if intent == "greeting":
            # Pass through: allow general LLM (system prompt) to craft intro instead of fixed template
            return None, lost_state, "intent:greeting-pass", _snapshot(idx, current)
        if intent == "policy":
            reply = (
                "분실물은 발견 후 경찰/유실물 센터에 인계되면 통상 일정 기간(예: 6개월 전후, 품목/법규에 따라 차이) 보관 후 처리됩니다. "
                "정확한 최신 보관 기간은 관할 경찰청/지자체 공지 또는 공식 유실물 통합포털을 확인해 주세요. \n"
                "물건 정보를 알려주시면 바로 구조화해서 검색 준비를 도와드릴게요."
            )
            return _persona_wrap(reply, 'ask'), lost_state, "intent:policy", _snapshot(idx, current)
        # other (off-topic): block general chat & gently/firmly redirect
        lowered = user_text.lower()
        keyword_hit = next((kw for kw in OFF_TOPIC_KEYWORDS if kw in lowered), None)
        count = lost_state.get("off_topic_count", 0) + 1
        lost_state["off_topic_count"] = count
        if count > SOFT_LOCK_THRESHOLD:
            lost_state["soft_lock"] = True
            _log_metric('intent.soft_lock', count=count)
            msg = "분실물 정보 없이 다른 주제 대화가 계속되어 잠시 잠금 상태입니다. 예: '어제 강남에서 검정 백팩 잃어버렸어요' 처럼 알려주시면 다시 도와드릴게요."
            return _persona_wrap(msg, 'ask'), lost_state, "intent:redirect-lock", _snapshot(idx, current)
        if keyword_hit:
            msg = f"해당 주제(예: {keyword_hit})는 이 서비스 범위를 벗어나요. 분실한 물건의 종류와 장소, 날짜 중 하나라도 알려주시면 도와드릴게요. 예: '3일 전 신촌에서 파란 휴대폰 잃어버렸어요'"
        elif count == 1:
            msg = (
                "이 서비스는 분실물 회수 지원에 집중해요. 어떤 물건을 어디서 언제쯤 잃어버렸는지 알려주시면 도와드릴게요. "
                "예: '어제 강남에서 파란 휴대폰 잃어버렸어요'"
            )
        elif count == 2:
            msg = (
                "먼저 분실물 기본 정보(카테고리/날짜/장소)가 필요해요. 예: '지난주 신촌에서 백팩 잃어버렸어요' 처럼 알려주세요." )
        else:
            msg = "분실물 정보(예: 언제, 어디서, 어떤 물건)를 입력해주셔야 계속 도와드릴 수 있어요."
        return _persona_wrap(msg, 'ask'), lost_state, "intent:redirect", _snapshot(idx, current)

    # confirm/cancel intents outside ready stage → treat politely as guidance
    if intent == 'confirm' and current.get('stage') != 'ready':
        # Provide guidance on missing fields
        missing = current.get('missing') or li_ext.compute_missing(current.get('extracted', {}))
        if missing:
            base = li_ext.build_missing_field_prompt(current.get('extracted', {}), missing)
            return _persona_wrap("아직 확인 전이에요. " + base, 'ask'), lost_state, 'intent:confirm-misplaced', _snapshot(idx, current)
    if intent == 'cancel' and current.get('stage') != 'ready':
        # Just encourage providing info
        return _persona_wrap("진행을 취소할 단계가 아직 아니에요. 물건 정보를 조금 더 알려주실래요?", 'ask'), lost_state, 'intent:cancel-misplaced', _snapshot(idx, current)

    # Confirmation handling (only when intent=item)
    if current.get("stage") == "ready" and intent == 'confirm':
        current["stage"] = "confirmed"
        snapshot = _snapshot(idx, current)
        # Run approximate external search now (after explicit user confirm)
        search_msg = "공개 습득물 근사 검색을 시작합니다."
        try:
            matches = external_search.approximate_external_matches(current.get("extracted") or {}, max_results=5)
            if matches:
                current['external_matches'] = matches
                summary_line = external_search.summarize_matches(matches, limit=3)
                search_msg += "\n\n" + summary_line + "\n더 보고 싶으시면 '후보 자세히'라고 말씀해주세요."
            else:
                search_msg += "\n(현재 정보로 즉시 유사 후보 없음)"
        except Exception as e:
            print('[external_search] confirm-stage search failed', e)
            search_msg += "\n(검색 중 오류가 발생했어요. 잠시 후 다시 시도해주세요.)"
        search_msg += "\n다른 분실물을 새로 등록하려면 '새 물건'이라고 입력하시면 됩니다."
        snapshot = _snapshot(idx, current)
        return (search_msg, lost_state, "lost-item-flow.v2", snapshot)
    if current.get("stage") == "ready" and intent == 'cancel':
        current["stage"] = "collecting"
        # fall through to ask again

    # If confirmed and user gives more info without start_new trigger -> start a new item
    if current.get("stage") == "confirmed" and not start_new:
        lost_state["items"].append({"extracted": {}, "missing": li_ext.compute_missing({}), "stage": "collecting"})
        lost_state["active_index"] = len(lost_state["items"]) - 1
        idx = lost_state["active_index"]
        current = lost_state["items"][idx]

    # If previously soft-locked and now user provides item intent, unlock
    if lost_state.get("soft_lock") and intent == 'item':
        lost_state.pop("soft_lock", None)
        _log_metric('intent.soft_lock_release')

    # Extraction decision
    prev_extracted = current.get("extracted", {}).copy()
    prev_missing = current.get("missing", [])
    if should_run_extraction(current, user_text):
        extracted = current.get("extracted", {}).copy()
        # Rule first
        rule_part = _rule_extract(user_text)
        _merge_extracted(extracted, rule_part)
        # Always call strict LLM for potential refinement (now vision-aware)
        missing_before = li_ext.compute_missing(extracted)
        _log_metric('extraction.llm_call', missing=len(missing_before), ambiguity='1' if 'lost_date_candidates' in extracted else '0')
        llm_part = _strict_llm_extract(user_text, extracted, image_urls=image_urls)
        prev_keys = set(extracted.keys())
        pre_values = {k: extracted.get(k) for k in ["category", "subcategory"] if extracted.get(k)}
        # Detect explicit correction intent when at ready stage and user negates/overrules
        correction_signal = False
        is_ready_stage = current.get("stage") == "ready"
        if is_ready_stage:
            if re.search(r"(수정|정정|아니라|아닌|다시|틀렸|바꿔|변경|맞(?!.*않)|오타)", user_text):
                correction_signal = True
        # Implicit correction heuristic: short message providing a single differing field value
        implicit = False
        if is_ready_stage and not correction_signal and llm_part:
            core_fields = ["category", "subcategory", "lost_date", "region"]
            diffs = [f for f in core_fields if f in llm_part and f in extracted and llm_part[f] != extracted[f]]
            # message length & token heuristics: very short or label-prefixed
            text_len = len(user_text.strip())
            token_count = len(re.split(r"\s+", user_text.strip())) if user_text.strip() else 0
            label_prefix = bool(re.match(r"^(장소|지역|날짜|종류|카테고리|subcategory)[:\s]", user_text.strip()))
            # avoid verbs indicating new narrative (잃, 찾, 분실 등) -> more likely new item
            verb_like = re.search(r"(잃|찾|분실|도와|검색)", user_text)
            if diffs and len(diffs) <= 2 and text_len <= 25 and token_count <= 5 and not verb_like:
                implicit = True
            elif diffs and label_prefix and not verb_like:
                implicit = True
        if implicit:
            correction_signal = True
        # Merge (allow override only for core fields during correction)
        _merge_extracted(
            extracted,
            llm_part,
            allow_override=correction_signal,
            override_keys=["category", "subcategory", "lost_date", "region"]
        )
        # Region post-processing: remove relative date words mistakenly captured as region, attempt recovery
        try:
            REL_DATE = {"어제","오늘","그저께","그제","이틀전","삼일전","3일전","2일전","최근","방금"}
            reg_val = extracted.get('region')
            if reg_val and reg_val in REL_DATE:
                extracted.pop('region', None)
            if not extracted.get('region'):
                # Attempt to recover place from user_text tokens
                import re as _re
                tokens = _re.split(r"\s+", user_text.strip())
                cand = None
                for t in tokens:
                    base = t
                    m = _re.match(r"(.+?)(?:에서|에)$", t)
                    if m:
                        base = m.group(1)
                        # Treat anything that had '에서' suffix as a location candidate directly if length ok
                        if 2 <= len(base) <= 15 and base not in REL_DATE:
                            cand = base
                            break
                    if len(base) < 2 or len(base) > 15:
                        continue
                    # Heuristic: ends with typical location suffix OR in gazetteer list
                    if base.endswith(("역","구","동","터미널","공항")) or base in {"종로","강남","신촌","홍대","잠실","건대","인천공항"}:
                        cand = base
                        break
                if cand:
                    extracted['region'] = cand
        except Exception as _e:
            print('[extraction.region_sanitize] error', _e)
        # Source tagging + conflict detection
        conflicts = current.get('conflicts') or {}
        vision_conf_map = {}
        if llm_part and isinstance(llm_part, dict):
            conf_dict = llm_part.get('confidences') if isinstance(llm_part.get('confidences'), dict) else {}
            # normalize confidence numbers
            for ck, cv in conf_dict.items():
                try:
                    vision_conf_map[ck] = float(cv)
                except Exception:
                    pass
            sources = current.setdefault('sources', {})
            for k in llm_part.keys():
                if k == 'confidences':
                    continue
                if k not in prev_keys and k not in sources and k in extracted:
                    sources[k] = 'vision' if image_urls else 'llm'
            # Detect conflicts for core fields (text value retained unless override criteria)
            for field in ["category", "subcategory"]:
                if field in llm_part and field in pre_values and llm_part[field] != pre_values[field]:
                    # store conflict structure
                    if field not in conflicts:
                        conflicts[field] = {
                            'text_value': pre_values[field],
                            'vision_value': llm_part[field],
                            'vision_confidence': vision_conf_map.get(field)
                        }
            # Auto override rule (category/subcategory high confidence override)
            for field in ["category", "subcategory"]:
                if field in conflicts:
                    vc = conflicts[field].get('vision_confidence') or 0.0
                    if vc >= VISION_AUTO_OVERRIDE_THRESHOLD:
                        # perform override; keep original as alt_<field>_text
                        original = conflicts[field]['text_value']
                        extracted[field] = conflicts[field]['vision_value']
                        extracted[f'alt_{field}_text'] = original
                        sources[field] = 'vision-override'
                        conflicts[field]['auto_overridden'] = True
                    else:
                        conflicts[field]['auto_overridden'] = False
            if conflicts:
                current['conflicts'] = conflicts
            current['sources'] = sources
        # Validate (will keep candidates if present and no final date)
        extracted = li_ext._validate(extracted)  # type: ignore[attr-defined]
        current["extracted"] = extracted
        current["missing"] = li_ext.compute_missing(extracted)

    # Decide next prompt (date 후보 로직 제거)
    extracted = current.get("extracted", {})
    missing = current.get("missing", [])
    if missing:
            # Determine next field
            order = [f for f in ["category", "subcategory", "lost_date", "region"] if f in missing]
            next_field = order[0] if order else None
            # Track ask attempts
            ask_counts = current.setdefault("ask_counts", {})
            changed = extracted.keys() != prev_extracted.keys() or any(extracted.get(k) != prev_extracted.get(k) for k in prev_extracted.keys())
            if next_field:
                if (not changed) and prev_missing == missing:
                    ask_counts[next_field] = ask_counts.get(next_field, 0) + 1
                else:
                    # reset counter when progress made or field changed
                    ask_counts[next_field] = 0
                # Guess after 2 unsuccessful asks (count >=2)
                if ask_counts.get(next_field, 0) >= 2:
                    _log_metric('guess.attempt', field=next_field)
                    guess = _guess_field(extracted, next_field)
                    if guess and guess.upper() != 'UNKNOWN':
                        extracted[next_field] = guess
                        # revalidate & recompute
                        extracted = li_ext._validate(extracted)  # type: ignore[attr-defined]
                        current["extracted"] = extracted
                        current["missing"] = li_ext.compute_missing(extracted)
                        missing = current["missing"]
                        ask_counts[next_field] = 0
                        _log_metric('guess.filled', field=next_field)
            current["stage"] = "collecting"
            base = li_ext.build_missing_field_prompt(extracted, missing)
            reply = _persona_wrap(base, 'ask')
    else:
        if current.get("stage") != "confirmed":
            current["stage"] = "ready"
            confirmation = _build_confirmation_summary(extracted)
            reply = _persona_wrap(confirmation, 'confirm')
            if not current.get("media_ids") and not current.get("asked_photo"):
                reply += "\n\n추가로 사진이 있다면 지금 최대 3장까지 올려주세요. 없으면 그냥 계속 말씀하거나 검색 진행 의사를 자연스럽게 표현해 주세요."
                current["asked_photo"] = True
        else:
            reply = None  # no need to say anything; allow general chat

    snapshot = _snapshot(idx, current)
    # Optional final response guard polishing (only for replies we generated here)
    if reply and ENABLE_STYLE_ENRICH:
        enriched = _style_enrich(reply, intent="item", item_snapshot=snapshot)
        if enriched:
            reply = enriched
    if reply and ENABLE_RESPONSE_GUARD:
        guarded = _guard_response(reply, intent="item", model_label="lost-item-flow.v2", item_snapshot=snapshot)
        if guarded:
            reply = guarded
    return reply, lost_state, "lost-item-flow.v2", snapshot


def _snapshot(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "index": idx,
        "stage": item.get("stage"),
        "extracted": item.get("extracted"),
        "missing": item.get("missing"),
    }


def _log_metric(event: str, **fields: Any) -> None:
    ts = datetime.now().isoformat()
    compact = ' '.join(f"{k}={v}" for k, v in fields.items())
    print(f"[metric] {ts} {event} {compact}")


def _build_confirmation_summary(extracted: Dict[str, Any]) -> str:
    parts = []
    cat = extracted.get('category'); sub = extracted.get('subcategory')
    if cat and sub: parts.append(f"종류: {cat} ({sub})")
    elif cat: parts.append(f"종류: {cat}")
    # 색상 출력 제거
    ld = extracted.get('lost_date');
    if ld: parts.append(f"날짜: {ld}")
    reg = extracted.get('region');
    if reg: parts.append(f"장소: {reg}")
    if not parts:
        return "정보 정리 중입니다."
    return "확인해주세요:\n" + "\n".join(f"- {p}" for p in parts) + "\n수정할 내용이 있으면 바로 적어주시고, 괜찮으면 계속 진행 의사를 자연스럽게 표현해주세요."


def _guess_field(extracted: Dict[str, str], field: str) -> str | None:  # pragma: no cover (LLM)
    """Use LLM to guess a plausible value for a missing field after repeated user non-response.
    Returns guessed string or None/UNKNOWN.
    """
    llm = get_llm()
    context_json = json.dumps(extracted, ensure_ascii=False)
    guidelines = {
        'category': '가능한 대분류 중 가장 가능성 높은 1개 (전자기기/의류/가방/지갑/액세서리).',
        'subcategory': '이미 category가 있다면 그 하위 소분류 중 가장 가능성 높은 1개.',
    # 색상 항목 제거
        'lost_date': '최근 10일 이내의 날짜 중 합리적인 1개 (YYYY-MM-DD). 지나치게 임의 느낌 피하기.',
        'region': '한국 내 일반적인 지명 1개 (예: 강남, 신촌, 서울). 너무 상세 주소 피함.'
    }
    system = (
        f"분실물 추론 보조기. 주어진 partial JSON: {context_json}\n" \
        f"추정 대상 필드: {field}\n" \
        f"규칙: 1) 한 단어 또는 짧은 구 2) 자신없으면 UNKNOWN 3) 추가 설명 금지 4) 출력은 값만.\n" \
        f"가이드: {guidelines.get(field,'값 1개')}"
    )
    try:
        out = llm.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": "추정값만 출력"}
        ]).strip()
    except Exception:
        return None
    # Sanitize multi-line -> first line
    if '\n' in out:
        out = out.split('\n',1)[0].strip()
    # Basic cleanup
    out = out.strip().strip('"').strip()
    if not out:
        return None
    if len(out) > 25:
        return None
    return out


def _load_response_guard_prompt() -> str:
    fname = f"app/prompts/response_guard_v{PROMPT_VERSION}.txt"
    path = Path(fname)
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _guard_response(draft: str, intent: str, model_label: str, item_snapshot: Dict[str, Any]) -> str:
    """Run lightweight LLM pass to enforce style/safety rules. Fallback to original on failure."""
    try:
        prompt_tmpl = _load_response_guard_prompt()
        if not prompt_tmpl:
            return draft
        llm = get_llm()
        item_json = json.dumps(item_snapshot.get("extracted") or {}, ensure_ascii=False)
        system = prompt_tmpl.format(
            INTENT=intent,
            MODEL_LABEL=model_label,
            ITEM_JSON=item_json,
            REPLY_DRAFT=draft.strip(),
        )
        # Use single system style: guard prompt includes all variables; user empty to avoid leakage
        out = llm.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": ""}
        ]).strip()
        if out:
            # basic sanity: avoid returning raw labels or json markers
            if out.count('{') > 2 and len(out) - len(out.replace('{', '')) > 2:
                return draft
            return out
        return draft
    except Exception:
        return draft


def _style_enrich(draft: str, intent: str, item_snapshot: Dict[str, Any]) -> str:
    """Lightweight style enrichment: expand tone with empathy & clarity while preserving facts.
    Keeps bullet structure, field labels, JSON fragments. Avoids adding yes/no rigid prompts.
    """
    try:
        llm = get_llm()
        item_json = json.dumps(item_snapshot.get("extracted") or {}, ensure_ascii=False)
        if ALLOW_CUTE_PERSONA:
            # Few-shot 스타일 예시는 간결하게 포함 (과한 토큰 낭비 방지)
            examples = (
                "원문: '정보 정리 중입니다.'\n"
                "개선: '살짝 코를 킁킁하면서 정리 중이에요… 잠깐만 기다려주세요 🐾'\n\n"
                # 색상 관련 예시 제거 (색상 추출 비활성화)
            ) if CUTE_TONE_LEVEL >= 2 else ""
            intensity = {
                1: "은은하고 절제된",
                2: "눈에 띄지만 과하지 않은",
                3: "조금 더 적극적이지만 여전히 자연스러운"
            }.get(CUTE_TONE_LEVEL, "눈에 띄지만 과하지 않은")
            max_emoji = {1:1,2:3,3:4}.get(CUTE_TONE_LEVEL,3)
            tail_expr = {1:1,2:2,3:3}.get(CUTE_TONE_LEVEL,2)
            system = (
                f"너는 분실물 도우미 '강아지' 캐릭터다. DRAFT를 더 따뜻하고 {intensity} 강아지스러운 말투로 재작성하되 사실/필드/숫자/날짜는 절대 변형하지 마. "
                "규칙:\n"
                f"1) 말투: 부드러운 존댓말+살짝 귀여운 어미(~요, ~했어요, ~할게요). 과한 애교 금지.\n"
                f"2) 강아지 표현(예: 꼬리 살랑, 코 킁킁, 열심히 찾아볼게요) 최대 {tail_expr}회.\n"
                f"3) 이모지 0~{max_emoji}개 (허용: 🐶 🐾 🔍 ✨). 연속 사용 금지.\n"
                "4) 필요하면 한 줄 공감 (걱정되셨죠 / 잘 찾아볼게요 등).\n"
                "5) 불릿/구조/핵심 키워드 유지, 불릿 안 값 절대 변경하지 말기.\n"
                "6) 새로운 '예/아니오' 강요 문구 만들지 말기.\n"
                "7) 추측/허위/과장 추가 금지. 모르는 건 건너뜀.\n"
                "8) 2~6문장, 380자 이내.\n"
                "9) 사진 요청 이미 있으면 반복하지 말기.\n"
                "10) 동일 이모지 반복 사용 자제.\n"
                "출력: 재작성 텍스트만.\n\n"
                f"간단 예시:\n{examples}" 
            )
        else:
            system = (
                "너는 답변 스타일 향상 모듈. 입력 초안(DRAFT)을 한국어로 더 자연스럽고 공감 있게 다듬되, 의미/사실/필드명/날짜/카테고리 등 핵심 정보는 절대 삭제/왜곡하지 마. "
                "규칙:\n"
                "1) 목록/불릿/코드/JSON 구조 유지.\n"
                "2) 과한 이모지 피하고 0~3개 이모지 허용.\n"
                "3) 사용자가 편안히 느끼도록 짧은 공감 1문장 가능.\n"
                "4) '예/아니오' 같은 이분법 강요 표현 추가하지 말 것.\n"
                "5) 2~5문장 또는 250자 이내.\n"
                "6) DRAFT 내 불릿은 그대로 두고 필요하면 불릿 위/아래에 간결 문장만 추가.\n"
                "7) 추측 추가 금지.\n"
                "8) 사진 요청 문구가 이미 있다면 중복 요청 피함.\n"
                "출력은 다듬어진 텍스트만."
            )
        user = f"ITEM_JSON: {item_json}\nINTENT: {intent}\nDRAFT:\n{draft}\n\n개선된 응답:"  # context bundling
        out = llm.generate([
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]).strip()
        if not out:
            return draft
        # Basic safety: ensure core field tokens not lost (if they existed)
        core_tokens = []
        for k, v in (item_snapshot.get("extracted") or {}).items():
            if isinstance(v, str) and v and v in draft:
                core_tokens.append(v)
        missing_core = [t for t in core_tokens if t not in out]
        if missing_core:
            return draft  # revert to be safe
        if len(out) > 800:
            return draft
        return out
    except Exception:
        return draft
