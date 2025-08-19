"""Extraction controller orchestrating friendly persona vs structured JSON extraction.

Responsibilities:
- Decide whether to run structured extraction LLM (JSON only) based on new user input & state.
- Run fast rule extraction first, then (conditionally) strict JSON LLM extraction and merge.
- Validate & normalize fields (delegates to lost_item_extractor._validate where possible).
- Determine next conversational action (ask missing field, disambiguate, confirm, or pass-through).
- Maintain per-item stages: collecting -> ready -> confirmed.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import re
import json
from datetime import datetime, timedelta
from pathlib import Path

from app.services import lost_item_extractor as li_ext
from app.domain import lost_item_schema as schema
from app.services.llm_providers import get_llm
from config import settings

DATE_ISO_RE = re.compile(r"20\d{2}-\d{2}-\d{2}$")

PERSONA_PREFIXES = [
    "제가 더 정확히 도와드리려면 ",
    "정확한 검색을 위해 ",
]

# Intent keyword heuristics
SHORT_GREETING_TOKENS = ["안녕", "안녕하세요", "hi", "hello", "ㅎㅇ"]
INTENT_LABELS = ["greeting", "policy", "item", "confirm", "cancel", "other"]
_INTENT_CACHE: Dict[str, str] = {}
_INTENT_CACHE_MAX = 500

# Toggle for final response guard LLM (lightweight style/safety polish)
ENABLE_RESPONSE_GUARD = True
# PROMPT_VERSION now driven by environment via config.Settings (PROMPT_VERSION env var)
PROMPT_VERSION = getattr(settings, 'PROMPT_VERSION', 1)

def _persona_wrap(prompt: str, kind: str) -> str:
    """Apply lightweight persona phrasing depending on reply type."""
    if kind == 'confirm':
        return prompt
    if kind == 'disambiguate':
        return "날짜가 조금 모호해요. " + prompt
    # ask (default)
    pref = PERSONA_PREFIXES[hash(prompt) % len(PERSONA_PREFIXES)]
    return pref + prompt


def _classify_intent_rule(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return "other"
    # short pure greeting only
    if any(t == g.lower() or (t.startswith(g.lower()) and len(t) <= len(g)+2) for g in SHORT_GREETING_TOKENS):
        return "greeting"
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
    if low in {t.lower() for t in CONFIRM_TOKENS}:
        _INTENT_CACHE[key] = "confirm"; return "confirm", "rule"
    if low in {t.lower() for t in CANCEL_TOKENS}:
        _INTENT_CACHE[key] = "cancel"; return "cancel", "rule"
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


def _strict_llm_extract(user_text: str, current: Dict[str, str]) -> Dict[str, str]:  # pragma: no cover (LLM)
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
    prompt_tmpl = _load_extraction_prompt()
    current_json = json.dumps(current, ensure_ascii=False) if current else "{}"
    if prompt_tmpl:
        placeholders = {
            'TODAY': today,
            'YESTERDAY': yesterday,
            'D3': d3,
            'CATEGORIES': ", ".join(schema.PRIMARY_CATEGORIES),
            'CURRENT_JSON': current_json,
            'USER_TEXT': user_text.strip(),
        }
        system = prompt_tmpl
        # simple safe replacement (avoids str.format JSON brace conflict)
        for k, v in placeholders.items():
            system = system.replace('{'+k+'}', v)
    else:
        system = "단일 JSON {category, subcategory, color, lost_date, region}; 없는 값 키 생략; today=" + today
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text.strip()}
    ]

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

    def _clean(parsed: Dict[str, Any]) -> Dict[str, str]:
        cleaned: Dict[str, str] = {}
        for k in ["category", "subcategory", "color", "lost_date", "region"]:
            v = parsed.get(k)
            if isinstance(v, str) and v.strip():
                cleaned[k] = v.strip()
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


def _merge_extracted(base: Dict[str, str], new: Dict[str, str]) -> Dict[str, str]:
    for k, v in new.items():
        if k not in base:
            base[k] = v
    return base


# _has_ambiguity: removed (was unused after logic refactor)


CONFIRM_TOKENS = {"예", "네", "진행", "검색", "yes", "y"}
CANCEL_TOKENS = {"아니오", "아니", "취소", "no", "n"}
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
    token = user_text.strip().lower()
    if token in {t.lower() for t in CONFIRM_TOKENS | CANCEL_TOKENS}:
        return False
    _log_metric('extraction.trigger', reason='always')
    return True


def process_message(user_text: str, lost_state: Dict[str, Any], start_new: bool) -> Tuple[str | None, Dict[str, Any], str, Dict[str, Any] | None]:
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
    if lost_state.get("soft_lock") and intent not in {"item"}:
        reply = _persona_wrap(
            "서비스가 분실물 회수 지원 모드로 잠시 잠겼어요. 분실한 물건의 종류, 장소, 날짜 중 하나라도 알려주시면 다시 도와드릴 수 있어요.",
            'ask'
        )
        return reply, lost_state, "intent:locked", _snapshot(idx, current)

    if intent in {"greeting", "policy", "other"}:
        if intent == "greeting":
            reply = _persona_wrap("만나서 반가워요! 분실물을 찾고 싶으시면 어떤 물건을 어디서 언제쯤 잃어버렸는지 말해 주세요.", 'ask')
            return reply, lost_state, "intent:greeting", _snapshot(idx, current)
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
                "먼저 분실물 기본 정보(카테고리/색상/날짜/장소)가 필요해요. 예: '지난주 신촌에서 검정 백팩 잃어버렸어요' 처럼 알려주세요." )
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
    if current.get("stage") == "ready":
        if user_lower in {"예", "네", "진행", "검색", "yes", "y"}:
            current["stage"] = "confirmed"
            snapshot = _snapshot(idx, current)
            return (
                "검색을 준비 중입니다. (경찰청 API 연동 예정) 다른 물건도 신고하시겠습니까? '새 물건' 입력으로 새로 시작해요.",
                lost_state,
                "lost-item-flow.v2",
                snapshot,
            )
        if user_lower in {"아니오", "아니", "취소", "no", "n"}:
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
        # Always call strict LLM for potential refinement
        missing_before = li_ext.compute_missing(extracted)
        _log_metric('extraction.llm_call', missing=len(missing_before), ambiguity='1' if 'lost_date_candidates' in extracted else '0')
        llm_part = _strict_llm_extract(user_text, extracted)
        _merge_extracted(extracted, llm_part)
        # Validate (will keep candidates if present and no final date)
        extracted = li_ext._validate(extracted)  # type: ignore[attr-defined]
        current["extracted"] = extracted
        current["missing"] = li_ext.compute_missing(extracted)

    # If user provided explicit date resolving candidates
    if "lost_date_candidates" in current.get("extracted", {}):
        token = user_text.strip()
        if DATE_ISO_RE.match(token):
            current["extracted"]["lost_date"] = token
            current["extracted"].pop("lost_date_candidates", None)
            current["missing"] = li_ext.compute_missing(current["extracted"])
        else:
            # maybe user picked one candidate verbatim
            cands = current["extracted"]["lost_date_candidates"].split(",")
            for c in cands:
                if c in user_text:
                    current["extracted"]["lost_date"] = c
                    current["extracted"].pop("lost_date_candidates", None)
                    current["missing"] = li_ext.compute_missing(current["extracted"])
                    break

    # Decide next prompt
    extracted = current.get("extracted", {})
    if "lost_date_candidates" in extracted:
        base = li_ext.build_missing_field_prompt(extracted, current.get("missing", []))
        reply = _persona_wrap(base, 'disambiguate')
        current["stage"] = "collecting"
    else:
        missing = current.get("missing", [])
        if missing:
            # Determine next field
            order = [f for f in ["category", "subcategory", "color", "lost_date", "region"] if f in missing]
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
                reply = _persona_wrap(_build_confirmation_summary(extracted), 'confirm')
            else:
                reply = None  # no need to say anything; allow general chat

    snapshot = _snapshot(idx, current)
    # Optional final response guard polishing (only for replies we generated here)
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
    friendly = []
    cat = extracted.get('category')
    sub = extracted.get('subcategory')
    if cat and sub:
        friendly.append(f"- **물건 종류**: {cat} ({sub})")
    elif cat:
        friendly.append(f"- **물건 종류**: {cat}")
    col = extracted.get('color')
    if col:
        friendly.append(f"- **색상**: {col}")
    ld = extracted.get('lost_date')
    if ld:
        friendly.append(f"- **잃어버린 날짜**: {ld}")
    reg = extracted.get('region')
    if reg:
        friendly.append(f"- **장소**: {reg}")
    if not friendly:
        return "수집된 정보를 정리하는 중이에요. 조금만 기다려주세요!"
    bullet = "\n".join(friendly)
    return (
        "**확인 한번만 부탁드려요!** 제가 꼬리를 살랑살랑 흔들면서 모은 정보는 아래와 같아요:\n\n" +
        bullet +
        "\n\n괜찮다면 **'예'** 라고 답해주시면 이제 루시가 주인님의 물건을 찾아볼 준비를 할게요! (예/아니오)"
    )


def _guess_field(extracted: Dict[str, str], field: str) -> str | None:  # pragma: no cover (LLM)
    """Use LLM to guess a plausible value for a missing field after repeated user non-response.
    Returns guessed string or None/UNKNOWN.
    """
    llm = get_llm()
    context_json = json.dumps(extracted, ensure_ascii=False)
    guidelines = {
        'category': '가능한 대분류 중 가장 가능성 높은 1개 (전자기기/의류/가방/지갑/액세서리).',
        'subcategory': '이미 category가 있다면 그 하위 소분류 중 가장 가능성 높은 1개.',
        'color': '일반적으로 많이 쓰이는 현실적인 색상 1개 (검정/파랑/흰색 등).',
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
