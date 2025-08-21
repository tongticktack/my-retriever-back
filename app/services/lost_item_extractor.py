"""Lost item rule-based utilities & prompting helpers.

Provides minimal rule extraction + missing-field prompt builder. Advanced LLM
extraction & orchestration handled in extraction_controller.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import re
from datetime import datetime, timedelta, date
from app.domain import lost_item_schema as schema

DATE_PAT = re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")
RELATIVE_NUM_PAT = re.compile(r"(\d+)\s*일\s*전")

RELATIVE_KEYWORDS = {
    "오늘": 0,
    "어제": 1,
    "그제": 2,
    "그저께": 2,
    "이틀 전": 2,
    "사흘 전": 3,
    "나흘 전": 4,
    "닷새 전": 5,
    "엿새 전": 6,
    "일주일 전": 7,
    "한 주 전": 7,
    "한주 전": 7,
    "한달 전": 30,
    "한 달 전": 30,
    "두달 전": 60,
    "두 달 전": 60,
}


def _resolve_relative_dates(text: str, today: date | None = None) -> List[str]:
    """Resolve Korean relative date expressions to ISO date string.
    Returns ALL candidates (to detect ambiguity). Priority ordering kept.
    """
    if today is None:
        today = datetime.now().date()
    candidates: List[str] = []
    # all numeric patterns (there could be multiple like 2일 전이나 3일 전)
    for m in RELATIVE_NUM_PAT.finditer(text):
        try:
            days = int(m.group(1))
            target = today - timedelta(days=days)
            iso = target.isoformat()
            if iso not in candidates:
                candidates.append(iso)
        except Exception:
            continue
    lowered = text.lower()
    for key in sorted(RELATIVE_KEYWORDS.keys(), key=len, reverse=True):
        if key in lowered:
            days = RELATIVE_KEYWORDS[key]
            iso = (today - timedelta(days=days)).isoformat()
            if iso not in candidates:
                candidates.append(iso)
    return candidates


def _normalize_region(token: str) -> str | None:
    """Heuristic region detection (no static whitelist) with extra guards.

    추가 보호 이유: 이전 로직이 '아이폰을' 같은 품목+조사 결합을 장소로 잘못 인식.

    Rules:
        - Reject ultra-generic tokens: 역, 근처, 주변, 여기, 거기, 어디.
        - Strip common postpositions (조사) at tail: 을/를/은/는/이/가/도/에/에서/으로/로 if length>2.
        - Reject if stripped form matches (or 포함) 대표 품목 키워드 (폰, 노트북, 지갑, 가방, 백팩, 이어폰, 시계, 반지 등).
        - Accept if token (or stripped) ends with place suffix (역, 터미널, 공원, 캠퍼스, 대학교, 시장, 공항, 항, 산, 강) and stem length >=2.
        - Else accept if 2~6 pure Hangul AND not item keyword contaminated.
        - Keep stripped form (without 조사) as region value.
    """
    raw = token.strip()
    if not raw:
        return None
    generic = {"역", "근처", "주변", "여기", "거기", "어디"}
    if raw in generic:
        return None
    # Strip common particles
    particles_multi = ["에서", "으로"]
    particles_single = ["을", "를", "은", "는", "이", "가", "도", "에", "로"]
    stem = raw
    for p in particles_multi:
        if stem.endswith(p) and len(stem) > len(p)+1:  # ensure some stem
            stem = stem[:-len(p)]
            break
    else:
        if len(stem) > 2 and stem[-1] in particles_single:
            stem = stem[:-1]
    stem = stem.strip()
    if not stem:
        return None
    # Reject obvious item tokens
    item_keywords = {"폰", "휴대폰", "아이폰", "노트북", "맥북", "지갑", "가방", "백팩", "이어폰", "에어팟", "시계", "반지", "목걸이", "귀걸이"}
    for kw in item_keywords:
        if kw in stem:
            return None
    suffixes = ["역", "터미널", "공원", "캠퍼스", "대학교", "시장", "공항", "항", "산", "강"]
    for suf in suffixes:
        if stem.endswith(suf) and len(stem) > len(suf):
            return stem
    # pure Hangul short token 2~6 chars
    if 2 <= len(stem) <= 6 and all('가' <= c <= '힣' for c in stem):
        return stem
    return None


# _scan_with_spans: removed (unused)


def _analyze_rule(user_text: str) -> Dict[str, str]:
    text = user_text.strip()
    found: Dict[str, str] = {}
    # direct category/subcategory literal match
    for cat in schema.PRIMARY_CATEGORIES:
        if cat in text:
            found["category"] = cat
            for sub in schema.SUBCATEGORIES.get(cat, []):
                if sub in text:
                    found["subcategory"] = sub
                    break
            break
    # synonym-based inference if not already set
    if "subcategory" not in found:
        lowered = text.lower()
        for alias, pair in schema.SUBCATEGORY_SYNONYMS.items():
            if alias in lowered:
                cat, sub = pair
                found["category"] = found.get("category") or cat
                # ensure sub fits category list
                if sub in schema.SUBCATEGORIES.get(found["category"], []):
                    found["subcategory"] = sub
                break
    # direct subcategory mention (user provided 소분류만) -> infer category
    if "subcategory" not in found:
        # Scan all defined subcategories; first match wins
        for cat, subs in schema.SUBCATEGORIES.items():
            for sub in subs:
                if sub and sub in text:  # simple containment is acceptable for current Korean terms
                    found["subcategory"] = sub
                    # Only set category if not already inferred via other means
                    if "category" not in found:
                        found["category"] = cat
                    break
            if "subcategory" in found:
                break
    # regions (synonyms); tokenize once (색상 제거로 간단화)
    tokens = re.split(r"[\s,./]+", text)
    for tok in tokens:
        norm_r = _normalize_region(tok)
        if norm_r:
            # Skip overly generic tokens like just '역'
            if norm_r == '역':
                continue
            found["region"] = norm_r
            break
    # 날짜는 LLM 추론에 위임: 규칙 기반 절대/상대 날짜 파싱 비활성화.
    # (명시 YYYY-MM-DD 형태라 해도 여기서는 채우지 않고 LLM 일관 로직에 맡김)
    return found




def _validate(extracted: Dict[str, str]) -> Dict[str, str]:
    """Ensure values belong to allowed sets; drop invalid, fix subcategory/category mismatch."""
    out: Dict[str, str] = {}
    cat = extracted.get("category")
    if cat in schema.PRIMARY_CATEGORIES:
        out["category"] = cat
    sub = extracted.get("subcategory")
    if sub and cat and sub in schema.SUBCATEGORIES.get(cat, []):
        out["subcategory"] = sub
    elif sub and not cat:
        # infer category from subcategory membership (robust fallback)
        for c, subs in schema.SUBCATEGORIES.items():
            if sub in subs:
                out["category"] = c
                out["subcategory"] = sub
                break
    # 색상 관련 필드 제거 (color / color_all 무시)
    region = extracted.get("region")
    if region and len(region) <= 20:  # simple sanity
        out["region"] = region
    lost_date = extracted.get("lost_date")
    if lost_date and re.match(r"20\d{2}-\d{2}-\d{2}$", lost_date):
        out["lost_date"] = lost_date
    return out

def compute_missing(extracted: Dict[str, str]) -> List[str]:
    return [f for f in schema.REQUIRED_FIELDS if f not in extracted]


def build_missing_field_prompt(extracted: Dict[str, str], missing: List[str]) -> str:
    # 날짜 후보 브랜치 제거 (LLM 단일 추론)
    if not missing:
        return "필요한 정보가 모두 수집되었습니다. 검색을 진행할까요?"
    order = [f for f in ["category", "subcategory", "lost_date", "region"] if f in missing]
    next_field = order[0]
    # Free-form guidance replacing explicit enumeration
    if next_field == "category":
        return (
            "분실한 물건이 어떤 것인지 한두 단어 또는 짧은 구로 자연스럽게 묘사해 주세요. "
            "예: '아이폰', '가죽 지갑', '노트북', '백팩'. 제가 그 설명으로 분류를 추론할게요."
        )
    if next_field == "subcategory":
        return (
            "조금 더 구체적으로 어떤 종류인지 적어주세요. 예: '아이폰', '여성용 지갑', '게이밍 노트북', '무선 이어폰'. "
            "간단히 쓰셔도 제가 소분류를 추론합니다."
        )
    if next_field == "lost_date":
        return ("잃어버린 날짜를 알려주세요. 정확한 날짜가 기억나지 않으면 '어제', '3일 전', '지난주' 처럼 표현해도 돼요.")
    if next_field == "region":
        return ("어디에서 잃어버렸는지 간단히 알려주세요. 예: '강남', '신촌역', '홍대'. 너무 긴 주소는 피하고 핵심 지명만 주세요.")
    # Fallback (should not normally hit)
    return "분실물 정보를 조금 더 알려주세요." 


# build_llm_system_prompt: removed (deprecated & unused)
