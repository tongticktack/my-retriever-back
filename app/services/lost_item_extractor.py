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


def _normalize_color(token: str) -> str | None:
    if token in schema.COLORS:
        return token
    return schema.COLOR_SYNONYMS.get(token)


def _normalize_region(token: str) -> str | None:
    """Region normalization preserving user suffixes (역/터미널/공원 등).
    Accept token if:
      - Exact match in REGIONS, OR
      - Ends with a common suffix and base (without suffix) is in REGIONS.
    Returns token as-is (no shortening) to honor prompt rule.
    """
    if token in schema.REGIONS:
        return token
    common_suffixes = ["역", "터미널", "공원", "캠퍼스", "대학교"]
    for suf in common_suffixes:
        if token.endswith(suf) and len(token) > len(suf):
            base = token[:-len(suf)]
            if base in schema.REGIONS:
                return token  # keep original with suffix
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
    # colors (collect all then choose representative)
    tokens = re.split(r"[\s,./]+", text)
    colors: List[str] = []
    for tok in tokens:
        norm_c = _normalize_color(tok)
        if norm_c and norm_c not in colors:
            colors.append(norm_c)
    if colors:
        # representative selection by priority order
        rep = None
        for p in schema.COLOR_PRIORITY:
            if p in colors:
                rep = p
                break
        found["color"] = rep or colors[0]
        if len(colors) > 1:
            found["color_all"] = ",".join(colors)
    # regions (synonyms)
    for tok in tokens:
        norm_r = _normalize_region(tok)
        if norm_r:
            found["region"] = norm_r
            break
    m = DATE_PAT.search(text)
    if m:
        y, mo, d = m.groups()
        try:
            dt = datetime(int(y), int(mo), int(d))
            found["lost_date"] = dt.date().isoformat()
        except Exception:
            pass
    if "lost_date" not in found:
        rel_candidates = _resolve_relative_dates(text)
        if rel_candidates:
            if len(rel_candidates) == 1:
                found["lost_date"] = rel_candidates[0]
            else:
                # store ambiguity as auxiliary key for later clarification
                found["lost_date_candidates"] = ",".join(rel_candidates)
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
        # try infer category from subcategory membership
        for c, subs in schema.SUBCATEGORIES.items():
            if sub in subs:
                out["category"] = c
                out["subcategory"] = sub
                break
    color = extracted.get("color")
    if color in schema.COLORS:
        out["color"] = color
    if "color_all" in extracted:
        # keep raw list (filtered to known colors)
        valid_multi = [c for c in extracted["color_all"].split(",") if c in schema.COLORS]
        if len(valid_multi) > 1:
            out["color_all"] = ",".join(valid_multi)
    region = extracted.get("region")
    if region and len(region) <= 20:  # simple sanity
        out["region"] = region
    lost_date = extracted.get("lost_date")
    if lost_date and re.match(r"20\d{2}-\d{2}-\d{2}$", lost_date):
        out["lost_date"] = lost_date
    # propagate candidate list if present and no final date chosen
    if "lost_date" not in out and "lost_date_candidates" in extracted:
        out["lost_date_candidates"] = extracted["lost_date_candidates"]
    return out


## NOTE: LLM refinement & orchestration removed from this module.
## Use extraction_controller for any advanced logic.


def compute_missing(extracted: Dict[str, str]) -> List[str]:
    return [f for f in schema.REQUIRED_FIELDS if f not in extracted]


def build_missing_field_prompt(extracted: Dict[str, str], missing: List[str]) -> str:
    # Handle date ambiguity prompt
    if "lost_date_candidates" in extracted:
        cand = extracted["lost_date_candidates"].split(",")
        return (
            "날짜가 모호합니다. 아래 후보 중 하나를 선택하거나 정확한 날짜(YYYY-MM-DD)를 입력해주세요:\n" +
            " / ".join(cand)
        )
    if not missing:
        return "필요한 정보가 모두 수집되었습니다. 경찰청 검색을 진행할까요? (예/아니오)"
    order = [f for f in ["category", "subcategory", "color", "lost_date", "region"] if f in missing]
    next_field = order[0]
    friendly_map = {
        "category": "대분류 (예: 전자기기/의류/가방/지갑/액세서리)",
        "subcategory": "소분류 (예: 휴대폰/백팩 등)",
        "color": "색상 (예: 검정, 파랑 등)",
        "lost_date": "잃어버린 날짜 (예: 2025-08-15)",
        "region": "지역 (예: 서울 강남, 홍대 등)",
    }
    context_lines: List[str] = []
    # 친절하고 공감 있는 어조로 단일 항목 요청
    context_lines.append("아직 몇 가지 정보가 더 필요해요.")
    context_lines.append(f"먼저 {friendly_map[next_field]}를(을) 알려주실 수 있을까요?")
    # 간단한 안내 문구 추가 (사용자가 헷갈리지 않도록)
    if next_field == "lost_date":
        context_lines.append("정확한 날짜를 모르시면 '어제', '3일 전' 처럼 말씀해도 괜찮아요.")
    elif next_field == "region":
        context_lines.append("지역은 너무 길지 않게 (예: 서울 강남 / 신촌) 정도로 적어주세요.")
    if next_field == "category":
        context_lines.append("가능한 대분류: " + ", ".join(schema.PRIMARY_CATEGORIES))
    elif next_field == "subcategory" and extracted.get("category"):
        context_lines.append("가능한 소분류: " + ", ".join(schema.SUBCATEGORIES.get(extracted["category"], [])))
    elif next_field == "color":
        context_lines.append("예시 색상: " + ", ".join(schema.COLORS[:6]))
    return "\n".join(context_lines)


# build_llm_system_prompt: removed (deprecated & unused)
