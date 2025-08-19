"""Placeholder schema lists for lost item info (to be replaced with real lists later)."""

PRIMARY_CATEGORIES = [
    "전자기기", "의류", "가방", "지갑", "액세서리"
]

SUBCATEGORIES = {
    "전자기기": ["휴대폰", "태블릿", "노트북", "이어폰", "카메라"],
    "의류": ["자켓", "셔츠", "바지", "모자"],
    "가방": ["백팩", "토트백", "크로스백"],
    "지갑": ["카드지갑", "장지갑", "동전지갑"],
    "액세서리": ["시계", "목걸이", "반지", "팔찌"],
}

# Brand/alias tokens that imply (category, subcategory)
# Used for lightweight deterministic inference when user doesn't supply exact 분류 단어.
SUBCATEGORY_SYNONYMS = {
    # 휴대폰
    "아이폰": ("전자기기", "휴대폰"),
    "iphone": ("전자기기", "휴대폰"),
    "갤럭시": ("전자기기", "휴대폰"),
    "galaxy": ("전자기기", "휴대폰"),
    # 노트북
    "맥북": ("전자기기", "노트북"),
    "macbook": ("전자기기", "노트북"),
    "랩탑": ("전자기기", "노트북"),
    "laptop": ("전자기기", "노트북"),
    # 이어폰
    "에어팟": ("전자기기", "이어폰"),
    "에어팟프로": ("전자기기", "이어폰"),
    "airpods": ("전자기기", "이어폰"),
    # 카메라
    "dslr": ("전자기기", "카메라"),
    "미러리스": ("전자기기", "카메라"),
    # 가방 종류 (synonyms)
    "배낭": ("가방", "백팩"),
    "백팩": ("가방", "백팩"),  # redundancy for completeness
    "토트": ("가방", "토트백"),
    "크로스": ("가방", "크로스백"),
    # 지갑
    "wallet": ("지갑", "카드지갑"),
    "지갑": ("지갑", "카드지갑"),  # default map when 지갑 alone (refinement later)
}

COLORS = ["빨강", "파랑", "초록", "검정", "흰색", "회색", "노랑", "갈색"]

# Synonym / alias normalization maps (simple, extend later)
COLOR_SYNONYMS = {
    "빨간": "빨강", "빨간색": "빨강", "레드": "빨강",
    "파란": "파랑", "파란색": "파랑", "블루": "파랑", "청색": "파랑",
    "초록색": "초록", "그린": "초록", "녹색": "초록",
    "검은": "검정", "검은색": "검정", "블랙": "검정",
    "화이트": "흰색", "하양": "흰색", "하얀": "흰색", "하얀색": "흰색",
    "그레이": "회색", "회": "회색", "회색빛": "회색",
    "노란": "노랑", "노란색": "노랑", "옐로": "노랑", "옐로우": "노랑",
    "브라운": "갈색", "갈색의": "갈색"
}

# Representative 색상 선정 우선순위 (첫 일치 사용)
COLOR_PRIORITY = COLORS  # 현재는 정의된 순서를 그대로 활용

REGIONS = [
    # 핵심 기본 지명
    "서울", "부산", "인천", "대구", "대전", "광주", "수원",
    # 자주 등장하는 세부/생활 지명 및 접미 포함 변형 (접미사 보존 지침 반영)
    "강남", "강남역", "홍대", "홍대입구", "신촌", "신촌역", "혜화역"
]

# REGION_SYNONYMS 제거: 프롬프트 지침(접미사 보존)에 맞춰 임의 축약 방지.

REQUIRED_FIELDS = ["category", "subcategory", "color", "lost_date", "region"]
