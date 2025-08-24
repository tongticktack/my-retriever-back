"""
분실물 검색을 위한 유사어/동의어 매핑
# 지역은 정확한 매칭만 사용 (동의어 확장 없음)
# 더 구체적인 정보를 일반적인 정보로 축소하지 않기 위해
"""
from typing import List, Set, Dict, Tuple, Optional
import re
from app.domain import lost_item_schema as schema

# 카테고리 동의어 매핑
CATEGORY_SYNONYMS = {
    '휴대폰': ['핸드폰', '스마트폰', '폰', '아이폰', '갤럭시', '스마트워치폰'],
    '지갑': ['월렛', '카드케이스', '돈지갑', '카드지갑'],
    '가방': ['백팩', '배낭', '숄더백', '토트백', '크로스백', '서류가방', '핸드백'],
    '카드': ['신용카드', '체크카드', '교통카드', '학생증', '사원증', '멤버십카드'],
    '열쇠': ['키', '차키', '집키', '사무실키', '자동차열쇠'],
    '전자기기': ['태블릿', '패드', '아이패드', '스마트워치', '에어팟', '이어폰'],
    '컴퓨터': ['노트북', '랩탑', '맥북', '울트라북'],
    '의류': ['옷', '셔츠', '바지', '치마', '코트', '자켓'],
    '신발': ['운동화', '구두', '슬리퍼', '부츠', '샌들'],
    '안경': ['선글라스', '뿔테안경', '콘택트렌즈케이스'],
}

# 브랜드/모델 동의어
BRAND_SYNONYMS = {
    '아이폰': ['iphone', '애플폰', '아이폰14', '아이폰15'],
    '갤럭시': ['galaxy', '삼성폰', '갤럭시s', '갤럭시노트'],
    '맥북': ['macbook', '애플노트북', 'macbook pro', 'macbook air'],
    '에어팟': ['airpods', '애플이어폰', '무선이어폰'],
    '아이패드': ['ipad', '애플태블릿'],
}

def expand_category_terms(category: str) -> List[str]:
    """카테고리 용어를 동의어로 확장"""
    if not category:
        return []
    
    category_lower = category.lower().strip()
    expanded = {category_lower}
    
    # 직접 매핑 확인
    for main_cat, synonyms in CATEGORY_SYNONYMS.items():
        if category_lower == main_cat.lower() or category_lower in [s.lower() for s in synonyms]:
            expanded.add(main_cat.lower())
            expanded.update(s.lower() for s in synonyms)
    
    # 부분 매칭으로 확장
    for main_cat, synonyms in CATEGORY_SYNONYMS.items():
        if category_lower in main_cat.lower() or any(category_lower in s.lower() for s in synonyms):
            expanded.add(main_cat.lower())
            expanded.update(s.lower() for s in synonyms)
    
    return list(expanded)

def expand_region_terms(region: str) -> List[str]:
    """지역 용어는 동의어 확장 없이 원본만 반환 (정확한 매칭)"""
    if not region:
        return []
    return [region.lower().strip()]

def expand_brand_terms(text: str) -> List[str]:
    """브랜드/모델명을 동의어로 확장"""
    if not text:
        return []
    
    text_lower = text.lower().strip()
    expanded = {text_lower}
    
    for main_brand, synonyms in BRAND_SYNONYMS.items():
        if text_lower == main_brand.lower() or text_lower in [s.lower() for s in synonyms]:
            expanded.add(main_brand.lower())
            expanded.update(s.lower() for s in synonyms)
    
    return list(expanded)

def is_category_match(item_category: str, search_terms: List[str]) -> bool:
    """카테고리가 검색 용어들과 매치되는지 확인"""
    if not item_category or not search_terms:
        return False
    
    item_cat_lower = item_category.lower()
    
    # 직접 매치
    for term in search_terms:
        if term.lower() in item_cat_lower:
            return True
    
    # 동의어 확장해서 매치
    item_expanded = expand_category_terms(item_category)
    for term in search_terms:
        term_expanded = expand_category_terms(term)
        if any(exp_term in item_cat_lower for exp_term in term_expanded):
            return True
        if any(exp_item in term.lower() for exp_item in item_expanded):
            return True
    
    return False

def is_region_match(item_region: str, search_terms: List[str]) -> bool:
    """지역이 검색 용어들과 매치되는지 확인 (정확한 매칭만)"""
    if not item_region or not search_terms:
        return False
    
    item_region_lower = item_region.lower()
    
    # 정확한 부분 매치만 사용
    for term in search_terms:
        if term.lower() in item_region_lower:
            return True
    
    return False

def get_category_from_subcategory(subcategory: str) -> Optional[str]:
    """subcategory로부터 해당하는 category를 찾기"""
    if not subcategory:
        return None
    
    subcategory_lower = subcategory.lower().strip()
    
    # SUBCATEGORY_SYNONYMS에서 직접 매핑 확인
    for alias, (cat, sub) in schema.SUBCATEGORY_SYNONYMS.items():
        if subcategory_lower == sub.lower() or subcategory_lower == alias.lower():
            return cat
    
    # SUBCATEGORIES에서 직접 검색
    for category, subcats in schema.SUBCATEGORIES.items():
        for subcat in subcats:
            if subcategory_lower == subcat.lower():
                return category
    
    return None

def expand_search_terms_with_subcategory(category: str, subcategory: str = None) -> Tuple[List[str], List[str]]:
    """category와 subcategory를 고려한 검색어 확장
    
    Returns:
        (expanded_categories, target_subcategories): 
        - expanded_categories: 검색할 카테고리 목록 (동의어 포함)
        - target_subcategories: 우선적으로 매칭할 subcategory 목록
    """
    expanded_categories = []
    target_subcategories = []
    
    # 1. Category 확장
    if category:
        expanded_categories = expand_category_terms(category)
    
    # 2. Subcategory가 있으면 추가 처리
    if subcategory:
        # subcategory에서 category 추론
        inferred_category = get_category_from_subcategory(subcategory)
        if inferred_category:
            # 추론된 category도 검색 대상에 추가
            inferred_expanded = expand_category_terms(inferred_category)
            expanded_categories.extend(inferred_expanded)
            # 중복 제거
            expanded_categories = list(set(expanded_categories))
        
        # target subcategories 설정
        target_subcategories = [subcategory.lower().strip()]
        
        # subcategory의 동의어도 추가
        for alias, (cat, sub) in schema.SUBCATEGORY_SYNONYMS.items():
            if subcategory.lower().strip() == sub.lower():
                target_subcategories.append(alias.lower())
            elif subcategory.lower().strip() == alias.lower():
                target_subcategories.append(sub.lower())
        
        # 중복 제거
        target_subcategories = list(set(target_subcategories))
    
    # 기본값 처리
    if not expanded_categories and category:
        expanded_categories = [category.lower().strip()]
    
    return expanded_categories, target_subcategories

def calculate_category_subcategory_score(item_category: str, search_category: str, search_subcategory: str = None) -> float:
    """카테고리와 서브카테고리를 고려한 매칭 점수 계산
    
    Args:
        item_category: DB 아이템의 카테고리
        search_category: 검색하려는 카테고리  
        search_subcategory: 검색하려는 서브카테고리 (optional)
    
    Returns:
        float: 0.0 ~ 1.0 사이의 매칭 점수
    """
    if not item_category or not search_category:
        return 0.0
    
    item_cat_lower = item_category.lower().strip()
    
    # 1. Category 매칭 점수
    category_score = 0.0
    expanded_categories, target_subcategories = expand_search_terms_with_subcategory(search_category, search_subcategory)
    
    # 카테고리 직접 매칭
    if item_cat_lower == search_category.lower().strip():
        category_score = 1.0
    else:
        # 동의어 매칭
        for exp_cat in expanded_categories:
            if exp_cat and exp_cat in item_cat_lower:
                category_score = 0.8
                break
    
    # 2. Subcategory 고려
    if search_subcategory and category_score > 0:
        # subcategory가 있는 경우, 해당 category 내에서 더 정확한 매칭을 위한 보너스
        # 실제 DB에는 subcategory 필드가 없으므로, category 매칭이 정확하면 보너스 점수
        
        # subcategory에서 추론한 category와 검색 category가 일치하는지 확인
        inferred_category = get_category_from_subcategory(search_subcategory)
        if inferred_category and inferred_category.lower() == search_category.lower():
            # subcategory 정보가 일관성 있게 제공된 경우 보너스
            category_score = min(category_score + 0.1, 1.0)
    
    return category_score

def calculate_similarity_score(item_text: str, search_terms: List[str]) -> float:
    """텍스트 유사도 점수 계산 (카테고리/브랜드 동의어만 고려, 지역은 정확 매칭)"""
    if not item_text or not search_terms:
        return 0.0
    
    item_lower = item_text.lower()
    total_score = 0.0
    
    for term in search_terms:
        # 직접 매치
        if term.lower() in item_lower:
            total_score += 1.0
            continue
        
        # 카테고리와 브랜드 동의어만 매치 (지역 제외)
        term_expanded = (expand_category_terms(term) + 
                        expand_brand_terms(term))
        
        for exp_term in term_expanded:
            if exp_term in item_lower:
                total_score += 0.8  # 동의어 매치는 약간 낮은 점수
                break
    
    # 정규화: 검색 용어 수로 나누기
    return min(total_score / len(search_terms), 1.0)
