"""이미지 인덱싱 로깅 설정"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json

# 로그 디렉토리 생성
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_indexing_logger(name: str = "image_indexing") -> logging.Logger:
    """이미지 인덱싱 전용 로거 설정"""
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 이미 핸들러가 설정되어 있으면 중복 방지
    if logger.handlers:
        return logger
    
    # 파일 핸들러 (일별 로테이션)
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=LOG_DIR / "image_indexing.log",
        when='midnight',
        interval=1,
        backupCount=30,  # 30일치 보관
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_indexing_event(logger: logging.Logger, event_type: str, details: dict):
    """인덱싱 이벤트 로깅"""
    timestamp = datetime.now().isoformat()
    log_data = {
        'timestamp': timestamp,
        'event_type': event_type,
        'details': details
    }
    
    logger.info(f"INDEXING_EVENT: {json.dumps(log_data, ensure_ascii=False)}")

def log_image_download(logger: logging.Logger, atc_id: str, url: str, success: bool, 
                        size_bytes: int = None, error: str = None):
    """이미지 다운로드 로깅"""
    details = {
        'atc_id': atc_id,
        'url': url,
        'success': success,
        'size_bytes': size_bytes,
        'error': error
    }
    
    if success:
        logger.info(f"이미지 다운로드 성공: {atc_id} ({size_bytes} bytes)")
    else:
        logger.warning(f"이미지 다운로드 실패: {atc_id} - {error}")

def log_embedding_generation(logger: logging.Logger, atc_id: str, success: bool, 
                            embedding_dim: int = None, error: str = None):
    """임베딩 생성 로깅"""
    details = {
        'atc_id': atc_id,
        'success': success,
        'embedding_dim': embedding_dim,
        'error': error
    }
    
    if success:
        logger.info(f"임베딩 생성 성공: {atc_id} (차원: {embedding_dim})")
    else:
        logger.error(f"임베딩 생성 실패: {atc_id} - {error}")

def log_faiss_operation(logger: logging.Logger, operation: str, atc_id: str, 
                        success: bool, index_size: int = None, error: str = None):
    """FAISS 작업 로깅"""
    details = {
        'operation': operation,
        'atc_id': atc_id,
        'success': success,
        'index_size': index_size,
        'error': error
    }
    
    if success:
        logger.info(f"FAISS {operation} 성공: {atc_id} (인덱스 크기: {index_size})")
    else:
        logger.error(f"FAISS {operation} 실패: {atc_id} - {error}")

def log_batch_summary(logger: logging.Logger, batch_info: dict):
    """배치 처리 요약 로깅"""
    summary = {
        'total_processed': batch_info.get('processed', 0),
        'success_count': batch_info.get('success', 0),
        'error_count': batch_info.get('error', 0),
        'total_indexed': batch_info.get('total_indexed', 0),
        'duration_seconds': batch_info.get('duration', 0),
        'avg_time_per_item': batch_info.get('avg_time_per_item', 0)
    }
    
    logger.info(f"배치 처리 완료: {json.dumps(summary, ensure_ascii=False)}")

# 전역 로거 인스턴스
indexing_logger = setup_indexing_logger()
