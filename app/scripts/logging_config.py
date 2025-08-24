# logging_config.py
import logging
import logging.config
import logging.handlers
from pathlib import Path
from datetime import datetime
import json
import contextvars

# 요청 단위 식별자(ContextVar로 보관)
_request_id_ctx = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get("-")
        return True

def set_request_id(req_id: str):
    _request_id_ctx.set(req_id)

def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)

# 로그 디렉터리
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def build_dict_config(json_fmt: bool = False) -> dict:
    fmt = (
        '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
        '"request_id":"%(request_id)s","msg":"%(message)s"}'
        if json_fmt
        else '%(asctime)s | %(levelname)s | %(name)s | rid=%(request_id)s | %(message)s'
    )

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {"()": RequestIdFilter},
        },
        "formatters": {
            "default": {
                "format": fmt,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "filters": ["request_id"],
            },
            "file_app": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "INFO",
                "formatter": "default",
                "filters": ["request_id"],
                "filename": str(LOG_DIR / "app.log"),
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8",
            },
            "file_image_indexing": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "INFO",
                "formatter": "default",
                "filters": ["request_id"],
                "filename": str(LOG_DIR / "image_indexing.log"),
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8",
            },
            "file_user_image_embedding": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "level": "INFO",
                "formatter": "default",
                "filters": ["request_id"],
                "filename": str(LOG_DIR / "user_image_embedding.log"),
                "when": "midnight",
                "interval": 1,
                "backupCount": 30,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # 루트 로거: 앱 전반
            "": {
                "level": "INFO",
                "handlers": ["console", "file_app"],
            },
            # 이미지 인덱싱 전용 로거
            "image_indexing": {
                "level": "INFO",
                "handlers": ["console", "file_image_indexing"],
                "propagate": False,
            },
            # 사용자 업로드 이미지 임베딩 전용 로거
            "media_embed": {
                "level": "INFO",
                "handlers": ["console", "file_user_image_embedding"],
                "propagate": False,
            },
            # uvicorn 로거 레벨 통일
            "uvicorn": {"level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"level": "INFO"},
        },
    }

def setup_logging(json_fmt: bool = False):
    logging.config.dictConfig(build_dict_config(json_fmt=json_fmt))

# ===== 이미지 인덱싱 보조 함수들 =====
def log_indexing_event(event_type: str, details: dict, logger: logging.Logger | None = None):
    logger = logger or get_logger("image_indexing")
    logger.info("INDEXING_EVENT: %s", json.dumps({
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details
    }, ensure_ascii=False))

def log_image_download(atc_id: str, url: str, success: bool,
                       size_bytes: int | None = None, error: str | None = None,
                       logger: logging.Logger | None = None):
    logger = logger or get_logger("image_indexing")
    if success:
        logger.info("이미지 다운로드 성공: %s (%s bytes) url=%s", atc_id, size_bytes, url)
    else:
        logger.warning("이미지 다운로드 실패: %s - %s url=%s", atc_id, error, url)

def log_embedding_generation(atc_id: str, success: bool,
                             embedding_dim: int | None = None, error: str | None = None,
                             logger: logging.Logger | None = None):
    logger = logger or get_logger("image_indexing")
    if success:
        logger.info("임베딩 생성 성공: %s (차원: %s)", atc_id, embedding_dim)
    else:
        logger.error("임베딩 생성 실패: %s - %s", atc_id, error)

def log_faiss_operation(operation: str, atc_id: str, success: bool,
                        index_size: int | None = None, error: str | None = None,
                        logger: logging.Logger | None = None):
    logger = logger or get_logger("image_indexing")
    if success:
        logger.info("FAISS %s 성공: %s (인덱스 크기: %s)", operation, atc_id, index_size)
    else:
        logger.error("FAISS %s 실패: %s - %s", operation, atc_id, error)

def log_batch_summary(batch_info: dict, logger: logging.Logger | None = None):
    logger = logger or get_logger("image_indexing")
    summary = {
        'total_processed': batch_info.get('processed', 0),
        'success_count': batch_info.get('success', 0),
        'error_count': batch_info.get('error', 0),
        'total_indexed': batch_info.get('total_indexed', 0),
        'duration_seconds': batch_info.get('duration', 0),
        'avg_time_per_item': batch_info.get('avg_time_per_item', 0),
    }
    logger.info("배치 처리 완료: %s", json.dumps(summary, ensure_ascii=False))