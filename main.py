# main.py
import json
import uuid
from time import time

import firebase_admin
from fastapi import FastAPI, Request
from firebase_admin import credentials

from config import settings
from logging_config import setup_logging, get_logger, set_request_id

# 1) 로깅 설정(최우선)
# 운영 환경에서 JSON 로그를 원하면 json_fmt=True
setup_logging(json_fmt=False)
logger = get_logger(__name__)

# 2) Firebase 초기화
cred_obj = None
bucket_from_json = None

try:
    if settings.FIREBASE_CREDENTIALS_JSON_STRING:
        cred_info = json.loads(settings.FIREBASE_CREDENTIALS_JSON_STRING)
        bucket_from_json = cred_info.get('storage_bucket') or cred_info.get('storageBucket')
        cred_obj = credentials.Certificate(cred_info)
        logger.info("Firebase credentials loaded from FIREBASE_CREDENTIALS_JSON_STRING.")
    elif settings.GOOGLE_APPLICATION_CREDENTIALS:
        try:
            with open(settings.GOOGLE_APPLICATION_CREDENTIALS, 'r', encoding='utf-8') as f:
                ci = json.load(f)
                bucket_from_json = ci.get('storage_bucket') or ci.get('storageBucket')
            cred_obj = credentials.Certificate(settings.GOOGLE_APPLICATION_CREDENTIALS)
            logger.info("Firebase credentials loaded from GOOGLE_APPLICATION_CREDENTIALS file.")
        except Exception as e:
            bucket_from_json = None
            logger.warning("Failed to read GOOGLE_APPLICATION_CREDENTIALS file: %s", e)

    if cred_obj:
        init_options: dict = {}
        chosen_bucket = bucket_from_json  # JSON의 storage_bucket 우선
        if chosen_bucket:
            init_options['storageBucket'] = chosen_bucket
            logger.info("Firebase init with storageBucket=%s", chosen_bucket)
        else:
            logger.info("Firebase init without explicit storageBucket (project-id fallback may apply).")

        firebase_admin.initialize_app(cred_obj, init_options or None)
        logger.info("Firebase initialized successfully.")
    else:
        logger.warning("Firebase credentials not found. Firebase features will be disabled.")
except Exception as e:
    logger.exception("Firebase initialization failed: %s", e)

# 3) FastAPI 앱
app = FastAPI(title="Loosy Lost & Found API")

# 4) 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    set_request_id(rid)

    start = time()
    path = request.url.path
    method = request.method
    query = request.url.query
    client_ip = getattr(request.client, 'host', '-') if request.client else '-'
    ua = request.headers.get('user-agent', '')[:120]

    body_preview = ""
    try:
        if method in {"POST", "PUT", "PATCH"}:
            body_bytes = await request.body()
            if body_bytes:
                body_preview = body_bytes[:300].decode('utf-8', 'ignore')
                if len(body_bytes) > 300:
                    body_preview += "..."
    except Exception:
        body_preview = "<unreadable>"

    if method == 'GET' and query:
        logger.info("REQ start %s %s?%s ip=%s ua=%r", method, path, query, client_ip, ua)
    else:
        logger.info("REQ start %s %s ip=%s ua=%r body=%r", method, path, client_ip, ua, body_preview)

    try:
        response = await call_next(request)
        return response
    finally:
        duration = (time() - start) * 1000
        status = getattr(locals().get('response', None), 'status_code', 'NA')
        size = '-'
        try:
            size = response.headers.get('content-length') if 'response' in locals() else '-'
        except Exception:
            pass

        if method == 'GET' and query:
            logger.info("REQ end %s %s?%s status=%s %.1fms size=%s", method, path, query, status, duration, size)
        else:
            logger.info("REQ end %s %s status=%s %.1fms size=%s", method, path, status, duration, size)

# 5) CORS (optional)
try:
    from fastapi.middleware.cors import CORSMiddleware
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware configured for %s", allowed_origins)
except Exception as e:
    logger.warning("CORS middleware not added: %s", e)

# 6) 라우터
from app.api import items, chat, lost_item, media
from app.services import faiss_index

app.include_router(items.router)
app.include_router(chat.router)
app.include_router(lost_item.router)
app.include_router(media.router)

# 7) 엔드포인트
@app.get("/")
def root():
    return {"message": "Loosy backend modular 구조 준비", "routes": [
        "/items/ingest",
        "/items/search/image",
        "/chat/session",
        "/chat/send",
        "/chat/history/{session_id}",
        "/lost/analyze",
        "/media/upload",
    ]}

@app.post("/admin/reindex")
def admin_reindex(force: bool = False):
    changed = faiss_index.reindex_all(force=force)
    logger.info("Admin reindex requested: force=%s, changed=%s, embedding_version=%s",
                force, changed, faiss_index.EMBEDDING_VERSION)
    return {"reindexed": changed, "embedding_version": faiss_index.EMBEDDING_VERSION}