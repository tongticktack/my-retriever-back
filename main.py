import firebase_admin
import json
from fastapi import FastAPI, Request
from fastapi import Form, Response, HTTPException
import httpx
from firebase_admin import credentials

from config import settings

# Firebase init (credentials JSON 기반 bucket 고정)
cred_obj = None
bucket_from_json = None
if settings.FIREBASE_CREDENTIALS_JSON_STRING:
    cred_info = json.loads(settings.FIREBASE_CREDENTIALS_JSON_STRING)
    bucket_from_json = cred_info.get('storage_bucket') or cred_info.get('storageBucket')
    cred_obj = credentials.Certificate(cred_info)
elif settings.GOOGLE_APPLICATION_CREDENTIALS:
    try:
        with open(settings.GOOGLE_APPLICATION_CREDENTIALS, 'r', encoding='utf-8') as f:
            ci = json.load(f)
            bucket_from_json = ci.get('storage_bucket') or ci.get('storageBucket')
    except Exception:
        bucket_from_json = None
    cred_obj = credentials.Certificate(settings.GOOGLE_APPLICATION_CREDENTIALS)

if cred_obj:
    init_options = {}
    # 환경변수 무시, credentials JSON 의 storage_bucket 만 사용 (없으면 project_id 기반 자동)
    chosen_bucket = bucket_from_json
    if not chosen_bucket:
        try:
            # project id 로 fall back
            from firebase_admin import _project_id  # internal; fallback if available
        except Exception:
            _project_id = None
        # firebase_admin.get_app() 이전이라 project id 추출 어려움 -> credentials JSON 없으면 이후 media_store 에서 유추
    if chosen_bucket:
        init_options['storageBucket'] = chosen_bucket
    firebase_admin.initialize_app(cred_obj, init_options or None)
else:
    print("WARNING: Firebase credentials not found. Firebase features will be disabled.")

app = FastAPI(title="Loosy Lost & Found API")

# Simple structured logging middleware (can be replaced by proper logger)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    from time import time
    start = time()
    path = request.url.path
    method = request.method
    try:
        response = await call_next(request)
        return response
    finally:
        duration = (time() - start) * 1000
        print(f"[req] {method} {path} {duration:.1f}ms")

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
except Exception as e:
    print("WARNING: CORS middleware not added:", e)

# Routers
from app.api import items, chat, lost_item, media
from app.services import faiss_index
app.include_router(items.router)
app.include_router(chat.router)
app.include_router(lost_item.router)
app.include_router(media.router)


@app.get("/")
def root():
    return {"message": "Loosy backend modular 구조 준비", "routes": [
        "/items/ingest",
        "/items/search/image",
        "/chat/session",
        "/chat/send",
        "/chat/history/{session_id}",
        "/lost/analyze",
        "/media/upload"
    ]}


@app.post("/admin/reindex")
def admin_reindex(force: bool = False):
    changed = faiss_index.reindex_all(force=force)
    return {"reindexed": changed, "embedding_version": faiss_index.EMBEDDING_VERSION}


@app.post("/proxy/lost112")
async def proxy_lost112(act_id: str = Form(...), fd_sn: str = Form("1")):
    """Server-side POST proxy to lost112 detail (browser can't send body via simple link).
    Returns raw HTML so frontend can open in new window or render in iframe.
    """
    target_url = "https://www.lost112.go.kr/find/findDetail.do"
    data = {"ACT_ID": act_id, "FD_SN": fd_sn}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(target_url, data=data)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"proxy_error: {e}")
    # Pass through status (treat non-200 as upstream issue still returning body)
    return Response(content=resp.text, media_type="text/html", status_code=resp.status_code)

