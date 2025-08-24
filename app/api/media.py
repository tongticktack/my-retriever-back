from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import hashlib
from PIL import Image
import io
import time
import uuid

# Firebase Storage 전용 모드: 메모리 모드 제거
try:
    from app.services import media_store  # lazy import
    media_store.get_bucket()
    USE_FIREBASE_STORAGE = True
    print("[media] Firebase storage 활성화")
except Exception as e:
    # 즉시 실패: 서비스 필수 구성 누락
    USE_FIREBASE_STORAGE = False
    print(f"[media] FATAL: Firebase storage 사용 불가 - {e}")

router = APIRouter(prefix="/media", tags=["media"])

ALLOWED_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/jpeg": ".jpeg", "image/webp": ".webp"}
MAX_BYTES = 10 * 1024 * 1024  # 10MB


class MediaUploadResponse(BaseModel):
    media_id: str
    url: str  # pseudo URL for now
    width: int
    height: int
    hash: str
    palette: List[str]
    content_type: str


class MediaMetaResponse(BaseModel):
    media_id: str
    url: str
    width: int
    height: int
    hash: str
    palette: List[str]
    content_type: str


def _phash_bytes(data: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(data)).convert("L").resize((16, 16))
        pixels = bytes(img.getdata())
        return hashlib.sha256(pixels).hexdigest()[:32]
    except Exception:
        return hashlib.sha256(data[:128]).hexdigest()[:32]


def _dominant_palette(img: Image.Image, k: int = 3) -> List[str]:
    # Quick & rough palette using resize + most common colors
    small = img.copy()
    small.thumbnail((64, 64))
    colors = small.getcolors(64 * 64)
    if not colors:
        return []
    colors.sort(reverse=True, key=lambda c: c[0])
    hexes = []
    for count, rgb in colors[:k]:
        if isinstance(rgb, int):  # L mode
            v = rgb
            hexes.append(f"#{v:02x}{v:02x}{v:02x}")
        else:
            r, g, b = rgb[:3]
            hexes.append(f"#{r:02x}{g:02x}{b:02x}")
    return hexes


@router.post("/upload", response_model=MediaUploadResponse)
async def upload_image(file: UploadFile = File(...)):
    if not USE_FIREBASE_STORAGE:
        raise HTTPException(500, detail="firebase_storage_not_configured")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, detail="unsupported_type")
    raw = await file.read()
    size = len(raw)
    if size > MAX_BYTES:
        raise HTTPException(413, detail="file_too_large")
    print(f"[media.upload] name={file.filename} ct={file.content_type} bytes={size}")
    try:
        meta = media_store.upload_image(raw, file.content_type or "application/octet-stream")
        print(f"[media.upload] stored media_id={meta.get('media_id')} hash={meta.get('hash')} url={meta.get('url')}")
        return MediaUploadResponse(**{k: meta[k] for k in ["media_id","url","width","height","hash","palette","content_type"]})
    except Exception as e:
        print(f"[media.upload] ERROR {e}")
        raise HTTPException(500, detail=f"upload_error:{e}")


def get_media_batch(media_ids: List[str]):
    """주어진 media_id 목록에 대한 메타데이터 배열 반환 (Firebase 전용).

    존재하지 않는 id 는 무시. 실패 시 빈 배열. 디버그 로그 포함.
    """
    if not USE_FIREBASE_STORAGE:
        print(f"[media.get_media_batch] ERROR firebase_storage_not_configured req={media_ids}")
        return []
    try:
        from app.services import media_store as _ms
        metas = _ms.batch_get(media_ids)
        allow = {"media_id","url","width","height","hash","palette","content_type"}
        shaped = [{k: m[k] for k in allow if k in m} for m in metas]
        if len(shaped) != len(media_ids):
            missing = set(media_ids) - {m['media_id'] for m in shaped}
            if missing:
                print(f"[media.get_media_batch] missing={list(missing)} found={len(shaped)}")
        return shaped
    except Exception as e:
        print(f"[media.get_media_batch] ERROR {e} req={media_ids}")
        return []


@router.get("/{media_id}/meta", response_model=MediaMetaResponse)
def get_media_meta(media_id: str):
    if not USE_FIREBASE_STORAGE:
        raise HTTPException(500, detail="firebase_storage_not_configured")
    meta = media_store.get_media(media_id)
    if not meta:
        print(f"[media.meta] not_found media_id={media_id}")
        raise HTTPException(404, detail="not_found")
    print(f"[media.meta] hit media_id={media_id}")
    return MediaMetaResponse(**{k: meta[k] for k in ["media_id","url","width","height","hash","palette","content_type"]})


@router.get("/{media_id}")
def get_media_raw(media_id: str):
    if not USE_FIREBASE_STORAGE:
        raise HTTPException(500, detail="firebase_storage_not_configured")
    meta = media_store.get_media(media_id)
    if not meta:
        print(f"[media.raw] not_found media_id={media_id}")
        raise HTTPException(404, detail="not_found")
    return {"redirect_url": meta.get("url")}
