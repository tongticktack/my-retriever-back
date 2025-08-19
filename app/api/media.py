from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
from typing import List
import hashlib
from PIL import Image
import io
import time
import uuid

# NOTE: Simple in-memory store placeholder.
# In production replace with cloud storage + persistent metadata (Firestore or DB).
_MEDIA_STORE = {}
USE_FIREBASE_STORAGE = False
try:
    from config import settings
    if settings.FIREBASE_STORAGE_BUCKET:
        from app.services import media_store
        USE_FIREBASE_STORAGE = True
except Exception:
    USE_FIREBASE_STORAGE = False

router = APIRouter(prefix="/media", tags=["media"])

ALLOWED_CONTENT_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}
MAX_BYTES = 5 * 1024 * 1024  # 5MB


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
    # Simple perceptual-ish hash: resize grayscale 16x16 and hash pixels
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
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, detail="unsupported_type")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, detail="file_too_large")
    if USE_FIREBASE_STORAGE:
        try:
            meta = media_store.upload_image(raw, file.content_type or "application/octet-stream")
            return MediaUploadResponse(**{k: meta[k] for k in ["media_id","url","width","height","hash","palette","content_type"]})
        except Exception as e:
            raise HTTPException(500, detail=f"upload_error:{e}")
    # fallback in-memory
    ph = _phash_bytes(raw)
    for mid, meta in _MEDIA_STORE.items():
        if meta.get("hash") == ph:
            return MediaUploadResponse(**meta)
    try:
        img = Image.open(io.BytesIO(raw)); img.load()
    except Exception:
        raise HTTPException(400, detail="invalid_image")
    width, height = img.size
    palette = _dominant_palette(img)
    media_id = str(uuid.uuid4())
    _MEDIA_STORE[media_id] = {
        "media_id": media_id,
        "url": f"memory://{media_id}",
        "width": width,
        "height": height,
        "hash": ph,
        "palette": palette,
        "created_at": time.time(),
        "data": raw,
        "content_type": file.content_type or "application/octet-stream",
    }
    return MediaUploadResponse(**{k: v for k, v in _MEDIA_STORE[media_id].items() if k not in {"data","created_at"}})


def get_media_batch(media_ids: List[str]):
    out = []
    for mid in media_ids:
        meta = _MEDIA_STORE.get(mid)
        if meta:
            out.append({k: v for k, v in meta.items() if k not in {"data"}})
    return out


@router.get("/{media_id}/meta", response_model=MediaMetaResponse)
def get_media_meta(media_id: str):
    if USE_FIREBASE_STORAGE:
        meta = media_store.get_media(media_id)
        if not meta:
            raise HTTPException(404, detail="not_found")
        return MediaMetaResponse(**{k: meta[k] for k in ["media_id","url","width","height","hash","palette","content_type"]})
    else:
        meta = _MEDIA_STORE.get(media_id)
        if not meta:
            raise HTTPException(404, detail="not_found")
        return MediaMetaResponse(**{k: v for k, v in meta.items() if k not in {"data"}})


@router.get("/{media_id}")
def get_media_raw(media_id: str):
    if USE_FIREBASE_STORAGE:
        meta = media_store.get_media(media_id)
        if not meta:
            raise HTTPException(404, detail="not_found")
        # Return redirect link pattern (client fetch directly)
        return {"redirect_url": meta.get("url")}
    else:
        meta = _MEDIA_STORE.get(media_id)
        if not meta:
            raise HTTPException(404, detail="not_found")
        data = meta.get("data")
        ct = meta.get("content_type", "application/octet-stream")
        return Response(content=data, media_type=ct)
