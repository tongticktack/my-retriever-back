from __future__ import annotations
import io
import hashlib
import time
import uuid
from typing import Dict, Any, List, Optional
from PIL import Image

from firebase_admin import storage, firestore
from . import chat_store
from . import embeddings, faiss_index
from app.scripts.logging_config import get_logger
from config import settings

# Firestore collection name
MEDIA_COLLECTION = "media"

# Lazy bucket init
_bucket = None

def get_bucket():
    global _bucket
    if _bucket is None:
        bucket_name = settings.FIREBASE_STORAGE_BUCKET
        try:
            import firebase_admin
            app = firebase_admin.get_app()
            # If app initialized with storageBucket option it will appear in options
            opt_bucket = app.options.get('storageBucket') if hasattr(app, 'options') else None
            if not bucket_name and opt_bucket:
                bucket_name = opt_bucket
            if not bucket_name:
                project_id = getattr(app, 'project_id', None) or app.project_id
                if project_id:
                    bucket_name = f"{project_id}.appspot.com"
        except Exception:
            pass
        if not bucket_name:
            raise RuntimeError("Storage bucket not configured and cannot derive project id")
        _bucket = storage.bucket(bucket_name)
    return _bucket


def _phash(data: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(data)).convert("L").resize((16,16))
        return hashlib.sha256(bytes(img.getdata())).hexdigest()[:32]
    except Exception:
        return hashlib.sha256(data[:128]).hexdigest()[:32]


def _palette(data: bytes, k: int = 3) -> List[str]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return []
    small = img.copy()
    small.thumbnail((64,64))
    colors = small.getcolors(64*64) or []
    colors.sort(reverse=True, key=lambda c: c[0])
    hexes = []
    for count, rgb in colors[:k]:
        if isinstance(rgb, int):
            v = rgb; hexes.append(f"#{v:02x}{v:02x}{v:02x}")
        else:
            r,g,b = rgb[:3]; hexes.append(f"#{r:02x}{g:02x}{b:02x}")
    return hexes


logger = get_logger("media_embed")


def upload_image(data: bytes, content_type: str) -> Dict[str, Any]:
    ph = _phash(data)
    db = chat_store.get_db()
    # Check duplicate by hash
    dup = db.collection(MEDIA_COLLECTION).where("hash", "==", ph).limit(1).stream()
    for d in dup:
        meta = d.to_dict() or {}
        meta['media_id'] = d.id
        return meta
    # New upload
    mid = str(uuid.uuid4())
    bucket = get_bucket()
    blob = bucket.blob(f"lost/{mid}")
    blob.upload_from_string(data, content_type=content_type)
    # Make it (optionally) publicly accessible (could switch to signed URL later)
    try:
        blob.make_public()
        url = blob.public_url
    except Exception:
        url = blob.path
    pal = _palette(data)
    # compute embedding (hash or provider) and normalize (embeddings handles it)
    emb_list = None
    image_vec = None
    t0 = time.time()
    try:
        image_vec = embeddings.embed_image(data)
        emb_list = image_vec.tolist()
        logger.info("user_image_embedding success mid=pending dim=%d ms=%.1f", len(emb_list), (time.time()-t0)*1000)
    except Exception as e:
        logger.error("user_image_embedding error mid=pending err=%s", e)
    # width/height
    try:
        im = Image.open(io.BytesIO(data))
        w,h = im.size
    except Exception:
        w=h=0
    meta = {
        'media_id': mid,
        'url': url,
        'width': w,
        'height': h,
        'hash': ph,
        'palette': pal,
        'content_type': content_type,
        'created_at': time.time(),
        'storage_path': blob.name,
        'embedding_version': faiss_index.EMBEDDING_VERSION,
        'embedding_provider': embeddings.current_provider(),
        'embedding': emb_list,
    }
    db.collection(MEDIA_COLLECTION).document(mid).set(meta)
    # Post-write log (now we know final mid)
    if emb_list is not None:
        logger.info("user_image_stored mid=%s hash=%s dim=%d palette=%s", mid, ph, len(emb_list), ','.join(pal))
    else:
        logger.warning("user_image_stored_no_embedding mid=%s hash=%s", mid, ph)
    return meta


def get_media(media_id: str) -> Optional[Dict[str, Any]]:
    db = chat_store.get_db()
    ref = db.collection(MEDIA_COLLECTION).document(media_id)
    snap = ref.get()
    if not snap.exists:
        return None
    data = snap.to_dict() or {}
    data['media_id'] = media_id
    return data


def batch_get(media_ids: List[str]) -> List[Dict[str, Any]]:
    out = []
    for mid in media_ids:
        m = get_media(mid)
        if m:
            out.append(m)
    return out


def get_embedding(media_id: str):
    m = get_media(media_id)
    if not m:
        return None
    emb = m.get('embedding')
    if isinstance(emb, list):
        try:
            import numpy as np
            arr = np.array(emb, dtype='float32')
            return arr
        except Exception:
            return None
    return None
