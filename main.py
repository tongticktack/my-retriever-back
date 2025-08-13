import firebase_admin
import json
from fastapi import FastAPI
from firebase_admin import credentials

from config import settings

# Firebase init (unchanged)
cred_obj = None
if settings.FIREBASE_CREDENTIALS_JSON_STRING:
    cred_info = json.loads(settings.FIREBASE_CREDENTIALS_JSON_STRING)
    cred_obj = credentials.Certificate(cred_info)
elif settings.GOOGLE_APPLICATION_CREDENTIALS:
    cred_obj = credentials.Certificate(settings.GOOGLE_APPLICATION_CREDENTIALS)

if cred_obj:
    firebase_admin.initialize_app(cred_obj)
else:
    print("WARNING: Firebase credentials not found. Firebase features will be disabled.")

app = FastAPI(title="Loosy Lost & Found API")

# Routers
from app.api import items, chat  # noqa: E402
app.include_router(items.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"message": "Loosy backend modular 구조 준비", "routes": ["/items/ingest", "/items/search/image", "/chat/session", "/chat/send", "/chat/history/{session_id}"]}