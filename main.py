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
from app.api import items, chat
app.include_router(items.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"message": "Loosy backend modular 구조 준비", "routes": ["/items/ingest", "/items/search/image", "/chat/session", "/chat/send", "/chat/history/{session_id}"]}