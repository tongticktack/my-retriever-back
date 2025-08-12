import firebase_admin
import json
from fastapi import FastAPI
from firebase_admin import credentials

from config import settings

# Docker 환경과 로컬 환경 모두를 위한 Firebase 초기화 로직
cred_obj = None

# 1. Docker 환경: 환경 변수에서 직접 JSON 문자열을 읽어 초기화
if settings.FIREBASE_CREDENTIALS_JSON_STRING:
    cred_info = json.loads(settings.FIREBASE_CREDENTIALS_JSON_STRING)
    cred_obj = credentials.Certificate(cred_info)
# 2. 로컬 환경: .env.local 파일에 지정된 경로의 파일을 읽어 초기화
elif settings.GOOGLE_APPLICATION_CREDENTIALS:
    cred_obj = credentials.Certificate(settings.GOOGLE_APPLICATION_CREDENTIALS)

if cred_obj:
    firebase_admin.initialize_app(cred_obj)
else:
    # 로컬 개발 시 .env.local 파일이 없거나, Docker 실행 시 환경변수가 주입되지 않은 경우
    # 서버가 시작은 되지만 Firebase 연동 기능은 동작하지 않음
    print("WARNING: Firebase credentials not found. Firebase features will be disabled.")

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to My Retriever Backend"}