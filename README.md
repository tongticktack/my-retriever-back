# My Retriever Backend

이 프로젝트는 FastAPI를 사용하여 구현된 My Retriever 프로젝트의 백엔드입니다.

## 기술 스택

- Python 3.10+
- FastAPI
- Uvicorn

## 설치 및 실행

### 1. 저장소 복제

```bash
git clone (저장소 URL)
cd my-retriever-back
```

### 2. 가상 환경 생성 및 활성화

프로젝트의 의존성을 독립적으로 관리하기 위해 가상 환경을 생성하고 활성화합니다.

```bash
# 가상 환경 생성
python3 -m venv venv

# 가상 환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상 환경 활성화 (Windows)
# venv\Scripts\activate
```

### 3. 의존성 설치

`requirements.txt` 파일에 명시된 라이브러리들을 설치합니다.

```bash
pip install -r requirements.txt
```

### 4. 서버 실행

개발 서버를 실행합니다. `--reload` 옵션은 코드 변경 시 서버를 자동으로 재시작해줍니다.

```bash
uvicorn main:app --reload
```

서버가 실행되면 브라우저에서 `http://127.0.0.1:8000` 주소로 접속하여 API 문서를 확인할 수 있습니다. (`/docs`)

## LLM / Gemini 설정

환경 변수로 LLM 제공자를 제어합니다.

필드:

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `LLM_PROVIDER` | 사용 LLM: `echo`, `gemini`, `openai`, `auto` | `auto` |
| `LLM_AUTO_PRIORITY` | auto 모드 체인 우선순위 (콤마 구분) | `gemini,openai,echo` |
| `GEMINI_API_KEY` | AI Studio 발급 키 (존재 시 Gemini 활성) | (없음) |
| `GEMINI_MODEL_NAME` | Gemini 모델명 | `gemini-2.0-flash` |
| `OPENAI_MODEL_NAME` | OpenAI Chat 모델명 | `gpt-4o-mini` |
| `LLM_MAX_HISTORY_MESSAGES` | 유지할 대화 히스토리 길이 (system 제외) | `20` |

auto 모드 동작:
1. `LLM_AUTO_PRIORITY` 순서대로 provider 초기화 & 호출을 시도합니다.
2. 각 provider 가 정상 응답(status == ok)하면 체인을 즉시 종료합니다.
3. 실패(예외) 또는 provider 내부 fallback(status != ok) 시 다음 provider 로 진행합니다.
4. `echo` 는 항상 최종 안전망으로 보장됩니다.

예시 (.env.local):

```env
LLM_PROVIDER=auto
LLM_AUTO_PRIORITY=gemini,openai,echo
GEMINI_API_KEY=your_ai_studio_key
GEMINI_MODEL_NAME=gemini-2.0-flash
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini
```

OpenAI 를 1순위로 바꾸고 싶다면:

```env
LLM_AUTO_PRIORITY=openai,gemini,echo
```

Gemini 를 비활성화하려면:

Gemini 비활성화: `GEMINI_API_KEY` 값을 제거하거나 비워두면 체인에서 자동 제외됩니다.

## 배포 (Deployment)

### Docker를 사용한 배포

Docker를 사용하면 애플리케이션과 실행 환경을 컨테이너로 패키징하여 일관성 있고 격리된 배포가 가능합니다. 프로젝트 루트에 포함된 `Dockerfile`을 사용하여 이미지를 빌드하고 컨테이너를 실행할 수 있습니다.

**사전 준비:**

- Docker가 설치되어 있어야 합니다.
- 프로젝트 루트 디렉토리에 Firebase에서 다운로드한 서비스 계정 키 파일이 `firebase-credentials.json`이라는 이름으로 존재해야 합니다.

**1. Docker 이미지 빌드**

```bash
# Docker 이미지 빌드
docker build -t my-retriever-back .
```

**2. Docker 컨테이너 실행**

컨테이너를 실행할 때, `-e` 옵션을 사용하여 `firebase-credentials.json` 파일의 내용을 환경 변수로 주입합니다. 이는 비밀 정보를 이미지에 포함시키지 않는 보안 모범 사례입니다.

```bash
# Docker 컨테이너 실행
docker run -d -p 8000:8000 \
  -e FIREBASE_CREDENTIALS_JSON_STRING="$(cat firebase-credentials.json)" \
  --name my-retriever-app \
  my-retriever-back
```

- `-d`: 컨테이너를 백그라운드에서 실행합니다.
- `-p 8000:8000`: 호스트의 8000번 포트를 컨테이너의 8000번 포트와 매핑합니다.
- `-e ...`: 컨테이너 내부에 환경 변수를 설정합니다. `cat` 명령어로 `firebase-credentials.json` 파일의 내용을 통째로 읽어 주입합니다.
- `--name`: 컨테이너에 `my-retriever-app`이라는 이름을 부여합니다.

## 이미지 인덱싱 실행 및 자동화

### 즉시 인덱싱 실행


아래 명령어로 DB 이미지 인덱싱을 즉시 실행할 수 있습니다.

```bash
python app/scripts/db_image_indexer.py
```

- 옵션: `--limit N` 으로 인덱싱할 최대 아이템 수를 지정할 수 있습니다.
   ```bash
   python app/scripts/db_image_indexer.py --limit 100
   ```
- 실행 결과와 통계가 콘솔에 출력됩니다.

### 크론(cron)으로 주기적 자동 실행


`app/scripts/db_image_indexer.py`를 crontab에 등록하여 매일 새벽 4시에 자동 실행할 수 있습니다.

#### 설정 방법

1. 실행 권한 부여:
   ```bash
   chmod +x app/scripts/db_image_indexer.py
   ```
2. crontab 편집:
   ```bash
   crontab -e
   ```
3. 아래 줄 추가 (경로는 실제 프로젝트 위치로 변경):
   ```
   0 4 * * * /usr/bin/python3 /path/to/my-retriever-back/app/scripts/db_image_indexer.py >> /path/to/my-retriever-back/logs/image_indexing.log 2>&1
   ```
- 로그는 `logs/image_indexing.log`에 기록됩니다.
- 상세 인덱싱 로그 및 통계는 스크립트에서 자동 기록됩니다.

## 운영/로그 관리
- 인덱싱 상세 로그(컬렉션, image_url, 통계)는 `app/scripts/db_image_indexer.py`에서 자동 기록됩니다.
- 모든 로그는 `logs/` 폴더에 저장되며, 인덱싱 상세 로그는 `logs/image_indexing.log`에 기록됩니다.
