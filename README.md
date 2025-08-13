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

공공데이터 api -> firestore

Firebase CLI 로그인:

  터미널에서 명령어를 실행, 웹 브라우저에서 본인의 Google 계정으로 로그인

    firebase login

Firebase 프로젝트 연결 확인:

  프로젝트에 정상적으로 접근 가능한지 확인

    firebase projects:list

      firebase 프로젝트 권한을 드려서 myretriver-c5fdd 프로젝트가 있어야 합니다.

필요한 라이브러리 설치:

  데이터 수집기(firebase_function/functions)에 필요한 라이브러리를 설치

    cd firebase_function/functions

      npm install