"""크론 작업 자동 설치 스크립트 (이미지 인덱싱 + 매치 알림)

이 스크립트는 매일 정해진 시간에 아래 두 작업을 실행하도록 crontab 을 멱등(idempotent)하게 구성합니다.
1) 이미지 인덱싱: app.scripts.db_image_indexer
2) 분실물 매치 알림: app.scripts.daily_match_notifier

주요 특징:
- 여러 번 실행해도 마커 블록 사이 내용을 갱신하므로 중복 라인 누적 없음
- 시간 변경 시 재실행만으로 업데이트
- 로그는 프로젝트 폴더 하위 logs/ 에 append

기본 사용 예 (프로젝트 루트에서):
        python -m app.scripts.install_cron \
            --python /usr/bin/python3 \
            --workspace /home/ec2-user/my-retriever-back \
            --index-time 03:30 \
            --notify-time 03:45

옵션 설명:
    --python       사용할 파이썬 인터프리터 절대 경로 (기본: 현재 인터프리터)
    --workspace    프로젝트 루트 절대 경로 (기본: 파일 경로 기준 상위)
    --index-time   인덱싱 실행 시각 HH:MM (기본 03:30)
    --notify-time  알림 실행 시각 HH:MM (기본 03:45)

환경 변수 (옵션):
    INDEX_TIME / NOTIFY_TIME  -> 동일 기능 (CLI 인자가 우선)
    CRON_USER (미사용/확장 여지) -> 현재 구현은 현재 사용자 crontab 에 설치

설치 후 확인:
    crontab -l    명령으로 다음 형태 블록 존재 여부 확인
        # >>> my-retriever scheduled tasks >>>
        WORKSPACE=/.../my-retriever-back
        30 3 * * * cd /... && /usr/bin/python3 -m app.scripts.db_image_indexer >> logs/image_indexing.log 2>&1
        45 3 * * * cd /... && /usr/bin/python3 -m app.scripts.daily_match_notifier >> logs/notifications.log 2>&1
        # <<< my-retriever scheduled tasks <<<

제거 방법:
    crontab -e 열어 마커(# >>> ... <<<) 사이 블록 삭제 후 저장, 또는 전체 crontab 제거

로그 위치:
    logs/image_indexing.log   (이미지 인덱싱)
    logs/notifications.log    (알림 스크립트)

사전 요구 사항:
    - 파이어베이스 자격 증명(FIREBASE_CREDENTIALS_JSON_STRING) 환경 변수
    - 알림 이메일 발송을 위한 SMTP_HOST (미설정 시 알림 스킵)

주의:
    - 서버 로컬 타임존 기준으로 크론이 동작하므로 TZ 확인 필요
    - 시간대(UTC vs KST) 차이가 있다면 원하는 실제 실행 시각 보정

"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

MARKER_BEGIN = "# >>> my-retriever scheduled tasks >>>"
MARKER_END = "# <<< my-retriever scheduled tasks <<<"


def build_cron_block(python: str, workspace: str, index_time: str, notify_time: str) -> str:
    ih, im = index_time.split(":")
    nh, nm = notify_time.split(":")
    env_export = f"WORKSPACE={workspace}"
    index_cmd = f"cd {workspace} && {python} -m app.scripts.db_image_indexer >> logs/image_indexing.log 2>&1"
    notify_cmd = f"cd {workspace} && {python} -m app.scripts.daily_match_notifier >> logs/notifications.log 2>&1"
    lines = [
        MARKER_BEGIN,
        f"{env_export}",
        f"{int(im)} {int(ih)} * * * {index_cmd}",  # minute hour * * *
        f"{int(nm)} {int(nh)} * * * {notify_cmd}",
        MARKER_END,
        "",
    ]
    return "\n".join(lines)


def install(cron_block: str, user: str | None = None):
    try:
        current = subprocess.check_output(["crontab", "-l"], text=True)
    except subprocess.CalledProcessError:
        current = ""
    # Remove existing block
    lines = []
    skip = False
    for line in current.splitlines():
        if line.strip() == MARKER_BEGIN:
            skip = True
            continue
        if line.strip() == MARKER_END:
            skip = False
            continue
        if not skip:
            lines.append(line)
    if lines and lines[-1].strip():
        lines.append("")
    lines.append(cron_block.rstrip())
    new_crontab = "\n".join(lines) + "\n"
    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True)
    if proc.returncode != 0:
        print("Failed to install crontab", file=sys.stderr)
        sys.exit(proc.returncode)
    print("Cron tasks installed/updated.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=sys.executable, help="Python interpreter path")
    p.add_argument("--workspace", default=str(Path(__file__).resolve().parents[2]), help="Workspace root path")
    p.add_argument("--index-time", default=os.getenv("INDEX_TIME", "03:30"), help="HH:MM")
    p.add_argument("--notify-time", default=os.getenv("NOTIFY_TIME", "03:45"), help="HH:MM")
    return p.parse_args()


def main():
    args = parse_args()
    block = build_cron_block(args.python, args.workspace, args.index_time, args.notify_time)
    install(block)


if __name__ == "__main__":
    main()
