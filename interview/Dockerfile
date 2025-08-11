# 1) 빌드 베이스
FROM python:3.12-slim

# 2) 기본 패키지(필요 시 ffmpeg 등 추가)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# 3) 워크디렉토리
WORKDIR /app

# 4) 의존성 먼저 복사/설치 (캐시 최적화)
COPY pyproject.toml poetry.lock* requirements.txt* ./
# Poetry 쓰면:
# RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
# requirements.txt 쓰면:
RUN if [ -f "requirements.txt" ]; then pip install --no-cache-dir -r requirements.txt; fi

# 5) 앱 코드 복사
COPY . .

# 6) 런타임 환경변수(영속 경로: 볼륨으로 연결)
ENV CHROMA_DIR=/app/chroma_data \
    PYTHONUNBUFFERED=1

# 7) 영속 디렉토리 생성(이미지 안에서는 빈 폴더만 유지)
RUN mkdir -p ${CHROMA_DIR}

# 8) 포트/엔트리포인트
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "src"]
