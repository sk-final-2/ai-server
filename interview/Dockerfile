# syntax=docker/dockerfile:1.7

############################
# 1) Builder: 의존성 휠 사전 빌드
############################
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.2

# 빌드에만 필요한 것 (최종 이미지엔 X)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Poetry 설치 및 요구사항 export
RUN pip install "poetry==$POETRY_VERSION"
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

# wheel 미리 빌드 (최종 이미지는 여기서만 복사)
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

############################
# 2) Runtime: 최소 런타임만
############################
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 런타임에 꼭 필요한 것만 (ffmpeg 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치 (Poetry X, 캐시 X)
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

# 앱 소스만 복사 (불필요한 파일은 .dockerignore로 제외)
COPY src/ ./src/
#COPY src/main.py . 
# 비루트 유저
RUN useradd -m appuser
USER appuser

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "src"]
