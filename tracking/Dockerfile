# 1. 베이스 이미지
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 패키지 설치 (OpenCV, Mediapipe 등 필요한 의존성 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    gcc \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. Poetry 설치
RUN pip install --no-cache-dir poetry

# 5. Poetry 환경 설정 (경고 제거)
ENV POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

# 6. pyproject.toml, poetry.lock 복사
COPY pyproject.toml poetry.lock ./

# 7. 의존성 설치
RUN poetry install --no-interaction --no-ansi

# 8. 소스 코드 복사
COPY ./src ./src

# 9. FastAPI 실행
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003"]