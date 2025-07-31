# 1. 베이스 이미지
FROM python:3.9-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. Poetry 설치
RUN pip install --no-cache-dir poetry

# 4. pyproject.toml과 lock 파일 복사
COPY pyproject.toml poetry.lock ./

# 5. 의존성 설치 (프로덕션 환경이라면 --only main)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# 6. 소스 코드 복사
COPY ./src ./src

# 7. OpenCV, DeepFace 실행에 필요한 추가 패키지
RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 8. FastAPI 실행 (Uvicorn)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
