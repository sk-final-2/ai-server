# Python 3.12 슬림 이미지 기반
FROM python:3.12-slim

# 시스템 패키지 설치 (ffmpeg 포함)
RUN apt-get update && \
    apt-get install -y ffmpeg gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Poetry 설치
ENV POETRY_VERSION=1.8.2
RUN pip install "poetry==$POETRY_VERSION"

# 작업 디렉토리 설정
WORKDIR /app

# pyproject.toml, poetry.lock 복사 후 의존성 설치
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# 앱 소스 복사
COPY . .

# 포트 오픈
EXPOSE 8001

# FastAPI 앱 실행
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--app-dir", "src"]
