# [1] Python 3.12 slim 기반 이미지
FROM python:3.12-slim

# [2] 작업 디렉토리 생성
WORKDIR /app

# [3] Poetry 설치
RUN pip install --upgrade pip && pip install poetry

# [4] Poetry 설정 - 가상환경 없이 전역에 설치
RUN poetry config virtualenvs.create false

# [5] pyproject.toml & poetry.lock 복사
COPY pyproject.toml poetry.lock ./

# [6] 의존성 설치
RUN poetry install --no-root

# [7] 실제 소스 코드 복사
COPY ./src ./src

# [8] FastAPI 실행 (포트 8002)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]