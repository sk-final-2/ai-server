# ===== builder: requirements만 추출 =====
FROM python:3.12-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry poetry-plugin-export \
 && poetry --version \
 && poetry export -f requirements.txt --without-hashes --only main -o /app/requirements.txt

# ===== runtime: 코드만 포함 (의존성/ffmpeg는 볼륨에) =====
FROM python:3.12-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    VENV_PATH=/opt/venv \
    BIN_PATH=/opt/bin \
    CHROMA_DIR=/app/chroma_data \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    CTRANSLATE2_ROOT=/root/.cache/ctranslate2 \
    PATH="/opt/bin:${PATH}"

# 코드 + 요구사항 + 엔트리포인트만
COPY src/ ./src
COPY --from=builder /app/requirements.txt ./requirements.txt
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 전부 볼륨으로
VOLUME ["/opt/venv", "/opt/bin", "/app/chroma_data", "/root/.cache/huggingface", "/root/.cache/ctranslate2"]

EXPOSE 8001
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "interview.main:app", "--host", "0.0.0.0", "--port", "8001"]