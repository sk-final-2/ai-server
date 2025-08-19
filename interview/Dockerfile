# ========= builder =========
FROM python:3.12-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./
RUN pip install --no-cache-dir poetry poetry-plugin-export \
 && poetry export -f requirements.txt --without-hashes --only main -o /app/requirements.txt

# torch/torchvision/torchaudio만 CPU 인덱스에서 분리
RUN awk '/^[[:space:]]*torch([=<> ]|$)/      {print > "/app/torch-req.txt"; next} \
         /^[[:space:]]*torchvision([=<> ]|$)/{print > "/app/torch-req.txt"; next} \
         /^[[:space:]]*torchaudio([=<> ]|$)/ {print > "/app/torch-req.txt"; next} \
         {print > "/app/req-notorch.txt"}' /app/requirements.txt

RUN pip wheel --no-cache-dir --prefer-binary -r /app/req-notorch.txt -w /wheels
RUN if [ -s /app/torch-req.txt ]; then \
      pip wheel --no-cache-dir --prefer-binary \
        --index-url https://download.pytorch.org/whl/cpu \
        -r /app/torch-req.txt -w /wheels ; \
    fi

RUN echo "=== WHEELS ===" && ls -l /wheels

# ========= runtime =========
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src

# ffmpeg 설치 (ffprobe 포함) + torchaudio가 쓰는 OpenMP 런타임
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 휠 + 요구사항 복사
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt /app/requirements.txt

# 오프라인 설치(휠만 사용)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt -v \
 && pip cache purge

# 앱 소스
COPY src/ /app/src

EXPOSE 8001
CMD ["uvicorn","main:app","--app-dir","/app/src","--host","0.0.0.0","--port","8001"]