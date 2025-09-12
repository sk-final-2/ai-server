FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121" \
    VLLM_NO_USAGE_STATS=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python 패키지
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "torch==2.3.1+cu121" && \
    python3 -m pip install \
      vllm \
      "transformers>=4.43" \
      "fastapi>=0.116,<0.117" \
      "uvicorn[standard]>=0.35,<0.36" \
      json-repair

# 앱/모델(로컬 가중치) 포함
COPY app ./app
COPY models ./models

# 앱이 참조할 모델 경로 (app/main.py의 경로와 맞춰야 함)
ENV MODEL_DIR=/app/models/llama3-awq-quantized-model
# T4에서 안정적으로 돌리기 위한 기본값(코드가 env를 읽게 되어있다면 사용)
ENV VLLM_QUANT=awq

EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD curl -fsS http://localhost:8002/ || exit 1

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8002"]
