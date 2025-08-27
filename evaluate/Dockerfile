
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # torch를 CUDA 12.1 전용 인덱스에서 받도록
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121" \
    VLLM_NO_USAGE_STATS=1

# Python + 필수 빌드 도구
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치
# 1) GPU용 torch 먼저 설치 (cu121)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "torch==2.3.1+cu121"

# 2) 나머지 라이브러리
#    - huggingface_hub[cli]는 불필요하므로 제외
RUN python3 -m pip install \
    vllm \
    "transformers>=4.43" \
    "fastapi>=0.116,<0.117" \
    "uvicorn[standard]>=0.35,<0.36" \
    json-repair

# 앱 복사 (모델은 볼륨 마운트로 외부에서 주입)
COPY app ./app

# 포트 노출
EXPOSE 8000

# 간단 헬스체크 (선택)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD curl -fsS http://localhost:8000/ || exit 1

# 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
