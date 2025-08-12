#!/usr/bin/env bash
set -euo pipefail

# 0) /opt/bin 준비
mkdir -p "${BIN_PATH}"

# 1) ffmpeg가 없으면 볼륨에 1회만 다운로드(정적 빌드)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "⬇️  downloading ffmpeg static..."
  # 필요시 다른 URL로 교체 (리눅스 x86_64 정적 바이너리)
  FFMPEG_URL="${FFMPEG_URL:-https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.1.1-...-linux64-gpl-shared.tar.xz}"
  TMP=/tmp/ffmpeg.txz
  apt-get update && apt-get install -y --no-install-recommends curl xz-utils >/dev/null
  curl -L "$FFMPEG_URL" -o "$TMP"
  tar -xf "$TMP" -C /tmp
  # 압축 안에 bin/ffmpeg 경로 맞춰서 복사 (빌드 파일에 따라 경로 다를 수 있음)
  cp /tmp/*/bin/ffmpeg "${BIN_PATH}/ffmpeg"
  chmod +x "${BIN_PATH}/ffmpeg"
  rm -rf "$TMP" /tmp/ffmpeg* /var/lib/apt/lists/*
fi

# 2) venv가 없으면 볼륨에 생성 + 의존성 설치
if [ ! -x "${VENV_PATH}/bin/python" ]; then
  echo "🐍 creating venv and installing deps..."
  python -m venv "${VENV_PATH}"
  "${VENV_PATH}/bin/pip" install --no-cache-dir --upgrade pip
  "${VENV_PATH}/bin/pip" install --no-cache-dir -r /app/requirements.txt
  # 토치/CUDA 체인 강제 제거(필요 없다고 했으니)
  "${VENV_PATH}/bin/pip" uninstall -y torch torchvision torchaudio triton \
    nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cusparse-cu12 \
    nvidia-cusolver-cu12 nvidia-nccl-cu12 nvidia-cusparselt-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 \
    nvidia-nvjitlink-cu12 || true
fi

# 3) 앱 실행
exec "${VENV_PATH}/bin/$@"
