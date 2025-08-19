#!/usr/bin/env sh
set -e

# (선택) BIN_PATH 쓸 거면 PATH에만 추가
#[ -n "${BIN_PATH}" ] && export PATH="${BIN_PATH}:${PATH}"

# 아무 설치도 하지 말고, 바로 앱 실행
exec "$@"