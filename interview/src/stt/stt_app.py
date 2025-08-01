from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
# import requests # requests 모듈이 더 이상 필요 없으므로 제거하거나 주석 처리합니다.
from src.stt.transcriber import convert_to_wav, transcribe_audio

app = FastAPI()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔗 너의 스프링 백엔드 주소로 바꿔줘 (이 변수도 더 이상 사용되지 않으므로 제거하거나 주석 처리합니다.)
# BACKEND_URL = os.getenv("BACKEND_URL")

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    # 파일명 및 확장자
    filename = file.filename
    ext = filename.split(".")[-1].lower()

    # 확장자 제한
    if ext not in ["webm", "mp4", "wav"]:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    # 파일 경로 설정
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(UPLOAD_DIR, "converted.wav")

    # 파일 저장
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 변환 처리
    if ext != "wav":
        convert_to_wav(input_path, output_path)
    else:
        output_path = input_path

    # STT 처리
    text = transcribe_audio(output_path)

    # ✨✨✨ 이 부분이 제거되거나 주석 처리되어야 합니다! ✨✨✨
    # # Spring 백엔드로 결과 전송
    # try:
    #     response = requests.post(
    #         BACKEND_URL,
    #         json={"filename": filename, "text": text},
    #         timeout=60
    #     )
    #     print("✅ Spring 응답:", response.status_code, response.text)
    # except Exception as e:
    #     print("❌ Spring 전송 실패:", e)
    # ✨✨✨ 제거 또는 주석 처리 끝 ✨✨✨

    # 클라이언트(Spring Boot)에게 응답
    # 이 부분은 유지합니다.
    return JSONResponse(content={"text": text})