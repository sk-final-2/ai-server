from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
# import requests # requests 모듈이 더 이상 필요 없으므로 제거하거나 주석 처리합니다.
from src.stt.transcriber import convert_to_wav, transcribe_audio
from src.stt.corrector import correct_transcript 
app = FastAPI()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
    corrected_text = correct_transcript(text)

    return JSONResponse(content={
        "original": text,
        "corrected": corrected_text
        })