# main.py
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
# 업로드된 모듈 import
from stt.corrector import correct_transcript
from stt.transcriber import stt_from_path

app = FastAPI(title="STT + Corrector Test API")

# 임시 저장 경로
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    try:
        # 1) 파일 저장
        ext = (file.filename or "audio").split(".")[-1].lower()
        if ext not in ["mp3", "mp4", "wav", "m4a", "webm"]:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

        temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{ext}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2) STT 실행
        transcript, segments = stt_from_path(temp_path)

        # 3) 교정 실행
        correction = correct_transcript(transcript)

        # 4) 결과 반환
        return JSONResponse({
            "original_transcript": transcript,
            "segments": segments,
            "correction": correction
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")
    finally:
        # 업로드된 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
def root():
    return {"message": "STT + Corrector API is running!"}
