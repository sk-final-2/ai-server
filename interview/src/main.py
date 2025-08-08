# 아직 수정 안함
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, uuid, shutil 
from stt.corrector import correct_transcript
from interview.model import InterviewState
from interview.chroma_qa import reset_chroma_all
from stt.transcriber import convert_to_wav, transcribe_audio
from interview.graph import graph_app  # 테스트용 그래프
from typing import Optional, Literal

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
def _temp_path(name: str) -> str:
    return os.path.join(UPLOAD_DIR, name)
# ✅ 세션 상태 저장
session_state = {}

# ✅ 첫 질문 요청용 Request 모델
class StateRequest(BaseModel):
    text: str                 # OCR 결과 텍스트
    career: Optional[str] = None
    interviewType: Optional[str] = None
    job: str
    level: Literal["상", "중", "하"] = "중"
    Language: Literal["KOREAN", "ENGLISH"] = "KOREAN"
    seq: int = 1
    interviewId: str
    count: int = 0            # 0이면 조건문 처리

@app.post("/first-ask")
async def first_ask(payload: StateRequest):
    reset_chroma_all()
    print("📦 [payload]:", payload.model_dump())
    try:
        state = InterviewState(
            interviewId=payload.interviewId,
            job=payload.job,
            text=payload.text,
            career=payload.career,
            interviewType=payload.interviewType,
            level=payload.level,
            Language=payload.Language,
            seq=payload.seq,
            count=payload.count,
            options_locked=False
        )

        result = graph_app.invoke(state.model_dump())
        if isinstance(result, dict):
            result = InterviewState(**result)

        session_state[payload.interviewId] = result

        return {
            "status": 200,
            "code": "SUCCESS",
            "message": "첫 번째 질문 생성 성공",
            "data": {
                "interviewId": payload.interviewId,
                "question": result.questions[-1] if result.questions else None,
                "seq": result.seq
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ STT 기반 꼬리 질문 생성
@app.post("/stt-ask")
async def stt_ask(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int = Form(...)
):
    try:
        # 1) 세션 존재 확인
        state = session_state.get(interviewId)
        if not state:
            raise HTTPException(status_code=404, detail="면접 세션이 없습니다. /first-ask를 먼저 호출하세요.")

        # 2) 파일 저장
        ext = (file.filename or "uploaded").split(".")[-1].lower()
        if ext not in ["mp4", "webm", "wav", "m4a", "mp3"]:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
        in_name = f"{uuid.uuid4().hex}.{ext}"
        in_path = _temp_path(in_name)
        with open(in_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 3) WAV 변환 + STT
        wav_path = _temp_path(f"{uuid.uuid4().hex}.wav")
        convert_to_wav(in_path, wav_path)                  # (ffmpeg/av 기반)
        raw_transcript = transcribe_audio(wav_path)        # faster-whisper 등
        corrected = correct_transcript(raw_transcript) or raw_transcript

        # 4) 그래프 진행(answer→analyze→next_question)
        state.last_answer = corrected                      # 최신 답변 주입
        result = graph_app.invoke(state.model_dump())      # LangGraph 0.6: dict in/out
        if isinstance(result, dict):
            result = InterviewState(**result)

        # 5) 세션 갱신
        session_state[interviewId] = result

        # 6) 분석/다음 질문 꺼내기 (없을 때 대비)
        analysis = getattr(result, "last_analysis", {}) or {}

        # 7) 명세서 포맷으로 반환
        return {
            "interviewId": interviewId,
            "seq": getattr(result, "seq", seq + 1),
            "interview_answer": corrected,
            "interview_answer_good": analysis.get("good", ""),
            "interview_answer_bad": analysis.get("bad", ""),
            "score": analysis.get("score", 0),
            "new_question": result.questions[-1] if result.questions else ""
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[stt-ask ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))