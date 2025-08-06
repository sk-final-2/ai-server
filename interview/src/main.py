from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from interview.model import InterviewState
from interview.graph import graph_app
from stt.transcriber import convert_to_wav, transcribe_audio
from stt.corrector import correct_transcript

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

# ✅ 세션 상태 저장
session_state = {}

# ✅ 첫 질문 요청용 Request 모델
class StateRequest(BaseModel):
    interviewId: str
    job: str
    text: str
    seq: int = 1

# ✅ 자기소개서 텍스트 기반 첫 질문 생성
@app.post("/first-ask")
async def first_ask(payload: StateRequest):
    try:
        state = InterviewState(
            interview_id=payload.interviewId,
            job=payload.job,
            text=payload.text,
            seq=payload.seq
        )

        result = graph_app.invoke(state.model_dump())

        # ✅ dict일 경우 다시 모델로 변환
        if isinstance(result, dict):
            result = InterviewState(**result)

        session_state[payload.interviewId] = result

        return {
            "interviewId": payload.interviewId,
            "interview_question": result.questions[-1] if result.questions else None
        }

    except Exception as e:
        print(f"[first-ask ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ✅ STT 기반 꼬리 질문 생성
@app.post("/stt-ask")
async def stt_ask(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int = Form(...)
):
    try:
        # 1. 파일 저장
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # 2. WAV 변환 + STT + 교정
        wav_path = os.path.join(UPLOAD_DIR, "converted.wav")
        convert_to_wav(input_path, wav_path)
        transcript = transcribe_audio(wav_path)
        corrected = correct_transcript(transcript)

        # 3. 상태 불러오기
        state = session_state.get(interviewId)
        if not state:
            raise HTTPException(status_code=404, detail="면접 세션이 없습니다.")

        state.answers.append(transcript)

        # 4. LangGraph 실행
        result = graph_app.invoke(state.model_dump())

        if isinstance(result, dict):
            result = InterviewState(**result)

        session_state[interviewId] = result

        return {
            "interviewId": interviewId,
            "seq": result.seq,
            "interview_answer": corrected,
            "interview_answer_good": result.last_analysis.get("good", ""),
            "interview_answer_bad": result.last_analysis.get("bad", ""),
            "score": result.last_analysis.get("score", 0),
            "new_question": result.questions[-1] if result.questions else ""
        }

    except Exception as e:
        print(f"[stt-ask ERROR] {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})