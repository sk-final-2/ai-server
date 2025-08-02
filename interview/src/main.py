from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid
from interview.graph import graph_app
from interview.model import InterviewState
from stt.transcriber import convert_to_wav, transcribe_audio
from interview.nodes import analyze_node, next_question_node


app = FastAPI()


# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔹 상태 관리 (간단한 예시 - 실제로는 DB나 Redis로 대체 가능)
session_state = {}


class StateRequest(BaseModel):
    interviewId: str
    job: str
    text: str
    seq: int = 1

@app.post("/first-ask")
async def first_ask(payload: StateRequest):
    try:
        print("📩 [first_ask] 요청 수신:", payload)

        state = InterviewState(
            interview_id=payload.interviewId,
            job=payload.job,
            text=payload.text,
            seq=payload.seq,
        )

        # 첫 질문 설정
        first_question = "자기소개 해보세요"
        state.questions.append(first_question)

        print("✅ [first_ask] 질문 생성 완료:", first_question)

        return {
            "interviewId": state.interview_id,
            "interview_question": first_question
        }

    except Exception as e:
        print(f"❌ [first-ask ERROR]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 2. STT 음성 분석 + 답변 저장 + 분석 + 다음 질문
@app.post("/stt-ask")
async def stt_ask(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int = Form(...)
):
    try:
        print("🎙️ [stt_ask] 요청 수신:", file.filename)

        # 1. 파일 저장
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 2. 변환 (webm/mp4 → wav)
        wav_path = os.path.join(UPLOAD_DIR, "converted.wav")
        convert_to_wav(input_path, wav_path)

        # 3. STT 변환
        transcript = transcribe_audio(wav_path)
        print("📝 [STT 결과]:", transcript)

        # 4. 상태 구성
        state = InterviewState(
            interview_id=interviewId,
            seq=seq,
            questions=[],    # 실제 구현에선 DB에서 불러와야 함
            answers=[]       # 마찬가지
        )

        # 5. 답변 저장
        state.answers.append(transcript)

        # 6. 분석 및 다음 질문 생성
        state = analyze_node(state)
        state = next_question_node(state)

        return {
            "interviewId": state.interview_id,
            "seq": state.seq,
            "interview_answer": transcript,
            "interview_answer_good": state.last_analysis.get("good", ""),
            "interview_answer_bad": state.last_analysis.get("bad", ""),
            "score": state.last_analysis.get("score", 0),
            "new_question": state.questions[-1] if state.questions else ""
        }

    except Exception as e:
        print(f"❌ [stt-ask ERROR]: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})