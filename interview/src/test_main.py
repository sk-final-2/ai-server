from fastapi import FastAPI, UploadFile, File, Form, HTTPException,Body 
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from stt.corrector import correct_transcript
from interview.model import InterviewState
from interview.chroma_qa import reset_chroma_all
from stt.transcriber import convert_to_wav, transcribe_audio
from interview.graph import graph_app  # 테스트용 그래프
from typing import Optional, Literal
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
session_state = {}

class StateRequest(BaseModel):
    text: str
    career: Optional[str] = None
    interviewType: Optional[str] = None 
    job: str
    level: Literal["상","중","하"] = "중"
    Language: Literal["KOREAN","ENGLISH"] = "KOREAN"
    seq: int = 1
    interviewId: str   
    count: int= 0 #수정 예정 

    
# print("📦 payload:", payload.model_dump())
# ✅ /first-ask: 텍스트 기반 첫 질문 생성 (LangGraph 기반)
session_state: dict[str, InterviewState] = {}   # 나중에 Redis 대체
@app.post("/first-ask")
async def first_ask(payload: StateRequest):
    print("📦 [payload.raw]:", payload)
    print("📦 [payload.dict]:", payload.model_dump())
    print(f"🧹 [/first-ask] reset_chroma_all() 호출 - interviewId={payload.interviewId}")
    reset_chroma_all()
    try:
        state = InterviewState(
            interviewId=payload.interviewId,
            job=payload.job,
            text=payload.text,
            career=payload.career,
            interviewType=payload.interviewType,
            seq=payload.seq,
            Language=payload.Language,
            level=payload.level,
            count=payload.count, # 일단 수정 예정
            options_locked=False
            )

        result = graph_app.invoke(state)
        if isinstance(result, dict):
            result = InterviewState(**result)

        session_state[payload.interviewId] = result

        return {
            "status": 200,
            "code": "SUCCESS",
            "message": "첫 번째 질문 생성 성공",
            "data": {
                "interviewId": payload.interviewId,
                "question": result.questions[-1],
                "seq": result.seq,
                "count": result.count
            }
}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

# ✅ stt-ask: 영상 업로드 → STT 분석 → 꼬리 질문 생성
class TextAskRequest(BaseModel):
    interviewId: str
    seq: int
    answer: str

# ✅ JSON 기반 텍스트 답변 처리 API
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

        # 4. 답변 추가 및 상태 갱신
        state.last_answer = transcript  # ✅ answer_node에서 참조
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
    
class TextAskRequest(BaseModel):
    interviewId: str
    seq: int
    answer: str

@app.post("/text-ask")
async def text_ask(payload: TextAskRequest):
    state = session_state.get(payload.interviewId)
    if not state:
        raise HTTPException(status_code=404, detail="면접 세션이 없습니다.")

    state.seq = payload.seq
    state.last_answer = payload.answer   # ✅ 핵심
    state.answer.append(payload.answer)

    result = graph_app.invoke(state)       # ✅ 모델 그대로
    if isinstance(result, dict):
        result = InterviewState(**result)

    session_state[payload.interviewId] = result

    analysis = result.last_analysis or {}   # ✅ 방어
    return {
        "interviewId": payload.interviewId,
        "seq": result.seq,
        "interview_answer": payload.answer,
        "interview_answer_good": analysis.get("good", ""),
        "interview_answer_bad": analysis.get("bad", ""),
        "score": analysis.get("score", 0),
        "new_question": result.questions[-1] if result.questions else "",
        "is_finished": result.is_finished
    }