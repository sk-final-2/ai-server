# main.py — 서비스용 최종본
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Literal
import os, uuid, shutil

from stt.corrector import correct_transcript
from interview.model import InterviewState
from stt.transcriber import convert_to_wav, transcribe_audio
from interview.graph import graph_app
from utils.chroma_setup import reset_chroma, get_collections, reset_interview  # ✅ 변경

# ✅ 앱 생명주기: 개발에서만 전역 초기화 + 컬렉션 준비
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("CHROMA_RESET_ON_START", "0") == "1":
        reset_chroma()
    # EF가 붙은 컬렉션을 미리 열어 초기화(안 열어도 작동은 함)
    app.state.qa_questions, app.state.qa_answers, app.state.qa_feedback = get_collections()
    yield

app = FastAPI(lifespan=lifespan)

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

# ✅ 세션 상태 저장 (메모리)
session_state = {}

# ✅ 첫 질문 요청용 Request 모델
class StateRequest(BaseModel):
    text: str                               # OCR 결과 텍스트
    career: Optional[str] = None
    interviewType: Optional[str] = None
    job: str
    level: Literal["상", "중", "하"] = "중"
    Language: Literal["KOREAN", "ENGLISH"] = "KOREAN"
    seq: int = 1
    interviewId: str
    count: int = 0                          # 0이면 노드에서 기본 로직(최대 20)

@app.post("/first-ask")
async def first_ask(payload: StateRequest, request: Request):
    print("📦 [payload]:", payload.model_dump())
    try:
        # ✅ 이 인터뷰의 기존 데이터만 초기화 (운영 안전)
        reset_interview(payload.interviewId)

        # ✅ 초기 상태 구성
        state = InterviewState(
            interviewId=payload.interviewId,
            job=payload.job,
            text=payload.text,
            career=payload.career,
            interviewType=payload.interviewType,
            level=payload.level,
            Language=payload.Language,
            seq=payload.seq or 1,
            count=payload.count,
            options_locked=False,
            # 초기화
            question=[],
            answer=[],
            last_answer=None,
            is_finished=False,
            step=0,
        )

        # ✅ 그래프 실행 (first_question_node 내부에서 save_question 호출)
        result = graph_app.invoke(state.model_dump())
        if isinstance(result, dict):
            result = InterviewState(**result)

        # ✅ 세션 캐시 갱신
        session_state[payload.interviewId] = result

        return {
            "status": 200,
            "code": "SUCCESS",
            "message": "첫 번째 질문 생성 성공",
            "data": {
                "interviewId": payload.interviewId,
                "question": result.question[-1] if result.question else None,
                "seq": result.seq,
            },
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
        # 1) 세션 확인
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
        convert_to_wav(in_path, wav_path)
        raw_transcript = transcribe_audio(wav_path)
        corrected = correct_transcript(raw_transcript) or raw_transcript

        # 4) 그래프 진행(answer→analyze→next_question)
        state.last_answer = corrected
        result = graph_app.invoke(state.model_dump())   # LangGraph 0.6: dict in/out
        if isinstance(result, dict):
            result = InterviewState(**result)

        # 5) 세션 갱신
        session_state[interviewId] = result

        # 6) 분석/다음 질문 꺼내기
        analysis = getattr(result, "last_analysis", {}) or {}

        # 7) 응답
        return {
            "interviewId": interviewId,
            "seq": getattr(result, "seq", seq + 1),
            "interview_answer": corrected,
            "interview_answer_good": analysis.get("good", ""),
            "interview_answer_bad": analysis.get("bad", ""),
            "score": analysis.get("score", 0),
            "new_question": result.question[-1] if result.question else "",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[stt-ask ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
#a