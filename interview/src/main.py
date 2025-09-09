# main.py — 서비스용 최종본
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Literal
import os, uuid, shutil
from utils.chroma_qa import save_turn  # ✅ 추가
from stt.corrector import correct_transcript
from interview.model import InterviewState
from stt.transcriber import stt_from_path
from interview.graph import graph_app
from utils.chroma_setup import reset_chroma, get_collections, reset_interview  # ✅ 변경
from fastapi.responses import JSONResponse
# OCR import
from utils.extractor import (
    extract_text_from_pdf_pymupdf,
    extract_text_from_txt,
    extract_text_from_docx
)
import tempfile, os, re
from utils.text_cleaner import clean_spacing

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
    ocrText: str                               # OCR 결과 텍스트
    career: Optional[str] = None
    interviewType: Optional[str] = None
    job: str
    level: Literal["상", "중", "하"] = "중"
    language: Literal["KOREAN", "ENGLISH"] = "KOREAN"
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
            ocrText=payload.ocrText,
            career=payload.career,
            interviewType=payload.interviewType,
            level=payload.level,
            language=payload.language,
            seq=payload.seq or 1,
            count=payload.count,
            options_locked=False,
            # 초기화
            question="",
            answer=[],
            last_answer=None,
            keepGoing=True, 
            step=0,
        )

        # ✅ 그래프 실행 (first_question_node 내부에서 save_question 호출)
        result = graph_app.invoke(state.model_dump())
        if isinstance(result, dict):
            result = InterviewState(**result)

        # ✅ 세션 캐시 갱신
        session_state[payload.interviewId] = result

        last_question = (
            result.questions[-1] if hasattr(result, "questions") and result.questions
            else result.question
        )

        topic = (
            state.topics[state.current_topic_index]["name"]
            if getattr(state, "topics", None)
            and 0 <= state.current_topic_index < len(state.topics)
            else ""
        )

        aspect = (
            state.aspects[state.aspect_index]
            if getattr(state, "aspects", None)
            and 0 <= state.aspect_index < len(state.aspects)
            else ""
        )
        
        save_turn(
            interviewId=payload.interviewId,
            seq=1,
            question=last_question,
            answer="",
            topic=topic,
            aspect=aspect,
            feedback=None,
        )
        print(f"📝 [first-ask] last_question={last_question}")
        return {
            "interviewId": payload.interviewId,
            "interview_question": result.question
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ STT 기반 꼬리 질문 생성
@app.post("/stt-ask")
async def stt_ask(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int | None = Form(None),   # ← 하위 호환용(무시)
    question: str = Form(None),     # ✅ (옵션) 동적 모드용 질문
):
    # 1) 세션 불러오기
    state = session_state.get(interviewId)
    if not state:
        raise HTTPException(status_code=404, detail="면접 세션이 없습니다. /first-ask를 먼저 호출하세요.")

    # 2) 업로드 파일 저장 (임시 보관)
    ext = (file.filename or "uploaded").split(".")[-1].lower()
    if ext not in ["mp4", "webm", "wav", "m4a", "mp3"]:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
    in_path = _temp_path(f"{uuid.uuid4().hex}.{ext}")
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 3) STT 실행 (numpy 기반)
    raw_transcript, _ = stt_from_path(in_path, language=state.language)

    # 4) 교정 실행 (언어별 교정 전략)
    corrected_dict = correct_transcript(raw_transcript, language=state.language)
    corrected = corrected_dict.get("corrected", raw_transcript) if isinstance(corrected_dict, dict) else raw_transcript
    
    # 4) 답변 업데이트 (→ DB 저장은 answer_node에서 처리됨)
    state.last_answer = corrected
    if not hasattr(state, "answer"):
        state.answer = []
    state.answer.append(corrected)

    # ✅ 4-1) 동적 모드(count=0)일 때만 임시 질문 보관
    if getattr(state, "count", 0) == 0 and question:
        state.last_question_for_dynamic = question
        print(f"📝 [stt-ask] 동적 모드용 질문 저장: {question}")

    # 5) 그래프 실행 (분석 → keepgoing → next_question)
    result = graph_app.invoke(state.model_dump())
    if isinstance(result, dict):
        result = InterviewState(**result)

    session_state[interviewId] = result

    # 6) 출력용 seq & 종료 여부
    seq_out = getattr(result, "step", None)
    if seq_out is None:
        seq_out = getattr(result, "seq", None)
    if seq_out is None:
        seq_out = 1

    analysis = result.last_analysis if hasattr(result, "last_analysis") else {}

    save_turn(
        interviewId=interviewId,
        seq=seq_out,
        question=result.questions[-1] if result.questions else "",
        answer=corrected,
        topic=state.topics[state.current_topic_index]["name"] if state.topics else None,
        aspect=state.aspects[state.aspect_index] if hasattr(state, "aspects") else None,
        feedback=analysis
    )
    print(f"💾 [save_turn 호출] seq={seq_out}, answer={corrected}")
    return {
        "interviewId": interviewId,
        "seq": seq_out,
        "interview_answer": corrected,
        "interview_answer_good": analysis.get("good", ""),
        "interview_answer_bad": analysis.get("bad", ""),
        "score": analysis.get("score", 0),
        "new_question": result.question if getattr(result, "question", None) else "",
        "keepGoing": getattr(result, "keepGoing", True),
    }

# OCR 메서드
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # 공백 정리
    return text.strip()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    _, ext = os.path.splitext(filename)

    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # 확장자별 텍스트 추출
        if ext == ".pdf":
            raw_text = extract_text_from_pdf_pymupdf(tmp_path)
        elif ext == ".txt":
            raw_text = extract_text_from_txt(tmp_path)
        elif ext == ".docx":
            raw_text = extract_text_from_docx(tmp_path)
        else:
            return JSONResponse(status_code=400, content={"error": f"{ext} 형식은 지원하지 않습니다."})

        cleaned_text = clean_spacing(raw_text)
        return {"ocr_output": cleaned_text}

    finally:
        # 임시 파일 정리
        os.remove(tmp_path)