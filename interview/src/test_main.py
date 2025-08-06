from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from io import BytesIO
from docx import Document
import fitz  # PyMuPDF
from stt.corrector import correct_transcript
from interview.model import InterviewState
from interview.chroma_qa import save_qa_pair
from stt.transcriber import convert_to_wav, transcribe_audio
from interview.graph import graph_app  # LangGraph FSM

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
    interviewId: str
    job: str
    text: str
    seq: int = 1

# ✅ 문서 텍스트 추출 함수
def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename
    content = BytesIO(file.file.read())

    if filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text

    elif filename.endswith(".docx"):
        document = Document(content)
        return "\n".join([p.text for p in document.paragraphs])

    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. (PDF 또는 DOCX)")

# ✅ /first-ask: 텍스트 기반 첫 질문 생성 (LangGraph 기반)
@app.post("/first-ask")
async def first_ask(payload: StateRequest):
    try:
        state = InterviewState(
            interview_id=payload.interviewId,
            job=payload.job,
            text=payload.text,
            seq=payload.seq
        )

        # ✅ dict로 변환해서 전달해야 LangGraph가 정상 작동함
        result = graph_app.invoke(state.dict())

        # ✅ dict로 올 경우 다시 모델로 변환
        if isinstance(result, dict):
            result = InterviewState(**result)

        session_state[payload.interviewId] = result

        return {
            "interviewId": payload.interviewId,
            "interview_question": result.questions[-1] if result.questions else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ /upload-resume: 문서 업로드 기반 첫 질문 생성
@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    job: str = Form(...),
    seq: int = Form(...)
):
    try:
        text = extract_text_from_file(file)

        state = InterviewState(
            interview_id=interviewId,
            job=job,
            text=text,
            seq=seq
        )

        result = graph_app.invoke(state.model_dump())

        # ✅ dict 변환
        if isinstance(result, dict):
            result.pop('_last_answer', None)
            result = InterviewState(**result)

        session_state[interviewId] = result

        return {
            "interviewId": interviewId,
            "interview_question": result.questions[-1] if result.questions else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ /stt-ask: 영상 업로드 → STT 분석 → 꼬리 질문 생성
@app.post("/stt-ask")
async def stt_ask(
    file: UploadFile = File(...),
    interviewId: str = Form(...),
    seq: int = Form(...)
):
    try:
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        wav_path = os.path.join(UPLOAD_DIR, "converted.wav")
        convert_to_wav(input_path, wav_path)
        transcript = transcribe_audio(wav_path)
        corrected = correct_transcript(transcript)

        state = session_state.get(interviewId)
        if not state:
            raise HTTPException(status_code=404, detail="면접 세션이 없습니다.")

        state.answers.append(transcript)
        updated_state = graph_app.invoke(state)
        session_state[interviewId] = updated_state

        return {
            "interviewId": interviewId,
            "seq": updated_state.seq,
            "interview_answer": corrected,
            "interview_answer_good": updated_state.last_analysis.get("good", ""),
            "interview_answer_bad": updated_state.last_analysis.get("bad", ""),
            "score": updated_state.last_analysis.get("score", 0),
            "new_question": updated_state.questions[-1] if updated_state.questions else ""
        }

    except Exception as e:
        print(f"[STT ERROR]: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})