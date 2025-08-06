from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os 
from io import BytesIO
from interview.chroma_qa import save_qa_pair
from interview.model import InterviewState
from interview.nodes import analyze_node, next_question_node, first_question_prompt, llm
from stt.transcriber import convert_to_wav, transcribe_audio
from stt.corrector import correct_transcript
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

# ✅ /first-ask: 자기소개서 텍스트 기반 질문 생성
@app.post("/first-ask")
async def first_ask(payload: StateRequest):
    try:
        state = InterviewState(
            interview_id=payload.interviewId,
            job=payload.job,
            text=payload.text,
            seq=payload.seq
        )

        prompt = first_question_prompt.format_messages(resume=payload.text)
        response = llm.invoke(prompt)
        first_question = response.content.strip()

        state.questions.append(first_question)
        session_state[payload.interviewId] = state

        # ✅ 첫 질문 저장
        try:
            save_qa_pair("[FIRST] " + first_question, "")
            print("✅ [ChromaDB 첫 질문 저장 완료]")
        except Exception as e:
            print(f"❌ [첫 질문 저장 실패]: {e}")

        return {
            "interviewId": payload.interviewId,
            "interview_question": first_question
        }

    except Exception as e:
        print(f"❌ [first-ask ERROR]: {e}")
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
            content = await file.read()
            f.write(content)

        wav_path = os.path.join(UPLOAD_DIR, "converted.wav")
        convert_to_wav(input_path, wav_path)
        transcript = correct_transcript(transcript)

        state = session_state.get(interviewId)
        if not state:
            raise HTTPException(status_code=404, detail="면접 세션이 없습니다.")
        # ✅ 이전 질문 저장
        prev_question = state.questions[-1] if state.questions else ""

        state.answers.append(transcript)
        state = analyze_node(state)
        state = next_question_node(state)
        session_state[interviewId] = state
         # ✅ 질문-답변 저장
        try:
            save_qa_pair(prev_question, transcript)
            print("✅ [ChromaDB QA 저장 완료]")
        except Exception as e:
            print(f"❌ [QA 저장 실패]: {e}")

        return {
            "interviewId": interviewId,
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