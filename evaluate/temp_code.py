# app/main.py

import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import os

# --- 1. 기본 설정 및 모델 로딩 ---

# 모델이 저장된 경로 설정 (Dockerfile이나 실행 위치 기준)
MODEL_DIR = "./models/llama3-awq-quantized-model"

# Pydantic 모델: API 요청 본문의 형식을 정의합니다.
class EvaluationRequest(BaseModel):
    question: str
    answer: str

# Pydantic 모델: API 응답 본문의 형식을 정의합니다.
class EvaluationResponse(BaseModel):
    score: float
    feedback: str
    improve: str

# FastAPI 앱 초기화
app = FastAPI(
    title="면접 평가 API",
    description="Llama 3 파인튜닝 모델과 vLLM을 사용한 AI 면접관 API",
    version="1.0.0"
)

# 시스템 프롬프트 (학습 때 사용한 것과 완벽히 동일해야 함)
SYSTEM_PROMPT = """당신은 전문 면접 평가자입니다. 당신의 임무는 제공된 면접 질문과 지원자의 답변을 분석하는 것입니다.
이 분석을 바탕으로 'score', 'feedback', 'improve'이라는 세 가지 키를 포함하는 JSON 객체를 생성해야 합니다.
- 'score': 답을 평가하는 숫자 점수.
- 'feedback': 답변의 강점에 대한 상세하고 건설적인 피드백.
- 'improve': 답변에서 개선할 수 있는 사항에 대한 제안."""

# --- 2. vLLM 엔진 초기화 ---
# FastAPI 서버가 시작될 때 단 한 번만 모델을 로드합니다.
@app.on_event("startup")
async def load_model():
    print("="*50)
    print("vLLM으로 최종 AWQ 모델을 로딩합니다...")
    
    global llm, tokenizer, sampling_params
    
    if not os.path.exists(MODEL_DIR):
        raise RuntimeError(f"모델 디렉토리를 찾을 수 없습니다: {MODEL_DIR}")

    llm = LLM(
        model=MODEL_DIR,
        quantization="awq",
        tensor_parallel_size=1, # 사용 가능한 GPU 수
        trust_remote_code=True,
        dtype="half" # float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=512
    )
    
    print("✅ 모델 로딩 및 서버 준비 완료!")
    print("="*50)


# --- 3. API 엔드포인트 정의 ---
@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    """
    제공된 질문(question)과 답변(answer)을 평가하여 JSON 형식으로 결과를 반환합니다.
    """
    try:
        # 1. 프롬프트 생성
        user_content = f"### 질문:\n{request.question}\n### 답변:\n{request.answer}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. vLLM으로 추론 실행
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        # 3. 결과 파싱 및 반환
        # 모델이 항상 완벽한 JSON을 생성한다는 보장이 없으므로, 파싱 에러 처리는 필수입니다.
        try:
            result_json = json.loads(generated_text)
            # score가 float이 아닐 경우를 대비한 형 변환
            result_json['score'] = float(result_json.get('score', 0.0))
            return result_json
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"JSON 파싱 에러: {e}")
            print(f"모델 원본 출력: {generated_text}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "모델이 유효한 JSON을 생성하지 못했습니다.",
                    "raw_output": generated_text
                }
            )

    except Exception as e:
        print(f"서버 내부 에러: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 에러 발생: {str(e)}")

# 서버 상태 확인을 위한 루트 엔드포인트
@app.get("/")
def read_root():
    return {"status": "AI 면접관 API 서버가 정상적으로 동작 중입니다."}