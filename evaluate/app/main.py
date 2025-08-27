# app/main.py
import os, re, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from json_repair import repair_json

MODEL_DIR = "./models/llama3-awq-quantized-model"

class EvaluationRequest(BaseModel):
    question: str
    answer: str

class EvaluationResponse(BaseModel):
    score: float
    feedback: str
    improve: str

app = FastAPI(
    title="면접 평가 API",
    description="Llama3(AWQ) + vLLM 기반 면접 평가",
    version="1.0.0"
)

SYSTEM_PROMPT = (
    "당신은 전문 면접 평가자입니다. 제공된 면접 질문과 답변을 분석한 뒤, "
    "반드시 아래 JSON만 출력하세요.\n"
    '{ "score": <0~100 숫자>, "feedback": "<장점 피드백>", "improve": "<개선 제안>" }'
)

def _extract_json(text: str) -> dict:
    """
    모델 출력에서 최초의 JSON 오브젝트를 뽑아내고, 불완전한 경우 json-repair로 보정한다.
    """
    # ```json ... ``` 혹은 그냥 { ... } 패턴 모두 대응
    codeblock = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if codeblock:
        raw = codeblock.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        raw = brace.group(0) if brace else text

    try:
        return json.loads(raw)
    except Exception:
        fixed = repair_json(raw)
        return json.loads(fixed)

@app.on_event("startup")
async def load_model():
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"모델 디렉토리를 찾을 수 없습니다: {MODEL_DIR}")

    # vLLM 초기화 (6GB VRAM 기준 안전값)
    app.state.llm = LLM(
        model=MODEL_DIR,
        quantization="awq_marlin",   # 권장 커널
        dtype="half",
        tensor_parallel_size=1,
        trust_remote_code=True,

        # ★ 메모리 튜닝 핵심
        gpu_memory_utilization=0.98,  # VRAM 예산 크게
        max_model_len=2048,           # KV캐시 줄이기
        max_num_batched_tokens=1024,  # 프리필 피크 낮추기
        swap_space=8,                 # 일부 CPU 오프로딩(GB)
        enforce_eager=True,           # torch.compile 오버헤드↓
        # kv_cache_dtype="fp8",       # 선택: 미지원이면 이 줄만 주석 처리
    )

    app.state.tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True
    )
    app.state.sampling = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=192  # 초반엔 짧게(피크 VRAM 완화)
    )

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(req: EvaluationRequest):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"### 질문:\n{req.question}\n\n### 답변:\n{req.answer}"}
        ]
        prompt = app.state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = app.state.llm.generate([prompt], app.state.sampling)
        generated = out[0].outputs[0].text.strip()

        data = _extract_json(generated)
        score = float(data.get("score", 0))
        feedback = str(data.get("feedback", "")).strip()
        improve = str(data.get("improve", "")).strip()
        return {"score": score, "feedback": feedback, "improve": improve}

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "inference_failed", "msg": str(e)})
