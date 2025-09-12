# app/main.py
import os
import re
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from json_repair import repair_json

def env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v is None or v != "" else default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

# 모델 경로: MODEL_PATH 우선, 없으면 MODEL_DIR, 둘 다 없으면 기본 경로
MODEL_PATH = env_str("MODEL_PATH", None) or env_str("MODEL_DIR", "/app/models/llama3-awq-quantized-model")

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
    "당신은 전문 면접 평가자입니다. 제공된 면접 질문과 지원자의 답변을 분석하십시오. "
    "분석 결과는 JSON 객체로 반환해야 하며 키는 'score', 'feedback', 'improve' 입니다. "
    '예: { "score": <1.0~5.0 (소수점 한 자리)>, "feedback": "<강점과 피드백>", "improve": "<개선 제안>" }'
)

def _extract_json(text: str) -> dict:
    """모델 출력에서 JSON을 추출하고, 불완전하면 json-repair로 보정."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else (re.search(r"\{.*\}", text, re.DOTALL).group(0) if re.search(r"\{.*\}", text, re.DOTALL) else text)
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(repair_json(raw))

@app.on_event("startup")
async def load_model():
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"모델 디렉토리를 찾을 수 없습니다: {MODEL_PATH}")

    # vLLM 파라미터 (환경변수로 오버라이드 가능)
    quant          = env_str("VLLM_QUANT", "awq")            # 모델 config의 awq와 일치
    dtype          = env_str("VLLM_DTYPE", "half")
    max_len        = env_int("VLLM_MAX_MODEL_LEN", 2048)
    max_btoks      = env_int("VLLM_MAX_BATCHED_TOKENS", max_len)
    if max_btoks < max_len:
        max_btoks = max_len                                  # 스케줄러 에러 방지
    gpu_util       = env_float("VLLM_GPU_UTIL", 0.98)
    swap_space_gb  = env_int("VLLM_SWAP_GB", 8)
    eager          = env_str("VLLM_EAGER", "true").lower() == "true"
    kv_cache_dtype = env_str("VLLM_KV_CACHE_DTYPE", None)    # 옵션

    llm_kwargs = dict(
        model=MODEL_PATH,
        quantization=quant,
        dtype=dtype,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_len,
        max_num_batched_tokens=max_btoks,
        swap_space=swap_space_gb,
        enforce_eager=eager,
        disable_log_stats=True,
    )
    if kv_cache_dtype:  # 지원되는 경우에만 적용
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype

    app.state.llm = LLM(**llm_kwargs)
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 샘플링 설정(필요시 ENV로 조정 가능)
    temperature = env_float("SAMPLING_TEMPERATURE", 0.1)
    top_p       = env_float("SAMPLING_TOP_P", 0.9)
    max_tokens  = env_int("SAMPLING_MAX_TOKENS", 192)

    app.state.sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
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
        prompt = app.state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = app.state.llm.generate([prompt], app.state.sampling)
        generated = out[0].outputs[0].text.strip()

        data = _extract_json(generated)
        score = float(data.get("score", 0))
        feedback = str(data.get("feedback", "")).strip()
        improve = str(data.get("improve", "")).strip()
        return {"score": score, "feedback": feedback, "improve": improve}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "inference_failed", "msg": str(e)})
