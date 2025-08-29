from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
import json, re, dirtyjson
from json_repair import repair_json

# ====================================================
# 🔹 언어 검증 / 정규화 유틸
# ====================================================

_HANGUL = r"[가-힣]"
_LATIN  = r"[A-Za-z]"
_CJK    = r"[\u4E00-\u9FFF\u3400-\u4DBF]"   # 한자
_JP     = r"[\u3040-\u30FF]"

def _ratio(text: str, pattern: str) -> float:
    if not text:
        return 0.0
    total = len(re.findall(r"\S", text))
    hits  = sum(len(m) for m in re.findall(pattern, text))
    return hits / max(total, 1)

def validate_language_text(text: str, target: Literal["KOREAN", "ENGLISH"]) -> bool:
    """텍스트가 지정 언어 규칙을 따르는지 검증"""
    hangul = _ratio(text, _HANGUL)
    latin  = _ratio(text, _LATIN)
    cjk    = _ratio(text, _CJK)
    jp     = _ratio(text, _JP)
    if target == "KOREAN":
        return (hangul >= 0.80) and ((cjk + jp + latin) <= 0.20)
    else:
        return (latin >= 0.80) and ((cjk + jp) <= 0.20)

def normalize_text(llm, text: str, target: Literal["KOREAN", "ENGLISH"], nounify: bool=False) -> str:
    """텍스트를 한국어나 영어로 정규화 (토픽용이면 명사형으로 변환 가능)"""
    if target == "ENGLISH":
        sys_rule = "Use English only. Keep the meaning. Output body only."
        if nounify:
            sys_rule += " Convert into a noun phrase (not a question)."
        user = "Rewrite in English only:\n" + (text or "")
    else:
        sys_rule = "오직 한국어만 사용. 의미 유지. 본문만 출력."
        if nounify:
            sys_rule += " 명사구로 변환. 질문문/조사/어미 제거."
        user = "한국어로 다시 쓰기:\n" + (text or "")

    # 토큰 절약: 너무 길면 자르기
    if len(user) > 350:
        user = user[:350].rsplit(" ", 1)[0] + "..."

    try:
        resp = (
            ChatPromptTemplate.from_messages([
                ("system", sys_rule),
                ("user", "{u}")
            ]) | llm.bind(max_tokens=60, temperature=0)
        ).invoke({"u": user})
        return getattr(resp, "content", str(resp)).strip()
    except Exception:
        return text or ""


# ====================================================
# 🔹 토픽 정규화
# ====================================================

def normalize_topic_str(text: str) -> str:
    """문장형 토픽을 명사형 키워드로 정규화"""
    t = text.strip()
    t = re.sub(r'에\s*(대한|관한|대해)', ' ', t)
    t = re.sub(r'(이|가)?\s*있는지$', '', t)
    t = re.sub(r'(하는지|했는지|될지|될까)$', '', t)
    t = re.sub(r'(인가요|인가|이냐)$', '', t)
    return re.sub(r'\s+', ' ', t).strip()


# ====================================================
# 🔹 JSON 파싱 (LLM 출력 보정 포함)
# ====================================================

def safe_parse_json_from_llm(content: str) -> dict:
    print("📨 [LLM 응답 원문]:", content)
    
    # 무조건 초기화
    cleaned = str(content).strip().replace("```json", "").replace("```", "").strip()

    # ✅ 자동 괄호 보정
    if cleaned.count("{") > cleaned.count("}"):
        cleaned += "}" * (cleaned.count("{") - cleaned.count("}"))

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            print("✅ [JSON 파싱 성공]:", parsed)
            return parsed
    except Exception as e:
        print("❌ [JSON 파싱 실패 - 1차]:", str(e))

    # ✅ fallback: dirtyjson
    try:
        parsed = dirtyjson.loads(cleaned)
        if isinstance(parsed, dict):
            print("✅ [JSON 파싱 성공 - dirtyjson]:", parsed)
            return parsed
    except Exception as e:
        print("❌ [JSON 파싱 실패 - dirtyjson]:", str(e))

    # 실패하면 빈 dict
    return {}



# ====================================================
# 🔹 인터뷰 규칙 매핑
# ====================================================

TYPE_RULE = {
    "TECHNICAL": "- 구체적인 기술적 방법, 과정, 결과, 문제 해결 경험에 집중할 것. 인성/가치관 관련 질문 금지.",
    "PERSONALITY": "- 지원자의 성격, 행동, 동기, 태도, 협업 방식에 초점을 맞출 것. 추상적 표현(가치관, 성장 가능성) 금지.",
    "MIXED": "- 기술과 인성 질문을 균형 있게 섞되 동일 유형만 반복하지 말 것."
}

LEVEL_RULE = {
    "하": "- 쉬운 질문. 부담 없는 경험 중심. 예: '가장 기억에 남는 프로젝트는 무엇이었나요?'",
    "중": "- 실제 상황 대처/적용을 묻는 질문. 예: '팀 프로젝트에서 갈등이 있었을 때 어떻게 해결했나요?'",
    "상": "- 심층적이고 압박 있는 질문. 예: '프로젝트 실패 원인과 본인의 책임은 무엇이라고 생각하나요?'"
}

LANG_RULE = {
    "KOREAN": (
        "- 출력은 반드시 한국어 질문 하나뿐. "
        "- 평서문(~다, ~있다) 금지, 질문 어미(~나요?, ~습니까?) 필수."
    ),
    "ENGLISH": (
        "- Output must be in English only. "
        "- Must be a single interview question ending with '?'. "
        "- No prefaces, no numbering, no explanations."
    )
}


# ====================================================
# 🔹 system_rule 생성기
# ====================================================

def system_rule(state) -> str:
    """면접 유형 + 난이도 + 언어에 맞는 system prompt 생성"""
    language = getattr(state, "language", "KOREAN")
    interviewType = getattr(state, "interviewType", "MIXED")
    career = getattr(state, "career", "신입")
    level = getattr(state, "level", "중")

    base = "너는 면접관이다. " if language == "KOREAN" else "You are an interviewer. "
    base += "- 출력은 반드시 질문 하나뿐이다. 답변, 해설, 설명, 메타 문구 금지. "
    base += TYPE_RULE.get(interviewType, "")
    base += LEVEL_RULE.get(level, "")
    base += LANG_RULE.get(language, "")

    # 경력 구분
    if career == "신입":
        base += " 신입이므로 학습 태도, 성장 가능성, 적응력에 초점을 맞춰라."
    elif career in ["경력", "경력직"]:
        base += " 경력직이므로 성과, 리더십, 문제 해결 경험에 초점을 맞춰라."

    return base


# ====================================================
# 🔹 질문 후처리
# ====================================================

def validate_question(q: str, lang: str = "KOREAN") -> bool:
    """질문이 자연스러운 어미 규칙을 따르는지 확인"""
    q = q.strip()
    if lang == "KOREAN":
        endings = ["나요?", "습니까?", "할까요?", "있나요?", "있습니까?", "무엇인가요?", "어떻게 생각하나요?"]
        return any(q.endswith(end) for end in endings)
    else:
        return q.endswith("?") and q.lower().split()[0] in [
            "what", "why", "how", "when", "where",
            "do", "does", "did", "can", "could", "would"
        ]

def clean_question(q: str) -> str:
    """불필요한 접두사, 메타설명 제거"""
    q = q.strip()
    q = re.sub(r'^(?:\d+|Q\d+|질문)[:\.\-\s]*', '', q, flags=re.IGNORECASE)  # 번호 제거
    q = re.sub(r'.*에 대한 질문(입니다|입니다\.)', '', q)                   # "에 대한 질문입니다" 제거
    q = q.strip('"“”')
    return q
