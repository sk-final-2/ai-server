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
    "TECHNICAL": (
        "- 선택한 직무의 대표 기술 스택, 알고리즘, 도구에 기반한 질문만 생성할 것\n"
        "- 질문 형식은 다양하게 섞을 것: (설명형, 비교형, 적용형, 대안형)\n"
        "- 단순 정의 암기 질문 대신, 반드시 상황·선택 이유·대안 중 하나 이상 포함\n"
        "- 실무 시뮬레이션 포함 가능: 배포/장애/성능 저하 등 실제 업무 맥락 가정\n"
        "- 직전 답변의 키워드가 있으면 그 부분을 더 깊이 파고드는 후속 질문을 30% 확률로 생성\n"
        "- 금지: 인성/동기/가치관/문화 적합 관련 질문"
    ),
    "PERSONALITY": (
        "- 지원자의 행동, 동기, 협업, 갈등 해결 경험을 사례 중심으로 묻는다 (STAR 구조 유도)\n"
        "- 질문 형식은 다양하게 섞을 것: (경험 설명, 상황 대처, 행동 이유, 결과 반성)\n"
        "- 실무 시뮬레이션 포함 가능: 마감 지연, 협업 갈등, 의사소통 문제 등 실제 팀 상황 가정\n"
        "- 직전 답변에서 언급된 경험/사례를 더 깊이 캐묻는 follow-up 질문을 30% 확률로 생성\n"
        "- 금지: 추상적 가치관, 장래성, 장단점, 포괄적 자기평가"
    ),
    "MIXED": ( 
        "- 기술과 인성 질문을 균형 있게 섞되 동일 유형만 반복하지 말 것."
        "- 기술 질문은 위의 TECHNICAL 규칙을, 인성 질문은 PERSONALITY 규칙을 각각 따를 것."
    )
}

LEVEL_RULE = {
    "하": (
        "- 기본 개념이나 단순 경험 회상 수준 질문\n"
        "- 예시: '최근에 공부한 기술 중 가장 흥미로웠던 것은 무엇인가요?'\n"
        "- 부담을 줄이고, 직무 기본기를 확인하는 질문"
    ),
    "중": (
        "- 실제 상황 적용, 트레이드오프, 선택 이유를 묻는 질문\n"
        "- 반드시 하나 이상의 구체적 사례를 전제로 질문\n"
        "- 예시: '팀 프로젝트에서 데이터베이스 선택 과정에서 어떤 기준을 고려하셨나요?'"
    ),
    "상": (
        "- 심층적, 압박 면접 스타일 가능 (20~30% 확률로 압박/곡선 질문 허용)\n"
        "- 한계·실패·리스크 관리·대안 비교를 묻는다\n"
        "- 예시: '프로젝트 성능이 기대 이하였을 때 본인의 책임은 무엇이라고 생각하시나요?'"
    )
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
        base += ( 
            " 지원자는 신입이다. 따라서 질문은 학습 태도, 새로운 기술/지식 습득 경험, "
            "팀 내 적응력, 협업 방식, 실패 후 극복 사례에 초점을 맞출 것. "
            "추상적 성장 가능성 대신 실제 경험 기반 질문을 생성해야 한다."
            "질문은 실제 기업 면접관이 신입 지원자에게 할 법한, 어렵고 추상적인 표현보다는 간단하고 직관적인 표현으로 생성한다. 예: '향후 목표는 무엇인가요?', '5년 뒤 어떤 모습이 되고 싶나요?'"
        )
        
    elif career in ["경력", "경력직"]:
        base += (
             " 지원자는 경력직이다. 따라서 질문은 실무 성과, 의사결정 과정, "
             "리더십/멘토링 경험, 갈등 해결, 조직 기여와 책임에 초점을 맞출 것."
             "가능하다면 수치나 구체적 지표를 끌어낼 수 있는 질문을 생성해야 한다."
    )

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
