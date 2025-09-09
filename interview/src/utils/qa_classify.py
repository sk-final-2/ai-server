# utils/qa_classify.py
import re, json
from langchain_core.prompts import ChatPromptTemplate

# === 언어 선택 (state.language: "ko" | "en" | "" 등) =========================
def _lang(state_lang: str) -> str:
    l = (state_lang or "").lower()
    if l.startswith("ko") or l in ("kr", "kor", "korean", "ko-kr"):
        return "ko"
    return "en"  # 기본 en

# === 프롬프트 (KO/EN) =======================================================
PROMPT_KO = """너는 면접관 보조다. 아래 Q/A와 최근 맥락을 보고,
이번 턴에서 다뤄야 할 질문 성격을 판정하라.

[역할]
- TECHNICAL: 도구/표준/지표/절차/설계/원인-해결을 파고드는 기술 중심
- PERSONALITY: 협업/의사소통/책임/갈등/동기/가치/동료와의 상호작용 같은 인성 중심

[서브타입 가이드]
- METHOD | TOOL_STANDARD | IMPACT | STAKEHOLDER | RISK_ETHICS

[입력]
- last_question: {last_q}
- last_answer: {last_a}
- topic_hint: {topic}
- current_type: {cur_type}
- recent_n_turns: {recent_text}

[출력(JSON)]
{{
  "prob_tech": 0.0~1.0,
  "prob_person": 0.0~1.0,
  "subtype": "<METHOD|TOOL_STANDARD|IMPACT|STAKEHOLDER|RISK_ETHICS>",
  "terminate": false,
  "confidence": 0.0~1.0,
  "rationale": "<짧은 이유>"
}}
- 확률은 합이 1에 가깝게.
- subtype은 위 5개 중 가장 적합한 하나.
"""

PROMPT_EN = """You are an interview co-pilot. Based on the Q/A and recent context,
classify what the next question should focus on.

[Types]
- TECHNICAL: technical depth on tools/standards/metrics/process/design/root-cause & fix
- PERSONALITY: soft factors like collaboration/communication/ownership/conflict/motivation/values

[Subtypes]
- METHOD | TOOL_STANDARD | IMPACT | STAKEHOLDER | RISK_ETHICS

[Input]
- last_question: {last_q}
- last_answer: {last_a}
- topic_hint: {topic}
- current_type: {cur_type}
- recent_n_turns: {recent_text}

[Output JSON]
{{
  "prob_tech": 0.0~1.0,
  "prob_person": 0.0~1.0,
  "subtype": "<METHOD|TOOL_STANDARD|IMPACT|STAKEHOLDER|RISK_ETHICS>",
  "terminate": false,
  "confidence": 0.0~1.0,
  "rationale": "<brief reason>"
}}
- Probabilities should sum close to 1.
- Pick exactly one subtype.
"""

# === 한/영 휴리스틱 =========================================================
# 숫자/지표/단위
_METRIC = re.compile(
    r"\b(\d+(\.\d+)?\s?%|\d+(\.\d+)?\s?(ms|s|sec|min|hour|hr|hrs|시간|일|주|월)|KPI|metric|지표|accuracy|precision|recall|F1|RMSE|MAE|latency|cost|원|만원|억|usd|\$)\b",
    re.I,
)
# 표준/도구/프레임워크
_STD = re.compile(
    r"\b(ISO|IEC|IEEE|GDPR|HIPAA|FDA|CE|UL|KS|standard|spec|규격|법규|지침|framework|library|tool|툴|도구)\b",
    re.I,
)
# 절차/프로세스/설계
_PROC = re.compile(
    r"(설계|분석|정의|측정|검증|배포|릴리즈|테스트|모니터링|개선|표준화|원인|해결|design|analysis|define|measure|verify|deploy|release|test|monitor|improve|root cause)",
    re.I,
)
# 협업/이해관계자
_STAKE = re.compile(
    r"(고객|사용자|협력사|부서|팀원|의사소통|조율|갈등|피드백|stakeholder|customer|user|client|team|cross[- ]?functional|communication|alignment|conflict|feedback)",
    re.I,
)
# 리스크/윤리/보안
_RISK = re.compile(
    r"(리스크|위험|사고|안전|품질|윤리|컴플라이언스|보안|개인정보|중단|장애|risk|hazard|safety|quality|ethic|compliance|security|privacy|outage|incident)",
    re.I,
)

def heuristic_scores(text: str) -> dict:
    text = text or ""
    tech_hits = int(bool(_METRIC.search(text))) + int(bool(_STD.search(text))) + int(bool(_PROC.search(text)))
    person_hits = int(bool(_STAKE.search(text))) + int(bool(_RISK.search(text)))
    total = max(1, tech_hits + person_hits)
    return {
        "prob_tech_h": tech_hits / total,
        "prob_person_h": person_hits / total,
        "subtype_h": (
            "IMPACT" if _METRIC.search(text) else
            "TOOL_STANDARD" if _STD.search(text) else
            "STAKEHOLDER" if _STAKE.search(text) else
            "RISK_ETHICS" if _RISK.search(text) else
            "METHOD"
        )
    }

# === LLM 분류 호출 ==========================================================
def classify_turn_with_llm(llm, state_language: str, last_q: str, last_a: str, topic: str="", cur_type: str="", recent_text: str="") -> dict:
    lang = _lang(state_language)
    tmpl = PROMPT_KO if lang == "ko" else PROMPT_EN
    prompt = ChatPromptTemplate.from_template(tmpl).format(
        last_q=last_q or "",
        last_a=last_a or "",
        topic=topic or "",
        cur_type=cur_type or "",
        recent_text=recent_text or ""
    )
    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", str(resp))
    try:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end+1])
    except Exception:
        data = {"prob_tech": 0.5, "prob_person": 0.5, "subtype": "METHOD", "terminate": False, "confidence": 0.2, "rationale": "fallback"}
    return data

# === 보정 + 결정 ============================================================
THETA_UP   = 0.62
THETA_DOWN = 0.38
COOLDOWN_TURNS = 2

def can_bridge(state, asked, max_q):
    cutoff = max_q // 2
    if asked < cutoff:
        return False, "before_cutoff"
    last = getattr(state, "last_bridge_turn", None)
    cur  = getattr(state, "seq", None)
    if last is not None and cur is not None and (cur - last) < COOLDOWN_TURNS:
        return False, "cooldown"
    if getattr(state, "terminate", False):
        return False, "terminate_pending"
    if getattr(state, "bridge_done", False):
        return False, "already_bridged_topic"
    return True, "ok"

def decide_next_type(state_language: str, prev_type: str, llm_result: dict, h_result: dict) -> dict:
    # prev_type 정규화
    _map = {"TECH": "TECHNICAL", "PERSON": "PERSONALITY", None: "PERSONALITY", "": "PERSONALITY"}
    prev_type = _map.get(prev_type, prev_type) or "PERSONALITY"

    # 가중 평균 보정
    w_llm, w_h = 0.6, 0.4
    ptech = w_llm*float(llm_result.get("prob_tech", 0.5)) + w_h*float(h_result["prob_tech_h"])
    pper  = w_llm*float(llm_result.get("prob_person", 0.5)) + w_h*float(h_result["prob_person_h"])

    next_type = prev_type
    if ptech >= THETA_UP:
        next_type = "TECHNICAL"
    elif pper >= THETA_UP:
        next_type = "PERSONALITY"
    # 중간대는 유지

    switched = (next_type != prev_type)
    # ✅ note는 항상 빈 문자열
    note = ""

    subtype = llm_result.get("subtype") or h_result["subtype_h"]

    return {
        "prob_tech": round(ptech, 3),
        "prob_person": round(pper, 3),
        "next_type": next_type,
        "subtype": subtype,
        "switched": switched,
        "bridge_note": note
    }