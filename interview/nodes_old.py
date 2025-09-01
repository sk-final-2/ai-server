from interview.model import InterviewState
from langchain_core.prompts import ChatPromptTemplate
from utils.chroma_qa import get_similar_question, save_answer, save_question
from utils.chroma_setup import reset_interview
from langchain_openai import ChatOpenAI
from typing import Union, Literal
import os, json, re
from utils.question_filter import is_redundant
from interview.question_bank import ASPECTS, FALLBACK_POOL
from dotenv import load_dotenv
from interview.predict_keepGoing import keepGoing

load_dotenv("src/interview/.env")

DYN_MIN_SEQ = int(os.getenv("DYN_MIN_SEQ", "3"))     # 최소 N문항은 진행
DYN_HARD_CAP = int(os.getenv("DYN_HARD_CAP", "20"))  # 동적 모드 최대 문항

def _should_stop_dynamic(state: InterviewState) -> bool:
    """count==0인 경우에만 호출: 충분히 평가 완료면 True."""
    seq = int(getattr(state, "seq", 0) or 1)
    
    print(f'♥{seq}')
    # 1) 최대치 넘으면 무조건 종료
    if seq > DYN_HARD_CAP:
        return True
    
    # 2) 최소치 넘으면 LLM한테 물어보기
    if seq >= DYN_MIN_SEQ:
        # last_q = (state.question[-1] if getattr(state, "question", None) else "") or ""
        # ans = state.last_answer or (state.answer[-1] if state.answer else "") or ""
        # la = getattr(state, "last_analysis", {}) or {}
        # good, bad, score = la.get("good", ""), la.get("bad", ""), la.get("score", 0)

        # if getattr(state, "language", "KOREAN") == "ENGLISH":
        #     sys_msg = (
        #         'Decide whether to end the interview now. '
        #         'You must output exactly {{"stop": true}} or {{"stop": false}}, only one of the two. '
        #         'Do not include any other text, explanations, quotes, or comments.'
        #     )
        #     user_msg = (
        #         "last_question: {q}\nlast_answer: {a}\nanalysis.good: {g}\nanalysis.bad: {b}\nscore: {s}\n"
        #         "Return ONLY JSON."
        #     )
        # else:
        #     sys_msg = (
        #         '면접을 지금 종료할지 결정하라. '
        #         '출력은 반드시 정확히 {{"stop": true}} 또는 {{"stop": false}} 둘 중 하나만. '
        #         '그 외 다른 텍스트, 설명, 따옴표, 주석을 절대 쓰지 말라.'
        #     )
        #     user_msg = (
        #         "마지막_질문: {q}\n마지막_답변: {a}\n분석.잘한점: {g}\n분석.개선점: {b}\n점수: {s}\n"
        #         "JSON만 반환."
        #     )

        # print("🔥 sys_msg 원본 =", repr(sys_msg))
        # print("🔥 user_msg 원본 =", repr(user_msg))

        # try:
        #     p = ChatPromptTemplate.from_messages([("system", sys_msg), ("user", user_msg)])
        #     resp = (p | llm.bind(max_tokens=12, temperature=0)).invoke({
        #         "q": last_q[:300], "a": ans[:300], "g": str(good)[:200], "b": str(bad)[:200], "s": score,
        #     })
        #     raw = (getattr(resp, "content", str(resp)) or "").strip()
        #     raw = raw.replace("```json", "").replace("```", "").strip()
        #     data = json.loads(raw) if raw.startswith("{") else {}
        #     return bool(data.get("stop", False))
        # except Exception as e:
        #     print("⚠️ [동적 종료 판단 실패 → 계속 진행]:", e)
        #     return False
        return True  # ⬅️ 임시로 True
    # 3) 최소치 이전이면 무조건 계속 진행
    return False
    
_HANGUL = re.compile(r"[가-힣]")  # 빠른 1차 체크용(간단)

def normalize_language(lang: str | None) -> str:
    if not lang:
        return "KOREAN"
    L = str(lang).strip().upper()
    if L in {"EN", "ENGLISH", "EN-US", "EN-GB"}:
        return "ENGLISH"
    if L in {"KO", "KOREAN", "KOR"}:
        return "KOREAN"
    return "KOREAN"

def system_rule(state) -> str:
    language = getattr(state, "language", "KOREAN")
    interviewType = getattr(state, "interviewType", "MIXED")
    career = getattr(state, "career", "신입")
    level = getattr(state, "level", "중")

    if language == "ENGLISH":
        base = ("You are an interviewer. Use ONLY English."
                " -Do not include any non-English words or characters (no CJK, no transliteration)."
                " -Proper nouns may be used as they are."
                " -Output exactly ONE sentence with no preface, numbering, quotes, or explanations."
                " -Ask a specific question about ONE of: core role competencies, recent work, a project,"
                " or a problem the candidate solved. Do not repeat or closely paraphrase the previous question."
                " -Do NOT evaluate/recap/declare.")

        if interviewType == "PERSONALITY":
            base += " Focus only on behavioral/personality interview questions (values, attitude, teamwork, communication)."
        elif interviewType == "TECHNICAL":
            base += " Focus only on TECHNICALnical competencies, project experience, and problem-solving skills."
        elif interviewType == "MIXED":
            base += " Balance both behavioral/personality and TECHNICALnical questions. Do not repeat the same type consecutively."

        if career == "신입":
            base += " The candidate is entry-level, so focus on learning attitude, growth potential, and adaptability to new environments rather than prior work experience."
        elif career == "경력":
            base += " The candidate is experienced, so focus on concrete achievements, project leadership, collaboration, and problem-solving experience."

        if level == "하":
            base += " Keep the questions simple, focusing on basic knowledge and straightforward experiences."
        elif level == "중":
            base += " Ask questions of medium difficulty that assess the candidate’s ability to apply skills in real projects and handle practical situations."
        elif level == "상":
            base += " Ask in-depth and challenging questions that evaluate advanced problem-solving, strategic thinking, and the ability to analyze complex scenarios."

        return base

    # KOREAN
    base = ("너는 면접관이다. 오직 한국어만 사용한다."
            " -영어, 한자, 일본어, 중국어 등 다른 언어 사용 금지."
            " -고유명사는 그대로 사용할 것."
            " -출력은 정확히 한 문장. 머리말/번호/따옴표/설명 금지."
            " -직무 핵심 역량·최근 업무·프로젝트·문제 해결 중 하나를 구체적으로 묻기."
            " -직전 질문을 반복하거나 비슷하게 바꾸지 말 것."
            " -평가·요약·진술문 금지.")

    if interviewType == "PERSONALITY":
        base += " 인성면접 질문만 하라 (가치관, 태도, 협업, 커뮤니케이션 관련)."
    elif interviewType == "TECHNICAL":
        base += " 기술면접 질문만 하라 (역량, 프로젝트 경험, 문제 해결 관련)."
    elif interviewType == "MIXED":
        base += " 인성과 기술 질문을 균형 있게 섞어서 하라. 동일한 유형만 반복하지 말라."

    if career == "신입":
        base += "지원자는 신입이므로 실무 경험보다는 학습 태도, 성장 가능성, 새로운 환경 적응력에 초점을 맞추라."
    elif career == "경력":
        base += "지원자는 경력직이므로 구체적인 성과, 프로젝트 리더십, 협업 및 문제 해결 경험에 초점을 맞추라."

    if level == "하":
        base += "“질문은 기본 지식과 단순 경험을 확인하는 쉬운 수준으로 하라."
    elif level == "중":
        base += "질문은 실제 프로젝트 적용 가능성이나 상황 대처 능력을 확인할 수 있는 중간 수준으로 하라."
    elif level == "상":
        base += "질문은 고난도 문제 해결, 전략적 사고, 복잡한 상황 분석 능력을 확인할 수 있는 심층적이고 어려운 수준으로 하라."

    return base


def enforce_language_ok(text: str, target: str) -> bool:
    if target == "ENGLISH":
        return not _HANGUL.search(text or "")
    if target == "KOREAN":
        return bool(_HANGUL.search(text or ""))
    return True

# ── 정밀 비율 검증(기존 함수 이름 유지) ──
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
    # 규칙: 기대 언어 비율 >= 0.8 && 금지 문자 비율 <= 0.2
    hangul = _ratio(text, _HANGUL)
    latin  = _ratio(text, _LATIN)
    cjk    = _ratio(text, _CJK)
    jp     = _ratio(text, _JP)
    if target == "KOREAN":
        return (hangul >= 0.80) and ((cjk + jp + latin) <= 0.20)
    else:  # ENGLISH
        return (latin >= 0.80) and ((cjk + jp) <= 0.20)

REASK_PROMPT_KO = (
    "방금 출력은 언어 규칙을 위반했습니다. 오직 한국어로, 질문 문장 1개만 다시 작성하세요. "
    "머리말/번호/따옴표/설명 금지."
)
REASK_PROMPT_EN = (
    "Your previous output violated the language rule. Re-write ONLY ONE question in English. "
    "No preface, numbering, quotes, or explanations."
)

# 분석 재요청 및 정규화(토큰 절약용 초간단 프롬프트)
REASK_ANALYSIS_KO = "언어 규칙 위반입니다. 의미를 유지하고 한국어로 간결하게 다시 작성하세요."
REASK_ANALYSIS_EN = "Language rule violated. Re-write in English only, preserving the meaning."

def normalize_text(llm, text: str, target: Literal["KOREAN", "ENGLISH"]) -> str:
    # 토큰 절약: 짧은 system+user 2줄
    if target == "ENGLISH":
        sys_rule = "Use English only. Keep the meaning. Output body only."
        user = "Rewrite in English only:\n" + (text or "")
    else:
        sys_rule = "오직 한국어만 사용. 의미 유지. 본문만 출력."
        user = "한국어로 다시 쓰기:\n" + (text or "")
    # 짧게 자르기(최대 300자) → 토큰 절약
    user = user[:350]
    try:
        resp = (ChatPromptTemplate.from_messages([("system", sys_rule), ("user", "{u}")]) | llm.bind(max_tokens=60, temperature=0)).invoke({"u": user})
        out = getattr(resp, "content", str(resp)).strip()
        return out
    except Exception:
        return text or ""

# LLM 설정 (토큰 절약: 낮은 temperature, 질문/분석 각각 max_tokens 제한)
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

def safe_parse_json_from_llm(content: str) -> dict:
    print("📨 [LLM 응답 원문]:", content)
    try:
        cleaned = content.strip().replace("```json", "").replace("```", "").strip()
        print("🧼 [클린된 문자열]:", cleaned)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            print("✅ [JSON 파싱 성공]:", parsed)
            return parsed
        else:
            print("❌ [파싱은 됐지만 dict 아님]:", parsed)
            return {}
    except Exception as e:
        print("❌ [JSON 파싱 예외]:", str(e))
        return {}

type_rule_map = {
    "TECHNICALNICAL": "- 기술적인 깊이를 평가할 수 있는 질문을 포함할 것",
    "PERSONALITY": "- 행동 및 가치관을 평가할 수 있는 질문을 포함할 것",
    "MIXED": "- 기술과 인성을 모두 평가할 수 있는 질문을 포함할 것"
}
def get_type_rule(state):
    return type_rule_map.get(state.interviewType, "")

def get_language_rule(lang: str):
    if lang == "KOREAN":
        return "출력은 반드시 한국어로만 작성하세요."
    elif lang == "ENGLISH":
        return "Output must be written in English only."
    else:
        return ""

#---------------------------------------------------------------------------------------------------------------------------------
def check_keepGoing(state: InterviewState) -> str:
    print("🧐 check_keepGoing 진입:", state.keepGoing)
    return "stop" if state.keepGoing is False else "continue"
    
def set_options_node(state: InterviewState) -> InterviewState:
    """🛠 면접 옵션(language, level, count, interviewType) 확정 노드"""
    if isinstance(state, dict):
        state = InterviewState(**state)

    print("\n======================")
    print("⚙️ [set_options_node] 옵션 설정 시작")
    print(f"입력 language: {state.language}, level: {state.level}, count: {state.count}, interviewType: {state.interviewType}")
    print("======================")

    # 기본값 처리 (명세서 값 그대로 사용)
    if not state.language:
        state.language = "KOREAN"
    if not state.level:
        state.level = "중"
    if state.count is None:
        state.count = 0
    if not state.interviewType:
        state.interviewType = "MIXED"
    if getattr(state, "keepGoing", None) is None:
        state.keepGoing = True

    state.options_locked = True
    print(f"✅ 최종 language: {state.language}, level: {state.level}, count: {state.count}, interviewType: {state.interviewType}")
    return state

def keepGoing_node(state: InterviewState) -> Union[InterviewState, None]:
    """count=0일 때 KoELECTRA + LLM 보조로 종료 여부 판단"""
    if isinstance(state, dict):
        state = InterviewState(**state)

    # count>0이면 그냥 통과
    if getattr(state, "count", None) != 0:
        print("➡️ [keepGoing_node] count>0 → 그대로 통과")
        return state

    # ✅ 동적 모드일 때 질문 선택 (임시 저장된 질문 > 기존 질문)
    question = getattr(state, "last_question_for_dynamic", None)
    if not question:
        question = state.question[-1] if getattr(state, "question", None) else ""

    answer = state.last_answer or ""

    try:
        # 1차: KoELECTRA 분류
        label = keepGoing(question, answer)
        print(f"🧩 [KoELECTRA 결과] label={label!r}")
        if label == "terminate":
            print("🔎 [keepGoing_node] KoELECTRA 종료 예측 → LLM 확인")

            # 2차: LLM 보조 확인
            if _should_stop_dynamic(state):
                print("🛑 [keepGoing_node] LLM도 종료 확인 → FSM 종료")
                state.keepGoing = False
                return state
            else:
                print("⚠️ [keepGoing_node] KoELECTRA는 종료 예측했지만, LLM은 계속 진행")
                state.keepGoing = True
                return state
        else:
            print("✅ [keepGoing_node] KoELECTRA 계속 진행 예측")
            state.keepGoing = True
            return state

    except Exception as e:
        print("⚠️ [keepGoing_node 오류] 예외 발생 → 계속 진행:", e)
        state.keepGoing = True
        return state

    finally:
        # ✅ 한 번 쓰고 버리기 → 다음 루프에 영향 안 주도록 제거
        if hasattr(state, "last_question_for_dynamic"):
            delattr(state, "last_question_for_dynamic")

def build_prompt(state: InterviewState):
    lang_sys = "한국어로 질문하세요." if state.language == "KOREAN" else "Ask in English."
    diff_rule = {
        "하": "개념 확인 위주로, 용어를 풀어서 묻고 힌트를 제공하세요.",
        "중": "직무 관련 구체 질문과 간단한 꼬리질문을 포함하세요.",
        "상": "모호성 허용, 반례·트레이드오프, 시스템 설계/깊은 CS 질문을 우선하세요."
    }[state.level]
    system = f"{lang_sys}\n질문 난이도: {state.level}\n규칙: {diff_rule}"
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{context}")
    ])

#---------------------------------------------------------------------------------------------------------------------------------
def first_question_node(state: InterviewState) -> InterviewState:
    print("✅ state.raw:", state.model_dump() if hasattr(state, "model_dump") else state)
    """🎯 첫 질문 생성 노드 (interviewId만 사용)"""
    try:
        if isinstance(state, dict):
            state = InterviewState(**state)

        # --- 입력 정리 ---
        job = (getattr(state, "job", "") or "").strip()
        if not job or job.lower() in {"string", "null"}:
            print("⚠️ [경고] 직무 정보 누락 → 기본값 '웹 개발자' 적용")
            state.job = job = "웹 개발자"

        resume_text = (
            getattr(state, "ocrText", None)
            or getattr(state, "resume", "")
            or ""
        ).strip()
        resume_text = resume_text[:800]  # ⬅ 토큰 절약 (1200→800)

        lang_code = getattr(state, "language", "KOREAN")
        lang = "한국어" if lang_code == "KOREAN" else "영어"

        print("\n======================")
        print("🎯 [first_question_node] 진입")
        print(f"💼 지원 직무: {job}")
        print(f"📄 이력서 텍스트 미리보기: {resume_text[:100] if resume_text else '❌ 없음'}")
        print("======================")

        # --- interviewId ---
        interviewId = getattr(state, "interviewId", None)
        if not interviewId:
            raise ValueError("❌ interviewId가 없습니다. (명세: interviewId)")

        # --- 프롬프트 ---
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "너는 면접관이다. 제공된 입력만 사용해 첫 질문을 만든다. "
             "자기소개서가 비어 있으면 직무/경력만으로 질문을 생성한다. "
             f"출력은 {lang}로 된 정확히 한 문장. 머리말/번호/따옴표/설명 금지. "
             "막연한 '자기소개' 금지, 역할의 핵심 역량·최근 업무·프로젝트·문제 해결 중 하나를 구체적으로 묻기."),
            ("system", system_rule(lang_code)),
            ("system", f"{get_type_rule(state)}"),
            ("user", "job: {job}\ncareer: {career}\nresume: '''{resume}'''"),
        ])

        variables = {
            "job": job,
            "career": getattr(state, "career", None) or "미기재",
            "resume": resume_text,
        }

        # --- LLM 실행(토큰/랜덤성 보수화) ---
        try:
            chain = prompt | llm.bind(max_tokens=200, temperature=0.2, top_p=0.8)
        except AttributeError:
            chain = prompt | llm

        print("🧠 [LLM 요청 시작]")
        response = chain.invoke(variables)
        question = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # 언어 미스매치 보정(정밀 검증)
        if not validate_language_text(question, lang_code):
            strong = "Respond ONLY in English. One sentence only." if lang_code == "ENGLISH" else "오직 한국어로 한 문장만 답하라."
            fix_prompt = ChatPromptTemplate.from_messages([
                ("system", strong),
                ("user", "Rewrite as ONE interview question only (no preface/numbering/quotes): {q}")
            ])
            question = ((fix_prompt | llm.bind(max_tokens=200, temperature=0)).invoke({"q": question}).content).strip()

        # --- 후처리: 한 문장 보장 ---
        if "\n" in question:
            question = question.splitlines()[0].strip()
        if question.count("?") > 1:
            question = question.split("?")[0].strip() + "?"
        if not question:
            question = (
                f"{job} 역할에서 최근 수행한 프로젝트와 본인 기여를 구체적으로 설명해 주세요."
                if lang_code == "KOREAN"
                else f"For the {job} role, describe your most recent project and your specific contribution."
            )

        print("📨 [생성된 질문]:", question)

        # ✅ seq 설정(첫 질문이면 1)
        seq = int(getattr(state, "seq", 0) or 1)
        state.seq = seq

        # (선택) 첫 질문에서만 기존 기록 초기화
        if seq == 1:
            reset_interview(interviewId)

        # ✅ 질문 저장
        save_question(
            interviewId,
            seq,
            question,
            job=getattr(state, "job", None),
            level=getattr(state, "level", None),
            language=getattr(state, "language", None),
        )

        # 상태 업데이트
        if not getattr(state, "question", None):
            state.question = []
        state.question.append(question)
        state.step = (getattr(state, "step", 0) or 0) + 1

        # 종료 판단
        cnt = int(getattr(state, "count", 0) or 0)
        if cnt > 0 and len(state.question) >= cnt:
            state.keepGoing = False

        return state

    except Exception as e:
        print("❌ [first_question_node 오류 발생]:", str(e))
        import traceback; traceback.print_exc()
        raise e
    

def answer_node(state: InterviewState) -> Union[InterviewState, None]:
    """답변 수집 노드 - 사용자 입력을 기다리는 상태"""
    if isinstance(state, dict):
        state_obj = InterviewState(**state)
    else:
        state_obj = state

    print("✍️ [answer_node] 사용자 답변 대기 중...")
    print(f"❓ 현재 질문: {state_obj.question[-1] if state_obj.question else 'None'}")
    print(f"📦 [answer_node 리턴 타입]: {type(state_obj)} / 값: {state_obj}")

    if not state_obj.last_answer:
        print("🛑 [answer_node] 답변이 없어 FSM 종료 → 외부 입력 대기")
        return None
     
    #question = state_obj.question[-1] if state_obj.question else "질문 없음"
    interviewId = getattr(state_obj, "interviewId", None) or getattr(state_obj, "interviewId", None)
    if not interviewId:
        raise ValueError("interviewId 없음(state_obj.interviewId / interviewId 확인)")

    seq = int(getattr(state_obj, "seq", 0) or 1)
    ans_text = (state_obj.last_answer or "").strip()

    save_answer(
        interviewId,
        seq,
        ans_text,
        job=getattr(state_obj, "job", None),
        level=getattr(state_obj, "level", None),
        language=getattr(state_obj, "language", None) or getattr(state_obj, "language", None),
    )

    print("✅ [answer_node] 답변 수신됨 → 다음 단계로")
    state_obj.answer.append(state_obj.last_answer)
    return state_obj

def analyze_node(state: InterviewState) -> InterviewState:
    """🧠 답변 분석 노드"""
    try:
        if isinstance(state, dict):
            state = InterviewState(**state)

        print("\n======================")
        print("🔍 [analyze_node] 진입")
        print("======================")

        answer = state.last_answer or (state.answer[-1] if state.answer else "")
        if not answer:
            print("⚠️ [경고] 분석할 답변이 없음")
            state.last_analysis = {"comment": "답변이 없어 분석할 수 없습니다."}
            return state

        print("📝 [분석 대상 답변]:", answer[:100], "...")

        # 언어별 시스템 지시 (토큰 절약, JSON 고정)
        if getattr(state, "language", "KOREAN") == "ENGLISH":
            sys_msg = (
                "You are an interview evaluator. Analyze the answer and produce exactly one 'good', one 'bad', and a 'score(0-100)'. "
                "Respond in English only and output ONLY this JSON:\n"
                "Do not include any non-English words or characters (no CJK, no transliteration)."
                "{{\n\"good\": \"what was good\",\n\"bad\": \"what needs improvement\",\n\"score\": number\n}}"
            )
        else:
            sys_msg = (
                "너는 면접 평가자다. 아래 답변을 분석해 '잘한 점', '개선이 필요한 점', '점수(0~100)'를 각각 하나씩 도출하라. "
                "한국어로만 답하고, 다음 JSON으로만 출력하라:\n"
                " -영어, 한자, 일본어, 중국어 등 다른 언어 사용 금지."
                " -고유명사는 그대로 사용할 것."
                "{{\n\"good\": \"잘한 점\",\n\"bad\": \"개선이 필요한 점\",\n\"score\": 점수숫자\n}}"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            ("human", "답변: {answer}")
        ])

        try:
            chain = prompt | llm.bind(max_tokens=250, temperature=0.2, top_p=0.8)
        except AttributeError:
            chain = prompt | llm

        print("🔍 [LLM 요청 시작]")
        response = chain.invoke({"answer": answer})
        content = response.content if hasattr(response, "content") else str(response)
        print("📨 [LLM 응답 원문]:", content)

        analysis_json = safe_parse_json_from_llm(content)
        if not isinstance(analysis_json, dict):
            analysis_json = {}

        # ✅ 상태에 저장
        state.last_analysis = {
            "good": analysis_json.get("good", ""),
            "bad": analysis_json.get("bad", ""),
            "score": analysis_json.get("score", 0)
        }

        # ✅ 언어 검증 & 필요 시 정규화(짧은 호출, 토큰 절약)
        tgt = "ENGLISH" if getattr(state, "language", "KOREAN") == "ENGLISH" else "KOREAN"
        for k in ("good", "bad"):
            v = state.last_analysis.get(k, "") or ""
            if v and not validate_language_text(v, tgt):
                state.last_analysis[k] = normalize_text(llm, v, tgt)

    except Exception as e:
        print("❌ [analyze_node 오류]:", str(e))
        import traceback
        traceback.print_exc()
        state.last_analysis = {"comment": f"분석 중 오류 발생: {str(e)}"}
    return state


def next_question_node(state: InterviewState) -> InterviewState:
    """➡️ 다음 질문 생성 노드 (측면 전환 + 다중 기준 중복 차단 + 안전 폴백)"""
    if isinstance(state, dict):
        state = InterviewState(**state)

    try:
        # 종료 조건
        if getattr(state, "count", None) and len(state.question) >= state.count:
            state.keepGoing = False
            print("🏁 질문 종료 (count 상한 도달)")
            state.step = getattr(state, "step", 0) + 1
            return state

        job = (getattr(state, "job", "") or "").strip() or "웹 개발자"
        lang_code = getattr(state, "language", "KOREAN")
        lang = "한국어" if lang_code == "KOREAN" else "영어"
        prev_q = state.question[-1] if state.question else ""

        interviewId = getattr(state, "interviewId", None)
        if not interviewId:
            raise ValueError("interviewId가 없습니다. (명세: interviewId)")

        aspect_idx = getattr(state, "aspect_index", 0) or 0
        aspect = ASPECTS[aspect_idx % len(ASPECTS)]
        print(f"🎛️ 대상 측면(aspect): {aspect} (index={aspect_idx})")

        system_prompt = (
            "너는 면접관이다. 제공된 정보만 사용해 다음 질문을 만든다. "
            f" 출력은 {lang}로 된 정확히 한 문장. 머리말/번호/따옴표/설명 금지. "
            "무조건 상대방이 답변을 할 수 있는 형태로 질문을 생성하라"
            "직전 질문과 의미가 거의 같은 질문 금지. "
            "반드시 지정된 측면에 대한 새로운 각도의 질문을 생성하라."
        )

        question_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "직무: {job}\n측면(aspect): {aspect}\n직전 질문: {prev_q}\n직전 답변(요약 가능): {prev_a}")
        ])

        previous_answer = (state.last_answer or (state.answer[-1] if state.answer else "")) or ""
        next_q = None
        for attempt in range(3):
            print(f"🧠 [LLM 요청] 시도 {attempt+1}/3")
            chain = question_prompt | llm
            res = chain.invoke({"job": job, "aspect": aspect, "prev_q": prev_q, "prev_a": previous_answer})
            candidate_q = (res.content or "").strip()
            if not candidate_q.endswith((".", "?", "!")):
                candidate_q += "?"

            print(f"🧪 [시도 {attempt+1}] 후보 질문: {candidate_q}")

            try:
                from utils.chroma_qa import get_similar_question, save_question
                check = get_similar_question(
                    interviewId=interviewId,
                    question=candidate_q,
                    k=5,
                    min_similarity=0.88,
                    verify_all=True,
                )
                embed_sim = 0.0
                if check.get("hits"):
                    embed_sim = max(h.get("sim", 0.0) for h in check["hits"] if h.get("text"))
                redundant = is_redundant(prev_q or "", candidate_q, embed_sim,
                                         cos_thr=0.95, jac_thr=0.60, ngram_thr=0.50) if prev_q else False

                if not (check.get("similar") or redundant):
                    next_q = candidate_q
                    break
                else:
                    sims = ", ".join(f"{h['sim']:.3f}" for h in (check.get("hits") or [])[:3] if 'sim' in h)
                    print(f"❌ 유사 질문 존재 (embed_sim≈{embed_sim:.3f}) | 보조중복={redundant} | knn top3: {sims}")
            except Exception as e:
                print("⚠️ 유사도 체크 오류 → 후보 채택(보수):", e)
                next_q = candidate_q
                break

        if not next_q:
            state.dup_streak = getattr(state, "dup_streak", 0) + 1
            state.aspect_index = (aspect_idx + 1) % len(ASPECTS)
            new_aspect = ASPECTS[state.aspect_index]
            from random import choice
            fb_list = FALLBACK_POOL.get(new_aspect, [])
            fb = choice(fb_list) if fb_list else "최근에 맡은 업무 중 본인이 주도적으로 개선한 한 가지를 간단히 설명해 주세요."
            print(f"⚠️ 재시도 실패 → 폴백({new_aspect}) 사용")
            next_q = fb
        else:
            state.dup_streak = 0
            state.aspect_index = (aspect_idx + 1) % len(ASPECTS)

        from utils.chroma_qa import save_question
        save_question(interviewId, len(state.question)+1, next_q,
                      job=job, level=getattr(state, "level", None), language=lang_code)
        state.question.append(next_q)
        state.seq = getattr(state, "seq", 0) + 1
        print(f"➡️ 질문 {len(state.question)} 생성 완료: {next_q}")

        if getattr(state, "count", None) and len(state.question) >= state.count:
            state.keepGoing = False
        elif not getattr(state, "count", None) and len(state.question) >= 20:
            state.keepGoing = False

    except Exception as e:
        print("❌ [next_question_node 예외 발생]:", str(e))
        import traceback; traceback.print_exc()
        state.keepGoing = False

    state.step = getattr(state, "step", 0) + 1
    return state
