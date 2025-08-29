from interview.model import InterviewState
from typing import Union
from utils.chroma_qa import save_answer
from langchain_core.prompts import ChatPromptTemplate
from interview.node.rules import safe_parse_json_from_llm, validate_language_text, normalize_text
from interview.config import llm

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
    print("✅ [node_name] state.question type:", type(state.question), "value:", state.question)
    return state_obj
#----------------------------------------------------------------------------------------------------------------------------------
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
            analysis_result = {"comment": "답변이 없어 분석할 수 없습니다."}
            return state

        print("📝 [분석 대상 답변]:", answer[:100], "...")

        # 언어별 시스템 지시 (토큰 절약, JSON 고정)
        if getattr(state, "language", "KOREAN") == "ENGLISH":
            sys_msg = (
                "You are an interview evaluator. Analyze the answer and produce exactly one 'good', one 'bad', and a 'score(0-100)'. "
                "Respond in English only and output ONLY this JSON:\n"
                "Use ONLY these three keys: 'good', 'bad', 'score'."
                " Do NOT include any explanation, text, or formatting outside the JSON."
                "Do not include any non-English words or characters (no CJK, no transliteration)."
                "{{\n\"good\": \"what was good\",\n\"bad\": \"what needs improvement\",\n\"score\": number\n}}"
            )
        else:
            sys_msg = (
                "너는 면접 평가자다. 아래 답변을 분석해 '잘한 점', '개선이 필요한 점', '점수(0~100)'를 각각 하나씩 도출하라. "
                "한국어로만 답하고, 다음 JSON으로만 출력하라:\n"
                "key는 'good', 'bad', 'score' 세 개만 사용한다."
                "JSON 이외의 설명, 텍스트, 포맷을 출력하지 말라."
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
        analysis_result = {
            "good": analysis_json.get("good", ""),
            "bad": analysis_json.get("bad", ""),
            "score": analysis_json.get("score", 0)
        }

        # ✅ 언어 검증 & 필요 시 정규화(짧은 호출, 토큰 절약)
        tgt = "ENGLISH" if getattr(state, "language", "KOREAN") == "ENGLISH" else "KOREAN"
        for k in ("good", "bad"):
            v = analysis_result.get(k, "") or ""
            if v and not validate_language_text(v, tgt):
                analysis_result[k] = normalize_text(llm, v, tgt)

    except Exception as e:
        print("❌ [analyze_node 오류]:", str(e))
        import traceback
        traceback.print_exc()
        analysis_result = {"comment": f"분석 중 오류 발생: {str(e)}"}
    state.last_analysis = analysis_result
    print("✅ [node_name] state.question type:", type(state.question), "value:", state.question)
    return state
