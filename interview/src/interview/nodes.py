from interview.model import InterviewState
from langchain_core.prompts import ChatPromptTemplate
from interview.chroma_qa import get_similar_qa, save_qa_pair
from langchain_openai import ChatOpenAI
from typing import Union
import os, json
from dotenv import load_dotenv

load_dotenv("src/interview/.env")

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
    "tech": "- 기술적인 깊이를 평가할 수 있는 질문을 포함할 것",
    "behavior": "- 행동 및 가치관을 평가할 수 있는 질문을 포함할 것",
    "mixed": "- 기술과 인성을 모두 평가할 수 있는 질문을 포함할 것"
}
def get_type_rule(state):
    return type_rule_map.get(state.interviewType, "")

def get_Language_rule(lang: str):
    if lang.lower() == "KOREAN":
        return "출력은 반드시 한국어로만 작성하세요."
    elif lang.lower() == "ENGLISH":
        return "Output must be written in English only."
    else:
        return ""
    
# LLM 설정       
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama3-8b-8192",
    temperature=0.7
)
def router_node(state: InterviewState) -> str:
    if not state.answer:
        print("🧭 [router_node] 첫 질문 생성 흐름")
        return "first_question"
    else:
        print("🧭 [router_node] 답변 분석 흐름")
        return "answer"
    
def set_options_node(state: InterviewState) -> InterviewState:
    """🛠 면접 옵션(Language, level, count, interviewType) 확정 노드"""
    if isinstance(state, dict):
        state = InterviewState(**state)

    print("\n======================")
    print("⚙️ [set_options_node] 옵션 설정 시작")
    print(f"입력 Language: {state.Language}, level: {state.level}, count: {state.count}, interviewType: {state.interviewType}")
    print("======================")

    # 기본값 처리 (명세서 값 그대로 사용)
    if not state.Language:
        state.Language = "KOREAN"        # 명세서 기준
    if not state.level:
        state.level = "중"               # 명세서 기준 (상/중/하)
    if state.count is None:
        state.count = 0                  # 0이면 동적 모드
    if not state.interviewType:
        state.interviewType = "MIXED"   # 기본값 (명세서에 맞춰 사용)

    # 잠금
    state.options_locked = True

    print(f"✅ 최종 Language: {state.Language}, level: {state.level}, count: {state.count}, interviewType: {state.interviewType}")
    return state

def build_prompt(state: InterviewState):
    lang_sys = "한국어로 질문하세요." if state.Language == "KOREAN" else "Ask in English."
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
    
def first_question_node(state: InterviewState) -> InterviewState:
    print("✅ state.raw:", state.model_dump())
    """🎯 첫 질문 생성 노드"""
    try:
        # ✅ Pydantic 객체 보장
        if isinstance(state, dict):
            state = InterviewState(**state)

        print("\n======================")
        print("🎯 [first_question_node] 진입")
        print(f"💼 지원 직무: {state.job}")
        print(f"📄 이력서 텍스트 미리보기: {state.text[:100] if state.text else state.resume[:100] if state.resume else '❌ 없음'}")
        print("======================")

        # ✅ 직무 기본값 처리
        if not state.job or state.job == "string":
            print("⚠️ [경고] 직무 정보 누락 → 기본값 '웹 개발자' 적용")
            state.job = "웹 개발자"

        # ✅ 이력서 텍스트 통합
        resume_text = state.text or state.resume or ""
        if not resume_text:
            raise ValueError("❌ 이력서 텍스트가 비어 있음")

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
                당신은 면접관입니다.
                아래 지원자의 자기소개서와 경력 여부를 바탕으로
                면접에서 시작할 첫 번째 질문을 { '한국어' if state.Language.lower() == 'KOREAN' else '영어' }로만 자연스럽게 생성하세요.
                - 질문은 한 문장, 명확하고 구체적으로.
                {get_type_rule(state)}
                {get_Language_rule(state.Language)}

                지원 직무: {{job}}
                경력 여부: {{career}}
                지원자의 자기소개서:
                {{resume}}
                """)   
                ])
        chain = prompt | llm

        # ✅ LLM 실행
        print("🧠 [LLM 요청 시작]")
        response = chain.invoke({"job": state.job, "career": state.career, "resume": resume_text})
        question = response.content.strip() if hasattr(response, "content") else str(response).strip()

        print("📨 [생성된 질문]:", question)
        if not question:
            raise ValueError("❌ 질문 생성 실패 (빈 응답)")

        # ✅ 상태 업데이트
        state.questions.append(question)
        save_qa_pair(question, "")
        state.step += 1
        # ✅ 종료 판단
        if state.count and len(state.questions) >= state.count:
            state.is_finished = True
        elif not state.count and len(state.questions) >= 20:
            state.is_finished = True

        return state

    except Exception as e:
        print("❌ [first_question_node 오류 발생]:", str(e))
        import traceback
        traceback.print_exc()
        raise e
    

def answer_node(state: InterviewState) -> Union[InterviewState, None]:
    """답변 수집 노드 - 사용자 입력을 기다리는 상태"""

    if isinstance(state, dict):
        state_obj = InterviewState(**state)
    else:
        state_obj = state

    print("✍️ [answer_node] 사용자 답변 대기 중...")
    print(f"❓ 현재 질문: {state_obj.questions[-1] if state_obj.questions else 'None'}")
    print(f"📦 [answer_node 리턴 타입]: {type(state_obj)} / 값: {state_obj}")

    # ❗ 답변이 없으면 FSM 종료 (나중에 이어서 실행해야 함)
    if not state_obj.last_answer:
        print("🛑 [answer_node] 답변이 없어 FSM 종료 → 외부 입력 대기")
        return None
     
    question = state_obj.questions[-1] if state_obj.questions else "질문 없음"
    save_qa_pair(question, state_obj.last_answer)


    # ✅ 답변이 있는 경우: 정상 진행
    print("✅ [answer_node] 답변 수신됨 → 다음 단계로")
    state_obj.answer.append(state_obj.last_answer)
    state_obj.step += 1
    return state_obj

def analyze_node(state: InterviewState) -> InterviewState:
    """🧠 답변 분석 노드"""
    try:
        # ✅ Pydantic 모델 보장
        if isinstance(state, dict):
            state = InterviewState(**state)

        print("\n======================")
        print("🔍 [analyze_node] 진입")
        print(f"📝 현재 step: {state.step}")
        print("======================")

        # ✅ 분석할 답변 가져오기
        answer = state.last_answer or (state.answer[-1] if state.answer else "")
        if not answer:
            print("⚠️ [경고] 분석할 답변이 없음")
            state.last_analysis = {"comment": "답변이 없어 분석할 수 없습니다."}
            state.step += 1
            return state

        print("📝 [분석 대상 답변]:", answer[:100], "...")

        # ✅ 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
너는 면접 평가자입니다. 아래의 답변을 분석해서 '잘한 점', '개선이 필요한 점', '점수(0~100)'를 각각 하나씩 도출하세요. 다른 말은 절대 하지말고,
형식은 꼭 다음 JSON 형식으로만 한국어로 출력하세요. 잘한 점이 없어도 작성해주세요.:
{{
  "good": "잘한 점",
  "bad": "개선이 필요한 점",
  "score": 점수숫자
}}
"""),
            ("human", "답변: {answer}")
        ])
        chain = prompt | llm

        # ✅ LLM 분석 요청
        print("🔍 [LLM 요청 시작]")
        response = chain.invoke({"answer": answer})
        print("📨 [LLM 응답 원문]:", response.content)

        # ✅ JSON 파싱 시도
        analysis_json = safe_parse_json_from_llm(response.content)
        if not analysis_json or not isinstance(analysis_json, dict):
            print("❌ [예외 경고] 분석 결과가 None 또는 dict 아님 →", analysis_json)
            analysis_json = {}
        

        # ✅ 상태에 저장
        state.last_analysis = {
            "good": analysis_json.get("good", ""),
            "bad": analysis_json.get("bad", ""),
            "score": analysis_json.get("score", 0)
        }

    except Exception as e:
        print("❌ [analyze_node 오류]:", str(e))
        import traceback
        traceback.print_exc()
        state.last_analysis = {"comment": f"분석 중 오류 발생: {str(e)}"}

    state.step += 1
    return state


def next_question_node(state: InterviewState) -> InterviewState:
    """➡️ 다음 질문 생성 노드 (유사도 필터 포함)"""
    question_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
너는 인공지능 면접관입니다.
지원자가 제출한 자기소개서와 직전에 한 답변을 참고하여
다음에 이어질 면접 질문을 { '한국어' if state.Language.lower() == 'KOREAN' else '영어' }로만 생성하세요.

조건:
- 구체적이고 맥락 있는 질문
- 이전 질문과 유사하지 않음
- 직무 또는 인성 관련 질문 위주
- 너무 포괄적인 질문 지양
- 형식은 질문 문장만 출력
{get_Language_rule(state.Language)}
{get_type_rule(state)}
"""),
    ("human", "{text}")
])
    try:
        if isinstance(state, dict):
            state = InterviewState(**state)

        if len(state.questions) >= state.count:
            state.is_finished = True
            state.step += 1
            print("🏁 질문 종료")
            return state

        previous_answer = state.last_answer or (state.answer[-1] if state.answer else "")
        resume_text = state.text or ""
        print("📝 [LLM 입력 준비] 답변:", previous_answer)
        print("📄 [LLM 입력 준비] 자기소개서 있음 여부:", bool(resume_text))

        next_q = None
        attempt = 0
        max_attempts = 3

        while attempt < max_attempts:
            try:
                type_rule_value = get_type_rule(state)
                result = (question_prompt | llm).invoke({
                    "text": state.text,
                    "answer": state.answer,
                    "type_rule": type_rule_value
                })
                candidate_q = result.content.strip()
                print(f"🧪 [시도 {attempt+1}] 후보 질문:", candidate_q)

                # ✅ 기존 유사 질문 확인
                similar_qas = get_similar_qa(candidate_q, k=1)
                if not similar_qas:
                    print("✅ 유사도 낮음 → 질문 채택")
                    next_q = candidate_q
                    break
                else:
                    print("❌ 유사한 Q/A 존재 → 재시도")
                    attempt += 1

            except Exception as e:
                print("⚠️ 질문 생성 실패:", str(e))
                attempt += 1

        # 🔚 재시도 실패 시 마지막 질문이라도 사용
        if not next_q:
            next_q = candidate_q if 'candidate_q' in locals() else "방금 답변에 대해 좀 더 설명해주시겠어요?"
            print("⚠️ 재시도 실패 → 마지막 질문 사용:", next_q)

        # ✅ 질문 추가 및 상태 갱신
        state.questions.append(next_q)
        print(f"➡️ 질문 {len(state.questions)} 생성 완료: {next_q}")
        if state.count and len(state.questions) >= state.count:
            state.is_finished = True
        elif not state.count and len(state.questions) >= 20:
            state.is_finished = True
        
    except Exception as e:
        print("❌ [next_question_node 예외 발생]:", str(e))
        import traceback
        traceback.print_exc()
        state.is_finished = True

    state.step += 1
    return state