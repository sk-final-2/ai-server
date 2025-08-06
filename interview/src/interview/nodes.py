from interview.model import InterviewState
from langchain_core.prompts import ChatPromptTemplate
from interview.chroma_qa import get_similar_qa
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv("src/interview/.env")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ✅ 모든 노드 함수는 dict를 받고 dict를 반환하게 변경
def first_question_node(state: dict) -> dict:
    state = InterviewState(**state)
    print("🎯 [first_question_node] 첫 질문 생성")
    print("📄 [이력서 텍스트]:", state.text[:200], "...")
    print("💼 [지원 직무]:", state.job)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 면접관입니다. 아래 지원자의 자기소개서를 바탕으로
면접에서 시작할 첫 번째 질문을 한국어로만 자연스럽게 생성하세요.
- 너무 광범위하지 않게, 한 문장으로 질문 형태로 끝내세요.
- 예시: \"자기소개서에 나온 프로젝트 중 가장 기억에 남는 경험은 무엇인가요?\"
지원자의 자기소개서:
{resume}
""")
    ])

    chain = prompt | llm
    response = chain.invoke({"resume": state.text})

    question = response.content.strip() if hasattr(response, "content") else str(response).strip()

    if not question:
        raise ValueError("질문 생성 실패")

    state.questions.append(question)
    state.step += 1
    return state.model_dump()

def answer_node(state: dict) -> dict:
    state = InterviewState(**state)
    print("✍️ [answer_node] 사용자 답변 수집")
    state.step += 1
    return state.model_dump()

def analyze_node(state: dict) -> dict:
    state = InterviewState(**state)
    print("🔍 [analyze_node] 답변 분석")
    answer = state.answers[-1] if state.answers else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
너는 면접 평가자입니다. 아래의 답변을 분석해서 '잘한 점', '개선이 필요한 점', '점수(0~100)'를 각각 하나씩 도출하세요.
형식은 다음 JSON 형식으로 출력하세요.
{
  "good": "잘한 점",
  "bad": "개선이 필요한 점",
  "score": 숫자
}
"""),
        ("human", "{answer}")
    ])

    chain = prompt | llm
    analysis = chain.invoke({"answer": answer}).content
    state.last_analysis = {"comment": analysis}
    state.step += 1
    return state.model_dump()

def next_question_node(state: dict) -> dict:
    state = InterviewState(**state)
    print("➡️ [next_question_node] 다음 질문 생성")

    if len(state.questions) >= 3:
        state.is_finished = True
    else:
        # 유사 질문 기반 추론 (선택)
        previous_answer = state.answers[-1] if state.answers else ""
        next_q = get_similar_qa(previous_answer)
        state.questions.append(next_q)

    state.step += 1
    return state.model_dump()