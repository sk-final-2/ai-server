from interview.model import InterviewState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import json
from dotenv import load_dotenv

load_dotenv(dotenv_path="src/interview/.env")

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama3-8b-8192",
    temperature=0.7
)

def safe_parse_json_from_llm(content: str) -> dict:
    print("📨 [LLM 응답 원문 - 다시 확인]:", content)

    # JSON 포맷을 감싸는 불필요한 구문 제거
    cleaned = content.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("❌ JSON 디코딩 실패:", e)
        raise
    
def first_question_node(state: InterviewState) -> InterviewState:
    print("🎯 [first_question_node] 첫 질문 생성")
    print("📄 [이력서 텍스트]:", state.text[:200], "...")  # 너무 길면 잘라서 출력
    print("💼 [지원 직무]:", state.job)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 면접관입니다. 아래 지원자의 자기소개서를 바탕으로
면접에서 시작할 첫 번째 질문을 한국어로만 자연스럽게 생성하세요.

- 너무 광범위하지 않게, 한 문장으로 질문 형태로 끝내세요.
- 예시: "자기소개서에 나온 프로젝트 중 가장 기억에 남는 경험은 무엇인가요?"

지원자의 자기소개서:
{resume}
""")
    ])

    chain = prompt | llm

    last_answer = state.answers[-1] if state.answers else ""

    try:
        response = chain.invoke({
            "resume": state.text,
            "job": state.job,
            "answer": last_answer
        })
        print("🧠 [LLM 응답]:", response)
        question = response.content
        if not question:
            print("❗ [경고] 질문 생성 실패: 빈 응답")
            raise ValueError("LLM 응답이 비어 있습니다.")

    except Exception as e:
        print("❌ [LLM 호출 실패]:", str(e))
        raise e  # 이걸 반드시 다시 던져야 FastAPI에서 500으로 기록됨

    state.questions.append(question)
    return state 

def answer_node(state: InterviewState) -> InterviewState:
    print("🗣️ [answer_node] 답변 수집 완료")
    if state.last_answer:
        state.answers.append(state.last_answer)
    else:
        print("⚠️ last_answer가 비어 있음")
    state.step += 1
    return state

def analyze_node(state: InterviewState) -> InterviewState:
    print("🔍 [analyze_node] 답변 분석")
    answer = state.answers[-1] if state.answers else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
너는 면접 평가자입니다. 아래의 답변을 분석해서 '잘한 점', '개선이 필요한 점', '점수(0~100)'를 각각 하나씩 도출하세요.
형식은 다음 JSON 형식으로 출력하세요.

{{
  "good": "잘한 점",
  "bad": "개선이 필요한 점",
  "score": 숫자
}}
"""),
        ("human", "{answer}")
    ])

    chain = prompt | llm
    response = chain.invoke({"answer": answer}).content
    print("📨 [LLM 응답]:", response)

    try:
        # JSON 형식으로 안전하게 파싱
        analysis = safe_parse_json_from_llm(response)

        # 개별 필드에 직접 저장
        state.interview_answer_good = analysis.get("good", "")
        state.interview_answer_bad = analysis.get("bad", "")
        state.score = analysis.get("score", 0)

        # 전체 분석 결과도 저장 (선택)
        state.last_analysis = analysis

    except Exception as e:
        print("❌ JSON 파싱 실패:", e)
        # 예외 발생 시 기본값 설정
        state.interview_answer_good = ""
        state.interview_answer_bad = ""
        state.score = 0
        state.last_analysis = {"error": str(e), "raw": response}

    return state

def next_question_node(state: InterviewState) -> InterviewState:
    print("➡️ [next_question_node] 다음 질문 생성")
    if len(state.questions) >= 3:
        state.is_finished = True
    else:
        last_answer = state.answers[-1] if state.answers else ""
        prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 면접관입니다. 지원자의 이전 답변과 자기소개서를 바탕으로 꼬리질문을 1개 생성하세요.
형식은 자연스러운 한국어 문장 하나로 출력하고, 질문 형식으로 끝내세요.
예: "그 경험에서 가장 힘들었던 점은 무엇인가요?"
"""),
    ("human", "답변: {answer}\n자기소개서: {resume}")
])
        chain = prompt | llm
        question = chain.invoke({"answer": last_answer, "resume": state.resume}).content
        state.questions.append(question)
    state.step += 1
    return state

def end_node(state: InterviewState) -> InterviewState:
    print("🏁 [end_node] 면접 종료")
    return state