from interview.model import InterviewState
from interview.chroma_qa import get_similar_qa, save_qa_pair
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import json
import re
from dotenv import load_dotenv

# .env 로드
load_dotenv(dotenv_path="src/interview/.env")

# LLM 초기화
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama3-8b-8192",
    temperature=0.7
)

# JSON 응답 안전 파싱
def safe_parse_json_from_llm(content: str) -> dict:
    print("📨 [LLM 응답 원문 - 다시 확인]:", content)
    cleaned = content.strip().replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(cleaned)
        print("✅ JSON 파싱 성공:", result)
        return result
    except json.JSONDecodeError as e:
        print(f"[!] JSON 파싱 실패: {e}")
        match = re.search(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', cleaned, re.DOTALL)
        if match:
            json_block = match.group(1)
            try:
                result = json.loads(json_block)
                print("✅ 2차 파싱 성공:", result)
                return result
            except json.JSONDecodeError as e2:
                print(f"[!] 2차 파싱 실패: {e2}")

    print("[!] Fallback 사용 (빈 딕셔너리)")
    return {}

# 프롬프트 정의
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
너는 면접관 AI야. 아래 사용자의 답변을 평가한 뒤, 아래 JSON 형식만 응답해.  
**절대 설명하거나 JSON 외 텍스트를 추가하지 마!**

반드시 이 형식을 따라:

```json
{
  "good": "긍정 피드백 한 줄",
  "bad": "부정 피드백 한 줄",
  "score": 정수(0~100)
}

""")
])

next_question_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 면접관입니다. 지원자의 이전 답변과 자기소개서를 바탕으로 꼬리질문을 1개 생성하세요.
형식은 자연스러운 한국어 문장 하나로 출력하고, 질문 형식으로 끝내세요.
예: "그 경험에서 가장 힘들었던 점은 무엇인가요?"
"""),
    ("human", "답변: {answer}\n자기소개서: {resume}")
])

first_question_prompt = ChatPromptTemplate.from_messages([
    ("system", """
당신은 면접관입니다. 아래 지원자의 자기소개서를 바탕으로
면접에서 시작할 첫 번째 질문을 한국어로만 자연스럽게 생성하세요.

- 너무 광범위하지 않게, 한 문장으로 질문 형태로 끝내세요.
- 예시: "자기소개서에 나온 프로젝트 중 가장 기억에 남는 경험은 무엇인가요?"

지원자의 자기소개서:
{resume}
""")
])
def first_question_node(state: InterviewState) -> InterviewState:
    print("🎯 [first_question_node] 첫 질문 생성")

    try:
        prompt = first_question_prompt.format_messages(resume=state.text)
        response = llm.invoke(prompt)
        question = response.content.strip()

        if not question or len(question) < 5:
            raise ValueError("질문 응답이 비정상적으로 짧음")

        print(f"✅ [첫 질문]: {question}")

    except Exception as e:
        print(f"❌ [첫 질문 생성 실패]: {e}")
        question = "자기소개서에 나온 프로젝트 중 기억에 남는 경험은 무엇인가요?"

    state.questions.append(question)
    state.step += 1
    return state

def answer_node(state: InterviewState) -> InterviewState:
    print("🗣️ [answer_node] 답변 수집 완료")
    if not state.answers:
        print("⚠️ [경고] answers가 비어 있음")
    state.step += 1
    return state

def analyze_node(state: InterviewState) -> InterviewState:
    print("🧠 [analyze_node] 답변 분석")
    answer = state.answers[-1] if state.answers else ""

    try:
        prompt = analysis_prompt.format_messages(answer=answer)
        response = llm.invoke(prompt)
        print("🧠 [LLM 응답]:", response.content)
        raw = response.content
        parsed = safe_parse_json_from_llm(raw)
    except Exception as e:
        print(f"💥 [예외 발생]: {e}")
        parsed = {
            "good": "분석 실패",
            "bad": "분석 실패",
            "score": 0,
            "feedback": "분석 실패"
        }

    state.last_analysis = parsed
    state.seq += 1
    state.step += 1
    return state

def next_question_node(state: InterviewState) -> InterviewState:
    print("➡️ [next_question_node] 다음 질문 생성")

    if len(state.questions) >= 3:
        state.is_finished = True
        return state

    answer = state.answers[-1] if state.answers else ""

    if not answer or answer.strip() == "":
        followup_q = "이전에 말씀하신 내용을 좀 더 구체적으로 설명해주시겠어요?"
    else:
        try:
            prompt = next_question_prompt.format_messages(answer=answer, resume=state.text)
            response = llm.invoke(prompt)
            followup_q = response.content.strip()
            if not followup_q:
                raise ValueError("빈 응답")
        except Exception as e:
            print(f"❌ [다음 질문 생성 실패]: {e}")
            followup_q = "이전에 말씀하신 내용을 좀 더 구체적으로 설명해주시겠어요?"

    # 🔹 질문 저장
    save_qa_pair(followup_q, answer)

    state.questions.append(followup_q)
    state.step += 1
    return state

def end_node(state: InterviewState) -> InterviewState:
    print("🏁 [end_node] 면접 종료")
    return state