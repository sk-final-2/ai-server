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