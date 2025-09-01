from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
load_dotenv("src/interview/.env")

# ✅ Whisper 결과 의미 보정 함수
def correct_transcript(raw_text: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 음성 인식 텍스트 교정기다.\n"
         "- 원래 발화자의 말투와 불필요한 추임새(음, 어, 그 등)는 보존한다.\n"
         "- 오타, 띄어쓰기, 문장부호만 교정한다.\n"
         "- 문장의 의미는 절대 바꾸지 않는다.\n"
         "출력: 교정된 한국어 문장 하나만."),
        ("human", "{raw_text}")
    ])


    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )

    chain = prompt | llm
    result = chain.invoke({"raw_text": raw_text}).content.strip()

    return {
        "raw": raw_text,
        "corrected": result
    }