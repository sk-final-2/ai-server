from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os,re

load_dotenv("src/interview/.env")

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.3
)
def postprocess_text(text: str, language: str = "ko") -> str:
    if language == "ko":
        # 1) 데이터셋 관련 오류
        text = re.sub(r"(데이터)\s*세세", r"\1셋", text)

        # 2) 숫자 오류
        text = re.sub(r"\b1개\b", "한계", text)
        text = re.sub(r"\b18\b", "십팔", text)

        # 3) AI/ML 관련
        text = text.replace("에이아이", "AI")
        text = text.replace("엠엘", "ML")
        text = text.replace("딥 러닝", "딥러닝")
        text = text.replace("머신 러닝", "머신러닝")
    return text

def correct_transcript(raw_text: str, language: str = "ko") -> dict:
    if language == "ko":
        system_prompt = (
            "너는 한국어 음성 인식 텍스트 교정기다.\n"
            "- 발화자의 말투와 추임새(음, 어, 그 등)는 보존한다.\n"
            "- 맞춤법, 띄어쓰기, 문장부호를 교정한다.\n"
            "- 숫자 인식 오류(예: '1개' → '한계')를 문맥에 맞게 바로잡는다.\n"
            "- 발음 오류나 잘못 인식된 단어는 문맥을 고려해 의미가 맞도록 교정한다.\n"
            "⚠️ 절대로 원문에 없는 새로운 문장을 추가하지 않는다.\n"
            "⚠️ 원문 문장의 개수와 순서를 유지한다.\n"
        )
    else:
        system_prompt = (
            "You are an English ASR transcript corrector.\n"
            "- Preserve filler words (um, uh, etc.).\n"
            "- Fix typos, spacing, and punctuation.\n"
            "- Do not change words based on possible mispronunciations.\n"
            "⚠️ Do not add new sentences or expand the content.\n"
            "⚠️ Keep the same number of sentences and their order.\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{raw_text}")
    ])

    chain = prompt | llm
    result = chain.invoke({"raw_text": raw_text}).content.strip()

    # ✅ 후처리 규칙 적용
    result = postprocess_text(result, language=language)

    return {
        "raw": raw_text,
        "corrected": result
    }