from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
load_dotenv(dotenv_path="src/stt/.env")

# ✅ Whisper 결과 의미 보정 함수
def correct_transcript(raw_text: str) -> str:
    prompt = ChatPromptTemplate.from_template("""
당신은 AI 문장 교정기입니다.  
다음 규칙을 반드시 지키세요:

- 문장에서 잘못 인식된 단어(예: 이래보고 → 일해보고)를 실제 의미에 맞게 수정해주세요.
- 오타를 문맥에 맞게 고쳐주세요.
- 단순한 표현 수정이나 의역은 하지 마세요.
- 출력은 반드시 한국어 문장 하나만 포함해주세요. 설명이나 다른 말은 절대 하지 마세요.

아래 예시를 참고하여 동작하세요:

입력: 이래보고 싶어서 매니저직에 지원했습니다.  
출력: 일해보고 싶어서 매니저직에 지원했습니다.

입력: 대학교에서 컴퓨러공학을 전공했습니다.  
출력: 대학교에서 컴퓨터공학을 전공했습니다.

입력: 편이팍을 위해 노력해왔습니다.  
출력: 편입학을 위해 노력해왔습니다.

입력: 고객니 욕구를 파악하고 대응했습니다.  
출력: 고객의 욕구를 파악하고 대응했습니다.

입력: 프론트앤드랑 백핸드 개발을 했습니다.  
출력: 프론트엔드랑 백엔드 개발을 했습니다.

입력: 저는 에이아이 관련 프로젝트를 수행했습니다.  
출력: 저는 AI 관련 프로젝트를 수행했습니다.

입력: 저희 팀은 협옹을 잘했습니다.  
출력: 저희 팀은 협업을 잘했습니다.

입력: 문제를 해결기 위해 노력했습니다.  
출력: 문제를 해결하기 위해 노력했습니다.

입력: 디지털 포맷을 벋어나  
출력: 디지털 포맷을 벗어나

입력: 고겍에게 더 좋은 가치를 주기 위해서  
출력: 고객에게 더 좋은 가치를 주기 위해서

입력: 그때 경험을 통해 저는 많이 배우게되  
출력: 그때 경험을 통해 저는 많이 배우게 돼

입력: 그래서 저느 노력했습니다.  
출력: 그래서 저는 노력했습니다.

입력: 프로젝트는 성공적으 로 마무리되었습니다.  
출력: 프로젝트는 성공적으로 마무리되었습니다.

입력: 리더로서 책임감이 강하다고 생각합니다  
출력: 리더로서 책임감이 강하다고 생각합니다.

입력: 데이터를 분석하고 모델을 훈룐시켰습니다.  
출력: 데이터를 분석하고 모델을 훈련시켰습니다.

이제 아래 문장을 같은 방식으로 수정하세요:

입력: "{raw}"  
출력:
""")

    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama3-8b-8192",
        temperature=0.3
    )

    chain = prompt | llm
    result = chain.invoke({"raw": raw_text})
    return result.content.strip()