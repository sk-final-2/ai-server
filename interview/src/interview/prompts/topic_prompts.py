def _personality_prompt(resume_text: str, language: str, topic: str, desc: str, aspect: str = "") -> str:
    if language == "KOREAN":
        return f"""
        너는 면접관이다.
        다음 자기소개서를 바탕으로 지원자의 인성 관련 주제를 3~5개 뽑아라.
        - 초점: 태도, 협업, 의사소통, 동기, 문제 대처 방식
        - 기술적 키워드는 절대 사용하지 말 것
        - 각 항목은 반드시 "key"와 "desc" 포함
        - "desc"는 구체적인 설명이어야 하며 절대 비워두지 말 것
        - 출력은 반드시 JSON 배열만
        자기소개서: {resume_text}
        현재 토픽: {topic}
        현재 토픽 관련 설명: {desc}
        """
    else:  # ENGLISH
        return f"""
        You are an interviewer.
        From the following resume, extract 3-5 personality-related topics.
        - Focus on attitude, teamwork, communication, motivation, and problem-solving approach
        - Do not use TECHNICAL keywords (e.g., data preprocessing, model architecture)
        - Each item must include both "key" and "desc"
        - "desc" must be a concrete explanation and must never be empty
        - Output must be a JSON array only
        Resume: {resume_text}
        Current topic: {topic}
        Topic description: {desc}
        """


def _TECHNICAL_prompt(resume_text: str, language: str, topic: str, desc: str, aspect: str = "") -> str:
    if language == "KOREAN":
        return f"""
        너는 면접관이다.
        다음 자기소개서를 바탕으로 지원자의 기술적 경험 주제를 3~5개 뽑아라.
        - 초점: 기술, 프로젝트 경험, 문제 해결 과정, 성과
        - 인성 관련 키워드는 절대 사용하지 말 것
        - 각 항목은 반드시 "key"와 "desc" 포함
        - 출력은 반드시 JSON 배열만
        자기소개서: {resume_text}
        현재 토픽: {topic}
        현재 토픽 관련 설명: {desc}
        """
    else:  # ENGLISH
        return f"""
        You are an interviewer.
        From the following resume, extract 3-5 TECHNICAL topics.
        - Focus on skills, project experience, problem-solving process, achievements
        - Do not include personality-related keywords (e.g., values, attitude)
        - Each item must include both "key" and "desc"
        - Output must be a JSON array only
        Resume: {resume_text}
        Current topic: {topic}
        Topic description: {desc}
        """

from langchain_core.prompts import ChatPromptTemplate
def _mixed_prompt(resume_text: str, language: str, cur_topic: str, aspect: str, desc: str):
    if language == "KOREAN":
        return ChatPromptTemplate.from_messages([
            ("system", "너는 인공지능 면접관이다.\n한국어로 다음 질문을 만들어라.\n\n조건:\n- 구체적이고 맥락 있는 질문(1~2문장)\n- 직전 질문과 포인트 중복 금지\n- 메타 표현 금지"),
            ("user", "지원 직무: {{ job }}\n현재 토픽: {{ cur_topic }}\n현재 토픽 설명: {{ topic_desc }}\n참고 관점(aspect): {{ subtype }}\n직전 질문: {{ prev_q }}\n직전 답변: {{ prev_a }}\n\n요청: {{ cur_topic }}와 관련된 **새로운 질문**을 한 문장 또는 두 문장으로 생성하라.")
        ], template_format="jinja2")
    else:
        return ChatPromptTemplate.from_messages([
            ("system", "You are an interviewer.\nGenerate the next interview question in English.\n\nConditions:\n- One or two sentences, context-aware\n- Do not repeat the previous point\n- Avoid meta expressions"),
            ("user", "Job: {{ job }}\nCurrent topic: {{ cur_topic }}\nDescription: {{ topic_desc }}\nAspect hint: {{ subtype }}\nPrevious question: {{ prev_q }}\nPrevious answer: {{ prev_a }}\n\nRequest: Create ONE new question about the current topic.")
        ], template_format="jinja2")


def get_topic_prompt(interviewType: str, resume_text: str, language: str, desc: str = "") -> str:
    """
    주어진 인터뷰 타입, 이력 텍스트, 언어, 설명을 바탕으로 토픽 추출용 프롬프트를 문자열로 반환한다.
    """
    lang_rule = "한국어" if language.upper() == "KOREAN" else "English"

    return f"""
너는 실제 면접관이다.
지원자의 이력서를 보고 {lang_rule}로 면접에서 활용할 **질문 소재(토픽)**를 JSON 배열로 출력하라.

조건:
- JSON 배열 형식으로 출력
- 각 항목은 {{"key": "토픽명", "desc": "설명"}} 구조
- {interviewType} 유형에 맞도록 토픽 다양성 반영
- 직무 관련 핵심 역량을 주제화
- 설명(desc)은 면접관이 **왜 궁금해할지**를 간단히 작성
- 출력 시 JSON 외 불필요한 문구 금지

입력:
이력: '''{resume_text}'''
추가 설명: {desc}
"""


def get_first_question_prompt(job: str, topic: str, desc: str, language: str) -> str:
    """
    첫 질문 생성을 위한 문자열 프롬프트 반환
    """
    lang_rule = "한국어" if language.upper() == "KOREAN" else "English"

    return f"""
너는 인공지능 면접관이다.
지원 직무: {job}
현재 토픽: {topic}
현재 토픽 설명: {desc}

조건:
- {lang_rule}로 질문 작성
- 반드시 한 문장만 작성
- 토픽 키워드를 그대로 쓰지 말고 의미를 풀어 표현
- 구체적이고 자연스러운 면접 질문으로 작성
"""


def get_followup_prompt(job: str, cur_topic: str, topic_desc: str, prev_q: str, prev_a: str, subtype: str, language: str) -> str:
    """
    후속 질문 생성을 위한 문자열 프롬프트 반환
    """
    lang_rule = "한국어" if language.upper() == "KOREAN" else "English"

    return f"""
너는 인공지능 면접관이다.
지원 직무: {job}
현재 토픽: {cur_topic}
현재 토픽 설명: {topic_desc}
참고 관점(aspect): {subtype}
직전 질문: {prev_q}
직전 답변: {prev_a}

조건:
- {lang_rule}로 질문 작성
- 반드시 한 문장
- 직전 답변에서 핵심을 더 깊이 파고드는 질문
- 중복 없이 새로운 포인트를 탐구
"""


def get_lateral_prompt(job: str, cur_topic: str, topic_desc: str, prev_q: str, prev_a: str, subtype: str, language: str) -> str:
    """
    새로운 각도에서 질문(Lateral) 생성을 위한 문자열 프롬프트 반환
    """
    lang_rule = "한국어" if language.upper() == "KOREAN" else "English"

    return f"""
너는 인공지능 면접관이다.
지원 직무: {job}
현재 토픽: {cur_topic}
현재 토픽 설명: {topic_desc}
집중할 측면(aspect): {subtype}
직전 질문: {prev_q}
직전 답변: {prev_a}

조건:
- {lang_rule}로 질문 작성
- 반드시 한 문장
- 현재 토픽을 유지하면서 다른 관점에서 새 질문 생성
- 직전 질문과 겹치지 않게 새로운 내용
"""