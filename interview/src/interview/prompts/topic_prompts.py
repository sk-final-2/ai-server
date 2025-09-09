def _personality_prompt(resume_text: str, language: str) -> str:
    if language == "KOREAN":
        return f"""
        너는 면접관이다.
        다음 자기소개서를 바탕으로 지원자의 인성 관련 주제를 3~5개 뽑아라.
        - 초점: 태도, 협업, 의사소통, 동기, 문제 대처 방식
        - 기술적 키워드는 절대 사용하지 말 것
        - 각 항목은 반드시 "key"와 "desc" 포함
        - "desc"는 구체적인 설명이어야 하며 절대 비워두지 말 것
        - 출력은 반드시 JSON 배열만
        자기소개서:
        {resume_text}
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
        Resume:
        {resume_text}
        """


def _TECHNICAL_prompt(resume_text: str, language: str) -> str:
    if language == "KOREAN":
        return f"""
        너는 면접관이다.
        다음 자기소개서를 바탕으로 지원자의 기술적 경험 주제를 3~5개 뽑아라.
        - 초점: 기술, 프로젝트 경험, 문제 해결 과정, 성과
        - 인성 관련 키워드는 절대 사용하지 말 것
        - 각 항목은 반드시 "key"와 "desc" 포함
        - 출력은 반드시 JSON 배열만
        자기소개서:
        {resume_text}
        """
    else:  # ENGLISH
        return f"""
        You are an interviewer.
        From the following resume, extract 3-5 TECHNICAL topics.
        - Focus on skills, project experience, problem-solving process, achievements
        - Do not include personality-related keywords (e.g., values, attitude)
        - Each item must include both "key" and "desc"
        - Output must be a JSON array only
        Resume:
        {resume_text}
        """


def _mixed_prompt(resume_text: str, language: str) -> str:
    if language == "KOREAN":
        return f"""
        너는 면접관이다.
        다음 자기소개서를 바탕으로 인성과 기술 주제를 고르게 3~5개 뽑아라.
        - 각 항목은 반드시 "key"와 "desc" 포함
        - 출력은 반드시 JSON 배열만
        자기소개서:
        {resume_text}
        """
    else:  # ENGLISH
        return f"""
        You are an interviewer.
        From the following resume, extract 3-5 balanced topics (both personality and TECHNICAL).
        - Each item must include both "key" and "desc"
        - Output must be a JSON array only
        Resume:
        {resume_text}
        """


def get_topic_prompt(interviewType: str, resume_text: str, language: str) -> str:
    if interviewType == "PERSONALITY":
        return _personality_prompt(resume_text, language)
    elif interviewType == "TECHNICAL":
        return _TECHNICAL_prompt(resume_text, language)
    else:  # MIXED
        return _mixed_prompt(resume_text, language)