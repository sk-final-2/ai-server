def get_bridge_prompt(prev_q: str, prev_a: str, new_type: str, lang: str = "KOREAN") -> str:
    if lang == "KOREAN":
        return f"""
        너는 면접관이다.
        직전 질문은 "{prev_q}"였고, 지원자의 답변은 "{prev_a}"였다.
        이제 새로운 면접 유형({ '기술면접' if new_type=='TECHNICAL' else 'PERSONALITY' })으로 전환해야 한다.

        규칙:
        - 직전 답변을 간단히 짚고, 새로운 유형과 자연스럽게 연결할 것
        - "이제 다른 주제로 넘어가겠습니다" 같은 메타 발언은 금지
        - 질문은 반드시 한 문장으로 출력
        - 반드시 한국어로 출력할 것
        - 질문은 반드시 { '기술적 경험, 방법, 문제 해결 과정' if new_type=='TECHNICAL' else '가치관, 태도, 협업, 성격' } 중심일 것
        """
    else:
        return f"""
        You are an interviewer.
        The previous question was: "{prev_q}" and the candidate's answer was: "{prev_a}".
        Now you must transition to a new interview type ({new_type}).

        Rules:
        - Briefly acknowledge the previous answer, then smoothly connect to the new type
        - Do not say things like "let's move on"
        - Output must be exactly one sentence
        - Must be in English
        - The question should reflect the {new_type} style
        """
