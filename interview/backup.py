        if topic:  # ✅ 토픽 기반
            if lang_code == "KOREAN":
                prompt = f"""
                너는 면접관이다.
                다음 토픽에 대해 첫 질문을 하나 만들어라.
                출력은 반드시 면접 질문 문장 1개만 하라. 
                f"출력은 {lang}로 된 정확히 한 문장. 머리말/번호/따옴표/설명 금지. "
                토픽: {topic}

                출력 조건:
                - f"{system_rule(state)} "
                - 위 주제를 기반으로 실제 면접 질문 하나를 만들어라.
                """
            else:
                prompt = f"""
                Based on this topic, create one interview question.
                You are an interviewer.
                Generate the first interview question about the following topic.
                Output must be exactly one interview question sentence only. 

                Topic: {topic}

                Output requirements:
                - f"{system_rule(state)} "
                - Must sound like real spoken English (e.g., "Can you tell me about...", "How did you...")
                - Only one English question sentence
                """
                
                
    # 자소서 유무 분기
    builder.add_edge("extract_topics", "first_question")
    builder.add_conditional_edges(
        "start_node",
        lambda state: "with_resume" if getattr(state, "resume", None) else "without_resume",
        {
            "with_resume": "extract_topics",
            "without_resume": "first_question"
        }
    )