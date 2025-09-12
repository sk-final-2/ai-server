def next_question_node(state: InterviewState) -> InterviewState:
    try:
        if isinstance(state, dict):
            state = InterviewState(**state)

        topics = getattr(state, "topics", [])
        if not topics:
            state.keepGoing = False
            return state

        job = (state.job or "").strip() or "웹 개발자"
        lang_code = state.language or "KOREAN"
        lang = "한국어" if lang_code == "KOREAN" else "영어"

        prev_q = state.questions[-1] if state.questions else ""
        prev_a = state.last_answer or (state.answer[-1] if state.answer else "")

        # 토픽/측면
        aspect_idx = getattr(state, "aspect_index", 0)
        aspect = state.aspects[aspect_idx % len(state.aspects)] if state.aspects else None
        topic = topics[state.current_topic_index] if state.current_topic_index < len(topics) else None

        # --- 새 토픽 인입 모드 판단 ---
        just_switched = bool(getattr(state, "just_switched_topic", False))

        # CHANGE: 전환 직후에는 이전 답변에 얽매이지 않도록 prev_a를 비우거나 약화
        if just_switched:
            prev_a_for_prompt = ""  # ← 새 토픽 인입: 꼬리질문 금지
        else:
            prev_a_for_prompt = prev_a[:800] + ("..." if len(prev_a) > 800 else "")

        # subtype 보정
        current_subtype = (getattr(state, "subtype", None) or "").strip()
        if not current_subtype:
            # CHANGE: 토픽 기반 기본 subtype 매핑
            topic_to_subtype = {
                "teamwork": "STAKEHOLDER",
                "communication": "STAKEHOLDER",
                "leadership": "STAKEHOLDER",
                "problem_solving": "METHOD",
                "technical_skills": "TOOL_STANDARD",
                "adaptability": "METHOD",
                "innovation": "IMPACT",
                "commitment": "IMPACT",
                "perseverance": "RISK_ETHICS",
            }
            key = (topic["name"] if topic else "").lower()
            current_subtype = topic_to_subtype.get(key) or "METHOD"

        summary_text = " ".join(item.desc for item in state.resume_summary) if state.resume_summary else ""
        system_prompt = (
            f"너는 인공지능 면접관이다.\n"
            f"{lang}으로 다음 질문을 만들어라.\n\n"
            "조건:\n"
            "- 구체적이고 맥락 있는 질문(1문장, 최대 2문장)\n"
            "- 바로 직전 질문과 포인트 중복 금지\n"
            "- 메타 표현 금지\n"
        )
        if topic:
            system_prompt += (
                f"\n현재 주제: {topic['name']}\n"
                f"참고 요약: {summary_text or '없음'}\n"
                f"참고 관점(aspect): {aspect}\n"
                "⚠️ 주제/요약/관점은 문장에 직접 노출하지 말 것."
            )

        from langchain_core.prompts import ChatPromptTemplate
        from interview.config import llm
        from utils.chroma_qa import get_similar_question

        # --- 프롬프트들 ---
        # FOLLOWUP 후보: 전환 직후엔 비활성화
        followup_prompt = ChatPromptTemplate.from_messages([
            ("system", system_rule(state)),
            ("system", system_prompt),
            ("human",
             "직무: {job}\n"
             "서브타입 힌트: {subtype}\n"
             "직전 질문: {prev_q}\n"
             "직전 답변: {prev_a}\n\n"
             "요청: 가능하면 직전 답변의 요지를 더 깊이 파는 **후속 질문** 1개를 생성하라.")
        ])

        # LATERAL 후보: 새 토픽 기반 질문
        lateral_prompt = ChatPromptTemplate.from_messages([
            ("system", system_rule(state)),
            ("system", system_prompt),
            ("human",
             "직무: {job}\n"
             "서브타입 힌트: {subtype}\n"
             "현재 토픽: {topic_name}\n"
             "직전 질문: {prev_q}\n"
             "직전 답변(선택): {prev_a}\n\n"
             "요청: **현재 토픽을 중심**으로, 적절한 측면(method/impact/stakeholder/risk/standard 중 하나)을 선택하여 질문 1개를 생성하라.")
        ])

        def gen_candidate(prompt, use_prev_a: str):
            res = (prompt | llm).invoke({
                "job": job,
                "prev_q": prev_q,
                "prev_a": use_prev_a,
                "subtype": current_subtype,
                "topic_name": topic["name"] if topic else "",
            })
            text = (getattr(res, "content", "") or "").strip()
            return clean_question(text)

        candidates = []
        if not just_switched:
            # 평소: FOLLOWUP/LATERAL 둘 다 생성
            candidates.append(("FOLLOWUP", gen_candidate(followup_prompt, prev_a_for_prompt)))
        # CHANGE: 전환 직후에는 FOLLOWUP를 만들지 않고, LATERAL만 생성
        candidates.append(("LATERAL",  gen_candidate(lateral_prompt, prev_a_for_prompt)))

        # 후보 스코어링 (간단/일관)
        best_q, best_kind, best_score = None, None, -1.0
        for kind, cand in candidates:
            if not cand or len(cand) < 5:
                continue
            check = get_similar_question(
                state.interviewId, cand, k=3, min_similarity=0.75,
                verify_all=True, subtype=current_subtype, job=state.job
            )
            if check.get("similar"):
                continue

            # 간단 점수: (새 토픽 모드면 LATERAL에 +가중치)
            score = 0.5
            if just_switched and kind == "LATERAL":
                score += 0.2    # CHANGE: 전환 직후는 LATERAL 우대
            elif (not just_switched) and kind == "FOLLOWUP":
                score += 0.05   # 평소엔 followup에 얕은 선호

            if score > best_score:
                best_q, best_kind, best_score = cand, kind, score

        final_q = best_q
        if not final_q:
            # 폴백: 새 토픽 인입이면 토픽-지향 템플릿, 아니면 일반 폴백
            if just_switched:
                # CHANGE: 토픽 인입용 폴백 예시
                topic_intro = {
                    "teamwork": "해당 프로젝트에서 본인이 맡은 역할과, 협업 과정에서 가장 중요하게 본 원칙 한 가지를 알려주세요.",
                    "problem_solving": "해당 프로젝트에서 발생했던 핵심 문제 한 가지와, 이를 해결하기 위해 선택한 접근 방식을 설명해 주세요.",
                    "technical_skills": "해당 프로젝트에서 가장 중요했던 기술 스택 한 가지와, 그 선택 이유를 구체적으로 말씀해 주세요.",
                    "communication": "해당 프로젝트에서 이해관계자와의 커뮤니케이션이 중요했던 순간 한 가지를 예로 들어 설명해 주세요.",
                    "leadership": "팀을 이끌거나 의사결정을 주도해야 했던 순간 한 가지와, 그때의 판단 근거를 들려주세요.",
                }
                key = (topic["name"] if topic else "").lower()
                final_q = topic_intro.get(key, "이번 토픽과 관련해 가장 핵심이었던 한 가지 사례를 구체적으로 말씀해 주시겠어요?")
            else:
                final_q = "방금 설명하신 내용에서 가장 효과적이었던 방법 한 가지를 예로 들어 구체적으로 설명해 주시겠어요?"

        print(f"✅ [선정] kind={best_kind} switched={just_switched} q={final_q}")

        # 상태 업데이트/저장
        state.question = final_q
        state.questions.append(final_q)

        # 저장 직전 보정 (항상 최신 토픽 사용)
        state.subtype = (getattr(state, "subtype", None) or current_subtype or "METHOD")
        final_topic_name = topics[state.current_topic_index]["name"] if 0 <= state.current_topic_index < len(topics) else (state.topic or None)

        from utils.chroma_qa import save_question
        save_question(
            state.interviewId,
            state.seq + 1,
            state.question,
            job=state.job,
            level=state.level,
            language=state.language,
            topic=final_topic_name,   # ✅ 전환 반영된 토픽으로 저장
            aspect=state.aspect,
            # subtype=state.subtype (필요 시)
        )

        # 카운터/인덱스 갱신
        cur = topics[state.current_topic_index]
        cur["asked"] = int(cur.get("asked", 0)) + 1
        state.aspect_index = (aspect_idx + 1) % len(state.aspects)

        # CHANGE: 전환 1턴 종료 → 플래그 해제
        if just_switched:
            state.just_switched_topic = False

        # 종료 조건/추가 전환은 기존 로직대로…

    except Exception as e:
        print("❌ [next_question_node 예외]:", str(e))
        import traceback; traceback.print_exc()
        state.keepGoing = False

    state.step += 1
    return state