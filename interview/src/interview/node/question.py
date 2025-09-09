from interview.question_bank import FALLBACK_POOL
from interview.model import InterviewState, ResumeItem
from interview.node.rules import system_rule
from utils.chroma_qa import save_question, get_similar_question
from utils.question_filter import is_redundant
from interview.config import llm
from langchain_core.prompts import ChatPromptTemplate
from utils.chroma_setup import reset_interview
from interview.node.rules import validate_language_text, clean_question, validate_question
from interview.prompts.topic_prompts import get_topic_prompt
import json,re

def extract_json_array(text: str):
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        return json.loads(match.group(0))
    except Exception as e:
        print("❌ JSON 파싱 실패:", e)
        print("⚠️ 원본 응답:", text)
        return []
    
def extract_topics_node(state):
    resume_text = state.resume or ""
    sum_prompt = get_topic_prompt(state.interviewType, resume_text, state.language)

    sum_resp = llm.invoke(sum_prompt)
    raw_sum = sum_resp.content if hasattr(sum_resp, "content") else str(sum_resp)
    print("📄 자소서 기반 토픽:", raw_sum)
    try:
        match = re.search(r'\[.*\]', raw_sum, re.DOTALL)
        json_str = match.group(0) if match else None
        parsed = json.loads(json_str) if json_str else []

        fixed = []
        for item in parsed:
            if isinstance(item, dict):
                fixed.append(
                    ResumeItem(
                        key=item.get("key") or item.get("theme") or item.get("topic") or "",
                        desc=item.get("desc") or item.get("value") or item.get("description") or ""
                    )
                )
            else:
                fixed.append(ResumeItem(key=str(item), desc=""))
        state.resume_summary = fixed
    except Exception as e:
        print("❌ JSON 파싱 실패:", e)
        print("⚠️ 원본 응답:", raw_sum)
        state.resume_summary = []

    # --- ✅ desc 중심으로 topic 세팅 ---
    state.topics = [
        {"name": item.key, "asked": 0, "max_questions": 3}
        for item in state.resume_summary if item.desc
    ]
    state.current_topic_index = 0 if state.topics else None

    print("📌 Resume summary:", state.resume_summary)
    print("PARSED TOPICS (desc 기반):", [t["name"] for t in state.topics])
    print("✅ state.topics 세팅 완료:", state.topics)
    return state

def setup_default_topics_node(state: InterviewState) -> InterviewState:
    try:
        if isinstance(state, dict):
            state = InterviewState(**state)

        job = (state.job or "웹 개발자").strip()
        lang_code = state.language or "KOREAN"
        interview_type = state.interviewType or "MIXED"
        level = state.level or "중"

        sys_prompt = system_rule(state)

        prompt = f"""
        {sys_prompt}

        지원 직무는 '{job}'이다. 자기소개서는 제공되지 않았다.
        이 직무와 관련된 면접 토픽을 3~5개 뽑아라.

        조건:
        - JSON 배열로 출력
        - 각 항목은 {{"key": "토픽명", "desc": "설명"}} 구조
        - {interview_type} 유형에 맞도록 토픽 다양성 반영
        - 난이도 {level}에 맞춰 쉬운 것부터 심화까지 포함
        - { '한국어' if lang_code == 'KOREAN' else '영어'} 질문에 적합한 주제
        """

        resp = llm.invoke(prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        topics = extract_json_array(raw)

        state.topics = [
            {"name": t.get("key", ""), "asked": 0, "max_questions": 3}
            for t in topics if isinstance(t, dict) and t.get("key")
        ]
        state.current_topic_index = 0 if state.topics else None

        print("✅ [setup_default_topics_node] topics:", state.topics)
        return state

    except Exception as e:
        print("❌ [setup_default_topics_node 오류]:", str(e))
        state.topics = []
        state.current_topic_index = None
        return state
#------------------------------------------------------------------------------------------------------------------------------    
def first_question_node(state: InterviewState) -> InterviewState:
    """🎯 첫 질문 생성 노드 (토픽 기반 + fallback + 언어 보정)"""
    try:
        if isinstance(state, dict):
            state = InterviewState(**state)

        idx = state.current_topic_index or 0
        job = (state.job or "").strip() or "웹 개발자"
        state.job = job
        lang_code = state.language or "KOREAN"
        print("\n======================")
        print("🎯 [first_question_node] 진입")
        print(f"💼 지원 직무: {job}")
        print("======================")
        print(f"🧭 INTERVIEW TYPE: {state.interviewType}")
        print(f"🧭 SELECTED ASPECT: {state.aspect}")
        interviewId = state.interviewId
        if not interviewId:
            raise ValueError("❌ interviewId가 없습니다.")

        # --- ✅ 토픽/측면 초기화 (질문 생성 전에 무조건 state에 세팅) ---
        if state.topics and idx < len(state.topics):
            state.topic = state.topics[idx]["name"]
        else:
            state.topic = None

        if state.aspects:
            state.aspect = state.aspects[state.aspect_index % len(state.aspects)]
        else:
            state.aspect = None

        print(f"📄 선택된 토픽: {state.topic}")
        print(f"📄 선택된 측면(aspect): {state.aspect}")
        current_subtype = getattr(state, "subtype", None)
        # --- 프롬프트 생성 ---
        if state.topic:  # ✅ topic 기반
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_rule(state)),
                    ("user", "지원 직무: {{ job }}\n토픽: {{ topic }}\n설명: {{ desc }}\n\n"
                              "위 정보를 바탕으로 실제 면접에서 사용할 질문을 생성해라.\n"
                              "- {{topic}}은 그대로 문장화하지 말 것 (단순 키워드임)\n"
                              "- 반드시 한 문장으로 작성\n"
                    )
                ],
                template_format="jinja2"
            )
            response = (prompt | llm).invoke({
                "job": job,
                "topic": state.topic,
                "desc": state.resume_summary[idx].desc if state.resume_summary and idx < len(state.resume_summary) else ""
            })
            raw_q = (response.content or "").strip()
        else:  # ❌ 토픽 없으면 직무/경력 기반
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_rule(state)),
                    ("user", "job: {{ job }}\ncareer: {{ career }}\nresume: '''{{ resume }}'''")
                ],
                template_format="jinja2"
            )
            variables = {
                "job": job,
                "career": state.career or "미기재",
                "resume": (state.ocrText or getattr(state, "resume", "") or "").strip()[:800],
            }
            response = (prompt | llm.bind(max_tokens=200, temperature=0.2, top_p=0.8)).invoke(variables)
            raw_q = (response.content or "").strip()

        # --- 언어 보정 ---
        question = raw_q
        if not validate_language_text(question, lang_code):
            strong = (
                "Respond ONLY in English. One sentence only."
                if lang_code == "ENGLISH"
                else "오직 한국어로 한 문장만 답하라."
            )
            fix_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", strong),
                    ("user", "Rewrite as ONE interview question only (no preface/numbering/quotes): {{ q }}")
                ],
                template_format="jinja2"
            )
            question = (
                (fix_prompt | llm.bind(max_tokens=200, temperature=0))
                .invoke({"q": question})
                .content.strip()
            )

        # --- 최종 후처리 ---
        question = clean_question(question)

        # fallback 질문
        if not question:
            question = (
                f"{job} 역할에서 최근 수행한 프로젝트와 본인 기여를 구체적으로 설명해 주세요."
                if lang_code == "KOREAN"
                else f"For the {job} role, describe your most recent project and your specific contribution."
            )

        # ✅ 상태 업데이트
        seq = int(state.seq or 1)
        state.seq = seq
        if seq == 1:
            reset_interview(interviewId)

        save_question(
            interviewId,
            seq,
            question,
            job=state.job,
            level=state.level,
            language=state.language,
            topic=state.topic,    # ✅ 항상 state 값
            aspect=state.aspect,   # ✅ 항상 state 값
            subtype=current_subtype
        )

        state.question = question
        state.questions.append(question)
        state.step = (state.step or 0) + 1

        if state.topic:
            state.topics[idx]["asked"] += 1
            state.last_question_for_dynamic = question

        # 종료 조건 처리
        if getattr(state, "last_label", None) == "terminate":
            if state.topics and state.current_topic_index < len(state.topics):
                print(f"🛑 terminate 신호 감지 → '{state.topics[state.current_topic_index]['name']}' 종료, 다음 토픽으로 이동")
                state.topics[state.current_topic_index]["asked"] = state.topics[state.current_topic_index].get("max_questions", 1)
                state.current_topic_index += 1
                state.last_label = None
                if state.current_topic_index >= len(state.topics):
                    state.keepGoing = False
                    return state

        cnt = int(state.count or 0)
        if cnt > 0 and seq >= cnt:
            state.keepGoing = False

        print("✅ [first_question_node] 최종 질문:", state.question)
        return state

    except Exception as e:
        print("❌ [first_question_node 오류 발생]:", str(e))
        import traceback; traceback.print_exc()
        raise e
#------------------------------------------------------------------------------------------------------------------------------    
def next_question_node(state: InterviewState) -> InterviewState:
    """➡️ 다음 질문 생성 노드 (terminate + max_questions 처리 + save_question 갱신 반영)"""
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

        # 현재 토픽/측면
        aspect_idx = getattr(state, "aspect_index", 0)
        aspect = state.aspects[aspect_idx % len(state.aspects)] if state.aspects else None
        topic = None
        if state.current_topic_index < len(topics):
            topic = topics[state.current_topic_index]
        print(f"🧭 INTERVIEW TYPE: {state.interviewType}")
        print(f"🧭 SELECTED ASPECT: {state.aspect}")
        current_subtype = getattr(state, "subtype", None)
        # --- system 프롬프트 구성 ---
        summary_text = " ".join(item.desc for item in state.resume_summary) if state.resume_summary else ""
        system_prompt = (
            f"너는 인공지능 면접관이다.\n"
            f"지원자가 제출한 자기소개서(선택적)와 직전 답변을 참고하여 "
            f"{lang}으로만 다음 질문을 만들어라.\n\n"
            "조건:\n"
            "- 구체적이고 맥락 있는 질문\n"
            "- 이전 질문과 유사하지 않음\n"
            "- 직무 관련 기술/경험 또는 인성 관련 질문 위주\n"
            "- '절대 금지: '이제 ~에 대해 묻겠습니다' / '다음으로 ~에 대해' / '제가 묻고 싶은 것은' 같은 메타 표현.\n"
        )
        if topic:
            system_prompt += (
                f"\n현재 주제: {topic['name']}\n"
                f"참고할 자기소개서 요약: {summary_text or '없음'}\n"
                f"참고 관점(aspect): {aspect}\n"
                "⚠️ 주제와 요약은 참고용으로만 사용하고, 질문 문장에 직접 노출하지 말 것."
            )

        # --- 질문 생성 ---
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", system_rule(state)),   # ✅ 언어 규칙
            ("system", system_prompt),          # ✅ 질문 규칙
            ("human", "직무: {job}\n직전 질문: {prev_q}\n직전 답변: {prev_a}")
        ])
        chain = question_prompt | llm
        res = chain.invoke({
            "job": job,
            "prev_q": prev_q,
            "prev_a": prev_a,
        })
        final_q = None
        for attempt in range(1, 6):
            res = chain.invoke({"job": job, "prev_q": prev_q, "prev_a": prev_a})
            candidate_q = (res.content or "").strip()
            candidate_q = clean_question(candidate_q)

            print(f"🔄 [재시도 {attempt}] 후보 질문: {candidate_q}")
            from utils.chroma_qa import get_similar_question, save_question
            save_question(
            state.interviewId,
            state.seq + 1,
            candidate_q,
            job=state.job,
            level=state.level,
            language=state.language,
            topic=topic["name"] if topic else None,
            aspect=aspect 
            )              
            
            check = get_similar_question(
                state.interviewId, candidate_q, k=3, min_similarity=0.75, verify_all=True, subtype=current_subtype, job=state.job, 
            )
            redundant = False
            if prev_q:
                embed_sim = max(
                    (h.get("sim", 0.0) for h in check.get("hits", []) if h.get("doc")), default=0.0
                )
                redundant = is_redundant(prev_q, candidate_q, embed_sim, cos_thr=0.95, jac_thr=0.6, ngram_thr=0.5)

            if not (check.get("similar") or redundant):
                final_q = candidate_q
                print(f"✅ [채택됨 - {attempt}번째 시도] {final_q}")
                break
            else:
                print(f"❌ [거부됨 - {attempt}번째 시도] 유사하거나 중복된 질문")

        if not final_q:
            fb_list = FALLBACK_POOL.get(state.aspects[state.aspect_index], [])
            final_q = fb_list[0] if fb_list else f"{job} 관련 다른 경험을 말씀해 주세요."
            print(f"⚠️ [Fallback 적용] 최종 질문: {final_q}")

        # 상태 업데이트
        state.question = final_q
        state.questions.append(final_q)
        
        save_question(
            state.interviewId,
            state.seq + 1,
            state.question,
            job=state.job,
            level=state.level,
            language=state.language,
            topic=state.topic,
            aspect=state.aspect
        )
        
        # --- ✅ terminate 기반 토픽 전환 ---
        if getattr(state, "last_label", None) == "terminate":
            print(f"🛑 terminate 신호 감지 → '{topics[state.current_topic_index]['name']}' 종료, 다음 토픽으로 이동")
            state.topics[state.current_topic_index]["asked"] = topics[state.current_topic_index].get("max_questions", 1)
            state.current_topic_index += 1
            state.bridge_done = False
            state.last_label = None
            if state.current_topic_index < len(topics):
                state.topic = topics[state.current_topic_index]["name"]
                state.aspect = state.aspects[state.aspect_index % len(state.aspects)] if state.aspects else None
                print(f"📌 [next_question_node] 토픽 전환 완료 → {state.topic}")
            else:
                state.keepGoing = False
                return state

        # --- ✅ max_questions 기반 토픽 전환 ---
        elif topic and topic.get("asked", 0) + 1 >= topic.get("max_questions", 3):
            print(f"🛑 '{topic['name']}' 토픽 질문 {topic.get('asked', 0)+1}개 완료 → 다음 토픽으로 전환")
            state.current_topic_index += 1
            state.bridge_done = False
            if state.current_topic_index < len(topics):
                state.topic = topics[state.current_topic_index]["name"]
                state.aspect = state.aspects[state.aspect_index % len(state.aspects)] if state.aspects else None
                print(f"📌 [next_question_node] 토픽 전환 완료 → {state.topic}")
                return next_question_node(state)
            else:
                state.keepGoing = False
                return state

        # --- 질문 카운트 증가 ---
        if topic:
            topic["asked"] = topic.get("asked", 0) + 1
        state.aspect_index = (aspect_idx + 1) % len(state.aspects)

        # --- ✅ save_question 호출 (항상 최신 state 값 저장) ---


        # --- 종료 조건 ---
        if state.count and len(state.questions) > state.count:
            state.keepGoing = False
        elif not state.count and topics:
            if all(t["asked"] >= t.get("max_questions", 1) for t in topics):
                state.keepGoing = False
        elif not state.count and not topics:
            if sum(1 for a in state.answer if a == "terminate") >= 2:
                state.keepGoing = False
            elif len(state.questions) >= 10:
                state.keepGoing = False

    except Exception as e:
        print("❌ [next_question_node 예외]:", str(e))
        import traceback; traceback.print_exc()
        state.keepGoing = False

    state.step += 1
    return state
