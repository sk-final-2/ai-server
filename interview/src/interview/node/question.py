from interview.model import InterviewState, ResumeItem
from interview.node.rules import system_rule
from utils.chroma_qa import save_question, get_similar_question
from interview.config import llm
from langchain_core.prompts import ChatPromptTemplate
from utils.chroma_setup import reset_interview
from interview.node.rules import validate_language_text, clean_question
from interview.prompts.topic_prompts import get_topic_prompt
from utils.question_filter import init_topics_for_session
import json,re, random

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
    topic_desc = state.topics[state.current_topic_index].get("desc", "") if state.topics else ""
    sum_prompt = get_topic_prompt(state.interviewType, resume_text, state.language, desc=topic_desc)

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
    state.topics = init_topics_for_session(state.topics, 3, 5)
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
        state.topics = init_topics_for_session(state.topics, 3, 5)
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

        # --- ✅ 토픽/측면 초기화 ---
        if state.topics and idx < len(state.topics):
            state.topic = state.topics[idx]["name"]
            topic_desc = state.topics[idx].get("desc", "")
        else:
            state.topic = None
            topic_desc = ""

        if state.aspects:
            state.aspect = random.choice(state.aspects)
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
                    ("user",
                        "지원 직무: {{ job }}\n"
                        "현재 토픽: {{ cur_topic }}\n"
                        "현재 토픽 설명: {{ topic_desc }}\n"
                        "참고 관점(aspect): {{ subtype }}\n\n"
                        "지시사항:"
                        "- 위 정보를 바탕으로 실제 면접에서 사용할 질문을 생성해라.\n"
                        "- 너는 기업 면접관이다. 항상 실제 채용 면접에서 사용할 법한 질문을 한다.\n"
                        "- {{cur_topic}}은 그대로 문장화하지 말 것 (단순 키워드임)\n"
                        "- 반드시 {{cur_topic}}을 중심으로 질문할 것\n"
                        "- 단, {{ topic_desc }} 내용을 참고하여 질문을 더 구체적이고 상황에 맞게 작성하라\n"
                        "- 반드시 한 문장으로 작성"
                    )
                ],
                template_format="jinja2"
            )

            variables = {
                "job": job,
                "cur_topic": state.topic,
                "topic_desc": topic_desc,
                "subtype": state.aspect or "METHOD",
            }
            messages = prompt.format_messages(**variables)   # ✅ dict 언패킹
            response = llm.bind(max_tokens=200, temperature=0.2, top_p=0.8).invoke(messages)
            raw_q = (getattr(response, "content", "") or str(response)).strip()

        else:  # ❌ 토픽 없으면 직무/경력 기반
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_rule(state)),
                    ("user",
                        "job: {{ job }}\n"
                        "career: {{ career }}\n"
                        "resume: '''{{ resume }}'''"
                    )
                ],
                template_format="jinja2"
            )
            variables = {
                "job": job,
                "career": state.career or "미기재",
                "resume": (state.ocrText or getattr(state, "resume", "") or "").strip()[:800],
            }
            messages = prompt.format_messages(**variables)   # ✅ dict 언패킹
            response = llm.bind(max_tokens=200, temperature=0.2, top_p=0.8).invoke(messages)
            raw_q = (getattr(response, "content", "") or "").strip()

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
            messages = fix_prompt.format_messages(q=question)   # ✅ 키워드 인자 방식
            response = llm.bind(max_tokens=200, temperature=0).invoke(messages)
            question = (getattr(response, "content", "") or "").strip()

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
            state.interviewId,
            seq,
            question,
            job=state.job,
            level=state.level,
            language=state.language,
            topic=state.topic or "",
            aspect=state.aspect or "",
            subtype=current_subtype or "",
        )

        state.question = question
        state.questions.append(question)
        state.step = (state.step or 0) + 1

        if state.topic:
            state.topics[idx]["asked"] = int(state.topics[idx].get("asked", 0)) + 1
            state.last_question_for_dynamic = question

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
    """➡️ 다음 질문 생성 노드 (전환/쿨다운/중복 방지 보완판)"""
    try:
        if isinstance(state, dict):
            state = InterviewState(**state)

        topics = getattr(state, "topics", [])
        if not topics:
            state.keepGoing = False
            return state

        cur_topic_data = topics[state.current_topic_index]
        cur_topic = cur_topic_data.get("name", "")
        topic_desc = cur_topic_data.get("desc", "") or ""
        job = (state.job or "").strip() or "웹 개발자"

        prev_q = state.questions[-1] if state.questions else ""
        prev_a = state.last_answer or (state.answer[-1] if state.answer else "")

        # 전환 여부 체크
        just_switched = bool(getattr(state, "just_switched_topic", False))
        prev_a_for_prompt = "" if just_switched else prev_a[:800]

        # subtype 보정
        current_subtype = (getattr(state, "subtype", None) or "").strip() or "METHOD"

        # --- 프롬프트 템플릿 ---
        followup_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 공손하고 간결하게 질문한다."),
            ("human",
             "직무: {job}\n"
             "현재 토픽: {cur_topic}\n"
             "현재 토픽 설명: {topic_desc}\n"
             "서브타입 힌트: {subtype}\n"
             "직전 질문: {prev_q}\n"
             "직전 답변: {prev_a}\n\n"
             "지시사항:"
             "- 직전 답변을 더 깊이 파는 후속 질문 1개 생성"
             "- 너는 기업 면접관이다. 항상 실제 채용 면접에서 사용할 법한 질문을 한다.\n"
             "- 답변에서 나온 내용을 기반으로, 토픽의 세부 맥락을 더 깊이 탐색하는 질문을 하라"
             "- Aspect는 참고용일 뿐, 토픽을 벗어나지 않는다"
             "- 한 문장으로, 실제 면접관처럼 공손하고 간결하게 질문하라")
        ])

        lateral_prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 공손하고 간결하게 질문한다."),
            ("human",
             "직무: {job}\n"
             "현재 토픽: {cur_topic}\n"
             "현재 토픽 설명: {topic_desc}\n"
             "집중할 측면: {subtype}\n"
             "직전 질문: {prev_q}\n"
             "직전 답변(선택): {prev_a}\n\n"
             "지시사항:"
             "- 반드시 [현재 토픽]을 중심으로 새로운 질문을 작성하라"
             "- 너는 기업 면접관이다. 항상 실제 채용 면접에서 사용할 법한 질문을 한다.\n"
             "- Aspect({subtype})는 보조적 각도로만 활용하고, 토픽 자체를 대체하지 않는다"
             "- 직전 질문과 중복되지 않는 방향으로 질문을 만들어라"
             "- 한 문장으로, 실제 면접관처럼 공손하고 간결하게 질문하라")
        ])

        def clean_question(q: str) -> str:
            return (q or "").strip()

        def gen_candidate(prompt, use_prev_a: str):
            variables = {
                "job": job,
                "prev_q": prev_q,
                "prev_a": use_prev_a,
                "subtype": current_subtype,
                "cur_topic": cur_topic,
                "topic_desc": topic_desc
            }
            messages = prompt.format_messages(**variables)  # ✅ dict 언패킹
            res = llm.invoke(messages)
            text = (getattr(res, "content", "") or str(res)).strip()
            return clean_question(text)

        # ------------------------------
        # [C] 후보 생성
        # ------------------------------
        candidates = []
        if not just_switched:
            for _ in range(3):
                candidates.append(("FOLLOWUP", gen_candidate(followup_prompt, prev_a_for_prompt)))
        for _ in range(2):
            candidates.append(("LATERAL", gen_candidate(lateral_prompt, prev_a_for_prompt)))

        # ------------------------------
        # [D] 후보 스코어링
        # ------------------------------
        best_q, best_kind, best_score = None, None, -1.0
        for kind, text in candidates:
            if not text or len(text) < 5:
                continue

            # ✅ 중복 체크
            check = get_similar_question(
                state.interviewId,
                text,
                k=3,  # 최근 3개 질문과 비교
                min_similarity=0.85,
                verify_all=True,
                subtype=state.subtype or "METHOD",
                job=state.job,
            )
            if check.get("similar"):
                print(f"⚠️ 중복 질문 제거됨: {text}")
                continue

            score = 0.7 if (just_switched and kind == "LATERAL") else (0.55 if kind == "FOLLOWUP" else 0.5)
            if score > best_score:
                best_q, best_kind, best_score = text, kind, score

        final_q = best_q or "방금 설명하신 내용에서 가장 인상적인 부분을 구체적으로 말씀해 주시겠어요?"
        print(f"✅ [선정] kind={best_kind}, score={best_score}, q={final_q}")

        # ------------------------------
        # [E] 상태 업데이트 & 저장
        # ------------------------------
        state.question = final_q
        state.questions.append(final_q)

        cur_topic_data["asked"] = int(cur_topic_data.get("asked", 0)) + 1
        state.seq = (state.seq or 0) + 1
        state.step = (state.step or 0) + 1

        # ✅ 저장 (항상 최신 토픽명 기준)
        final_topic_name = topics[state.current_topic_index]["name"]
        state.subtype = (getattr(state, "subtype", None) or current_subtype or "METHOD")

        save_question(
            state.interviewId,
            state.seq,
            state.question,
            job=state.job,
            level=state.level,
            language=state.language,
            topic=final_topic_name,
            aspect=state.aspect,
            subtype=state.subtype,
        )

        # ------------------------------
        # [F] 전환 조건 검사
        # ------------------------------
        if cur_topic_data.get("asked", 0) >= cur_topic_data.get("max_questions", 3):
            print(f"🛑 '{cur_topic_data['name']}' 토픽 질문 {cur_topic_data['asked']}개 완료 → 다음 토픽")
            state.current_topic_index += 1
            state.bridge_done = False
            if state.current_topic_index < len(topics):
                state.topic = topics[state.current_topic_index]["name"]
                state.aspect = random.choice(state.aspects) if state.aspects else None
                state.just_switched_topic = True
                state.last_bridge_turn = state.seq
                print(f"📌 [next_question_node] 토픽 전환 완료 → {state.topic}")
            else:
                state.keepGoing = False

        return state

    except Exception as e:
        print("❌ [next_question_node 예외]:", str(e))
        import traceback; traceback.print_exc()
        state.keepGoing = False
        return state