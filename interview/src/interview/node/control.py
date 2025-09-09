from interview.model import InterviewState
from typing import Union
from interview.predict_keepGoing import keepGoing
from utils.constants import PERSONALITY_ASPECTS, TECHNICAL_ASPECTS, MIXED_ASPECTS
import os
from utils.qa_classify import (
    classify_turn_with_llm, heuristic_scores,
    can_bridge, decide_next_type
)
DYN_MIN_SEQ = int(os.getenv("DYN_MIN_SEQ", "3"))     # 최소 N문항은 진행
DYN_HARD_CAP = int(os.getenv("DYN_HARD_CAP", "20"))  # 동적 모드 최대 문항

def start_node(state):
    """
    자소서(resume) 유무에 따라 분기
    """
    return state

def set_options_node(state: InterviewState) -> InterviewState:
    """🛠 면접 옵션(language, level, count, interviewType) 확정 노드"""
    if getattr(state, "ocrText", None) and not getattr(state, "resume", None):
        state.resume = state.ocrText
    if isinstance(state, dict):
        state = InterviewState(**state)

    print("\n======================")
    print("⚙️ [set_options_node] 옵션 설정 시작")
    print(f"입력 language: {state.language}, level: {state.level}, count: {state.count}, interviewType: {state.interviewType}")
    print("======================")

    # 기본값 처리 (명세서 값 그대로 사용)
    interview_type = state.interviewType
        
    if not state.language:
        state.language = "KOREAN"
    if not state.level:
        state.level = "중"
    if state.count is None:
        state.count = 0
    if not state.interviewType:
        state.interviewType = "MIXED"
    if getattr(state, "keepGoing", None) is None:
        state.keepGoing = True
    

    if interview_type == "PERSONALITY":
        state.aspects = PERSONALITY_ASPECTS
    elif interview_type == "TECHNICAL":
        state.aspects = TECHNICAL_ASPECTS
    else:  # MIXED
        state.aspects = MIXED_ASPECTS
    # ✅ 여기서 resume_summary를 인터뷰 타입 + 언어 기준으로 필터링
    if hasattr(state, "resume_summary") and state.resume_summary:
        print("📌 [set_options_node] resume_summary 필터링 적용 전:", state.resume_summary)
        state.resume_summary = filter_resume_topics(state.resume_summary, state.interviewType, state.language)
        print("📌 [set_options_node] resume_summary 필터링 적용 후:", state.resume_summary)
        
    state.options_locked = True
    print(f"✅ 최종 language: {state.language}, level: {state.level}, count: {state.count}, interviewType: {state.interviewType}")
    return state

def bridge_node(state: "InterviewState") -> "InterviewState":
    if isinstance(state, dict):
        state = InterviewState(**state)
    if getattr(state, "interviewType", "") != "MIXED":
        return state

    # 현재 토픽 확인
    cur_topic = None
    if getattr(state, "topics", None) and 0 <= getattr(state, "current_topic_index", 0) < len(state.topics):
        cur_topic = state.topics[state.current_topic_index]
    if not cur_topic:
        return state

    asked = int(cur_topic.get("asked", 0))
    max_q = int(cur_topic.get("max_questions", 3))

    ok, reason = can_bridge(state, asked, max_q)
    if not ok:
        print(f"⏸️ bridge skip: {reason}")
        return state

    last_q = getattr(state, "last_question", "") or getattr(state, "question", "")
    last_a = getattr(state, "last_answer", "")
    topic  = getattr(state, "topic", "")

    # prev_type 정규화 (혼입 방지)
    cur_t_raw = getattr(state, "qtype", "") or "PERSONALITY"
    _map = {"TECH": "TECHNICAL", "PERSON": "PERSONALITY", "": "PERSONALITY", None: "PERSONALITY"}
    cur_t = _map.get(cur_t_raw, cur_t_raw)

    recent_text = (
        " ".join([a.get("text","") if isinstance(a, dict) else str(a)
                  for a in getattr(state, "answers", [])[-2:]])
        if getattr(state, "answers", None) else ""
    )

    # 1) LLM 분류(KO/EN 자동 선택)
    from interview.config import llm
    llm_res = classify_turn_with_llm(llm, getattr(state, "language", ""), last_q, last_a, topic, cur_t, recent_text)

    # 2) 휴리스틱
    h_res = heuristic_scores(f"{last_q} {last_a}")

    # 3) 결정 (TECHNICAL / PERSONALITY 라벨 사용)
    decision = decide_next_type(getattr(state, "language", ""), cur_t, llm_res, h_res)
    print(f"🔀 decision={decision}")

    next_type = decision["next_type"]
    switched = (next_type != cur_t)

    # 상태 갱신
    state.qtype = next_type
    state.subtype = decision.get("subtype") or (getattr(state, "subtype", "") or "METHOD")
    state.bridge_note = ""  # 요청대로 note는 사용하지 않음
    state.bridge_switched = switched

    # 전환 '성공'시에만 잠금/타임스탬프
    if switched:
        state.bridge_done = True
        state.last_bridge_turn = getattr(state, "seq", 0) or 0

    # 기본값 보정(후속 노드 안전)
    state.qtype = state.qtype or "PERSONALITY"
    state.subtype = state.subtype or "METHOD"
    state.bridge_note = ""  # 항상 빈 문자열 유지

    # 로깅
    print(f"🧭 현재 질문 유형: {state.qtype} / subtype={state.subtype}")
    return state

def check_keepGoing(state: InterviewState) -> str:
    print("🧐 check_keepGoing 진입:", state.keepGoing)
    return "stop" if state.keepGoing is False else "continue"

def keepGoing_node(state: InterviewState) -> Union[InterviewState, None]:
    """count=0일 때 KoELECTRA로 종료 여부 판단 (terminate는 무조건 유지)"""
    if isinstance(state, dict):
        state = InterviewState(**state)

    # count>0이면 그냥 통과
    if getattr(state, "count", None) != 0:
        print("➡️ [keepGoing_node] count>0 → 그대로 통과")
        return state

    # ✅ 동적 모드일 때 질문 가져오기
    question = getattr(state, "last_question_for_dynamic", None)
    if not question:
        question = state.questions[-1] if state.questions else ""

    answer = state.last_answer or ""

    try:
        # 1차: KoELECTRA 분류
        label = keepGoing(question, answer)
        print(f"🧩 [KoELECTRA 결과] label={label!r}")

        if label == "terminate":
            print("🛑 [keepGoing_node] KoELECTRA terminate → terminate 확정")
            state.last_label = "terminate"
            state.keepGoing = True   # 👉 terminate 신호만 넘기고, 실제 전환은 next_question_node에서 처리
        else:
            print("✅ [keepGoing_node] KoELECTRA 계속 진행 예측")
            state.last_label = "continue"
            state.keepGoing = True

    except Exception as e:
        print("⚠️ [keepGoing_node 오류] 예외 발생 → 계속 진행:", e)
        state.last_label = "continue"
        state.keepGoing = True

    finally:
        if hasattr(state, "last_question_for_dynamic"):
            delattr(state, "last_question_for_dynamic")

    return state
            
def filter_resume_topics(summary, interviewType: str, language: str = "KOREAN"):
    if not summary:
        return []

    # 🔑 한국어/영어 키워드 세트
    personality_keywords = {
        "KOREAN": ["협업", "팀", "소통", "동기", "성장", "가치관", "리더십", "커뮤니케이션"],
        "ENGLISH": ["collaboration", "team", "communication", "motivation", "growth", "values", "leadership"]
    }
    TECHNICAL_keywords = {
        "KOREAN": ["모델", "데이터", "알고리즘", "성능", "분석", "전처리", "구현"],
        "ENGLISH": ["model", "data", "algorithm", "performance", "analysis", "preprocessing", "implementation"]
    }

    lang_key = "KOREAN" if language.upper() == "KOREAN" else "ENGLISH"

    def contains_keywords(text: str, keywords: list[str]) -> bool:
        return any(k.lower() in text.lower() for k in keywords)

    # ✅ 인터뷰 타입별 필터링
    if interviewType == "PERSONALITY":
        return [s for s in summary if contains_keywords(s.key + " " + s.desc, personality_keywords[lang_key])]

    elif interviewType == "TECHNICAL":
        return [s for s in summary if contains_keywords(s.key + " " + s.desc, TECHNICAL_keywords[lang_key])]

    elif interviewType == "MIXED":
        return summary  # 브릿지 노드에서 비율 제어

    return summary

