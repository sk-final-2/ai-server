from interview.model import InterviewState
from typing import Union
from interview.predict_keepGoing import keepGoing
from utils.constants import PERSONALITY_ASPECTS, TECHNICAL_ASPECTS, MIXED_ASPECTS
import os

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

def bridge_node(state: InterviewState) -> InterviewState:
    """🔀 MIXED 면접: 토픽 내에서 Aspect 전환"""
    if isinstance(state, dict):
        state = InterviewState(**state)

    if state.interviewType != "MIXED":
        return state

    # 현재 토픽 확인
    cur_topic = state.topics[state.current_topic_index] if state.topics else None
    if not cur_topic:
        return state

    asked = cur_topic.get("asked", 0)          # 지금까지 해당 토픽에서 질문한 수
    max_q = cur_topic.get("max_questions", 3)  # 이 토픽에서 허용된 질문 수
    cutoff = max_q // 2                        # 절반 시점 (예: 3이면 1~2번째에서 발동)

    # 이미 전환했으면 재발동 금지
    if getattr(state, "bridge_done", False):
        return state

    # 브릿지 발동 조건: 현재 토픽에서 절반 이상 질문했을 때
    if asked >= cutoff:
        if state.aspects == TECHNICAL_ASPECTS:
            print("🔀 브릿지 발동(토픽 내): TECHNICAL → PERSONALITY")
            state.aspects = PERSONALITY_ASPECTS
        else:
            print("🔀 브릿지 발동(토픽 내): PERSONALITY → TECHNICAL")
            state.aspects = TECHNICAL_ASPECTS

        state.aspect_index = 0
        state.bridge_switched = True
        state.bridge_done = True   # ✅ 이 토픽에서는 한 번만 발동
    # ✅ 현재 질문 유형 로그 출력
    if state.aspects == TECHNICAL_ASPECTS:
        print("🧭 현재 질문 유형: TECHNICAL")
    elif state.aspects == PERSONALITY_ASPECTS:
        print("🧭 현재 질문 유형: PERSONALITY")
    else:
        print(f"🧭 현재 질문 유형: UNKNOWN ({state.aspects})")
        
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

