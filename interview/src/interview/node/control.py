from interview.model import InterviewState
from typing import Union
from interview.predict_keepGoing import keepGoing
from utils.constants import PERSONALITY_ASPECTS, TECHNICAL_ASPECTS, MIXED_ASPECTS
import os

DYN_MIN_SEQ = int(os.getenv("DYN_MIN_SEQ", "3"))     # 최소 N문항은 진행
DYN_HARD_CAP = int(os.getenv("DYN_HARD_CAP", "20"))  # 동적 모드 최대 문항

def _should_stop_dynamic(state: InterviewState) -> bool:
    """count==0인 경우에만 호출: 충분히 평가 완료면 True."""
    seq = int(getattr(state, "seq", 0) or 1)
    
    print(f'♥{seq}')
    # 1) 최대치 넘으면 무조건 종료
    if seq > DYN_HARD_CAP:
        return True
    
    # 2) 최소치 넘으면 LLM한테 물어보기
    if seq >= DYN_MIN_SEQ:
        return True  # ⬅️ 임시로 True
    # 3) 최소치 이전이면 무조건 계속 진행
    return False

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
    """🔀 MIXED 면접: 일정 시점 이후 type 전환"""
    if isinstance(state, dict):
        state = InterviewState(**state)

    if state.interviewType != "MIXED":
        return state

    total = state.count or 10
    idx = len(state.questions)

    cutoff = total // 2  # 절반 이후에만 전환
    # 이미 전환했으면 다시 안 바꾸게 플래그
    if getattr(state, "bridge_done", False):
        return state

    if idx >= cutoff:
        if state.aspects == TECHNICAL_ASPECTS:
            print("🔀 브릿지 발동: TECHNICAL → PERSONALITY 전환")
            state.aspects = PERSONALITY_ASPECTS
        else:
            print("🔀 브릿지 발동: PERSONALITY → TECHNICAL 전환")
            state.aspects = TECHNICAL_ASPECTS
        state.aspect_index = 0
        state.bridge_switched = True  # ✅ 프롬프트에서 체크
        state.bridge_done = True      # ✅ 다시는 안 바꾸게
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

