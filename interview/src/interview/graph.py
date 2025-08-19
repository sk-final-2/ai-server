# test_graph.py
from typing import Literal
from langgraph.graph import StateGraph
from interview.nodes import (
    set_options_node,
    first_question_node,
    answer_node,
    analyze_node,
    next_question_node,
    keepGoing_node
)
from interview.model import InterviewState


def check_question_history(state: InterviewState) -> Literal["first", "followup"]:
    """질문 이력 유무로 분기"""
    return "first" if len(state.question) == 0 else "followup"


def check_count_mode(state: InterviewState) -> Literal["dynamic", "fixed"]:
    """count 값으로 동적/고정 모드 분기"""
    return "dynamic" if getattr(state, "count", 0) == 0 else "fixed"


def check_keepGoing(state: InterviewState) -> Literal["stop", "continue"]:
    """keepGoing 플래그로 종료/계속 분기"""
    return "stop" if not getattr(state, "keepGoing", True) else "continue"


def create_graph():
    builder = StateGraph(InterviewState)

    # 노드 등록
    builder.add_node("set_options", set_options_node)
    builder.add_node("first_question", first_question_node)
    builder.add_node("answer", answer_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("keepGoing", keepGoing_node)
    builder.add_node("next_question", next_question_node)

    # 1) 엔트리 포인트: 항상 옵션 설정부터 시작
    builder.set_entry_point("set_options")

    # 2) 옵션 설정 후, 질문 이력에 따라 분기
    builder.add_conditional_edges(
        "set_options",
        check_question_history,
        {
            "first": "first_question",
            "followup": "answer",
        },
    )

    # 3) answer → analyze (답변 분석은 항상 수행)
    builder.add_edge("answer", "analyze")

    # 4) analyze → count 모드 분기
    builder.add_conditional_edges(
        "analyze",
        check_count_mode,
        {
            "dynamic": "keepGoing",   # 동적 모드 → KoELECTRA 판별
            "fixed": "next_question", # 고정 모드 → 바로 다음 질문
        },
    )

    # 5) keepGoing → stop | continue
    builder.add_conditional_edges(
        "keepGoing",
        check_keepGoing,
        {
            "stop": "__end__",
            "continue": "next_question",
        },
    )

    # 6) 종료 지점
    builder.set_finish_point("first_question")
    builder.set_finish_point("next_question")

    return builder.compile()


graph_app = create_graph()