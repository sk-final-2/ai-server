from typing import Literal
from langgraph.graph import StateGraph
from interview.node.question import extract_topics_node, setup_default_topics_node ,first_question_node, next_question_node
from interview.node.analyze import analyze_node, answer_node
from interview.node.control import set_options_node, keepGoing_node, check_keepGoing, start_node, bridge_node
from interview.model import InterviewState

def check_question_history(state: InterviewState) -> Literal["first", "followup"]:
    """질문 이력 유무로 분기"""
    return "first" if len(state.question) == 0 else "followup"


def check_count_mode(state: InterviewState) -> Literal["dynamic", "fixed"]:
    """count 값으로 동적/고정 모드 분기"""
    return "dynamic" if getattr(state, "count", 0) == 0 else "fixed"

def check_bridge(state: InterviewState) -> str:
    print(f"[check_bridge] interviewType={state.interviewType}, count={state.count}, q_len={len(state.questions)}")
    return "bridge" if str(state.interviewType).upper() == "MIXED" else "next_question"

def check_keepGoing(state: InterviewState) -> Literal["stop", "continue"]:
    """keepGoing 플래그로 종료/계속 분기"""
    return "stop" if not getattr(state, "keepGoing", True) else "continue"


def create_graph():
    builder = StateGraph(InterviewState)

    # 노드 등록
    builder.add_node("set_options", set_options_node)
    builder.add_node("default_topic", setup_default_topics_node)
    builder.add_node("first_question", first_question_node)
    builder.add_node("answer", answer_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("keepGoing", keepGoing_node)
    builder.add_node("next_question", next_question_node)
    builder.add_node("start_node", start_node)
    builder.add_node("extract_topics", extract_topics_node)
    builder.add_node("bridge", bridge_node)
    # 1) entry point
    builder.set_entry_point("set_options")

    # 자소서 유무 분기
    builder.add_edge("extract_topics", "first_question")
    builder.add_conditional_edges(
        "start_node",
        lambda state: "with_resume" if getattr(state, "resume", None) else "without_resume",
        {
            "with_resume": "extract_topics",
            "without_resume": "default_topic"
        }
    )
    builder.add_edge("default_topic", "first_question")

    # set_options → start or answer
    builder.add_conditional_edges(
        "set_options",
        check_question_history,
        {
            "first": "start_node",
            "followup": "answer",
        },
    )

    # answer → analyze
    builder.add_edge("answer", "analyze")

    # analyze → dynamic | bridge | next_question
    builder.add_conditional_edges(
        "analyze",
        lambda state: (
            "keepGoing"
            if check_count_mode(state) == "dynamic"
            else check_bridge(state)
        ),
        {
            "keepGoing": "keepGoing",
            "bridge": "bridge",
            "next_question": "next_question",
        },
    )

    # keepGoing → stop | bridge/next_question
    builder.add_conditional_edges(
        "keepGoing",
        lambda state: (
            "bridge" if state.interviewType == "MIXED" and not getattr(state, "bridge_done", False)
            else ("stop" if check_keepGoing(state) == "stop" else "next_question")
        ),
        {
            "stop": "__end__",
            "bridge": "bridge",
            "next_question": "next_question",
        },
    )

    # bridge → next_question
    builder.add_edge("bridge", "next_question")



    # 종료 지점
    builder.set_finish_point("first_question")
    builder.set_finish_point("next_question")

    return builder.compile()


graph_app = create_graph()