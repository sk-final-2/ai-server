from langgraph.graph import StateGraph, END
from interview.model import InterviewState
from interview.nodes import (
    first_question_node,
    answer_node,
    analyze_node,
    next_question_node,
)

def router_node(state: dict) -> str:
    if state.get("is_finished"):
        print("\U0001F3C1 [router_node] 인터뷰 종료")
        return "__end__"
    elif state.get("step", 0) == 0:
        print("\U0001F9ED [router_node] 첫 질문 생성 흐름")
        return "first_question"
    else:
        print("\U0001F9ED [router_node] 답변 수집 흐름")
        return "answer"

def create_graph():
    print("✅ FSM 컴파일 시작")
    builder = StateGraph(dict)

    builder.add_node("router", router_node)
    builder.add_node("first_question", first_question_node)
    builder.add_node("answer", answer_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("next_question", next_question_node)

    builder.set_entry_point("router")
    builder.add_conditional_edges("router", router_node, {
        "first_question": "first_question",
        "answer": "answer",
        "__end__": END
    })

    builder.add_edge("first_question", "answer")
    builder.add_edge("answer", "analyze")
    builder.add_edge("analyze", "next_question")
    builder.add_edge("next_question", "router")

    return builder.compile()

graph_app = create_graph()