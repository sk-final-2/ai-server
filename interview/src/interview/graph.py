from langgraph.graph import StateGraph, END
from src.interview.nodes import (
    first_question_node, answer_node, analyze_node, next_question_node, end_node
)
from src.interview.model import InterviewState

# 최대 질문 수 설정
MAX_SEQ = 3

def is_final(state: InterviewState):
    # questions의 길이가 MAX_SEQ에 도달하면 종료
    return state.seq >= MAX_SEQ

graph_builder = StateGraph(InterviewState)

# 노드 추가
graph_builder.add_node("first_question_node", first_question_node)
graph_builder.add_node("answer_node", answer_node)
graph_builder.add_node("analyze_node", analyze_node)
graph_builder.add_node("next_question_node", next_question_node)
graph_builder.add_node("end_node", end_node)

# 시작점 설정
graph_builder.set_entry_point("first_question_node")

# 첫 질문 생성 -> 사용자 답변 노드
graph_builder.add_edge("first_question_node", "answer_node")
# 사용자 답변 -> 답변 분석 노드
graph_builder.add_edge("answer_node", "analyze_node")
# 답변 분석 -> 다음 질문 생성 노드
graph_builder.add_edge("analyze_node", "next_question_node")

# 다음 질문 생성 후 분기:
# 질문 횟수가 MAX_SEQ보다 작으면 다시 answer_node로 루프
# 질문 횟수가 MAX_SEQ 이상이면 end_node로 이동
graph_builder.add_conditional_edges(
    "next_question_node",
    is_final,
    {
        True: "end_node",
        False: "answer_node"
    }
)

# 종료 노드
graph_builder.add_edge("end_node", END)

# 그래프 컴파일 (한 번만!)
graph_app = graph_builder.compile()