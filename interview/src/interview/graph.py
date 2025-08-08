from langgraph.graph import StateGraph
from interview.model import InterviewState
from interview.nodes import (
    router_node,
    first_question_node,
    answer_node,
    analyze_node,
    next_question_node
)

def create_first_graph():
    builder = StateGraph(InterviewState)
    builder.add_node("first_question", first_question_node)
    builder.set_entry_point("first_question")
    builder.set_finish_point("first_question")
    return builder.compile()

# 후속 질문 FSM
def create_followup_graph():
    builder = StateGraph(InterviewState)
    builder.add_node("answer", answer_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("next_question", next_question_node)
    builder.set_entry_point("answer")
    builder.add_edge("answer", "analyze")
    builder.add_edge("analyze", "next_question")
    builder.set_finish_point("next_question")
    return builder.compile()

    # 4. FSM 컴파일
    #graph_app = builder.compile()
    #print("✅ FSM 생성 완료")
    #return graph_app
first_graph = create_first_graph()
followup_graph = create_followup_graph()
# 사용 예시
if __name__ == "__main__":
    # 초기 상태
    initial_state = {
        "text": "샘플 이력서 텍스트...",
        "job": "웹 개발자",
        "seq": 0,
        "questions": [],
        "answer": [],
        "step": 0,
        "is_finished": False,
        "last_analysis": None
    }
    
    # 그래프 실행
    result = first_graph.invoke(initial_state)
    print("최종 결과:", result)