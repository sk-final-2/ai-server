# main.py
from .graph import create_graph

if __name__ == "__main__":
    app = create_graph()
    state = {
        "step": 1,
        "questions": [],
        "answers": [],
        "last_analysis": "",
        "is_finished": False
    }

    print("\n🤖 면접을 시작합니다!\n")

    while not state["is_finished"]:
        result = app.invoke(state)
        state = result
