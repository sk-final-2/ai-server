from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from interview.graph import first_graph, followup_graph   # 나눠진 그래프 불러오기
from interview.test_graph import a_graph
# 시각화 함수
def visualize_graph(graph_app, filename: str):
    png = graph_app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    with open(filename, "wb") as f:
        f.write(png)
    display(Image(png))

# 각각 시각화
visualize_graph(first_graph, "first_graph.png")
visualize_graph(followup_graph, "followup_graph.png")
visualize_graph(a_graph, "aing_graph.png")