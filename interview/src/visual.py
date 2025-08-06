from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from interview.graph import graph_app  # 네 그래프 불러오기
from langchain_core.runnables.graph import MermaidDrawMethod

with open("graph.png", "wb") as f:
    f.write(
        graph_app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    )
# Mermaid 구조를 PNG 이미지로 시각화
display(
    Image(
        graph_app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    )
)