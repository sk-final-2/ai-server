import chromadb
import uuid

# PersistentClient를 사용하여 데이터가 디스크에 저장되도록 수정합니다.
# 경로를 짧고 단순하게 지정하여 Windows 경로 길이 제한 오류를 방지합니다.
try:
    # 절대 경로 또는 상대 경로를 짧게 지정
    client = chromadb.PersistentClient(path="./chroma_data")
    collection = client.get_or_create_collection(name="qa_logs")
except Exception as e:
    print(f"ChromaDB 초기화 실패: {e}. 인메모리 클라이언트로 전환합니다.")
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="qa_logs")

def save_qa_pair(question: str, answer: str):
    doc = f"Q: {question}\nA: {answer}"
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc]
    )

def get_similar_qa(answer: str, k=2):
    results = collection.query(
        query_texts=[answer],
        n_results=k
    )
    return [doc for sublist in results["documents"] for doc in sublist]
