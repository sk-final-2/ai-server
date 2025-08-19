# interview/chroma_setup.py  (SLIM)
import os
import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 🔧 저장 경로
# 🔧 영속 경로: 환경변수 없으면 로컬 ./chroma_data 사용
CHROMA_DIR = os.getenv("CHROMA_DIR", "/app/chroma_data")
CHROMA_DIR = str(Path(CHROMA_DIR))          # 경로 정규화
Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)  # 폴더 보장

# 🔧 디바이스: "cpu" 또는 "cuda"
DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# ✅ 내장 임베딩 함수 객체(충돌 방지용 name() 제공)
EF = SentenceTransformerEmbeddingFunction(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    device=DEVICE,
    # normalize_embeddings=True  # chromadb 버전에 따라 옵션 존재
)

_client = chromadb.PersistentClient(path=CHROMA_DIR)

def get_collections():
    """앱에서 쓸 컬렉션 핸들만 반환 (질문/답변=임베딩, 피드백=키값)"""
    qa_question = _client.get_or_create_collection(
        name="qa_question",
        metadata={"hnsw:space": "cosine"},
        embedding_function=EF,
    )
    qa_answer = _client.get_or_create_collection(
        name="qa_answer",
        metadata={"hnsw:space": "cosine"},
        embedding_function=EF,
    )
    qa_feedback = _client.get_or_create_collection(
        name="qa_feedback"  # 임베딩 불필요
    )
    return qa_question, qa_answer, qa_feedback

def reset_chroma():
    """(개발용) 전역 초기화: 컬렉션 삭제 후 재생성"""
    for name in ["qa_logs", "qa_question", "qa_answer", "qa_feedback"]:
        try:
            _client.delete_collection(name)
        except Exception:
            pass
    get_collections()
    print("🧹 Chroma reset complete (qa_question/qa_answer/qa_feedback)")

def reset_interview(interviewId: str):
    """(운영용) 특정 인터뷰 데이터만 삭제"""
    for name in ["qa_question", "qa_answer", "qa_feedback"]:
        try:
            _client.get_collection(name).delete(where={"interviewId": interviewId})
        except Exception:
            pass
    print(f"🧹 cleared interviewId={interviewId}")
