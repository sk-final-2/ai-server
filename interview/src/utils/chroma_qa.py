import re, math
from typing import Optional, List, Dict, Any
from utils.chroma_setup import get_collections, EF

# 같은 EF/경로를 쓰는 컬렉션 핸들 공유
# (앱 시작 시 lifespan에서 reset_chroma() 이후 get_collections()가 1회 호출돼 있으면 더 좋아요)
_question, _answers, _feedback = get_collections()

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _tolist(x):
    # numpy.ndarray -> list, 그 외는 그대로
    try:
        return x.tolist()  # ndarray면 가능
    except Exception:
        return x

def _first_inner_list(x):
    """
    x가 [[...]] 또는 ndarray([[...]]) 형태면 내부 1차 리스트를 꺼내고,
    그 외면 리스트 자체를 반환. 빈값/비리스트면 [].
    """
    x = _tolist(x)
    if not isinstance(x, list):
        return []
    if x and isinstance(x[0], (list, tuple)):
        return list(x[0])
    return x

def _cosine(a, b) -> float:
    num = sum(x*y for x, y in zip(a, b))
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return num / (den + 1e-9)

# -----------------------
# 저장 유틸 (메타 + 결정적 ID)
# -----------------------
def save_question(
    interviewId: str,
    seq: int,
    question: str,
    job: Optional[str] = None,
    level: Optional[str] = None,
    language: Optional[str] = None,
) -> None:
    _id = f"{interviewId}:{seq}:q"
    _question.add(
        ids=[_id],
        documents=[question],
        metadatas=[{
            "interviewId": interviewId,
            "seq": seq,
            "type": "question",
            "job": job, "level": level, "language": language,
        }],
    )

def save_answer(
    interviewId: str,
    seq: int,
    answer: str,
    job: Optional[str] = None,
    level: Optional[str] = None,
    language: Optional[str] = None,
) -> None:
    _id = f"{interviewId}:{seq}:a"
    _answers.add(
        ids=[_id],
        documents=[answer],
        metadatas=[{
            "interviewId": interviewId,
            "seq": seq,
            "type": "answer",
            "job": job, "level": level, "language": language,
        }],
    )

# -----------------------
# 질문 유사도(세션 스코프)
# -----------------------
def get_similar_question(
    interviewId: str,
    question: str,
    k: int = 5,
    min_similarity: float = 0.88,
    verify_all: bool = True,
) -> Dict[str, Any]:
    """
    1) KNN(top-k)로 빠른 체크 → 임계값 넘으면 바로 similar=True
    2) (옵션) 임계값 미달이면 세션 '전체 질문 임베딩' 전수 비교
    return: dict(similar, top_sim, match, method, hits)
    """

    # ---------- 1) KNN 빠른 체크 ----------
    res = _question.query(
        query_texts=[question],
        n_results=k,
        where={"interviewId": interviewId},
        include=["documents", "distances"],
    )
    docs_raw  = res.get("documents", [])
    dists_raw = res.get("distances", [])

    docs  = _first_inner_list(docs_raw)
    dists = _first_inner_list(dists_raw)

    hits: List[Dict[str, Any]] = []
    qn = _norm(question)

    # 길이 기반으로만 체크 (배열 truth 금지)
    if len(docs) and len(dists):
        for doc, dist in zip(docs, dists):
            doc = "" if doc is None else str(doc)
            sim = 1.0 - float(dist)  # cosine
            if _norm(doc) == qn:     # 동일 문장 제외
                continue
            hits.append({"doc": doc, "sim": sim, "dist": float(dist)})

    hits.sort(key=lambda x: x["sim"], reverse=True)

    if len(hits) and hits[0]["sim"] >= min_similarity:
        return {
            "similar": True,
            "top_sim": float(hits[0]["sim"]),
            "match": hits[0]["doc"],
            "method": "knn",
            "hits": hits,
        }

    if not verify_all:
        return {"similar": False, "top_sim": 0.0, "match": None, "method": "knn", "hits": hits}

    # ---------- 2) 전체 전수 비교 ----------
    rows = _question.get(
        where={"interviewId": interviewId},
        include=["documents", "embeddings"],
        limit=10000, offset=0
    )
    all_docs = _tolist(rows.get("documents", [])) or []
    all_embs = _tolist(rows.get("embeddings", [])) or []
    # 일부 버전은 리스트가 이중으로 감싸져 올 수 있음
    if all_docs and isinstance(all_docs[0], list) and isinstance(all_embs[0], list):
        all_docs = all_docs
        all_embs = all_embs
    elif all_docs and not isinstance(all_docs[0], list):
        # 평평하면 OK
        pass

    if not all_docs:
        return {"similar": False, "top_sim": 0.0, "match": None, "method": "all", "hits": hits}

    qvec = EF([question])[0]
    qvec = _tolist(qvec)

    best_sim, best_doc = 0.0, None
    for doc, emb in zip(all_docs, all_embs):
        if emb is None or (hasattr(emb, "__len__") and len(emb) == 0):
            continue
        emb = _tolist(emb)
        doc = "" if doc is None else str(doc)
        if _norm(doc) == qn:
            continue
        sim = _cosine(qvec, emb)
        if sim > best_sim:
            best_sim, best_doc = sim, doc

    return {
        "similar": best_sim >= min_similarity,
        "top_sim": float(best_sim),
        "match": best_doc,
        "method": "all",
        "hits": hits,
    }

# -----------------------
# 피드백 저장/조회
# -----------------------
def save_feedback(
    interviewId: str,
    seq: int,
    good: str,
    bad: str,
    score: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """한 질문/답변당 피드백 1건 → 같은 id면 업데이트"""
    _id = f"{interviewId}:{seq}:f"
    score = max(0, min(100, int(score) if isinstance(score, (int, str)) else 0))
    meta = {
        "interviewId": interviewId,
        "seq": seq,
        "good": good,
        "bad": bad,
        "score": score,
    }
    if extra_meta:
        meta.update(extra_meta)

    try:
        existed = _feedback.get(ids=[_id])
        if existed and existed.get("ids"):
            _feedback.update(
                ids=[_id],
                metadatas=[meta],
                documents=[f"good:{good}\nbad:{bad}\nscore:{score}"],
            )
            return
    except Exception:
        pass

    _feedback.add(
        ids=[_id],
        metadatas=[meta],
        documents=[f"good:{good}\nbad:{bad}\nscore:{score}"],
    )

def list_feedback(interviewId: str) -> List[Dict[str, Any]]:
    res = _feedback.get(
        where={"interviewId": interviewId},
        include=["metadatas"],
        limit=10000,
        offset=0,
    )
    items = res.get("metadatas") or []
    items.sort(key=lambda x: x.get("seq", 0))
    return items