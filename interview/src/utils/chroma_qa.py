import re, math, json, time 
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

def _sanitize_metadata(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        if v is None:
            out[k] = ""  # None 방지
        elif isinstance(v, bool):
            out[k] = bool(v)
        elif isinstance(v, int):
            out[k] = int(v)
        elif isinstance(v, float):
            # NaN/inf 방지
            out[k] = 0.0 if (math.isnan(v) or math.isinf(v)) else float(v)
        elif isinstance(v, str):
            out[k] = v
        else:
            # list/dict 등은 문자열로
            out[k] = json.dumps(v, ensure_ascii=False)
    return out

def _build_where(interviewId: str, subtype: str | None, job: str | None, min_seq: int | None):
    conds = [{"interviewId": interviewId}]
    if subtype:  # 빈 문자열이면 넣지 않기
        conds.append({"subtype": subtype})
    if job:
        conds.append({"job": job})
    if isinstance(min_seq, int):
        conds.append({"seq": {"$gte": int(min_seq)}})

    if len(conds) == 1:
        return conds[0]            # 조건 1개면 그대로
    return {"$and": conds}         # 2개 이상이면 $and로 감싸기
# -----------------------
# 저장 유틸 (메타 + 결정적 ID)
# -----------------------
def save_question(
    interviewId: str,
    seq: int,
    question: str,
    job: str | None = None,
    level: str | None = None,
    language: str | None = None,
    topic: str | None = None,
    aspect: str | None = None,
    subtype: str | None = None,
) -> None:
    _id = f"{interviewId}:{int(seq)}:q"
    meta_raw = {
        "interviewId": interviewId or "",
        "seq": int(seq),
        "type": "question",
        "job": job or "",
        "level": level or "",
        "language": language or "",
        "topic": topic or "",
        "aspect": aspect or "",
        "subtype": subtype or "",
        "timestamp": time.time(),  # float OK
    }
    meta = _sanitize_metadata(meta_raw)
    print("🧾[save_question meta]", {k: (type(v).__name__, v) for k, v in meta.items()})  # 디버그

    _question.add(
        ids=[_id],
        documents=[question or ""],   # 문서도 None 방지
        metadatas=[meta],             # ← 반드시 sanitize 후 넣기
    )

def save_answer(
    interviewId: str,
    seq: int,
    answer: str,
    job: Optional[str] = None,
    level: Optional[str] = None,
    language: Optional[str] = None,
    subtype: Optional[str] = None,
) -> None:
    _id = f"{interviewId}:{seq}:a"
    _answers.add(
        ids=[_id],
        documents=[answer],
        metadatas=[{
            "interviewId": interviewId,
            "seq": seq,
            "type": "answer",
            "job": job, "level": level, "language": language,"subtype": subtype or "",
        }],
    )
# -----------------------
# 질문-답변-피드백 턴 저장
def save_turn(
    interviewId: str,
    seq: int,
    question: str,
    answer: str,
    good: str = "",
    bad: str = "",
    score: int = 0,
    topic: str = "",
    aspect: str = "",
    job: Optional[str] = None,
    level: Optional[str] = None,
    language: Optional[str] = None,
    feedback: Optional[Dict[str, Any]] = None,
    subtype: Optional[str] = None,
) -> None:
    """질문-답변-피드백 전체 턴 저장"""
    _id = f"{interviewId}:{seq}:t"

    meta = {
        "interviewId": interviewId,
        "seq": seq,
        "type": "turn",
        "question": question or "",
        "answer": answer or "",
        "good": good or "",
        "bad": bad or "",
        "score": max(0, min(100, int(score) if isinstance(score, (int, str)) else 0)),
        "topic": topic or "",
        "aspect": aspect or "",
        "job": job or "",
        "level": level or "",
        "language": language or "",
        "subtype": subtype or "",
        "feedback": json.dumps(feedback) if feedback else "",
    }

    doc = f"Q: {question}\nA: {answer}\n[Feedback] good: {good} | bad: {bad} | score: {score}"

    try:
        existed = _feedback.get(ids=[_id])
        if existed and existed.get("ids"):
            _feedback.update(
                ids=[_id],
                metadatas=[meta],
                documents=[doc],
            )
            print(f"🔄 [Turn Update] interviewId={interviewId}, seq={seq}")
            print("   질문:", question)
            print("   답변:", answer)
            print("   토픽:", topic)
            print("   측면(aspect):", aspect)
            print("   피드백:", feedback)
            return
    except Exception:
        pass

    _feedback.add(
        ids=[_id],
        metadatas=[meta],
        documents=[doc],
    )
    print(f"💾 [Turn Saved] interviewId={interviewId}, seq={seq}")
    print("   질문:", question)
    print("   답변:", answer)
    print("   토픽:", topic)
    print("   측면(aspect):", aspect)
    print("   피드백:", feedback)
# -----------------------
# 질문 유사도(세션 스코프)
# -----------------------
def get_similar_question(
    interviewId: str,
    question: str,
    k: int = 10,
    min_similarity: float = 0.75,
    verify_all: bool = False,
    *,
    subtype: str | None = None,
    job: str | None = None,
    min_seq: int | None = None,
) -> Dict[str, Any]:
    """
    1) KNN(top-k)로 빠른 체크 → 임계값 넘으면 바로 similar=True
    2) (옵션) 임계값 미달이면 같은 where 범위에서 '전수 비교'
    return: dict(similar, top_sim, match, method, hits)
    """

    print(f"\n🔎 [DEBUG:get_similar_question] interviewId={interviewId}")
    print(f"   ▶︎ New Question: {question}")

    # ---------- where 필터 구성 ----------
    where = _build_where(interviewId, subtype, job, min_seq)

    # ---------- 1) KNN 빠른 체크 ----------
    res = _question.query(
        query_texts=[question],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    ) or {}

    docs_raw  = res.get("documents", [])
    dists_raw = res.get("distances", [])
    docs  = _first_inner_list(docs_raw)
    dists = _first_inner_list(dists_raw)

    hits: List[Dict[str, Any]] = []
    qn = _norm(question)

    if docs and dists:
        for doc, dist in zip(docs, dists):
            doc  = "" if doc  is None else str(doc)
            dist = 1.0 if dist is None else float(dist)
            sim  = 1.0 - dist  # cosine_similarity = 1 - cosine_distance
            if _norm(doc) == qn:  # 완전 동일 문장은 제외
                continue
            hits.append({"doc": doc, "sim": sim, "dist": dist})

    hits.sort(key=lambda x: x["sim"], reverse=True)

    if hits:
        print("   ▶︎ KNN hits:")
        for h in hits:
            print(f"      - '{h['doc']}' | sim={h['sim']:.4f}")

    if hits and hits[0]["sim"] >= float(min_similarity):
        print(f"✅ KNN Top match: {hits[0]['doc']} (sim={hits[0]['sim']:.4f}) >= {min_similarity}")
        return {
            "similar": True,
            "top_sim": float(hits[0]["sim"]),
            "match": hits[0]["doc"],
            "method": "knn",
            "hits": hits,
        }

    if not verify_all:
        return {"similar": False, "top_sim": 0.0, "match": None, "method": "knn", "hits": hits}

    # ---------- 2) 같은 where 범위에서 전수 비교 ----------
    rows = _question.get(
        where=where,
        include=["documents", "embeddings"],
        limit=10000, offset=0
    ) or {}

    all_docs = _tolist(rows.get("documents", [])) or []
    all_embs = _tolist(rows.get("embeddings", [])) or []

    if not all_docs:
        print("⚠️ No documents found for this where-filter in Chroma")
        return {"similar": False, "top_sim": 0.0, "match": None, "method": "all", "hits": hits}

    qvec = EF([question])[0]
    qvec = _tolist(qvec)

    best_sim, best_doc = 0.0, None
    for doc, emb in zip(all_docs, all_embs):
        if not emb:
            continue
        emb = _tolist(emb)
        doc = "" if doc is None else str(doc)
        if _norm(doc) == qn:
            continue
        sim = _cosine(qvec, emb)
        if sim > best_sim:
            best_sim, best_doc = sim, doc

    if best_doc:
        print(f"   ▶︎ Best overall match: '{best_doc}' (sim={best_sim:.4f})")

    return {
        "similar": best_sim >= float(min_similarity),
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
