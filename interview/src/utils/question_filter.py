import re, random
from typing import List, Tuple
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_STOPWORDS = {"은","는","이","가","을","를","에","에서","으로","와","과","도","만","하며","고","또는","및","등","의","뭘","뭔","무엇","무엇이","무엇을","무엇입니까","무엇이었나요"}

def _normalize(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return t

def keywords(text: str) -> List[str]:
    toks = _normalize(text).lower().split()
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def ngram_overlap(a: List[str], b: List[str], n=3) -> float:
    na = set(_ngrams(a, n))
    nb = set(_ngrams(b, n))
    if not na or not nb:
        return 0.0
    return len(na & nb) / max(1, len(na | nb))


def _tokenize(text: str) -> List[str]:
    """간단한 토큰화: 알파벳/숫자 기준 단어 단위 분리"""
    return re.findall(r"\w+", text.lower())

def lexical_overlap_score(a: str, b: str) -> float:
    """
    두 문장 간의 단순 단어 중복 비율 (Jaccard 유사도).
    값 범위: 0.0 ~ 1.0
    """
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def cosine_similarity_score(a: str, b: str) -> float:
    """
    두 문장 간의 코사인 유사도(TF-IDF 기반).
    값 범위: 0.0 ~ 1.0
    """
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer().fit([a, b])
    mat = vec.transform([a, b])
    sim = cosine_similarity(mat[0], mat[1])[0][0]
    return float(sim)

def init_topics_for_session(all_topics, k_min=3, k_max=5):
    """세션 시작 시 랜덤으로 3~5개 토픽만 선택"""
    if not all_topics:
        return []
    k = random.randint(k_min, min(k_max, len(all_topics)))
    selected = random.sample(all_topics, k)
    # 초기화
    for t in selected:
        t["asked"] = 0
        t["max_questions"] = 3  # 기본 3개씩 물어보기
    return selected