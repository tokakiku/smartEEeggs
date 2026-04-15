from typing import List, Optional

from fastapi import APIRouter

router = APIRouter()

_rag_engine = None
_rag_error: Optional[str] = None

_FALLBACK_DOCS = [
    "TCP three-way handshake is used to establish a connection.",
    "HTTP is an application layer protocol for web data transfer.",
    "IP protocol is responsible for network layer delivery.",
    "DNS resolves domain names to IP addresses.",
]


def _fallback_search(query: str, top_k: int = 2) -> List[str]:
    q = (query or "").strip().lower()
    if not q:
        return _FALLBACK_DOCS[:top_k]

    scored = []
    for doc in _FALLBACK_DOCS:
        score = 2 if q in doc.lower() else 0
        score += sum(1 for token in q.split() if token and token in doc.lower())
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def _search_with_engine(query: str) -> Optional[List[str]]:
    global _rag_engine, _rag_error

    if _rag_error:
        return None

    if _rag_engine is None:
        try:
            from rag.rag_engine import RAGEngine

            _rag_engine = RAGEngine()
        except Exception as exc:
            _rag_error = str(exc)
            return None

    return _rag_engine.search(query)


@router.get("/rag_search")
def rag_search(query: str):
    results = _search_with_engine(query)
    engine = "rag_engine"
    warning = None

    if results is None:
        results = _fallback_search(query=query)
        engine = "fallback"
        warning = _rag_error

    response = {
        "query": query,
        "results": results,
        "engine": engine,
    }
    if warning:
        response["warning"] = warning
    return response
