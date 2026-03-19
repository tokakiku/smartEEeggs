from fastapi import APIRouter
from rag.rag_engine import RAGEngine

router = APIRouter()

rag = RAGEngine()

@router.get("/rag_search")
def rag_search(query: str):

    results = rag.search(query)

    return {
        "query": query,
        "results": results
    }