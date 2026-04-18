import mongomock

from services.hybrid_retrieval_service import HybridRetrievalService
from services.mongo_kb_service import MongoKBService


class _FakeVectorIndex:
    def __init__(self, hits=None, raise_on_search: bool = False):
        self._hits = hits or []
        self._raise_on_search = raise_on_search
        self._active_entries = 3

    def count_active(self) -> int:
        return self._active_entries

    def build_from_mongo(self, kb_service, force=False):
        return {"status": "ok", "added": 0}

    def search(self, query: str, top_k: int = 10, layers=None, min_score: float = 0.0):
        if self._raise_on_search:
            raise RuntimeError("vector unavailable")
        return [item for item in self._hits if item.get("vector_score", 0) >= min_score][:top_k]


def _seed_kb() -> MongoKBService:
    kb = MongoKBService(
        client=mongomock.MongoClient(),
        db_name="ruijie_kb_hybrid_test",
        enabled=True,
        required=True,
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "机器学习", "subject": "机器学习"},
            "sections": [{"section_title": "1.2 假设空间"}],
            "knowledge_points": [{"name": "假设空间"}],
            "chunks": [{"chunk_id": "tb-1", "text": "假设空间定义与版本空间"}],
        },
        metadata={
            "doc_id": "textbook-1",
            "source_file": "机器学习.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="resource",
        payload={
            "resource_info": {"title": "机器学习导论", "subject": "机器学习"},
            "pages": [
                {
                    "page_no": 5,
                    "page_title": "假设空间",
                    "page_summary": "假设空间与归纳偏好",
                    "page_role": "definition_page",
                    "knowledge_points": ["假设空间"],
                }
            ],
            "chunks": [{"chunk_id": "res-1", "text": "PPT 第 5 页解释假设空间"}],
        },
        metadata={
            "doc_id": "resource-1",
            "source_file": "Chap01绪论.pptx",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    return kb


def test_hybrid_retrieval_merges_lexical_and_vector():
    kb = _seed_kb()
    vector_hits = [
        {
            "layer": "textbook",
            "doc_id": "textbook-1",
            "source_file": "机器学习.pdf",
            "title": "机器学习",
            "chunk_id": "tb-1",
            "text": "梯度下降法和假设空间都属于机器学习基础内容",
            "vector_score": 0.72,
        }
    ]
    svc = HybridRetrievalService(kb_service=kb, vector_index=_FakeVectorIndex(hits=vector_hits))
    payload = svc.retrieve_hybrid(query="假设空间", subject="机器学习", top_k=3)

    assert payload["debug"]["retrieval_mode"] == "hybrid"
    assert payload["debug"]["vector_hits"] >= 1
    assert payload["counts"]["textbook"] >= 1

    textbook_item = payload["results"]["textbook"][0]
    assert textbook_item["retrieval_mode"] in {"both", "lexical", "vector"}
    assert "detail_api" in textbook_item
    assert "vector_score" in textbook_item


def test_hybrid_retrieval_falls_back_to_lexical_when_vector_unavailable():
    kb = _seed_kb()
    svc = HybridRetrievalService(kb_service=kb, vector_index=_FakeVectorIndex(raise_on_search=True))
    payload = svc.retrieve_hybrid(query="假设空间", subject="机器学习", top_k=3)

    assert payload["debug"]["retrieval_mode"] == "lexical_fallback"
    assert payload["counts"]["textbook"] >= 1
