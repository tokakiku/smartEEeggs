import mongomock
import numpy as np

from services.hybrid_search_service import HybridSearchService
from services.milvus_service import MilvusUnavailableError
from services.mongo_kb_service import MongoKBService


class _FakeEmbeddingService:
    dimension = 4

    def embed_query(self, query: str):
        _ = query
        return np.asarray([[0.11, 0.22, 0.33, 0.44]], dtype=np.float32)


class _FakeMilvusService:
    def __init__(self, hits=None, raise_error: bool = False, has_collection: bool = True):
        self._hits = hits or []
        self._raise_error = raise_error
        self._has_collection = has_collection

    def has_collection(self) -> bool:
        if self._raise_error:
            raise MilvusUnavailableError("milvus offline")
        return self._has_collection

    def search(self, query_vector, top_k: int = 10, layer: str | None = None, **kwargs):
        _ = query_vector
        _ = kwargs
        if self._raise_error:
            raise MilvusUnavailableError("milvus offline")
        rows = self._hits
        if layer:
            rows = [row for row in rows if row.get("layer") == layer]
        return rows[:top_k]


def _build_kb() -> MongoKBService:
    return MongoKBService(
        client=mongomock.MongoClient(),
        db_name="hybrid_search_test",
        enabled=True,
        required=True,
    )


def _seed_docs(kb: MongoKBService) -> None:
    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "Machine Learning", "course_code": "ML101"},
            "teaching_key_points": ["gradient descent", "optimization"],
            "knowledge_points": ["gradient descent", "loss function"],
            "course_modules": [
                {
                    "module_name": "Optimization Basics",
                    "key_points": ["gradient descent"],
                    "difficult_points": ["learning rate tuning"],
                }
            ],
        },
        metadata={
            "doc_id": "sy-1",
            "source_file": "syllabus_ml.pdf",
            "source_type": "local_file",
            "parser_name": "unit_test",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "ML Foundations", "subject": "Machine Learning"},
            "chapters": [{"chapter_id": "ch1", "chapter_title": "Optimization"}],
            "sections": [{"section_id": "sec-1", "chapter_id": "ch1", "section_title": "Gradient Descent"}],
            "knowledge_points": [{"name": "gradient descent"}],
            "chunks": [{"chunk_id": "tb-1-c1", "text": "Gradient descent updates parameters iteratively."}],
        },
        metadata={
            "doc_id": "tb-1",
            "source_file": "textbook_ml.pdf",
            "source_type": "local_file",
            "parser_name": "unit_test",
        },
    )
    kb.save_extraction_result(
        layer="resource",
        payload={
            "resource_info": {"title": "Lecture Slides", "subject": "Machine Learning"},
            "pages": [
                {
                    "page_id": "p1",
                    "page_title": "Gradient Descent Visual",
                    "page_summary": "Visual intuition for gradient descent.",
                    "page_role": "definition_page",
                    "knowledge_points": ["gradient descent"],
                }
            ],
        },
        metadata={
            "doc_id": "rs-1",
            "source_file": "resource_ml.pptx",
            "source_type": "local_file",
            "parser_name": "unit_test",
        },
    )
    kb.save_extraction_result(
        layer="hotspot",
        payload={
            "hotspot_info": {"title": "LLM training trends", "publish_date": "2026-01-10"},
            "hotspot_item": [
                {
                    "title": "Optimizer design for LLM training",
                    "summary": "Practical usage of gradient descent variants.",
                    "related_knowledge_points": ["gradient descent"],
                    "keywords": ["llm", "optimizer"],
                    "event_type": "industry_application",
                }
            ],
        },
        metadata={
            "doc_id": "hs-1",
            "source_file": "hotspot_ml.json",
            "source_type": "local_file",
            "parser_name": "unit_test",
        },
    )


def test_hybrid_search_orchestrates_vector_graph_and_mongo():
    kb = _build_kb()
    _seed_docs(kb)
    milvus = _FakeMilvusService(
        hits=[
            {
                "doc_id": "tb-1",
                "layer": "textbook",
                "chunk_id": "tb-1-c1",
                "source_file": "textbook_ml.pdf",
                "title": "ML Foundations",
                "subject": "Machine Learning",
                "chunk_text": "Gradient descent updates parameters iteratively.",
                "score": 0.92,
            },
            {
                "doc_id": "rs-1",
                "layer": "resource",
                "chunk_id": "p1-summary",
                "source_file": "resource_ml.pptx",
                "title": "Lecture Slides",
                "subject": "Machine Learning",
                "chunk_text": "Visual intuition for gradient descent.",
                "score": 0.77,
            },
        ]
    )
    service = HybridSearchService(
        kb_service=kb,
        embedding_service=_FakeEmbeddingService(),
        milvus_service=milvus,
    )

    payload = service.orchestrate_search(query="gradient descent method", top_k=5)

    assert payload["debug"]["vector_available"] is True
    assert payload["debug"]["graph_available"] is True
    assert len(payload["vector_hits"]) >= 1
    assert len(payload["graph_hits"]["edges"]) >= 1
    assert len(payload["merged_doc_ids"]) >= 1
    assert any(doc.get("doc_id") == "tb-1" for doc in payload["mongo_docs"])
    assert _FakeEmbeddingService.dimension > 0
    assert payload["assembled_context"]["text"]


def test_hybrid_search_graceful_fallback_when_vector_unavailable():
    kb = _build_kb()
    _seed_docs(kb)
    service = HybridSearchService(
        kb_service=kb,
        embedding_service=_FakeEmbeddingService(),
        milvus_service=_FakeMilvusService(raise_error=True),
    )

    payload = service.orchestrate_search(query="gradient descent", top_k=5)

    assert payload["debug"]["vector_available"] is False
    assert "milvus offline" in str(payload["debug"].get("vector_backend_reason"))
    assert payload["debug"]["graph_available"] is True
    assert len(payload["graph_hits"]["edges"]) >= 1
    assert len(payload["mongo_docs"]) >= 1
