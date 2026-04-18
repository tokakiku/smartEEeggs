import mongomock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.mongo_kb_service import MongoKBService


def _seed_kb() -> MongoKBService:
    kb = MongoKBService(client=mongomock.MongoClient(), db_name="ruijie_kb_rag_test", enabled=True, required=True)

    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "Machine Learning"},
            "course_modules": [
                {
                    "module_name": "Foundations",
                    "key_points": ["hypothesis space", "inductive bias"],
                    "difficult_points": ["selecting a suitable hypothesis space"],
                }
            ],
            "teaching_schedule": [{"topic": "hypothesis space"}],
        },
        metadata={
            "doc_id": "syllabus-ml-1",
            "source_file": "ml_syllabus.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {
                "book_title": "ML Intro",
                "subject": "Machine Learning",
                "textbook_role": "main",
                "edition": "2nd",
                "authors": ["Alice"],
            },
            "sections": [{"section_title": "1.2 Hypothesis Space"}],
            "knowledge_points": [{"name": "hypothesis space"}],
            "chunks": [{"text": "Hypothesis space defines candidate functions for learning."}],
        },
        metadata={
            "doc_id": "textbook-ml-1",
            "source_file": "ml_intro.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {
                "book_title": "ML Workbook",
                "subject": "Machine Learning",
                "textbook_role": "supplementary",
                "edition": "1st",
                "authors": ["Bob"],
            },
            "sections": [{"section_title": "Exercise: Hypothesis Space"}],
            "knowledge_points": [{"name": "hypothesis space"}],
            "chunks": [{"text": "Workbook exercises for hypothesis space."}],
        },
        metadata={
            "doc_id": "textbook-ml-2",
            "source_file": "ml_workbook.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    kb.save_extraction_result(
        layer="resource",
        payload={
            "resource_info": {"title": "Lecture 01", "subject": "Machine Learning"},
            "pages": [
                {
                    "page_no": 32,
                    "page_title": "Hypothesis Space",
                    "page_summary": "Definition and examples of hypothesis spaces.",
                    "page_role": "definition_page",
                    "knowledge_points": ["hypothesis space"],
                }
            ],
            "reusable_units": [{"unit_title": "Hypothesis space examples"}],
        },
        metadata={
            "doc_id": "resource-ml-1",
            "source_file": "lecture01.pptx",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    return kb


def test_kb_rag_search_returns_answer_and_contexts(monkeypatch):
    kb = _seed_kb()
    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    resp = client.get("/kb/rag_search", params={"query": "hypothesis space", "top_k": 3})
    assert resp.status_code == 200

    payload = resp.json()
    assert isinstance(payload["answer"], str)
    assert payload["answer"].strip()
    assert len(payload["contexts"]) >= 1
    assert payload["counts"]["textbook"] >= 1

    context = payload["contexts"][0]
    assert "layer" in context
    assert "text" in context
    assert context["text"].strip()
    textbook_contexts = [item for item in payload["contexts_used"] if item.get("layer") == "textbook"]
    assert textbook_contexts
    assert textbook_contexts[0].get("textbook_role") == "main"


def test_kb_rag_search_no_match_still_returns_answer(monkeypatch):
    kb = _seed_kb()
    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    resp = client.get("/kb/rag_search", params={"query": "nonexistent keyword"})
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["contexts"] == []
    assert "鏈绱㈠埌" in payload["answer"]

def test_kb_rag_search_textbook_role_driven_by_syllabus(monkeypatch):
    kb = MongoKBService(client=mongomock.MongoClient(), db_name="ruijie_kb_rag_syllabus_role", enabled=True, required=True)

    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "Machine Learning"},
            "course_modules": [{"module_name": "Optimization", "key_points": ["gradient descent"]}],
            "teaching_materials": {
                "main_textbooks": ["ML Core Textbook"],
                "reference_textbooks": ["ML Workbook"],
            },
        },
        metadata={
            "doc_id": "syllabus-role-1",
            "source_file": "ml_syllabus_role.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {"is_primary": True, "course_name": "Machine Learning"},
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {
                "book_title": "ML Core Textbook",
                "subject": "Machine Learning",
                "edition": "3rd",
                "authors": ["Alice"],
            },
            "sections": [{"section_title": "Gradient Descent"}],
            "knowledge_points": [{"name": "gradient descent"}],
            "chunks": [{"text": "Gradient descent updates parameters along negative gradient direction."}],
        },
        metadata={
            "doc_id": "tb-role-main",
            "source_file": "ml_core_textbook.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {
                "book_title": "ML Workbook",
                "subject": "Machine Learning",
                "edition": "1st",
                "authors": ["Bob"],
            },
            "sections": [{"section_title": "Exercises"}],
            "knowledge_points": [{"name": "gradient descent"}],
            "chunks": [{"text": "Workbook exercises on gradient descent."}],
        },
        metadata={
            "doc_id": "tb-role-supp",
            "source_file": "ml_workbook_role.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    resp = client.get("/kb/rag_search", params={"query": "gradient descent", "subject": "Machine Learning", "top_k": 4})
    assert resp.status_code == 200
    payload = resp.json()

    textbook_contexts = [item for item in payload["contexts_used"] if item.get("layer") == "textbook"]
    assert textbook_contexts
    assert textbook_contexts[0].get("doc_id") == "tb-role-main"
    assert textbook_contexts[0].get("textbook_role") == "main"
    assert textbook_contexts[0].get("textbook_role_source") == "syllabus_main"

