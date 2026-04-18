import mongomock

from services.cross_layer_retrieval_service import CrossLayerRetrievalService
from services.mongo_kb_service import MongoKBService


def _build_kb() -> MongoKBService:
    return MongoKBService(client=mongomock.MongoClient(), db_name="ruijie_kb_priority_test", enabled=True, required=True)


def test_textbook_priority_prefers_main_book():
    kb = _build_kb()
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {
                "book_title": "Machine Learning Main Textbook",
                "subject": "Machine Learning",
                "textbook_role": "main",
            },
            "sections": [{"section_title": "1.2 Hypothesis Space"}],
            "knowledge_points": [{"name": "hypothesis space"}],
            "chunks": [{"text": "Hypothesis space defines candidate functions."}],
        },
        metadata={
            "doc_id": "textbook-main",
            "source_file": "ml_main_2025.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {
                "book_title": "Machine Learning Workbook",
                "subject": "Machine Learning",
                "textbook_role": "supplementary",
            },
            "sections": [{"section_title": "2.1 Hypothesis Space Exercises"}],
            "knowledge_points": [{"name": "hypothesis space"}],
            "chunks": [{"text": "Workbook section on hypothesis space."}],
        },
        metadata={
            "doc_id": "textbook-workbook",
            "source_file": "ml_workbook_2018.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    service = CrossLayerRetrievalService(kb_service=kb)
    results = service.retrieve_across_layers(
        query="hypothesis space",
        subject="Machine Learning",
        layers=["textbook"],
        top_k=5,
    )
    assert len(results["textbook"]) >= 2
    assert results["textbook"][0]["doc_id"] == "textbook-main"


def test_resource_priority_prefers_core_slides():
    kb = _build_kb()
    kb.save_extraction_result(
        layer="resource",
        payload={
            "resource_info": {"title": "ML Lecture 2025", "subject": "Machine Learning"},
            "pages": [
                {
                    "page_no": 4,
                    "page_title": "Hypothesis Space",
                    "page_summary": "Definition and formulation of hypothesis space.",
                    "page_role": "definition_page",
                    "knowledge_points": ["hypothesis space"],
                }
            ],
            "reusable_units": [{"unit_title": "Hypothesis space definition"}],
        },
        metadata={
            "doc_id": "resource-main",
            "source_file": "lecture_2025.pptx",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="resource",
        payload={
            "resource_info": {"title": "ML Exercise Slides", "subject": "Machine Learning"},
            "pages": [
                {
                    "page_no": 7,
                    "page_title": "Hypothesis Space Exercises",
                    "page_summary": "Practice questions for hypothesis space.",
                    "page_role": "exercise_page",
                    "knowledge_points": ["hypothesis space"],
                }
            ],
            "reusable_units": [{"unit_title": "Exercise set"}],
        },
        metadata={
            "doc_id": "resource-exercise",
            "source_file": "exercise_2025.pptx",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    service = CrossLayerRetrievalService(kb_service=kb)
    results = service.retrieve_across_layers(
        query="hypothesis space",
        subject="Machine Learning",
        layers=["resource"],
        top_k=5,
    )
    assert len(results["resource"]) >= 2
    assert results["resource"][0]["doc_id"] == "resource-main"


def test_hotspot_priority_deduplicates_same_event_and_prefers_recent():
    kb = _build_kb()
    kb.save_extraction_result(
        layer="hotspot",
        payload={
            "hotspot_info": {
                "title": "OpenClaw renamed and passed 100k stars",
                "publish_date": "2026-04-10",
                "url": "https://news.aibase.com/news/25122",
            },
            "hotspot_item": [
                {
                    "summary": "OpenClaw renamed and passed 100k stars in GitHub community.",
                    "event_type": "product_release",
                    "related_knowledge_points": ["AI product ecosystem"],
                    "keywords": ["OpenClaw", "GitHub"],
                    "teaching_usage": ["discussion"],
                }
            ],
            "chunks": [{"text": "OpenClaw renamed and reached 100k stars."}],
        },
        metadata={
            "doc_id": "hotspot-recent",
            "source_file": "openclaw_recent.html",
            "source_type": "web_url",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="hotspot",
        payload={
            "hotspot_info": {
                "title": "OpenClaw renamed and passed 100k stars",
                "publish_date": "2025-01-10",
                "url": "https://news.example.com/openclaw",
            },
            "hotspot_item": [
                {
                    "summary": "OpenClaw renamed and passed 100k stars in GitHub community.",
                    "event_type": "product_release",
                    "related_knowledge_points": ["AI product ecosystem"],
                    "keywords": ["OpenClaw", "GitHub"],
                    "teaching_usage": ["extended_reading"],
                }
            ],
            "chunks": [{"text": "OpenClaw renamed and reached 100k stars."}],
        },
        metadata={
            "doc_id": "hotspot-old-duplicate",
            "source_file": "openclaw_old.html",
            "source_type": "web_url",
            "parser_name": "unstructured",
        },
    )

    service = CrossLayerRetrievalService(kb_service=kb)
    results = service.retrieve_across_layers(
        query="OpenClaw",
        layers=["hotspot"],
        top_k=5,
    )
    assert len(results["hotspot"]) == 1
    assert results["hotspot"][0]["doc_id"] == "hotspot-recent"
