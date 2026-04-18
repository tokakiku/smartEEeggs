import uuid
from pathlib import Path

import mongomock
import pytest

from services.ingest_service import DocumentIngestService, IngestValidationError
from services.mongo_kb_service import MongoKBService
from services.storage_service import StorageService


def _build_storage() -> StorageService:
    backend_dir = Path(__file__).resolve().parents[2]
    base_dir = backend_dir / "data" / "test_outputs" / f"kb_{uuid.uuid4().hex}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return StorageService(base_dir=base_dir)


def _build_kb() -> MongoKBService:
    client = mongomock.MongoClient()
    return MongoKBService(client=client, db_name="ruijie_kb_test", enabled=True, required=True)


def test_mongo_kb_upsert_and_search():
    kb = _build_kb()

    payload = {
        "hotspot_info": {"title": "OpenClaw renamed", "source": "AIbase"},
        "hotspot_item": [
            {
                "event_type": "product_release",
                "related_knowledge_points": ["大模型应用", "开源社区"],
            }
        ],
    }
    metadata = {
        "doc_id": "doc-hotspot-1",
        "source_file": "aibase_25122.html",
        "source_type": "web_url",
        "parser_name": "unstructured",
    }

    first = kb.save_extraction_result("hotspot", payload, metadata)
    second = kb.save_extraction_result("hotspot", payload, metadata)

    assert first["status"] in {"upserted", "updated"}
    assert second["status"] == "updated"
    assert kb.get_collection_counts()["hotspot"] == 1

    items = kb.search_documents(layer="hotspot", event_type="product_release", knowledge_point="开源社区")
    assert len(items) == 1
    assert items[0]["source_file"] == "aibase_25122.html"

    doc = kb.get_document(layer="hotspot", doc_id="doc-hotspot-1")
    assert doc is not None
    assert doc["data"]["hotspot_info"]["title"] == "OpenClaw renamed"


def test_hotspot_static_source_policy_blocks_dynamic_domain():
    service = DocumentIngestService(storage_service=_build_storage(), kb_service=_build_kb())
    html = b"<html><body><h1>dynamic page</h1><p>demo text</p></body></html>"

    with pytest.raises(IngestValidationError):
        service.ingest_document(
            file_bytes=html,
            file_name="caip_43150.html",
            layer="hotspot",
            source_type="web_url",
            source_name="https://caip.org.cn/news/detail?id=43150",
        )


def test_set_primary_syllabus_switches_same_course_docs():
    kb = _build_kb()

    payload = {
        "course_info": {"course_name": "机器学习", "course_code": "ML101", "applicable_major": ["人工智能"]},
        "course_modules": [{"module_name": "优化基础", "key_points": ["梯度下降法"]}],
    }
    kb.save_extraction_result(
        layer="syllabus",
        payload=payload,
        metadata={
            "doc_id": "syllabus-v1",
            "source_file": "ml_syllabus_2023.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {"course_name": "机器学习", "course_code": "ML101", "academic_year": "2023-2024"},
        },
    )
    kb.save_extraction_result(
        layer="syllabus",
        payload=payload,
        metadata={
            "doc_id": "syllabus-v2",
            "source_file": "ml_syllabus_2024.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {
                "course_name": "机器学习",
                "course_code": "ML101",
                "academic_year": "2024-2025",
                "is_primary": False,
            },
        },
    )

    result = kb.set_primary_syllabus(doc_id="syllabus-v2")
    assert result["status"] == "ok"
    assert result["doc_id"] == "syllabus-v2"

    v2_doc = kb.get_document(layer="syllabus", doc_id="syllabus-v2")
    v1_doc = kb.get_document(layer="syllabus", doc_id="syllabus-v1")
    assert v2_doc is not None and v1_doc is not None
    assert bool((v2_doc.get("syllabus_meta") or {}).get("is_primary")) is True
    assert bool((v1_doc.get("syllabus_meta") or {}).get("is_primary")) is False


def test_textbook_meta_saved_and_switch_main_textbook():
    kb = _build_kb()

    payload = {
        "textbook_info": {
            "book_title": "机器学习（主教材）",
            "subject": "机器学习",
            "textbook_role": "main",
            "edition": "第2版",
            "authors": ["张三"],
        },
        "sections": [{"section_title": "1.2 梯度下降法"}],
        "knowledge_points": [{"name": "梯度下降法"}],
        "chunks": [{"text": "梯度下降法通过负梯度更新参数。"}],
    }
    payload2 = {
        "textbook_info": {
            "book_title": "机器学习习题与解析",
            "subject": "机器学习",
            "textbook_role": "supplementary",
            "edition": "第1版",
            "authors": ["李四"],
        },
        "sections": [{"section_title": "2.1 习题：梯度下降法"}],
        "knowledge_points": [{"name": "梯度下降法"}],
        "chunks": [{"text": "习题册中的梯度下降法练习。"}],
    }

    kb.save_extraction_result(
        layer="textbook",
        payload=payload,
        metadata={
            "doc_id": "textbook-main-v1",
            "source_file": "ml_main_textbook.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload=payload2,
        metadata={
            "doc_id": "textbook-supp-v1",
            "source_file": "ml_supp_textbook.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    doc_main = kb.get_document(layer="textbook", doc_id="textbook-main-v1")
    doc_supp = kb.get_document(layer="textbook", doc_id="textbook-supp-v1")
    assert doc_main is not None and doc_supp is not None
    assert (doc_main.get("textbook_meta") or {}).get("textbook_role") == "main"
    assert (doc_supp.get("textbook_meta") or {}).get("textbook_role") == "supplementary"

    switch = kb.set_primary_textbook(doc_id="textbook-supp-v1", subject="机器学习")
    assert switch["status"] == "ok"
    assert switch["doc_id"] == "textbook-supp-v1"
    assert (switch.get("textbook_meta") or {}).get("textbook_role") == "main"

    doc_main_after = kb.get_document(layer="textbook", doc_id="textbook-main-v1")
    doc_supp_after = kb.get_document(layer="textbook", doc_id="textbook-supp-v1")
    assert doc_main_after is not None and doc_supp_after is not None
    assert (doc_main_after.get("textbook_meta") or {}).get("textbook_role") == "supplementary"
    assert (doc_supp_after.get("textbook_meta") or {}).get("textbook_role") == "main"
