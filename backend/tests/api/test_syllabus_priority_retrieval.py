import mongomock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.mongo_kb_service import MongoKBService


def _seed_kb() -> MongoKBService:
    kb = MongoKBService(client=mongomock.MongoClient(), db_name="ruijie_kb_syllabus_priority_test", enabled=True, required=True)

    base_payload = {
        "course_info": {"course_name": "鏈哄櫒瀛︿範", "course_code": "ML101", "applicable_major": ["鏈哄櫒瀛︿範"]},
        "course_modules": [{"module_name": "浼樺寲鍩虹", "key_points": ["姊害涓嬮檷娉?], "difficult_points": ["瀛︿範鐜囬€夋嫨"]}],
        "teaching_schedule": [{"topic": "姊害涓嬮檷娉?}],
    }

    kb.save_extraction_result(
        layer="syllabus",
        payload=base_payload,
        metadata={
            "doc_id": "syllabus-ml-v1",
            "source_file": "ml_syllabus_2023.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {
                "course_name": "鏈哄櫒瀛︿範",
                "course_code": "ML101",
                "school": "绀轰緥澶у",
                "department": "浜哄伐鏅鸿兘瀛﹂櫌",
                "major": "鏈哄櫒瀛︿範",
                "academic_year": "2023-2024",
                "semester": "绉嬪",
                "version": "v1",
                "teacher": "鏁欏笀A",
                "effective_date": "2023-09-01",
                "is_primary": False,
            },
        },
    )
    kb.save_extraction_result(
        layer="syllabus",
        payload=base_payload,
        metadata={
            "doc_id": "syllabus-ml-v2",
            "source_file": "ml_syllabus_2024.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {
                "course_name": "鏈哄櫒瀛︿範",
                "course_code": "ML101",
                "school": "绀轰緥澶у",
                "department": "浜哄伐鏅鸿兘瀛﹂櫌",
                "major": "鏈哄櫒瀛︿範",
                "academic_year": "2024-2025",
                "semester": "绉嬪",
                "version": "v2",
                "teacher": "鏁欏笀B",
                "effective_date": "2024-09-01",
                "is_primary": True,
            },
        },
    )
    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "妯″紡璇嗗埆涓庢満鍣ㄥ涔?, "course_code": "PRML201", "applicable_major": ["妯″紡璇嗗埆"]},
            "course_modules": [{"module_name": "浼樺寲鏂规硶", "key_points": ["姊害涓嬮檷娉?]}],
        },
        metadata={
            "doc_id": "syllabus-prml",
            "source_file": "prml_syllabus_2024.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {
                "course_name": "妯″紡璇嗗埆涓庢満鍣ㄥ涔?,
                "course_code": "PRML201",
                "academic_year": "2024-2025",
                "version": "v1",
                "is_primary": False,
            },
        },
    )

    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "鏈哄櫒瀛︿範", "subject": "鏈哄櫒瀛︿範"},
            "sections": [{"section_title": "2.1 姊害涓嬮檷娉?}],
            "knowledge_points": [{"name": "姊害涓嬮檷娉?}],
            "chunks": [{"text": "姊害涓嬮檷娉曢€氳繃璐熸搴︽柟鍚戣凯浠ｄ紭鍖栨崯澶卞嚱鏁般€?}],
        },
        metadata={
            "doc_id": "textbook-ml-1",
            "source_file": "鏈哄櫒瀛︿範.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    return kb


def _build_client(monkeypatch, kb: MongoKBService) -> TestClient:
    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    return TestClient(app)


def test_syllabus_primary_priority_in_retrieve(monkeypatch):
    kb = _seed_kb()
    client = _build_client(monkeypatch, kb)

    resp = client.get("/kb/retrieve", params={"query": "姊害涓嬮檷娉?, "subject": "鏈哄櫒瀛︿範", "top_k": 5})
    assert resp.status_code == 200
    payload = resp.json()

    syllabus_items = payload["results"]["syllabus"]
    assert len(syllabus_items) >= 2
    assert syllabus_items[0]["doc_id"] == "syllabus-ml-v2"
    assert syllabus_items[0]["is_primary"] is True


def test_set_primary_api_switches_priority(monkeypatch):
    kb = _seed_kb()
    client = _build_client(monkeypatch, kb)

    switch_resp = client.post("/kb/syllabus/syllabus-ml-v1/primary", params={"course_code": "ML101"})
    assert switch_resp.status_code == 200
    assert switch_resp.json()["doc_id"] == "syllabus-ml-v1"

    retrieve_resp = client.get("/kb/retrieve", params={"query": "姊害涓嬮檷娉?, "subject": "鏈哄櫒瀛︿範", "top_k": 5})
    assert retrieve_resp.status_code == 200
    syllabus_items = retrieve_resp.json()["results"]["syllabus"]
    assert syllabus_items[0]["doc_id"] == "syllabus-ml-v1"
    assert syllabus_items[0]["is_primary"] is True


def test_rag_context_prefers_primary_syllabus(monkeypatch):
    kb = _seed_kb()
    client = _build_client(monkeypatch, kb)

    resp = client.get("/kb/rag_search", params={"query": "姊害涓嬮檷娉?, "subject": "鏈哄櫒瀛︿範", "top_k": 5, "max_contexts": 8})
    assert resp.status_code == 200
    payload = resp.json()

    syllabus_contexts = [item for item in payload["contexts"] if item.get("layer") == "syllabus"]
    assert syllabus_contexts
    assert syllabus_contexts[0].get("doc_id") == "syllabus-ml-v2"
    assert syllabus_contexts[0].get("is_primary") is True

