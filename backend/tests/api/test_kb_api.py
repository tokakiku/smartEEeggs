import mongomock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.mongo_kb_service import MongoKBService


def _seed_kb() -> MongoKBService:
    kb = MongoKBService(client=mongomock.MongoClient(), db_name="ruijie_kb_api_test", enabled=True, required=True)
    kb.save_extraction_result(
        layer="resource",
        payload={
            "resource_info": {"title": "鏈哄櫒瀛︿範瀵艰", "subject": "鏈哄櫒瀛︿範"},
            "pages": [{"page_role": "definition_page", "knowledge_points": ["鏈哄櫒瀛︿範"]}],
        },
        metadata={
            "doc_id": "resource-doc-1",
            "source_file": "Chap01缁.pptx",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    return kb


def test_kb_search_and_get_doc(monkeypatch):
    kb = _seed_kb()

    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    search_resp = client.get("/kb/search", params={"layer": "resource", "knowledge_point": "鏈哄櫒瀛︿範"})
    assert search_resp.status_code == 200
    payload = search_resp.json()
    assert payload["total"] == 1
    assert payload["items"][0]["source_file"] == "Chap01缁.pptx"

    doc_resp = client.get("/kb/doc/resource/resource-doc-1")
    assert doc_resp.status_code == 200
    doc_payload = doc_resp.json()
    assert doc_payload["doc_id"] == "resource-doc-1"
    assert doc_payload["data"]["resource_info"]["title"] == "鏈哄櫒瀛︿範瀵艰"


def test_set_main_textbook_api(monkeypatch):
    kb = _seed_kb()
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "鏈哄櫒瀛︿範涓绘暀鏉?, "subject": "鏈哄櫒瀛︿範", "textbook_role": "main"},
            "knowledge_points": [{"name": "姊害涓嬮檷娉?}],
            "sections": [{"section_title": "姊害涓嬮檷娉?}],
            "chunks": [{"text": "姊害涓嬮檷娉曞畾涔?}],
        },
        metadata={
            "doc_id": "tb-main",
            "source_file": "ml_main.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "鏈哄櫒瀛︿範涔犻鍐?, "subject": "鏈哄櫒瀛︿範", "textbook_role": "supplementary"},
            "knowledge_points": [{"name": "姊害涓嬮檷娉?}],
            "sections": [{"section_title": "涔犻 姊害涓嬮檷娉?}],
            "chunks": [{"text": "姊害涓嬮檷娉曚範棰?}],
        },
        metadata={
            "doc_id": "tb-supp",
            "source_file": "ml_supp.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    resp = client.post("/kb/textbook/tb-supp/main", params={"subject": "鏈哄櫒瀛︿範"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert payload["doc_id"] == "tb-supp"
    assert payload["textbook_meta"]["textbook_role"] == "main"

    target = kb.get_document(layer="textbook", doc_id="tb-supp")
    old_main = kb.get_document(layer="textbook", doc_id="tb-main")
    assert (target.get("textbook_meta") or {}).get("textbook_role") == "main"
    assert (old_main.get("textbook_meta") or {}).get("textbook_role") == "supplementary"

