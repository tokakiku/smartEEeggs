import mongomock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.mongo_kb_service import MongoKBService


def _seed_kb() -> MongoKBService:
    kb = MongoKBService(client=mongomock.MongoClient(), db_name="ruijie_kb_retrieve_test", enabled=True, required=True)
    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "鏈哄櫒瀛︿範"},
            "course_modules": [
                {
                    "module_name": "瀛︿範鐞嗚鍩虹",
                    "key_points": ["鍋囪绌洪棿", "褰掔撼鍋忓ソ"],
                    "difficult_points": ["鍋囪绌洪棿閫夋嫨"],
                }
            ],
            "teaching_schedule": [{"topic": "鍋囪绌洪棿"}],
            "teaching_materials": {
                "main_textbooks": ["鏈哄櫒瀛︿範锛堜富鏁欐潗锛?, "鏈哄櫒瀛︿範.pdf"],
                "reference_textbooks": ["鏈哄櫒瀛︿範涔犻涓庤В鏋?, "鏈哄櫒瀛︿範涔犻.pdf"],
            },
            "textbooks": ["鏈哄櫒瀛︿範锛堜富鏁欐潗锛?, "鏈哄櫒瀛︿範涔犻涓庤В鏋?],
        },
        metadata={
            "doc_id": "syllabus-1",
            "source_file": "鏁欏澶х翰.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {
                "is_primary": True,
                "course_name": "鏈哄櫒瀛︿範",
                "course_code": "ML101",
                "academic_year": "2024-2025",
                "version": "2024鐗?,
            },
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {
                "book_title": "鏈哄櫒瀛︿範锛堜富鏁欐潗锛?,
                "subject": "鏈哄櫒瀛︿範",
                "edition": "绗?鐗?,
                "authors": ["寮犱笁"],
            },
            "sections": [{"section_title": "1.2 鍋囪绌洪棿"}],
            "knowledge_points": [{"name": "鍋囪绌洪棿"}],
            "chunks": [{"text": "鍋囪绌洪棿瀹氫箟涓庣増鏈┖闂?}],
        },
        metadata={
            "doc_id": "textbook-1",
            "source_file": "鏈哄櫒瀛︿範.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {
                "book_title": "鏈哄櫒瀛︿範涔犻涓庤В鏋?,
                "subject": "鏈哄櫒瀛︿範",
                "edition": "绗?鐗?,
                "authors": ["鏉庡洓"],
            },
            "sections": [{"section_title": "涔犻锛氬亣璁剧┖闂?}],
            "knowledge_points": [{"name": "鍋囪绌洪棿"}],
            "chunks": [{"text": "閫氳繃涔犻鐞嗚В鍋囪绌洪棿銆?}],
        },
        metadata={
            "doc_id": "textbook-2",
            "source_file": "鏈哄櫒瀛︿範涔犻.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="resource",
        payload={
            "resource_info": {"title": "鏈哄櫒瀛︿範瀵艰", "subject": "鏈哄櫒瀛︿範"},
            "pages": [
                {
                    "page_no": 5,
                    "page_title": "鍋囪绌洪棿",
                    "page_summary": "鍋囪绌洪棿涓庡綊绾冲亸濂?,
                    "page_role": "definition_page",
                    "knowledge_points": ["鍋囪绌洪棿"],
                }
            ],
            "reusable_units": [{"unit_title": "鍋囪绌洪棿璁茶В鍗曞厓"}],
        },
        metadata={
            "doc_id": "resource-1",
            "source_file": "Chap01缁.pptx",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="hotspot",
        payload={
            "hotspot_info": {"title": "OpenClaw 鏇村悕", "url": "https://news.aibase.com/news/25122"},
            "hotspot_item": [
                {
                    "summary": "OpenClaw 鏇村悕骞剁獊鐮?10 涓?stars",
                    "event_type": "product_release",
                    "related_knowledge_points": ["澶фā鍨嬪簲鐢?, "寮€婧愮ぞ鍖?],
                    "keywords": ["OpenClaw", "GitHub"],
                    "teaching_usage": ["discussion"],
                }
            ],
            "chunks": [{"text": "OpenClaw renamed and exceeded 100k stars."}],
        },
        metadata={
            "doc_id": "hotspot-1",
            "source_file": "aibase_25122_raw.html",
            "source_type": "web_url",
            "parser_name": "unstructured",
        },
    )
    return kb


def test_kb_retrieve_across_layers(monkeypatch):
    kb = _seed_kb()
    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    resp = client.get("/kb/retrieve", params={"query": "鍋囪绌洪棿", "top_k": 3})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["counts"]["syllabus"] >= 1
    assert payload["counts"]["textbook"] >= 1
    assert payload["counts"]["resource"] >= 1

    syllabus_item = payload["results"]["syllabus"][0]
    assert "matched_modules" in syllabus_item
    assert "detail_api" in syllabus_item

    textbook_item = payload["results"]["textbook"][0]
    assert "matched_sections" in textbook_item
    assert "matched_knowledge_points" in textbook_item
    assert textbook_item.get("textbook_role") == "main"
    assert textbook_item.get("textbook_role_source") == "syllabus_main"

    resource_item = payload["results"]["resource"][0]
    assert "matched_pages" in resource_item
    assert "matched_page_roles" in resource_item


def test_kb_retrieve_hotspot_static_only(monkeypatch):
    kb = _seed_kb()
    kb.save_extraction_result(
        layer="hotspot",
        payload={
            "hotspot_info": {"title": "CAIP 鍔ㄦ€侀〉闈?, "url": "https://caip.org.cn/news/detail?id=43150"},
            "hotspot_item": [
                {
                    "summary": "鍔ㄦ€佺綉椤垫渚?,
                    "event_type": "other_event",
                    "related_knowledge_points": ["澶фā鍨嬪簲鐢?],
                }
            ],
        },
        metadata={
            "doc_id": "hotspot-caip",
            "source_file": "caip_43150.html",
            "source_type": "web_url",
            "parser_name": "unstructured",
        },
    )

    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    resp = client.get("/kb/retrieve", params={"query": "澶фā鍨嬪簲鐢?, "layers": "hotspot"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["counts"]["hotspot"] == 1
    assert payload["results"]["hotspot"][0]["source_file"] == "aibase_25122_raw.html"


def test_kb_retrieve_textbook_fallback_when_syllabus_has_no_materials(monkeypatch):
    kb = MongoKBService(client=mongomock.MongoClient(), db_name="ruijie_kb_retrieve_fallback", enabled=True, required=True)
    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "鏈哄櫒瀛︿範"},
            "course_modules": [{"module_name": "浼樺寲鍩虹", "key_points": ["姊害涓嬮檷娉?]}],
        },
        metadata={
            "doc_id": "syllabus-fallback-1",
            "source_file": "fallback_syllabus.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {"is_primary": True, "course_name": "鏈哄櫒瀛︿範"},
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "鏈哄櫒瀛︿範鏁欐潗", "subject": "鏈哄櫒瀛︿範"},
            "sections": [{"section_title": "姊害涓嬮檷娉?}],
            "knowledge_points": [{"name": "姊害涓嬮檷娉?}],
            "chunks": [{"text": "姊害涓嬮檷娉曠敤浜庡弬鏁颁紭鍖栥€?}],
        },
        metadata={
            "doc_id": "tb-fallback-main",
            "source_file": "ml_core.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "鏈哄櫒瀛︿範涔犻鍐?, "subject": "鏈哄櫒瀛︿範"},
            "sections": [{"section_title": "涔犻锛氭搴︿笅闄嶆硶"}],
            "knowledge_points": [{"name": "姊害涓嬮檷娉?}],
            "chunks": [{"text": "姊害涓嬮檷娉曚範棰樿缁冦€?}],
        },
        metadata={
            "doc_id": "tb-fallback-supp",
            "source_file": "ml_workbook.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    resp = client.get("/kb/retrieve", params={"query": "姊害涓嬮檷娉?, "layers": "textbook"})
    assert resp.status_code == 200
    payload = resp.json()
    items = payload["results"]["textbook"]
    assert len(items) >= 2
    assert items[0]["doc_id"] == "tb-fallback-main"
    assert items[0].get("textbook_role") in {"main", None}
    assert items[1]["doc_id"] == "tb-fallback-supp"
    assert items[1].get("textbook_role") == "supplementary"


def test_kb_retrieve_multi_syllabus_uses_best_syllabus_for_textbook_role(monkeypatch):
    kb = MongoKBService(client=mongomock.MongoClient(), db_name="ruijie_kb_retrieve_multi_syllabus", enabled=True, required=True)
    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "鏈哄櫒瀛︿範"},
            "teaching_materials": {"main_textbooks": ["鏈哄櫒瀛︿範A"], "reference_textbooks": ["鏈哄櫒瀛︿範B"]},
        },
        metadata={
            "doc_id": "syllabus-ml-2024",
            "source_file": "ml_2024.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {"is_primary": True, "course_name": "鏈哄櫒瀛︿範", "academic_year": "2024-2025"},
        },
    )
    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "妯″紡璇嗗埆涓庢満鍣ㄥ涔?},
            "teaching_materials": {"main_textbooks": ["妯″紡璇嗗埆涓庢満鍣ㄥ涔燙"]},
        },
        metadata={
            "doc_id": "syllabus-prml",
            "source_file": "prml.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
            "syllabus_meta": {"is_primary": False, "course_name": "妯″紡璇嗗埆涓庢満鍣ㄥ涔?, "academic_year": "2023-2024"},
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "鏈哄櫒瀛︿範A", "subject": "鏈哄櫒瀛︿範"},
            "sections": [{"section_title": "姊害涓嬮檷娉?}],
            "knowledge_points": [{"name": "姊害涓嬮檷娉?}],
            "chunks": [{"text": "鏈哄櫒瀛︿範A 涓殑姊害涓嬮檷娉曘€?}],
        },
        metadata={
            "doc_id": "tb-ml-A",
            "source_file": "ml_a.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )
    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "鏈哄櫒瀛︿範B", "subject": "鏈哄櫒瀛︿範"},
            "sections": [{"section_title": "涔犻"}],
            "knowledge_points": [{"name": "姊害涓嬮檷娉?}],
            "chunks": [{"text": "鏈哄櫒瀛︿範B 涔犻銆?}],
        },
        metadata={
            "doc_id": "tb-ml-B",
            "source_file": "ml_b.pdf",
            "source_type": "local_file",
            "parser_name": "unstructured",
        },
    )

    import api.course_api as kb_api_module

    monkeypatch.setattr(kb_api_module, "kb_service", kb)
    app = FastAPI()
    app.include_router(kb_api_module.router)
    client = TestClient(app)

    resp = client.get("/kb/retrieve", params={"query": "鏈哄櫒瀛︿範", "top_k": 5})
    assert resp.status_code == 200
    payload = resp.json()
    textbook_items = payload["results"]["textbook"]
    assert textbook_items[0]["doc_id"] == "tb-ml-A"
    assert textbook_items[0].get("textbook_role") == "main"
    assert textbook_items[0].get("textbook_role_source") == "syllabus_main"

