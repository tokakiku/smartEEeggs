from pathlib import Path

import mongomock

from extractors.graph_triple_extractor import GraphTripleExtractor
from services.graph_extraction_service import GraphExtractionService
from services.graph_store_service import GraphStoreService
from services.mongo_kb_service import MongoKBService


def _build_kb() -> MongoKBService:
    client = mongomock.MongoClient()
    return MongoKBService(client=client, db_name="graph_pipeline_test", enabled=True, required=True)


def _seed_docs(kb: MongoKBService) -> None:
    kb.save_extraction_result(
        layer="syllabus",
        payload={
            "course_info": {"course_name": "机器学习", "course_code": "ML101"},
            "teaching_goals": ["理解机器学习基本概念"],
            "teaching_key_points": ["机器学习", "梯度下降"],
            "teaching_difficult_points": ["偏差方差"],
            "knowledge_points": ["机器学习", "梯度下降"],
            "course_modules": [
                {
                    "module_name": "监督学习",
                    "description": "监督学习基本方法",
                    "key_points": ["机器学习", "梯度下降"],
                    "difficult_points": ["偏差方差"],
                    "learning_requirements": ["理解损失函数"],
                }
            ],
        },
        metadata={
            "doc_id": "syllabus-doc-1",
            "source_file": "syllabus.pdf",
            "source_type": "local_file",
            "parser_name": "unit_test",
        },
    )

    kb.save_extraction_result(
        layer="textbook",
        payload={
            "textbook_info": {"book_title": "机器学习教程", "subject": "机器学习"},
            "chapters": [
                {
                    "chapter_id": "ch1",
                    "chapter_title": "监督学习基础",
                    "sections": [
                        {
                            "section_id": "ch1-sec1",
                            "chapter_id": "ch1",
                            "section_title": "梯度下降算法",
                            "raw_text": "学习率和迭代步长是常见超参数",
                            "knowledge_points": ["梯度下降", "机器学习"],
                        }
                    ],
                }
            ],
            "sections": [],
            "knowledge_points": [{"name": "梯度下降", "chapter_id": "ch1"}],
            "relations": [{"source": "梯度下降", "target": "机器学习", "relation": "related_to", "confidence": 0.9}],
        },
        metadata={
            "doc_id": "textbook-doc-1",
            "source_file": "textbook.pdf",
            "source_type": "local_file",
            "parser_name": "unit_test",
        },
    )

    kb.save_extraction_result(
        layer="resource",
        payload={
            "resource_info": {"title": "机器学习导论", "subject": "机器学习", "course_topic": "机器学习"},
            "pages": [
                {
                    "page_id": "page-1",
                    "page_title": "梯度下降直观解释",
                    "page_summary": "通过可视化理解梯度下降",
                    "page_role": "definition_page",
                    "knowledge_points": ["梯度下降", "机器学习"],
                }
            ],
            "reusable_units": [
                {
                    "unit_id": "unit-1",
                    "page_id": "page-1",
                    "unit_title": "梯度下降讲解单元",
                    "unit_summary": "讲解梯度下降",
                    "knowledge_points": ["梯度下降"],
                    "recommended_use": ["课堂讨论"],
                }
            ],
            "relations": [{"source": "page-1", "target": "梯度下降", "relation": "related_to", "confidence": 0.9}],
        },
        metadata={
            "doc_id": "resource-doc-1",
            "source_file": "resource.pptx",
            "source_type": "local_file",
            "parser_name": "unit_test",
        },
    )

    kb.save_extraction_result(
        layer="hotspot",
        payload={
            "hotspot_info": {"title": "自动驾驶中的机器学习应用", "publish_date": "2026-01-01"},
            "hotspot_item": [
                {
                    "title": "自动驾驶中的机器学习应用",
                    "summary": "自动驾驶落地案例",
                    "event_type": "industry_application",
                    "related_knowledge_points": ["机器学习"],
                    "keywords": ["机器学习", "自动驾驶"],
                    "teaching_usage": ["discussion"],
                }
            ],
            "relations": [{"source": "自动驾驶中的机器学习应用", "target": "机器学习", "relation": "related_to"}],
        },
        metadata={
            "doc_id": "hotspot-doc-1",
            "source_file": "hotspot.json",
            "source_type": "local_file",
            "parser_name": "unit_test",
        },
    )


def test_graph_pipeline_minimal(tmp_path: Path) -> None:
    kb = _build_kb()
    _seed_docs(kb)

    extraction_service = GraphExtractionService(kb_service=kb, triple_extractor=GraphTripleExtractor())
    extraction_result = extraction_service.extract_from_mongo()

    assert extraction_result["docs_by_layer"] == {
        "syllabus": 1,
        "textbook": 1,
        "resource": 1,
        "hotspot": 1,
    }
    triples = extraction_result["triples"]
    assert len(triples) > 0

    graph_store = GraphStoreService()
    graph_store.add_triples(triples)
    cross_edges = graph_store.build_cross_layer_edges(min_score=0.9, max_matches_per_source=5)

    assert graph_store.graph.number_of_nodes() > 0
    assert graph_store.graph.number_of_edges() > 0
    assert cross_edges > 0

    query_result = graph_store.query_entity("机器学习", max_neighbors=20)
    assert len(query_result["matches"]) > 0
    assert len(query_result["neighbors"]) > 0
    assert any(row.get("doc_ids") for row in query_result["neighbors"])

    triples_path = extraction_service.save_triples_json(triples, tmp_path / "triples.json")
    graph_json_path = graph_store.export_graph_json(tmp_path / "graph.json")
    graphml_path = graph_store.export_graphml(tmp_path / "graph.graphml")

    assert triples_path.exists()
    assert graph_json_path.exists()
    assert graphml_path.exists()

