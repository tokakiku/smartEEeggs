from extractors.hotspot_extractor import HotspotExtractor
from schema.parsed_document_schema import ParsedDocument


def test_hotspot_extractor_basic():
    # 热点样例应能抽到事件类型、知识点、教学用途和证据片段
    parsed_doc = ParsedDocument(
        doc_id="doc-hotspot-1",
        file_name="reuters_ai_deforestation_2026-03-01.html",
        file_type="html",
        layer="hotspot",
        title="How AI predictive models are helping to prevent deforestation",
        raw_text=(
            "2026-03-01 Source: Reuters\n"
            "Researchers and NGOs are deploying machine learning models on satellite imagery to detect early signs of illegal forest clearing.\n"
            "The system helps rangers prioritize inspections and has improved intervention speed in pilot regions.\n"
            "The report says the approach could support climate governance and biodiversity protection.\n"
        ),
        elements=[],
        metadata={},
    )

    extractor = HotspotExtractor()
    result = extractor.extract(parsed_doc)

    assert "hotspot_info" in result
    assert "hotspot_item" in result
    assert "chunks" in result
    assert "relations" in result
    assert "hotspot_index" in result

    assert result["hotspot_info"]["title"]
    assert result["hotspot_info"]["publish_date"] == "2026-03-01"

    items = result["hotspot_item"]
    assert len(items) == 1
    item = items[0]
    assert item["summary"]
    assert item["event_type"] in {
        "industry_application",
        "research_breakthrough",
        "other_event",
    }
    assert isinstance(item["related_knowledge_points"], list)
    assert len(item["teaching_usage"]) >= 1
    assert len(item["evidence_snippets"]) >= 1

    assert len(result["chunks"]) >= 1
    assert len(result["relations"]) >= 3
