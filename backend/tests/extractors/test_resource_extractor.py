from extractors.resource_extractor import ResourceExtractor
from schema.parsed_document_schema import ParsedDocument, ParsedElement


def test_resource_extractor_basic():
    # 资源样例应能抽出页面对象、角色、可复用单元与索引
    parsed_doc = ParsedDocument(
        doc_id="doc-resource-1",
        file_name="Chap01绪论.pptx",
        file_type="pptx",
        layer="resource",
        title="机器学习导论",
        raw_text=(
            "机器学习导论\n"
            "机器学习是什么\n"
            "定义：机器学习是从数据中学习模型的方法。\n"
            "文献筛选的故事\n"
            "案例：通过机器学习提高筛选效率。\n"
        ),
        elements=[
            ParsedElement(
                element_id="p1-t",
                type="Title",
                text="机器学习导论",
                page_no=1,
                metadata={},
            ),
            ParsedElement(
                element_id="p1-c",
                type="NarrativeText",
                text="机器学习是什么\n定义：机器学习是从数据中学习模型的方法。",
                page_no=1,
                metadata={},
            ),
            ParsedElement(
                element_id="p2-t",
                type="Title",
                text="文献筛选的故事",
                page_no=2,
                metadata={"has_image": True},
            ),
            ParsedElement(
                element_id="p2-c",
                type="NarrativeText",
                text="案例：通过机器学习提高筛选效率。",
                page_no=2,
                metadata={},
            ),
        ],
        metadata={},
    )

    extractor = ResourceExtractor()
    result = extractor.extract(parsed_doc)

    assert "resource_info" in result
    assert "pages" in result
    assert "reusable_units" in result
    assert "chunks" in result
    assert "relations" in result
    assert "resource_index" in result

    assert len(result["pages"]) >= 2
    assert len(result["reusable_units"]) >= 2
    assert len(result["chunks"]) >= 2
    assert len(result["relations"]) >= 4

    roles = {page["page_role"] for page in result["pages"]}
    assert "title_page" in roles or "definition_page" in roles
    assert "case_page" in roles

    for page in result["pages"]:
        assert isinstance(page["knowledge_points"], list)
        assert isinstance(page["tags"], list)
        assert page["page_title"]
        assert page["page_summary"]

    for unit in result["reusable_units"]:
        assert unit["unit_id"]
        assert unit["unit_summary"]
        assert isinstance(unit["recommended_use"], list)
        assert 0 <= unit["reusability"] <= 1

    page_roles_index = result["resource_index"]["page_roles"]
    assert isinstance(page_roles_index, dict)
    assert any(page_roles_index.values())
