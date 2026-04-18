from extractors.textbook_extractor import TextbookExtractor
from schema.parsed_document_schema import ParsedDocument


def _build_textbook_sample_doc() -> ParsedDocument:
    # 构造教材层最小可运行样例（非真实 PDF）。
    return ParsedDocument(
        doc_id="doc-textbook-v1",
        file_name="ml_textbook_sample.docx",
        file_type="docx",
        layer="textbook",
        source_type="upload",
        source_name="unit-test",
        title="机器学习（第2版）",
        raw_text=(
            "机器学习（第2版）\n"
            "作者：周志华\n"
            "\n"
            "第一章 绪论\n"
            "1.1 什么是机器学习\n"
            "机器学习是让计算机从数据中学习规律的方法。\n"
            "（一）监督学习\n"
            "监督学习通过带标签数据训练模型。\n"
            "1.2 模型评估\n"
            "常见指标包括准确率、召回率和F1值。\n"
            "\n"
            "第二章 线性模型\n"
            "2.1 线性回归\n"
            "线性回归用于连续值预测，梯度下降是常用优化方法。\n"
            "2.2 逻辑回归\n"
            "逻辑回归用于分类任务。"
        ),
        elements=[],
        metadata={},
    )


def test_textbook_extractor_identifies_chapter_and_section():
    # 应能识别出章节与小节结构。
    extractor = TextbookExtractor()
    result = extractor.extract(_build_textbook_sample_doc())

    assert len(result["chapters"]) == 2
    assert result["chapters"][0]["chapter_index"] == "第一章"
    assert result["chapters"][1]["chapter_index"] == "第二章"

    section_titles = [section["section_title"] for section in result["sections"]]
    assert "什么是机器学习" in section_titles
    assert "线性回归" in section_titles


def test_textbook_extractor_extracts_knowledge_points():
    # 知识点应以概念名/方法名为主，不应为空。
    extractor = TextbookExtractor()
    result = extractor.extract(_build_textbook_sample_doc())

    kp_names = [kp["name"] for kp in result["knowledge_points"]]
    assert len(kp_names) >= 3
    assert "监督学习" in kp_names
    assert "线性回归" in kp_names
    assert "逻辑回归" in kp_names

    info = result["textbook_info"]
    assert info["textbook_role"] in {"main", "supplementary"}
    assert float(info.get("priority_score") or 0.0) > 0.0


def test_textbook_extractor_builds_chunks_and_relations():
    # 应生成 chunk 与基础关系边。
    extractor = TextbookExtractor()
    result = extractor.extract(_build_textbook_sample_doc())

    assert len(result["chunks"]) >= 4
    assert all(chunk["doc_id"] == "doc-textbook-v1" for chunk in result["chunks"])
    assert all(chunk["chapter_id"] for chunk in result["chunks"])
    assert all(chunk["section_id"] for chunk in result["chunks"])

    relations = result["relations"]
    assert len(relations) > 0
    assert any(
        relation["relation"] == "contains" and relation["source"] == "chapter-1"
        for relation in relations
    )
    assert any(relation["relation"] == "related_to" for relation in relations)
