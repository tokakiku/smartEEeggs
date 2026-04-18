from io import BytesIO
from zipfile import ZipFile

from parsers.unstructured_parser import UnstructuredParser


class _FakeMeta:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _FakeElement:
    def __init__(self, category, text, metadata):
        self.category = category
        self.text = text
        self.metadata = _FakeMeta(metadata)


def test_unstructured_parser_pdf(monkeypatch):
    # 使用 mock 的 unstructured 输出验证 ParsedDocument 转换逻辑
    parser = UnstructuredParser()

    def _mock_partition(_file_bytes, _file_name):
        return [
            _FakeElement("Title", "机器学习导论", {"page_number": 1, "element_id": "e1"}),
            _FakeElement("NarrativeText", "本章介绍机器学习基本概念。", {"page_number": 1, "element_id": "e2"}),
        ]

    monkeypatch.setattr(parser, "_partition_with_unstructured", _mock_partition)
    parsed = parser.parse(
        file_bytes=b"%PDF-1.4 mock",
        file_name="sample.pdf",
        layer="syllabus",
        source_type="upload",
        source_name="unit-test",
    )

    assert parsed.file_type == "pdf"
    assert parsed.title == "机器学习导论"
    assert len(parsed.elements) == 2
    assert parsed.elements[0].type == "Title"
    assert "机器学习基本概念" in parsed.raw_text


def test_unstructured_parser_pptx_fallback(monkeypatch):
    # 无 unstructured 时，PPTX 回退解析应保留按页文本与 page_no
    parser = UnstructuredParser()
    monkeypatch.setattr(
        parser,
        "_partition_with_unstructured",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    slide_1 = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
           xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cSld>
        <p:spTree>
          <p:sp><p:txBody><a:p><a:r><a:t>机器学习导论</a:t></a:r></a:p></p:txBody></p:sp>
          <p:sp><p:txBody><a:p><a:r><a:t>机器学习是什么</a:t></a:r></a:p></p:txBody></p:sp>
        </p:spTree>
      </p:cSld>
    </p:sld>"""
    slide_2 = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
           xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cSld>
        <p:spTree>
          <p:sp><p:txBody><a:p><a:r><a:t>文献筛选的故事</a:t></a:r></a:p></p:txBody></p:sp>
        </p:spTree>
      </p:cSld>
    </p:sld>"""

    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("ppt/slides/slide1.xml", slide_1)
        archive.writestr("ppt/slides/slide2.xml", slide_2)

    parsed = parser.parse(
        file_bytes=buffer.getvalue(),
        file_name="slides.pptx",
        layer="resource",
        source_type="upload",
        source_name="unit-test",
    )

    page_nos = sorted({item.page_no for item in parsed.elements if item.page_no})
    assert parsed.file_type == "pptx"
    assert parsed.metadata.get("parser_name") == "fallback_text_parser"
    assert page_nos == [1, 2]
    assert "机器学习导论" in parsed.raw_text
    assert "文献筛选的故事" in parsed.raw_text
