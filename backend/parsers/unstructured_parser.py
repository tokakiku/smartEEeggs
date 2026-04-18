from __future__ import annotations

"""基于 unstructured 的统一文档解析器。"""

from io import BytesIO
import re
import uuid
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from schema.parsed_document_schema import ParsedDocument, ParsedElement
from utils.text_cleaner import clean_text, split_paragraphs
from utils.pdf_reader import read_pdf_pages, read_pdf_text

from parsers.base_parser import BaseParser
from parsers.file_type_utils import detect_file_type


class UnstructuredParser(BaseParser):
    """统一文档解析器。"""
    # 基于 unstructured 的统一文档解析器

    def parse(
        self,
        file_bytes: bytes,
        file_name: str,
        layer: str,
        source_type: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> ParsedDocument:
        if not file_bytes:
            raise ValueError("empty file bytes")

        file_type = detect_file_type(file_name)
        doc_id = uuid.uuid4().hex

        parsed_elements: List[ParsedElement] = []
        parser_name = "unstructured"
        warnings: List[str] = []

        try:
            raw_elements = self._partition_with_unstructured(file_bytes, file_name)
            parsed_elements = self._convert_elements(raw_elements, file_name)
        except Exception as exc:
            # 当 unstructured 不可用或解析失败时，退化为轻量文本解析，保证流程可运行。
            parser_name = "fallback_text_parser"
            warnings.append(f"unstructured parser fallback: {exc}")
            parsed_elements = self._fallback_parse(file_bytes, file_name, file_type)

        raw_text = clean_text("\n".join([item.text for item in parsed_elements if item.text.strip()]))
        title = self._guess_title(parsed_elements, file_name)

        return ParsedDocument(
            doc_id=doc_id,
            file_name=file_name,
            file_type=file_type,
            layer=layer,
            source_type=source_type,
            source_name=source_name,
            title=title,
            raw_text=raw_text,
            elements=parsed_elements,
            metadata={
                "parser_name": parser_name,
                "warnings": warnings,
                "element_count": len(parsed_elements),
            },
        )

    def _partition_with_unstructured(self, file_bytes: bytes, file_name: str) -> List[Any]:
        # 直接调用 unstructured 的 auto partition，自动按扩展名走对应处理器。
        from unstructured.partition.auto import partition

        # 兼容新版本 unstructured：使用 filename / metadata_filename 参与文件类型识别。
        return list(
            partition(
                file=BytesIO(file_bytes),
                metadata_filename=file_name,
            )
        )

    def _convert_elements(self, elements: List[Any], file_name: str) -> List[ParsedElement]:
        # 将 unstructured element 转换成系统统一 ParsedElement。
        converted: List[ParsedElement] = []
        for idx, element in enumerate(elements):
            element_type = getattr(element, "category", None) or element.__class__.__name__
            text = getattr(element, "text", None)
            if text is None:
                text = str(element)
            text = text or ""

            metadata = self._extract_element_metadata(element)
            page_no = metadata.get("page_number") or metadata.get("page_no")
            element_id = metadata.get("element_id") or f"{idx + 1}"
            metadata.setdefault("source_file", file_name)

            converted.append(
                ParsedElement(
                    element_id=str(element_id),
                    type=str(element_type),
                    text=text.strip(),
                    page_no=page_no if isinstance(page_no, int) else None,
                    metadata=metadata,
                )
            )
        return converted

    def _extract_element_metadata(self, element: Any) -> Dict[str, Any]:
        # 兼容 unstructured metadata 对象和普通 dict。
        metadata_obj = getattr(element, "metadata", None)
        if metadata_obj is None:
            return {}
        if hasattr(metadata_obj, "to_dict"):
            try:
                return dict(metadata_obj.to_dict() or {})
            except Exception:
                return {}
        if isinstance(metadata_obj, dict):
            return dict(metadata_obj)
        return {}

    def _fallback_parse(self, file_bytes: bytes, file_name: str, file_type: str) -> List[ParsedElement]:
        # 退化解析策略：PDF 走 pypdf，HTML 用 BeautifulSoup，其他文件按文本解码。
        text = ""
        if file_type == "pdf":
            pages = read_pdf_pages(file_bytes)
            if pages:
                return self._build_pdf_fallback_elements(pages, file_name)
            text = read_pdf_text(file_bytes)
        elif file_type == "pptx":
            slides = self._extract_pptx_slides(file_bytes)
            if slides:
                return self._build_pptx_fallback_elements(slides, file_name)
        elif file_type == "html":
            text = self._extract_html_text(file_bytes)
        if not text:
            text = file_bytes.decode("utf-8", errors="ignore")
        text = clean_text(text)

        paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
        if not paragraphs:
            return []

        elements: List[ParsedElement] = []
        for idx, paragraph in enumerate(paragraphs):
            element_type = "Title" if idx == 0 else "NarrativeText"
            elements.append(
                ParsedElement(
                    element_id=f"fallback-{idx + 1}",
                    type=element_type,
                    text=paragraph,
                    page_no=1,
                    metadata={"source_file": file_name, "parser": "fallback_text_parser"},
                )
            )
        return elements

    def _build_pdf_fallback_elements(self, pages: List[str], file_name: str) -> List[ParsedElement]:
        # PDF 回退时按页切分元素，保留 page_no 便于后续前置页过滤。
        elements: List[ParsedElement] = []
        index = 0
        for page_no, page_text in enumerate(pages, start=1):
            paragraphs = split_paragraphs(page_text)
            for paragraph in paragraphs:
                index += 1
                element_type = "Title" if index == 1 else "NarrativeText"
                elements.append(
                    ParsedElement(
                        element_id=f"fallback-{index}",
                        type=element_type,
                        text=paragraph,
                        page_no=page_no,
                        metadata={
                            "source_file": file_name,
                            "parser": "fallback_text_parser",
                            "page_no": page_no,
                        },
                    )
                )
        return elements

    def _extract_pptx_slides(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        # PPTX 回退解析：读取 slide XML，提取每页文本和页面标记
        slides: List[Dict[str, Any]] = []
        try:
            with ZipFile(BytesIO(file_bytes)) as archive:
                slide_files = [
                    name
                    for name in archive.namelist()
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                ]
                if not slide_files:
                    return []

                def slide_sort_key(path: str) -> int:
                    match = re.search(r"slide(\d+)\.xml$", path)
                    return int(match.group(1)) if match else 10**9

                ns = {
                    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
                    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
                }

                for slide_file in sorted(slide_files, key=slide_sort_key):
                    slide_no = slide_sort_key(slide_file)
                    xml_bytes = archive.read(slide_file)
                    root = ET.fromstring(xml_bytes)
                    texts = [
                        (node.text or "").strip()
                        for node in root.findall(".//a:t", ns)
                        if (node.text or "").strip()
                    ]
                    has_image = bool(root.findall(".//p:pic", ns))
                    has_table = bool(root.findall(".//a:tbl", ns))
                    has_formula = bool(root.findall(".//m:oMath", ns) or root.findall(".//m:oMathPara", ns))
                    slides.append(
                        {
                            "slide_no": slide_no,
                            "texts": texts,
                            "has_image": has_image,
                            "has_table": has_table,
                            "has_formula": has_formula,
                        }
                    )
        except Exception:
            return []
        return slides

    def _build_pptx_fallback_elements(self, slides: List[Dict[str, Any]], file_name: str) -> List[ParsedElement]:
        # 将 PPTX slide 结果转换为 ParsedElement，保留 slide_no 作为 page_no
        elements: List[ParsedElement] = []
        for slide in slides:
            slide_no = int(slide.get("slide_no") or 1)
            texts = slide.get("texts") or []
            has_image = bool(slide.get("has_image"))
            has_table = bool(slide.get("has_table"))
            has_formula = bool(slide.get("has_formula"))

            if not texts:
                placeholder = "图示页" if has_image else "空白页"
                elements.append(
                    ParsedElement(
                        element_id=f"pptx-{slide_no}-1",
                        type="NarrativeText",
                        text=placeholder,
                        page_no=slide_no,
                        metadata={
                            "source_file": file_name,
                            "parser": "fallback_text_parser",
                            "page_no": slide_no,
                            "has_image": has_image,
                            "has_table": has_table,
                            "has_formula": has_formula,
                        },
                    )
                )
                continue

            for idx, text in enumerate(texts, start=1):
                element_type = "Title" if idx == 1 else "NarrativeText"
                elements.append(
                    ParsedElement(
                        element_id=f"pptx-{slide_no}-{idx}",
                        type=element_type,
                        text=text,
                        page_no=slide_no,
                        metadata={
                            "source_file": file_name,
                            "parser": "fallback_text_parser",
                            "page_no": slide_no,
                            "has_image": has_image,
                            "has_table": has_table,
                            "has_formula": has_formula,
                        },
                    )
                )
        return elements

    def _extract_html_text(self, file_bytes: bytes) -> str:
        # HTML 提取优先走 BeautifulSoup；缺包时回退纯文本解码。
        try:
            from bs4 import BeautifulSoup

            html = file_bytes.decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            return soup.get_text(separator="\n")
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")

    def _guess_title(self, elements: List[ParsedElement], file_name: str) -> Optional[str]:
        # 优先选择第一个 Title 元素作为文档标题。
        for item in elements:
            if item.type.lower() == "title" and item.text.strip():
                return item.text.strip()

        for item in elements:
            if item.text.strip():
                return item.text.strip().split("\n")[0][:120]

        return file_name
