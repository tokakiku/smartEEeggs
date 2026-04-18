import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

"""教材层抽取器：抽取章节、知识点、chunks 与关系。"""

from extractors.base_extractor import BaseExtractor
from schema.parsed_document_schema import ParsedDocument
from schema.textbook_schema import (
    TextbookChapter,
    TextbookChunk,
    TextbookExtractionResult,
    TextbookInfo,
    TextbookKnowledgePoint,
    TextbookRelation,
    TextbookSection,
)
from utils.text_cleaner import clean_text, split_paragraphs


class TextbookExtractor(BaseExtractor):
    """教材层 extractor。"""
    # 教材层 extractor：抽取章节、小节、知识点、chunks 与基础关系

    STRICT_CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百零〇\d]+章(?:\s+|[：:])?.*$")
    CHAPTER_EN_PATTERN = re.compile(r"^chapter\s*\d+(?:\s+|[：:\-])?.*$", re.IGNORECASE)
    ARABIC_CHAPTER_PATTERN = re.compile(r"^(\d{1,2})\s+([^\d].{1,40})$")
    STRICT_SECTION_PATTERN = re.compile(r"^\d+\.\d+(?:\.\d+)?\s*[^\s].*$")
    SECTION_CN_PATTERN = re.compile(r"^第[一二三四五六七八九十百零〇\d]+节(?:\s+|[：:])?.*$")
    SECTION_ITEM_PATTERN = re.compile(r"^[（(][一二三四五六七八九十]+[)）]\s*[^\s].*$")

    CHAPTER_PATTERNS = [
        STRICT_CHAPTER_PATTERN,
        CHAPTER_EN_PATTERN,
    ]
    SECTION_PATTERNS = [
        SECTION_CN_PATTERN,
        STRICT_SECTION_PATTERN,
        SECTION_ITEM_PATTERN,
    ]
    NUMBER_PREFIX_PATTERN = re.compile(
        r"^(第[一二三四五六七八九十百零〇\d]+章|第[一二三四五六七八九十百零〇\d]+节|chapter\s*\d+|\d+\.\d+(?:\.\d+)?|"
        r"[（(][一二三四五六七八九十]+[)）]|\d+)\s*[-.、:：]?\s*",
        re.IGNORECASE,
    )
    KNOWLEDGE_KEYWORDS = [
        "学习",
        "算法",
        "模型",
        "方法",
        "回归",
        "分类",
        "聚类",
        "网络",
        "梯度",
        "损失",
        "决策树",
        "贝叶斯",
        "向量机",
        "神经",
        "优化",
        "特征",
        "降维",
        "评估",
        "过拟合",
        "欠拟合",
    ]
    KP_STOP_WORDS = {
        "本章小结",
        "小结",
        "本节小结",
        "习题",
        "思考题",
        "课后练习",
        "练习",
        "参考文献",
        "附录",
        "导读",
        "绪论",
        "前言",
    }
    FRONT_MATTER_KEYWORDS = [
        "版权",
        "版权所有",
        "isbn",
        "出版",
        "出版社",
        "发行",
        "印刷",
        "定价",
        "前言",
        "序",
        "序言",
        "再版",
        "修订",
        "丛书",
        "责任编辑",
        "目 录",
        "目录",
        "contents",
    ]
    CHAPTER_HINT_KEYWORDS = [
        "绪论",
        "导读",
        "引言",
        "概述",
        "基础",
        "方法",
        "模型",
        "学习",
        "算法",
        "理论",
        "实验",
        "附录",
        "chapter",
        "part",
    ]
    DIRECTORY_TITLES = {"目录", "目 录", "contents"}

    def extract(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        # 主流程：先构建章节树，再抽知识点，再做分块和关系
        lines = self._prepare_lines(parsed_doc)
        chapters_data = self._parse_chapters(lines)
        if not chapters_data:
            chapters_data = [self._build_default_chapter()]

        knowledge_points = self._extract_knowledge_points(chapters_data)
        section_to_kp_names = self._build_section_kp_name_map(knowledge_points)
        self._fill_section_knowledge_points(chapters_data, section_to_kp_names)
        chunks = self._build_chunks(parsed_doc, chapters_data, section_to_kp_names)
        relations = self._build_relations(chapters_data, knowledge_points, chunks)
        textbook_info = self._build_textbook_info(parsed_doc, chapters_data)

        chapter_models: List[TextbookChapter] = []
        section_models: List[TextbookSection] = []
        for chapter in chapters_data:
            chapter_sections: List[TextbookSection] = []
            for section in chapter["sections"]:
                section_model = TextbookSection(
                    section_id=section["section_id"],
                    chapter_id=chapter["chapter_id"],
                    section_index=section["section_index"],
                    section_title=section["section_title"],
                    raw_text=section["raw_text"],
                    knowledge_points=section.get("knowledge_points", []),
                )
                section_models.append(section_model)
                chapter_sections.append(section_model)

            chapter_models.append(
                TextbookChapter(
                    chapter_id=chapter["chapter_id"],
                    chapter_index=chapter["chapter_index"],
                    chapter_title=chapter["chapter_title"],
                    raw_text=chapter["raw_text"],
                    sections=chapter_sections,
                )
            )

        result_model = TextbookExtractionResult(
            textbook_info=textbook_info,
            chapters=chapter_models,
            sections=section_models,
            knowledge_points=knowledge_points,
            chunks=chunks,
            relations=relations,
        )
        if hasattr(result_model, "model_dump"):
            return result_model.model_dump()
        return result_model.dict()

    def _prepare_lines(self, parsed_doc: ParsedDocument) -> List[str]:
            # PDF 优先按页过滤前置内容；其他类型走原始文本切分。
        if parsed_doc.file_type == "pdf":
            lines = self._prepare_pdf_lines(parsed_doc)
            if lines:
                return lines

        if parsed_doc.raw_text.strip():
            return split_paragraphs(parsed_doc.raw_text)

        lines: List[str] = []
        for element in parsed_doc.elements:
            text = clean_text(element.text)
            if text:
                lines.extend(split_paragraphs(text))
        return lines

    def _prepare_pdf_lines(self, parsed_doc: ParsedDocument) -> List[str]:
        # PDF 前置页过滤：封面/版权/目录/前言尽量剔除，并定位正文起点。
        pages = self._group_lines_by_page(parsed_doc)
        if not pages:
            return []

        page_numbers = sorted(pages.keys())
        if not page_numbers:
            return []

        page_stats = {page_no: self._analyze_page(lines) for page_no, lines in pages.items()}
        body_start_page = self._find_body_start_page(page_numbers, page_stats)

        selected: List[str] = []
        for page_no in page_numbers:
            if page_no < body_start_page:
                continue
            lines = pages[page_no]
            stats = page_stats[page_no]
            if self._should_skip_page_after_body_start(stats):
                continue
            for line in lines:
                normalized = self._normalize_line(line)
                if not normalized:
                    continue
                selected.append(normalized)
        return selected

    def _group_lines_by_page(self, parsed_doc: ParsedDocument) -> Dict[int, List[str]]:
        # 将解析元素按页聚合，后续用于前置页判定。
        pages: Dict[int, List[str]] = {}
        for index, element in enumerate(parsed_doc.elements, start=1):
            text = clean_text(element.text or "")
            if not text:
                continue
            page_no = element.page_no if isinstance(element.page_no, int) and element.page_no > 0 else index
            page_lines = [self._normalize_line(item, strip_toc_page_no=False) for item in split_paragraphs(text)]
            clean_lines = [item for item in page_lines if item]
            if not clean_lines:
                continue
            pages.setdefault(page_no, []).extend(clean_lines)
        return pages

    def _analyze_page(self, lines: List[str]) -> Dict[str, int]:
        # 统计页面特征，用于区分目录/前置页和正文页。
        stats = {
            "line_count": 0,
            "chapter_hits": 0,
            "section_hits": 0,
            "toc_hits": 0,
            "front_hits": 0,
            "body_hits": 0,
            "metadata_hits": 0,
        }
        for line in lines:
            raw_line = clean_text(line)
            if not raw_line:
                continue
            is_toc_line = self._looks_like_toc_entry(raw_line)
            line = self._normalize_line(raw_line, strip_toc_page_no=True)
            if not line:
                continue
            stats["line_count"] += 1

            if self._is_front_matter_line(line):
                stats["front_hits"] += 1
            if self._is_metadata_line(line):
                stats["metadata_hits"] += 1
            if is_toc_line:
                stats["toc_hits"] += 1
            if self._is_candidate_body_line(line):
                stats["body_hits"] += 1
            if self._is_chapter_heading(line):
                stats["chapter_hits"] += 1
            if self._is_section_heading(line):
                stats["section_hits"] += 1
        return stats

    def _find_body_start_page(self, page_numbers: List[int], page_stats: Dict[int, Dict[str, int]]) -> int:
        # 正文起点策略：优先选“有章节标题且非目录页”的最早页面。
        for page_no in page_numbers:
            stats = page_stats.get(page_no, {})
            if stats.get("chapter_hits", 0) >= 1 and stats.get("toc_hits", 0) <= 2:
                return page_no

        for page_no in page_numbers:
            stats = page_stats.get(page_no, {})
            if (
                stats.get("section_hits", 0) >= 3
                and stats.get("toc_hits", 0) <= 2
                and stats.get("body_hits", 0) >= 3
            ):
                return page_no

        for page_no in page_numbers:
            stats = page_stats.get(page_no, {})
            if stats.get("front_hits", 0) == 0 and stats.get("metadata_hits", 0) <= 2:
                return page_no
        return page_numbers[0]

    def _should_skip_page_after_body_start(self, stats: Dict[str, int]) -> bool:
        # 正文开始后仍可能出现目录/版权残留页，这里做二次过滤。
        line_count = max(stats.get("line_count", 0), 1)
        toc_hits = stats.get("toc_hits", 0)
        front_hits = stats.get("front_hits", 0)
        body_hits = stats.get("body_hits", 0)
        metadata_hits = stats.get("metadata_hits", 0)

        if toc_hits >= max(4, line_count // 3) and body_hits <= 2:
            return True
        if front_hits >= max(3, line_count // 3) and body_hits <= 2:
            return True
        if metadata_hits >= max(4, line_count // 2) and body_hits <= 1:
            return True
        return False

    def _normalize_line(self, line: str, strip_toc_page_no: bool = True) -> str:
        # 清洗单行文本，并去掉目录式行尾页码。
        normalized = clean_text(line)
        if not normalized:
            return ""
        normalized = re.sub(r"\s{2,}", " ", normalized)
        if strip_toc_page_no and self._looks_like_toc_entry(normalized):
            normalized = re.sub(r"\.{2,}\s*\d+\s*$", "", normalized)
            normalized = re.sub(r"[·•]{2,}\s*\d+\s*$", "", normalized)
            normalized = re.sub(r"\s+\d{1,4}\s*$", "", normalized)
        return normalized.strip()

    def _is_front_matter_line(self, line: str) -> bool:
        lower = line.lower()
        return any(keyword in lower for keyword in self.FRONT_MATTER_KEYWORDS)

    def _is_metadata_line(self, line: str) -> bool:
        # 过滤出版日期、ISBN、页码、版权信息等元数据行。
        if re.search(r"\bISBN\b", line, flags=re.IGNORECASE):
            return True
        if re.search(r"(出版社|出版|印刷|版次|定价|责任编辑|版权所有|版权)", line):
            return True
        if re.search(r"^\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?$", line):
            return True
        if re.search(r"^\d{1,3}\s*/\s*\d{1,4}$", line):
            return True
        if re.search(r"^第?\s*\d+\s*页$", line):
            return True
        return False

    def _looks_like_toc_entry(self, line: str) -> bool:
        # 目录条目判定：点线/页码尾缀/典型目录格式。
        if re.search(r"\.{2,}\s*\d+\s*$", line):
            return True
        if re.search(r"[·•]{2,}\s*\d+\s*$", line):
            return True
        if re.search(r"^(第[一二三四五六七八九十百零〇\d]+章|第[一二三四五六七八九十百零〇\d]+节).*\s+\d{1,4}$", line):
            return True
        if re.search(r"^\d+\.\d+(?:\.\d+)?\s+.*\s+\d{1,4}$", line):
            return True
        return False

    def _is_candidate_body_line(self, line: str) -> bool:
        # 正文行启发：含中文句段且非纯符号行。
        if len(line) < 6:
            return False
        if self._is_metadata_line(line):
            return False
        if self._looks_like_toc_entry(line):
            return False
        chinese_chars = len(re.findall(r"[\u4e00-\u9fa5]", line))
        return chinese_chars >= 4

    def _parse_chapters(self, lines: List[str]) -> List[Dict[str, Any]]:
        # 按“章 -> 节”顺序构建教材结构，避免全文乱切
        chapters: List[Dict[str, Any]] = []
        current_chapter: Optional[Dict[str, Any]] = None
        current_section: Optional[Dict[str, Any]] = None
        preface_lines: List[str] = []

        for raw_line in lines:
            line = clean_text(raw_line)
            if not line:
                continue

            if self._is_chapter_heading(line):
                current_chapter = self._start_chapter(chapters, line)
                if preface_lines:
                    current_chapter["content_lines"].extend(preface_lines)
                    preface_lines = []
                current_section = None
                continue

            if self._is_section_heading(line):
                if current_chapter is None:
                    current_chapter = self._start_chapter(chapters, "第1章 导读")
                current_section = self._start_section(current_chapter, line)
                continue

            if current_chapter is None:
                preface_lines.append(line)
                continue
            if current_section is None:
                current_section = self._start_section(
                    current_chapter,
                    self._build_default_section_heading(current_chapter),
                )

            current_section["content_lines"].append(line)
            current_chapter["content_lines"].append(line)

        if not chapters and preface_lines:
            current_chapter = self._start_chapter(chapters, "第1章 导读")
            current_section = self._start_section(
                current_chapter,
                self._build_default_section_heading(current_chapter),
            )
            current_section["content_lines"].extend(preface_lines)
            current_chapter["content_lines"].extend(preface_lines)

        self._finalize_outline(chapters)
        chapters = self._cleanup_outline(chapters)
        return chapters

    def _start_chapter(self, chapters: List[Dict[str, Any]], heading_line: str) -> Dict[str, Any]:
        # 创建章节节点并初始化容器
        chapter_number = len(chapters) + 1
        chapter_index, chapter_title = self._extract_chapter_index_and_title(heading_line, chapter_number)
        chapter = {
            "chapter_id": f"chapter-{chapter_number}",
            "chapter_index": chapter_index,
            "chapter_title": chapter_title,
            "content_lines": [heading_line],
            "sections": [],
        }
        chapters.append(chapter)
        return chapter

    def _start_section(self, chapter: Dict[str, Any], heading_line: str) -> Dict[str, Any]:
        # 在当前章节下创建小节节点
        section_number = len(chapter["sections"]) + 1
        section_index, section_title = self._extract_section_index_and_title(heading_line, section_number)
        section = {
            "section_id": f"{chapter['chapter_id']}-section-{section_number}",
            "section_index": section_index,
            "section_title": section_title,
            "content_lines": [],
            "raw_text": "",
            "knowledge_points": [],
        }
        chapter["sections"].append(section)
        chapter["content_lines"].append(heading_line)
        return section

    def _finalize_outline(self, chapters: List[Dict[str, Any]]) -> None:
        # 写回章节和小节 raw_text，并保证至少有一个 section
        for chapter in chapters:
            if not chapter["sections"]:
                chapter["sections"].append(
                    {
                        "section_id": f"{chapter['chapter_id']}-section-1",
                        "section_index": "1",
                        "section_title": "导读",
                        "content_lines": [],
                        "raw_text": "",
                        "knowledge_points": [],
                    }
                )

            for section in chapter["sections"]:
                section["raw_text"] = clean_text("\n".join(section["content_lines"]))
            chapter["raw_text"] = clean_text("\n".join(chapter["content_lines"]))

    def _cleanup_outline(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 清理目录残留并对 chapter/section 去重（正文优先）。
        chapter_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        chapter_order: List[Tuple[str, str]] = []

        for chapter in chapters:
            chapter_title = self._clean_heading_title(chapter.get("chapter_title", ""))
            chapter_index = self._clean_heading_title(chapter.get("chapter_index", ""))
            chapter_raw = clean_text(chapter.get("raw_text", ""))
            chapter_sections = self._cleanup_sections(chapter.get("sections", []))

            if self._is_directory_chapter(chapter_title, chapter_sections, chapter_raw):
                continue

            if not chapter_sections and chapter_raw:
                chapter_sections = [self._build_default_section_from_chapter(chapter_raw)]

            if not chapter_sections and not chapter_raw:
                continue

            key = (
                self._normalize_index_for_dedup(chapter_index),
                self._normalize_title_for_dedup(chapter_title),
            )
            if not key[0] and not key[1]:
                continue

            if key not in chapter_map:
                chapter_map[key] = {
                    "chapter_id": chapter.get("chapter_id", ""),
                    "chapter_index": chapter_index,
                    "chapter_title": chapter_title,
                    "content_lines": list(chapter.get("content_lines", [])),
                    "raw_text": chapter_raw,
                    "sections": chapter_sections,
                }
                chapter_order.append(key)
                continue

            merged = chapter_map[key]
            merged["raw_text"] = self._merge_text_block(merged.get("raw_text", ""), chapter_raw)
            merged["content_lines"] = self._merge_lines(
                merged.get("content_lines", []),
                chapter.get("content_lines", []),
            )
            merged["sections"] = self._merge_sections(merged.get("sections", []), chapter_sections)

        cleaned: List[Dict[str, Any]] = []
        for key in chapter_order:
            chapter = chapter_map[key]
            chapter["sections"] = self._cleanup_sections(chapter.get("sections", []))
            if not chapter["sections"] and chapter.get("raw_text"):
                chapter["sections"] = [self._build_default_section_from_chapter(chapter["raw_text"])]
            if not chapter["sections"]:
                continue
            cleaned.append(chapter)

        return self._reindex_outline(cleaned)

    def _cleanup_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 小节去重：同 index+title 合并，优先保留有正文。
        section_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        section_order: List[Tuple[str, str]] = []

        for section in sections:
            section_index = self._clean_heading_title(section.get("section_index", ""))
            section_title = self._clean_heading_title(section.get("section_title", ""))
            raw_text = clean_text(section.get("raw_text", ""))

            if self._is_directory_section(section_index, section_title, raw_text):
                continue

            key = (
                self._normalize_index_for_dedup(section_index),
                self._normalize_title_for_dedup(section_title),
            )
            if not key[0] and not key[1]:
                continue
            item = {
                "section_id": section.get("section_id", ""),
                "section_index": section_index,
                "section_title": section_title,
                "content_lines": list(section.get("content_lines", [])),
                "raw_text": raw_text,
                "knowledge_points": [],
            }
            if key not in section_map:
                section_map[key] = item
                section_order.append(key)
                continue

            merged = section_map[key]
            merged["raw_text"] = self._merge_text_block(merged.get("raw_text", ""), raw_text)
            merged["content_lines"] = self._merge_lines(
                merged.get("content_lines", []),
                section.get("content_lines", []),
            )

        result: List[Dict[str, Any]] = []
        for key in section_order:
            section = section_map[key]
            section["raw_text"] = clean_text(section.get("raw_text", ""))
            if self._is_directory_section(
                section.get("section_index", ""),
                section.get("section_title", ""),
                section.get("raw_text", ""),
            ):
                continue
            result.append(section)
        return result

    def _merge_sections(self, sections_a: List[Dict[str, Any]], sections_b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 合并章节重复时的小节列表。
        return self._cleanup_sections(list(sections_a) + list(sections_b))

    def _reindex_outline(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 重排 chapter_id/section_id，保证输出稳定。
        reindexed: List[Dict[str, Any]] = []
        for chapter_no, chapter in enumerate(chapters, start=1):
            chapter_id = f"chapter-{chapter_no}"
            new_chapter = {
                "chapter_id": chapter_id,
                "chapter_index": chapter.get("chapter_index") or str(chapter_no),
                "chapter_title": chapter.get("chapter_title") or f"第{chapter_no}章",
                "content_lines": list(chapter.get("content_lines", [])),
                "raw_text": clean_text(chapter.get("raw_text", "")),
                "sections": [],
            }
            for section_no, section in enumerate(chapter.get("sections", []), start=1):
                new_chapter["sections"].append(
                    {
                        "section_id": f"{chapter_id}-section-{section_no}",
                        "chapter_id": chapter_id,
                        "section_index": section.get("section_index") or str(section_no),
                        "section_title": section.get("section_title") or "导读",
                        "content_lines": list(section.get("content_lines", [])),
                        "raw_text": clean_text(section.get("raw_text", "")),
                        "knowledge_points": [],
                    }
                )
            if new_chapter["sections"]:
                reindexed.append(new_chapter)
        return reindexed

    def _build_default_section_from_chapter(self, chapter_raw: str) -> Dict[str, Any]:
            # 章节无小节时补一条兜底小节。
        return {
            "section_id": "",
            "section_index": "1",
            "section_title": "导读",
            "content_lines": split_paragraphs(chapter_raw),
            "raw_text": clean_text(chapter_raw),
            "knowledge_points": [],
        }

    def _build_default_section_heading(self, chapter: Dict[str, Any]) -> str:
        # 缺少 section 标题时，用章节编号生成更稳定的默认小节编号，避免全书都落到 1.1 导读。
        chapter_index = self._normalize_index_for_dedup(chapter.get("chapter_index", ""))
        chapter_no_match = re.search(r"\d+", chapter_index)
        if chapter_no_match:
            return f"{chapter_no_match.group(0)}.1 导读"
        return "1.1 导读"

    def _clean_heading_title(self, text: str) -> str:
        # 清洗 heading 文本：去掉首尾页码与目录点线。
        value = self._normalize_line(text, strip_toc_page_no=True)
        value = self._strip_leading_page_no_prefix(value)
        value = self._strip_trailing_page_no(value)
        value = re.sub(r"\s{2,}", " ", value).strip()
        return value

    def _normalize_index_for_dedup(self, text: str) -> str:
        # 章节/小节编号标准化：统一空格和标点，便于去重键稳定。
        value = self._clean_heading_title(text).lower()
        if not value:
            return ""
        value = value.replace("（", "(").replace("）", ")")
        value = value.replace("．", ".").replace("。", ".")
        value = re.sub(r"\s+", "", value)
        return value

    def _normalize_title_for_dedup(self, text: str) -> str:
        # 标题标准化：去除 OCR 造成的中文间空格和弱标点，减少重复章节/小节残留。
        value = self._clean_heading_title(text).lower()
        if not value:
            return ""
        value = value.replace("（", "(").replace("）", ")")
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"[·•\-_—–:：,，;；。\.、\"'“”‘’!?？！`~]", "", value)
        return value

    def _is_directory_section(self, section_index: str, section_title: str, raw_text: str) -> bool:
        # 判断 section 是否目录项或无效项。
        title = (section_title or "").strip().lower()
        if not title:
            return True
        if title in self.DIRECTORY_TITLES:
            return True
        if any(token in title for token in ["目 录", "目录", "contents"]):
            return True
        if "课件" in title and len(raw_text.strip()) < 20:
            return True

        body = raw_text.strip()
        if not body and re.match(r"^\d+\.\d+(?:\.\d+)?$", section_index):
            return True
        if not body and re.match(r"^[（(][一二三四五六七八九十]+[)）]$", section_index):
            return True
        return False

    def _is_directory_chapter(self, chapter_title: str, sections: List[Dict[str, Any]], chapter_raw: str) -> bool:
        # 判断 chapter 是否目录页/封面页误判。
        title = (chapter_title or "").strip().lower()
        if not title:
            return True
        if title in self.DIRECTORY_TITLES:
            return True
        if any(token in title for token in ["目 录", "目录", "contents"]):
            return True
        if "课件" in title and len(chapter_raw.strip()) < 80:
            return True

        if not sections:
            return True
        non_empty_sections = sum(1 for item in sections if len(item.get("raw_text", "").strip()) >= 20)
        if non_empty_sections == 0 and len(sections) >= 2:
            return True
        return False

    def _is_directory_like_title(self, text: str) -> bool:
        # 目录类标题判定：用于过滤 book_title=目录 这类污染。
        title = self._clean_heading_title(text).lower()
        compact = title.replace(" ", "")
        if not compact:
            return True
        if compact in {"目录", "contents"}:
            return True
        if "目录" in compact or "contents" in compact:
            return True
        return False

    def _merge_text_block(self, base: str, incoming: str) -> str:
        # 合并正文文本，避免重复追加。
        base_text = clean_text(base)
        new_text = clean_text(incoming)
        if not base_text:
            return new_text
        if not new_text or new_text in base_text:
            return base_text
        if base_text in new_text:
            return new_text
        return clean_text(f"{base_text}\n{new_text}")

    def _merge_lines(self, lines_a: List[str], lines_b: List[str]) -> List[str]:
        # 保序合并行列表。
        merged: List[str] = []
        seen = set()
        for line in list(lines_a) + list(lines_b):
            text = clean_text(line)
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
        return merged

    def _build_default_chapter(self) -> Dict[str, Any]:
        # 没有识别到章节时兜底，保证结构稳定
        return {
            "chapter_id": "chapter-1",
            "chapter_index": "1",
            "chapter_title": "第1章 导读",
            "raw_text": "",
            "sections": [
                {
                    "section_id": "chapter-1-section-1",
                    "section_index": "1",
                    "section_title": "导读",
                    "content_lines": [],
                    "raw_text": "",
                    "knowledge_points": [],
                }
            ],
        }

    def _extract_knowledge_points(self, chapters: List[Dict[str, Any]]) -> List[TextbookKnowledgePoint]:
        # 规则抽取知识点：节标题优先，再补充小标题/定义句
        knowledge_points: List[TextbookKnowledgePoint] = []
        seen: set[Tuple[str, str]] = set()

        for chapter in chapters:
            chapter_id = chapter["chapter_id"]
            for section in chapter["sections"]:
                section_id = section["section_id"]
                section_text = section.get("raw_text", "")
                candidates = self._build_kp_candidates(section["section_title"], section_text)
                for candidate in candidates:
                    dedup_key = (section_id, candidate)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    knowledge_points.append(
                        TextbookKnowledgePoint(
                            kp_id=f"kp-{len(knowledge_points) + 1}",
                            name=candidate,
                            chapter_id=chapter_id,
                            section_id=section_id,
                            aliases=[],
                            description=self._build_kp_description(candidate, section_text),
                            source_text=section_text,
                        )
                    )
        return knowledge_points

    def _build_kp_candidates(self, section_title: str, section_text: str) -> List[str]:
        # 组装知识点候选，避免抽出整句噪声
        candidates: List[str] = []

        title_candidate = self._normalize_kp_name(section_title)
        if self._is_valid_kp_candidate(title_candidate, require_keyword=False):
            candidates.append(title_candidate)

        for line in split_paragraphs(section_text):
            heading_candidate = self._extract_heading_candidate(line)
            if self._is_valid_kp_candidate(heading_candidate, require_keyword=True):
                candidates.append(heading_candidate)

            define_candidate = self._extract_definition_candidate(line)
            if self._is_valid_kp_candidate(define_candidate, require_keyword=True):
                candidates.append(define_candidate)

        return self._deduplicate(candidates)

    def _extract_heading_candidate(self, line: str) -> str:
        # 从层级标题或列表项中提取短名称
        match = re.match(
            r"^(?:\d+\.\d+\.\d+|\d+\.\d+|[（(][一二三四五六七八九十]+[)）]|"
            r"[一二三四五六七八九十]+、|[-•●])\s*(.+)$",
            line,
        )
        if not match:
            return ""
        return self._normalize_kp_name(match.group(1))

    def _extract_definition_candidate(self, line: str) -> str:
        # 从“X是/指/定义为”句式中提取术语名
        match = re.match(r"^([A-Za-z\u4e00-\u9fa5][A-Za-z0-9\u4e00-\u9fa5\-\s]{1,24}?)(?:是|指|定义为)", line)
        if not match:
            return ""
        return self._normalize_kp_name(match.group(1))

    def _normalize_kp_name(self, text: str) -> str:
        # 统一清洗知识点名称，尽量保持“概念名/方法名”
        candidate = clean_text(text)
        if not candidate:
            return ""

        candidate = self.NUMBER_PREFIX_PATTERN.sub("", candidate).strip()
        candidate = re.sub(r"[：:]", " ", candidate)
        candidate = re.split(r"[，,。；;（）()]", candidate, maxsplit=1)[0].strip()
        candidate = re.sub(r"\s{2,}", " ", candidate)
        candidate = re.sub(r"^(本章|本节|本小节)\s*", "", candidate)
        candidate = self._strip_trailing_page_no(candidate)
        if "包括" in candidate:
            candidate = candidate.split("包括", maxsplit=1)[0].strip()
        if "主要" in candidate:
            candidate = candidate.split("主要", maxsplit=1)[0].strip()
        return candidate[:40].strip()

    def _is_valid_kp_candidate(self, candidate: str, require_keyword: bool) -> bool:
        # 过滤空值、标题模板词和过长句子
        if not candidate:
            return False
        if candidate in self.KP_STOP_WORDS:
            return False
        if candidate.strip().lower() in {"目录", "目 录", "contents", "index"}:
            return False
        if len(candidate) < 2 or len(candidate) > 30:
            return False
        if re.match(r"^\d+(?:\.\d+)?$", candidate):
            return False
        if re.search(r"[。；;，,]", candidate):
            return False
        if self._is_noisy_kp_candidate(candidate):
            return False
        if any(verb in candidate for verb in ["用于", "通过", "包括", "实现", "能够", "可以"]):
            return False

        if not require_keyword:
            return True
        return any(keyword in candidate for keyword in self.KNOWLEDGE_KEYWORDS)

    def _is_noisy_kp_candidate(self, candidate: str) -> bool:
        text = clean_text(candidate or "")
        if not text:
            return True
        # 过滤明显 OCR 噪声与不可读字符组合
        if re.search(r"[�Þ¤�]", text):
            return True
        useful = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", text)
        if len(useful) / max(1, len(text)) < 0.55:
            return True
        # 排除纯碎片化字符序列（例如：VCÞ住...）
        if re.search(r"[A-Za-z]{1,3}[^\u4e00-\u9fffA-Za-z0-9]{1,4}[A-Za-z]{0,3}", text) and len(text) <= 10:
            return True
        return False

    def _build_kp_description(self, kp_name: str, section_text: str) -> str:
        # 从 section 中挑选包含术语的首句作为描述
        for sentence in re.split(r"[。；;!\n]", section_text):
            sentence = sentence.strip()
            if sentence and kp_name in sentence:
                return sentence[:120]
        return section_text[:120]

    def _build_section_kp_name_map(self, knowledge_points: List[TextbookKnowledgePoint]) -> Dict[str, List[str]]:
        # 将知识点按 section 分组，便于 section/chunk 回填
        mapping: Dict[str, List[str]] = {}
        for kp in knowledge_points:
            if not kp.section_id:
                continue
            mapping.setdefault(kp.section_id, []).append(kp.name)
        for section_id, names in mapping.items():
            mapping[section_id] = self._deduplicate(names)
        return mapping

    def _fill_section_knowledge_points(
        self,
        chapters: List[Dict[str, Any]],
        section_to_kp_names: Dict[str, List[str]],
    ) -> None:
        # 把知识点名称写回 section，保证输出结构可直接消费
        for chapter in chapters:
            for section in chapter["sections"]:
                section["knowledge_points"] = section_to_kp_names.get(section["section_id"], [])

    def _build_chunks(
        self,
        parsed_doc: ParsedDocument,
        chapters: List[Dict[str, Any]],
        section_to_kp_names: Dict[str, List[str]],
    ) -> List[TextbookChunk]:
        # 分块规则：先按小节切，超长小节再按段落切
        chunks: List[TextbookChunk] = []
        chunk_counter = 0
        for chapter in chapters:
            chapter_id = chapter["chapter_id"]
            chapter_title = chapter["chapter_title"]
            for section in chapter["sections"]:
                section_id = section["section_id"]
                section_title = section["section_title"]
                section_text = section.get("raw_text") or section_title
                knowledge_points = section_to_kp_names.get(section_id, [])
                text_pieces = self._split_section_text(section_text)

                for index, piece in enumerate(text_pieces, start=1):
                    chunk_counter += 1
                    chunk_text = clean_text(f"{chapter_title} / {section_title}\n{piece}")
                    tags = self._deduplicate(["textbook", chapter_title, section_title] + knowledge_points)
                    chunks.append(
                        TextbookChunk(
                            chunk_id=f"tb-{parsed_doc.doc_id}-{chunk_counter}",
                            doc_id=parsed_doc.doc_id,
                            chapter_id=chapter_id,
                            section_id=section_id,
                            chapter=chapter_title,
                            section=section_title,
                            text=chunk_text,
                            knowledge_points=knowledge_points,
                            tags=tags,
                            metadata={
                                "source_file": parsed_doc.file_name,
                                "chunk_index_in_section": index,
                                "section_index": section["section_index"],
                                "chapter_index": chapter["chapter_index"],
                            },
                        )
                    )
        return chunks

    def _split_section_text(self, section_text: str, max_chars: int = 420) -> List[str]:
        # 控制 chunk 大小，避免过碎或整章超大块
        text = clean_text(section_text)
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return [text[:max_chars]]

        chunks: List[str] = []
        buffer = ""
        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) + 1 <= max_chars:
                buffer = f"{buffer}\n{paragraph}".strip()
                continue

            if buffer:
                chunks.append(buffer)

            if len(paragraph) <= max_chars:
                buffer = paragraph
            else:
                sentence_buffer = ""
                for sentence in re.split(r"[。；;]", paragraph):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(sentence_buffer) + len(sentence) + 1 <= max_chars:
                        sentence_buffer = f"{sentence_buffer}。{sentence}".strip("。")
                    else:
                        if sentence_buffer:
                            chunks.append(sentence_buffer)
                        sentence_buffer = sentence
                buffer = sentence_buffer

        if buffer:
            chunks.append(buffer)
        return chunks

    def _build_relations(
        self,
        chapters: List[Dict[str, Any]],
        knowledge_points: List[TextbookKnowledgePoint],
        chunks: List[TextbookChunk],
    ) -> List[TextbookRelation]:
        # 构建基础关系：chapter->section, section->kp, section->chunk, kp->kp
        relations: List[TextbookRelation] = []
        seen: set[Tuple[str, str, str]] = set()

        section_to_kp_ids: Dict[str, List[str]] = {}
        for kp in knowledge_points:
            if kp.section_id:
                section_to_kp_ids.setdefault(kp.section_id, []).append(kp.kp_id)

        section_to_chunk_ids: Dict[str, List[str]] = {}
        for chunk in chunks:
            section_to_chunk_ids.setdefault(chunk.section_id, []).append(chunk.chunk_id)

        for chapter in chapters:
            chapter_id = chapter["chapter_id"]
            for section in chapter["sections"]:
                section_id = section["section_id"]
                self._append_relation(relations, seen, chapter_id, section_id, "contains", 0.95)

                for kp_id in section_to_kp_ids.get(section_id, []):
                    self._append_relation(relations, seen, section_id, kp_id, "contains", 0.92)

                for chunk_id in section_to_chunk_ids.get(section_id, []):
                    self._append_relation(relations, seen, section_id, chunk_id, "contains", 0.9)

                kp_ids = section_to_kp_ids.get(section_id, [])
                for left_kp_id, right_kp_id in combinations(kp_ids, 2):
                    self._append_relation(relations, seen, left_kp_id, right_kp_id, "related_to", 0.75)

        return relations

    def _append_relation(
        self,
        relations: List[TextbookRelation],
        seen: set[Tuple[str, str, str]],
        source: str,
        target: str,
        relation: str,
        confidence: float,
    ) -> None:
        # 关系去重，防止同一边重复输出
        if not source or not target:
            return
        key = (source, target, relation)
        if key in seen:
            return
        seen.add(key)
        relations.append(
            TextbookRelation(
                source=source,
                target=target,
                relation=relation,
                confidence=confidence,
            )
        )

    def _build_textbook_info(self, parsed_doc: ParsedDocument, chapters: List[Dict[str, Any]]) -> TextbookInfo:
        # 汇总教材元信息，后续可直接挂到库表
        raw_head = clean_text(parsed_doc.raw_text[:2000])
        filename_stem = clean_text(Path(parsed_doc.file_name).stem)
        edition = ""
        edition_match = re.search(r"第[一二三四五六七八九十百零〇\d]+版", raw_head)
        if edition_match:
            edition = edition_match.group(0)

        authors = self._extract_authors(raw_head)
        title_candidates = [
            parsed_doc.title or "",
            self._guess_title_from_chapters(chapters),
            Path(parsed_doc.file_name).stem,
            parsed_doc.file_name,
        ]
        book_title = ""
        for candidate in title_candidates:
            cleaned = clean_text(candidate or "")
            if not cleaned:
                continue
            if self._is_directory_like_title(cleaned):
                continue
            book_title = cleaned
            break
        if not book_title:
            book_title = clean_text(Path(parsed_doc.file_name).stem) or parsed_doc.file_name

        academic_year = self._extract_academic_year(f"{raw_head}\n{filename_stem}")
        textbook_role = self._infer_textbook_role(book_title=book_title, source_file=parsed_doc.file_name, text=raw_head)
        priority_score = self._compute_textbook_priority_score(
            textbook_role=textbook_role,
            edition=edition,
            academic_year=academic_year,
        )

        return TextbookInfo(
            book_title=book_title,
            source_file=parsed_doc.file_name,
            source_type=parsed_doc.source_type,
            subject=self._infer_subject(book_title, raw_head),
            textbook_role=textbook_role,
            priority_score=priority_score,
            edition=edition or None,
            academic_year=academic_year,
            authors=authors,
        )

    def _infer_textbook_role(self, book_title: str, source_file: str, text: str) -> str:
        value = f"{book_title} {source_file} {text}".lower()
        if any(token in value for token in ["习题", "练习", "辅导", "参考书", "参考资料", "题解", "workbook"]):
            return "supplementary"
        return "main"

    def _extract_authors(self, text: str) -> List[str]:
        # 从常见“作者/主编”行中抽作者名
        patterns = [
            re.compile(r"(?:作者|主编|编著)[:：]\s*([^\n；;。]+)"),
            re.compile(r"^([^\n]{2,30})\s*(?:著|编)$", re.MULTILINE),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            raw_authors = match.group(1)
            names = re.split(r"[、,，/ ]+", raw_authors)
            return [name.strip() for name in names if name.strip()]
        return []

    def _extract_academic_year(self, text: str) -> Optional[str]:
        # 统一抽取教材年份或学年信息，便于后续主辅教材排序。
        value = str(text or "")
        if not value:
            return None
        normalized = value.replace("—", "-").replace("–", "-")
        for pattern in [
            re.compile(r"(20\d{2}\s*[-/]\s*20\d{2}\s*学年?)"),
            re.compile(r"(20\d{2}\s*[-/]\s*20\d{2})"),
            re.compile(r"(20\d{2}\s*年)"),
        ]:
            match = pattern.search(normalized)
            if not match:
                continue
            candidate = re.sub(r"\s+", "", match.group(1))
            return candidate
        return None

    def _compute_textbook_priority_score(
        self,
        textbook_role: str,
        edition: str,
        academic_year: Optional[str],
    ) -> float:
        # 主教材显式优先；再叠加版本与年份信号，输出稳定可比较分数。
        score = 0.0
        role = str(textbook_role or "").strip().lower()
        if role == "main":
            score += 100.0
        elif role == "supplementary":
            score += 45.0
        else:
            score += 60.0

        edition_text = str(edition or "")
        edition_nums = [int(item) for item in re.findall(r"\d+", edition_text)]
        if edition_nums:
            score += min(12.0, edition_nums[-1] * 1.2)

        years = [int(item) for item in re.findall(r"(20\d{2}|19\d{2})", str(academic_year or ""))]
        if years:
            latest = max(years)
            if 1990 <= latest <= 2100:
                score += min(10.0, (latest - 2000) * 0.25)
        return round(score, 4)

    def _guess_title_from_chapters(self, chapters: List[Dict[str, Any]]) -> str:
        # 如果上游没有 title，使用首章标题兜底
        if not chapters:
            return ""
        return chapters[0].get("chapter_title", "")

    def _infer_subject(self, title: str, text: str) -> str:
        # 基于标题和文本关键词猜测学科
        content = f"{title} {text}"
        if "机器学习" in content or "神经网络" in content:
            return "机器学习"
        if "数据结构" in content:
            return "数据结构"
        if "数据库" in content:
            return "数据库"
        if "操作系统" in content:
            return "操作系统"
        return "通用教材"

    def _is_chapter_heading(self, line: str) -> bool:
        # 章节标题匹配，避免误判 section 行
        raw_line = self._normalize_line(line, strip_toc_page_no=False)
        raw_line = self._strip_leading_page_no_prefix(raw_line)
        line = self._normalize_line(line, strip_toc_page_no=True)
        line = self._strip_leading_page_no_prefix(line)
        if not line:
            return False
        if self._looks_like_toc_entry(raw_line) or self._is_metadata_line(line):
            return False
        if self._is_section_heading(line):
            return False

        if self.STRICT_CHAPTER_PATTERN.match(line):
            return True
        if self.CHAPTER_EN_PATTERN.match(line):
            return True

        match = self.ARABIC_CHAPTER_PATTERN.match(line)
        if match:
            index = int(match.group(1))
            title = match.group(2).strip()
            if not (1 <= index <= 40):
                return False
            if len(title) > 18:
                return False
            if len([token for token in re.split(r"\s+", title) if token]) > 3:
                return False
            if not any(keyword in title.lower() for keyword in self.CHAPTER_HINT_KEYWORDS):
                return False
            if re.search(r"(年|月|日|出版社|出版|版|目录|目 录)", title):
                return False
            if re.search(r"\d{3,}", title):
                return False
            if re.search(r"\s+\d{1,4}$", title):
                return False
            return True
        return False

    def _is_section_heading(self, line: str) -> bool:
        # 小节标题匹配，兼容中文和数字编号
        raw_line = self._normalize_line(line, strip_toc_page_no=False)
        line = self._normalize_line(line, strip_toc_page_no=True)
        if not line:
            return False
        if self._is_metadata_line(line):
            return False
        if self._looks_like_toc_entry(raw_line):
            return False

        for pattern in self.SECTION_PATTERNS:
            if pattern.match(line):
                return True
        return False

    def _extract_chapter_index_and_title(self, heading_line: str, fallback_index: int) -> Tuple[str, str]:
        # 抽章节序号与标题
        line = self._normalize_line(heading_line)
        line = self._strip_leading_page_no_prefix(line)
        line = self._strip_trailing_page_no(line)

        match = re.match(r"^(第[一二三四五六七八九十百零〇\d]+章)\s*[：:\s]?(.*)$", line)
        if match:
            index = match.group(1)
            title = match.group(2).strip() or line
            return index, title

        match = re.match(r"^(chapter\s*\d+)\s*[：:\s-]?(.*)$", line, re.IGNORECASE)
        if match:
            index = match.group(1)
            title = match.group(2).strip() or line
            return index, title

        match = re.match(r"^(\d+)\s+(.+)$", line)
        if match:
            index = match.group(1)
            title = self._strip_trailing_page_no(match.group(2).strip())
            return index, title

        return str(fallback_index), line

    def _extract_section_index_and_title(self, heading_line: str, fallback_index: int) -> Tuple[str, str]:
        # 抽小节序号与标题
        line = self._normalize_line(heading_line)
        line = self._strip_leading_page_no_prefix(line)
        line = self._strip_trailing_page_no(line)

        match = re.match(r"^(第[一二三四五六七八九十百零〇\d]+节)\s*[：:\s]?(.*)$", line)
        if match:
            index = match.group(1)
            title = match.group(2).strip() or line
            return index, title

        match = re.match(r"^(\d+\.\d+(?:\.\d+)?)\s*(.*)$", line)
        if match:
            index = match.group(1)
            title = self._strip_trailing_page_no(match.group(2).strip()) or line
            return index, title

        match = re.match(r"^([（(][一二三四五六七八九十]+[)）])\s*(.*)$", line)
        if match:
            index = match.group(1)
            title = self._strip_trailing_page_no(match.group(2).strip()) or line
            return index, title

        return str(fallback_index), line

    def _strip_trailing_page_no(self, text: str) -> str:
        # 去掉标题尾部目录页码残留。
        result = text
        result = re.sub(r"\.{2,}\s*\d+\s*$", "", result)
        result = re.sub(r"[·•]{2,}\s*\d+\s*$", "", result)
        result = re.sub(r"\s+\d{1,4}\s*$", "", result)
        return result.strip()

    def _strip_leading_page_no_prefix(self, text: str) -> str:
        # 去掉标题开头可能混入的页码编号（如“2 第1章绪论”）。
        result = re.sub(r"^\d{1,4}\s+(?=第[一二三四五六七八九十百零〇\d]+章)", "", text).strip()
        result = re.sub(r"^\d{1,4}\s+(?=chapter\s*\d+)", "", result, flags=re.IGNORECASE).strip()
        return result

    def _deduplicate(self, values: List[str]) -> List[str]:
        # 保序去重
        result: List[str] = []
        seen = set()
        for value in values:
            cleaned = clean_text(value)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
        return result
