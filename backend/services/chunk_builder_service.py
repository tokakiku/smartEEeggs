from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from schema.vector_schema import VectorChunkRecord


def _to_str(value: Any) -> str:
    return str(value or "").strip()


class ChunkBuilderService:
    """
    从 Mongo 四层知识库文档构建统一向量化 chunk。
    设计目标：
    1) 结构优先 chunking，固定长度兜底
    2) 每条 chunk 都带可追溯 metadata
    3) 过滤明显噪声（目录页、版权页、纯符号页）
    """

    REQUIRED_FIELDS = ["doc_id", "layer", "chunk_id", "source_file", "title", "subject", "chunk_text"]

    def __init__(
        self,
        fallback_chunk_size: Optional[int] = None,
        fallback_chunk_overlap: Optional[int] = None,
        min_chunk_chars: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
    ) -> None:
        self.fallback_chunk_size = int(
            fallback_chunk_size or os.getenv("VECTOR_CHUNK_SIZE", "900")
        )
        self.fallback_chunk_overlap = int(
            fallback_chunk_overlap or os.getenv("VECTOR_CHUNK_OVERLAP", "120")
        )
        self.min_chunk_chars = int(min_chunk_chars or os.getenv("VECTOR_MIN_CHUNK_CHARS", "40"))
        self.max_chunks_per_doc = int(max_chunks_per_doc or os.getenv("VECTOR_MAX_CHUNKS_PER_DOC", "2400"))

    def build_chunks_from_document(self, layer: str, mongo_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        normalized_layer = _to_str(layer).lower()
        if normalized_layer not in {"syllabus", "textbook", "resource", "hotspot"}:
            return []

        data = mongo_doc.get("data") or {}
        base = self._build_base_meta(normalized_layer, mongo_doc, data)
        chunk_seed = self._doc_seed(normalized_layer, base["doc_id"], base["source_file"])

        if normalized_layer == "syllabus":
            candidates = self._build_syllabus_chunks(data=data, base=base, seed=chunk_seed)
        elif normalized_layer == "textbook":
            candidates = self._build_textbook_chunks(data=data, base=base, seed=chunk_seed)
        elif normalized_layer == "resource":
            candidates = self._build_resource_chunks(data=data, base=base, seed=chunk_seed)
        else:
            candidates = self._build_hotspot_chunks(data=data, base=base, seed=chunk_seed)

        deduped = self._dedupe_and_limit(candidates, max_chunks=self.max_chunks_per_doc)
        return [item.model_dump() for item in deduped]

    def _build_base_meta(self, layer: str, mongo_doc: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        source_file = _to_str(mongo_doc.get("source_file"))
        doc_id = _to_str(mongo_doc.get("doc_id")) or source_file or f"{layer}-unknown-doc"
        title = _to_str(mongo_doc.get("title"))
        subject = _to_str(mongo_doc.get("subject"))

        if not title:
            if layer == "syllabus":
                title = _to_str(data.get("course_name")) or _to_str((data.get("course_info") or {}).get("course_name"))
            elif layer == "textbook":
                title = _to_str((data.get("textbook_info") or {}).get("book_title"))
            elif layer == "resource":
                title = _to_str((data.get("resource_info") or {}).get("title"))
            elif layer == "hotspot":
                title = _to_str((data.get("hotspot_info") or {}).get("title"))
        if not title:
            title = source_file or doc_id

        if not subject:
            if layer == "textbook":
                subject = _to_str((data.get("textbook_info") or {}).get("subject"))
            elif layer == "resource":
                resource_info = data.get("resource_info") or {}
                subject = _to_str(resource_info.get("subject")) or _to_str(resource_info.get("course_topic"))

        syllabus_meta = mongo_doc.get("syllabus_meta") or {}
        textbook_meta = mongo_doc.get("textbook_meta") or {}
        course_info = data.get("course_info") or {}
        textbook_info = data.get("textbook_info") or {}

        return {
            "doc_id": doc_id,
            "layer": layer,
            "source_file": source_file or doc_id,
            "title": title,
            "subject": subject,
            "course_name": _to_str(syllabus_meta.get("course_name"))
            or _to_str(data.get("course_name"))
            or _to_str(course_info.get("course_name")),
            "course_code": _to_str(syllabus_meta.get("course_code"))
            or _to_str(data.get("course_code"))
            or _to_str(course_info.get("course_code")),
            "textbook_role": _to_str(textbook_meta.get("textbook_role"))
            or _to_str(textbook_info.get("textbook_role")),
            "is_primary": bool(syllabus_meta.get("is_primary")) if isinstance(syllabus_meta, dict) else None,
        }

    def _build_syllabus_chunks(self, data: Dict[str, Any], base: Dict[str, Any], seed: str) -> List[VectorChunkRecord]:
        chunks: List[VectorChunkRecord] = []
        module_items = data.get("course_modules") or []
        for idx, module in enumerate(module_items, start=1):
            if not isinstance(module, dict):
                continue
            module_name = _to_str(module.get("module_name")) or f"module_{idx}"
            text = self._join_lines(
                [
                    f"课程模块：{module_name}",
                    f"模块学时：{_to_str(module.get('hours'))}" if _to_str(module.get("hours")) else "",
                    f"模块描述：{_to_str(module.get('description'))}" if _to_str(module.get("description")) else "",
                    self._render_list("学习要求", module.get("learning_requirements")),
                    self._render_list("重点", module.get("key_points")),
                    self._render_list("难点", module.get("difficult_points")),
                    self._render_list("作业", module.get("assignments")),
                ]
            )
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=self._make_chunk_id(seed, "module", idx),
                    chunk_text=text,
                    chapter="course_modules",
                    section=module_name,
                    knowledge_points=module.get("key_points") or [],
                    metadata={
                        "module_index": module.get("module_index"),
                        "module_hours": module.get("hours"),
                    },
                )
            )

        chunks.extend(
            self._build_list_chunks(
                base=base,
                seed=seed,
                chunk_scope="goals",
                section_name="teaching_goals",
                title_text="教学目标",
                values=data.get("teaching_goals") or [],
                group_size=6,
            )
        )
        chunks.extend(
            self._build_list_chunks(
                base=base,
                seed=seed,
                chunk_scope="knowledge",
                section_name="knowledge_points",
                title_text="知识点",
                values=data.get("knowledge_points") or [],
                group_size=12,
            )
        )
        chunks.extend(
            self._build_list_chunks(
                base=base,
                seed=seed,
                chunk_scope="keypoints",
                section_name="teaching_key_points",
                title_text="教学重点",
                values=data.get("teaching_key_points") or [],
                group_size=10,
            )
        )
        chunks.extend(
            self._build_list_chunks(
                base=base,
                seed=seed,
                chunk_scope="difficult",
                section_name="teaching_difficult_points",
                title_text="教学难点",
                values=data.get("teaching_difficult_points") or [],
                group_size=8,
            )
        )

        schedule_items = data.get("teaching_schedule") or []
        if schedule_items:
            rendered_rows: List[str] = []
            for row in schedule_items:
                if isinstance(row, dict):
                    order = _to_str(row.get("order"))
                    topic = _to_str(row.get("topic"))
                    hours = _to_str(row.get("hours"))
                    method = _to_str(row.get("teaching_method"))
                    row_text = f"{order}. {topic}" if order else topic
                    extras = []
                    if hours:
                        extras.append(f"学时:{hours}")
                    if method:
                        extras.append(f"方式:{method}")
                    if extras:
                        row_text = f"{row_text} ({' / '.join(extras)})"
                else:
                    row_text = _to_str(row)
                if row_text:
                    rendered_rows.append(row_text)
            chunks.extend(
                self._build_list_chunks(
                    base=base,
                    seed=seed,
                    chunk_scope="schedule",
                    section_name="teaching_schedule",
                    title_text="教学进度",
                    values=rendered_rows,
                    group_size=8,
                )
            )

        if not chunks:
            raw_sections = data.get("raw_sections") or {}
            raw_text = "\n".join([_to_str(value) for value in raw_sections.values() if _to_str(value)])
            chunks.extend(self._build_fallback_chunks(base=base, seed=seed, text=raw_text, scope="fallback"))
        return chunks

    def _build_textbook_chunks(self, data: Dict[str, Any], base: Dict[str, Any], seed: str) -> List[VectorChunkRecord]:
        chunks: List[VectorChunkRecord] = []

        content_chunks = data.get("chunks") or data.get("content_chunks") or []
        for idx, item in enumerate(content_chunks, start=1):
            payload = item if isinstance(item, dict) else {"text": item}
            text = _to_str(payload.get("text"))
            if not text:
                continue
            chapter = _to_str(payload.get("chapter")) or _to_str(payload.get("chapter_id"))
            section = _to_str(payload.get("section")) or _to_str(payload.get("section_id"))
            chunk_id = _to_str(payload.get("chunk_id")) or self._make_chunk_id(seed, "content", idx)
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=chunk_id,
                    chunk_text=text,
                    chapter=chapter or "chunks",
                    section=section,
                    knowledge_points=payload.get("knowledge_points") or [],
                    tags=payload.get("tags") or [],
                    metadata=payload.get("metadata") or {},
                )
            )

        for idx, section_item in enumerate(data.get("sections") or [], start=1):
            if not isinstance(section_item, dict):
                continue
            section_title = _to_str(section_item.get("section_title")) or _to_str(section_item.get("section_index"))
            chapter_id = _to_str(section_item.get("chapter_id"))
            text = self._join_lines(
                [
                    f"章节：{chapter_id}" if chapter_id else "",
                    f"小节：{section_title}" if section_title else "",
                    _to_str(section_item.get("raw_text")),
                    self._render_list("知识点", section_item.get("knowledge_points")),
                ]
            )
            chunk_id = _to_str(section_item.get("section_id")) or self._make_chunk_id(seed, "section", idx)
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=chunk_id,
                    chunk_text=text,
                    chapter=chapter_id or "sections",
                    section=section_title,
                    knowledge_points=section_item.get("knowledge_points") or [],
                    metadata={"section_index": section_item.get("section_index")},
                )
            )

        for idx, chapter_item in enumerate(data.get("chapters") or [], start=1):
            if not isinstance(chapter_item, dict):
                continue
            chapter_title = _to_str(chapter_item.get("chapter_title")) or _to_str(chapter_item.get("chapter_index"))
            text = self._join_lines(
                [
                    f"章节：{chapter_title}" if chapter_title else "",
                    _to_str(chapter_item.get("raw_text")),
                ]
            )
            chunk_id = _to_str(chapter_item.get("chapter_id")) or self._make_chunk_id(seed, "chapter", idx)
            chunks.extend(
                self._build_fallback_chunks(
                    base=base,
                    seed=seed,
                    text=text,
                    scope=f"chapter-{idx}",
                    chapter=chapter_title or "chapters",
                    section=chapter_title,
                    forced_chunk_id=chunk_id,
                )
            )

        for idx, kp in enumerate(data.get("knowledge_points") or [], start=1):
            if not isinstance(kp, dict):
                continue
            kp_name = _to_str(kp.get("name"))
            text = self._join_lines(
                [
                    f"知识点：{kp_name}" if kp_name else "",
                    f"章节：{_to_str(kp.get('chapter_id'))}" if _to_str(kp.get("chapter_id")) else "",
                    f"小节：{_to_str(kp.get('section_id'))}" if _to_str(kp.get("section_id")) else "",
                    self._render_list("别名", kp.get("aliases")),
                    f"描述：{_to_str(kp.get('description'))}" if _to_str(kp.get("description")) else "",
                    _to_str(kp.get("source_text")),
                ]
            )
            chunk_id = _to_str(kp.get("kp_id")) or self._make_chunk_id(seed, "kp", idx)
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=chunk_id,
                    chunk_text=text,
                    chapter=_to_str(kp.get("chapter_id")),
                    section=_to_str(kp.get("section_id")) or kp_name,
                    knowledge_points=[kp_name] if kp_name else [],
                    metadata={"aliases": kp.get("aliases") or []},
                )
            )
        return chunks

    def _build_resource_chunks(self, data: Dict[str, Any], base: Dict[str, Any], seed: str) -> List[VectorChunkRecord]:
        chunks: List[VectorChunkRecord] = []
        pages = data.get("pages") or []
        for idx, page in enumerate(pages, start=1):
            if not isinstance(page, dict):
                continue
            page_title = _to_str(page.get("page_title")) or f"第{idx}页"
            page_role = _to_str(page.get("page_role"))
            page_no = self._to_int(page.get("page_no"))
            summary = _to_str(page.get("page_summary"))
            page_text = _to_str(page.get("page_text"))

            summary_text = self._join_lines(
                [
                    f"页面标题：{page_title}",
                    f"页面角色：{page_role}" if page_role else "",
                    f"页面摘要：{summary}" if summary else "",
                    self._render_list("知识点", page.get("knowledge_points")),
                ]
            )
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=_to_str(page.get("page_id")) or self._make_chunk_id(seed, "page-summary", idx),
                    chunk_text=summary_text,
                    chapter="pages",
                    section=page_title,
                    page_no=page_no,
                    page_role=page_role,
                    knowledge_points=page.get("knowledge_points") or [],
                    tags=page.get("tags") or [],
                    metadata={"chunk_type": "page_summary"},
                )
            )

            body_candidates = self._split_text(page_text)
            for body_idx, piece in enumerate(body_candidates, start=1):
                chunks.append(
                    self._make_chunk(
                        base=base,
                        chunk_id=self._make_chunk_id(seed, f"page-{idx}-body", body_idx),
                        chunk_text=self._join_lines([f"页面：{page_title}", piece]),
                        chapter="pages",
                        section=page_title,
                        page_no=page_no,
                        page_role=page_role,
                        knowledge_points=page.get("knowledge_points") or [],
                        tags=page.get("tags") or [],
                        metadata={"chunk_type": "page_body"},
                    )
                )

        if not chunks:
            for idx, item in enumerate(data.get("chunks") or [], start=1):
                payload = item if isinstance(item, dict) else {"text": item}
                text = _to_str(payload.get("text"))
                if not text:
                    continue
                chunks.append(
                    self._make_chunk(
                        base=base,
                        chunk_id=_to_str(payload.get("chunk_id")) or self._make_chunk_id(seed, "chunk", idx),
                        chunk_text=text,
                        section=_to_str(payload.get("page_title")) or _to_str(payload.get("section")),
                        page_no=self._to_int(payload.get("page_no")),
                        page_role=_to_str(payload.get("page_role")),
                        knowledge_points=payload.get("knowledge_points") or [],
                        tags=payload.get("tags") or [],
                        metadata=payload.get("metadata") or {},
                    )
                )
        return chunks

    def _build_hotspot_chunks(self, data: Dict[str, Any], base: Dict[str, Any], seed: str) -> List[VectorChunkRecord]:
        chunks: List[VectorChunkRecord] = []
        info = data.get("hotspot_info") or {}
        publish_date = _to_str(info.get("publish_date"))
        info_title = _to_str(info.get("title")) or base["title"]

        headline_text = self._join_lines(
            [
                f"热点标题：{info_title}" if info_title else "",
                f"发布日期：{publish_date}" if publish_date else "",
                f"来源：{_to_str(info.get('source'))}" if _to_str(info.get("source")) else "",
            ]
        )
        if headline_text:
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=self._make_chunk_id(seed, "headline", 1),
                    chunk_text=headline_text,
                    section=info_title or "headline",
                    publish_date=publish_date,
                    metadata={"chunk_type": "headline"},
                )
            )

        for idx, item in enumerate(data.get("hotspot_item") or [], start=1):
            if not isinstance(item, dict):
                continue
            title = _to_str(item.get("title")) or info_title or f"hotspot_item_{idx}"
            summary = _to_str(item.get("summary"))
            event_type = _to_str(item.get("event_type"))
            text = self._join_lines(
                [
                    f"标题：{title}",
                    f"事件类型：{event_type}" if event_type else "",
                    f"摘要：{summary}" if summary else "",
                    self._render_list("关联知识点", item.get("related_knowledge_points")),
                    self._render_list("关键词", item.get("keywords")),
                    self._render_list("教学用途", item.get("teaching_usage")),
                ]
            )
            chunk_id = _to_str(item.get("hotspot_id")) or self._make_chunk_id(seed, "item", idx)
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=chunk_id,
                    chunk_text=text,
                    section=title,
                    publish_date=publish_date,
                    event_type=event_type,
                    knowledge_points=item.get("related_knowledge_points") or [],
                    tags=item.get("tags") or [],
                    metadata={
                        "keywords": item.get("keywords") or [],
                        "news_role": item.get("news_role"),
                    },
                )
            )

            for ev_idx, snippet in enumerate(item.get("evidence_snippets") or [], start=1):
                snippet_text = _to_str(snippet)
                if not snippet_text:
                    continue
                chunks.append(
                    self._make_chunk(
                        base=base,
                        chunk_id=self._make_chunk_id(seed, f"evidence-{idx}", ev_idx),
                        chunk_text=self._join_lines([f"标题：{title}", snippet_text]),
                        section=title,
                        publish_date=publish_date,
                        event_type=event_type,
                        metadata={"chunk_type": "evidence_snippet"},
                    )
                )

        for idx, item in enumerate(data.get("chunks") or [], start=1):
            payload = item if isinstance(item, dict) else {"text": item}
            text = _to_str(payload.get("text"))
            if not text:
                continue
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=_to_str(payload.get("chunk_id")) or self._make_chunk_id(seed, "body", idx),
                    chunk_text=text,
                    section=_to_str(payload.get("section")) or info_title,
                    publish_date=publish_date,
                    event_type=_to_str(payload.get("event_type")),
                    knowledge_points=payload.get("related_knowledge_points") or payload.get("knowledge_points") or [],
                    tags=payload.get("tags") or [],
                    metadata=payload.get("metadata") or {},
                )
            )
        return chunks

    def _build_list_chunks(
        self,
        base: Dict[str, Any],
        seed: str,
        chunk_scope: str,
        section_name: str,
        title_text: str,
        values: Iterable[Any],
        group_size: int,
    ) -> List[VectorChunkRecord]:
        normalized_values = [_to_str(item) for item in values if _to_str(item)]
        if not normalized_values:
            return []

        chunks: List[VectorChunkRecord] = []
        for idx, group in enumerate(self._group_items(normalized_values, max(1, group_size)), start=1):
            text = self._join_lines(
                [
                    f"{title_text}：",
                    *[f"{item_idx}. {item}" for item_idx, item in enumerate(group, start=1)],
                ]
            )
            chunks.append(
                self._make_chunk(
                    base=base,
                    chunk_id=self._make_chunk_id(seed, chunk_scope, idx),
                    chunk_text=text,
                    chapter=section_name,
                    section=title_text,
                    knowledge_points=group if section_name in {"knowledge_points", "teaching_key_points"} else [],
                    metadata={"chunk_type": chunk_scope, "source_section": section_name},
                )
            )
        return chunks

    def _build_fallback_chunks(
        self,
        base: Dict[str, Any],
        seed: str,
        text: str,
        scope: str,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
        forced_chunk_id: Optional[str] = None,
    ) -> List[VectorChunkRecord]:
        pieces = self._split_text(text)
        output: List[VectorChunkRecord] = []
        for idx, piece in enumerate(pieces, start=1):
            chunk_id = forced_chunk_id if idx == 1 and forced_chunk_id else self._make_chunk_id(seed, scope, idx)
            output.append(
                self._make_chunk(
                    base=base,
                    chunk_id=chunk_id,
                    chunk_text=piece,
                    chapter=chapter,
                    section=section,
                    metadata={"chunk_type": "fallback", "scope": scope},
                )
            )
        return output

    def _make_chunk(
        self,
        base: Dict[str, Any],
        chunk_id: str,
        chunk_text: str,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
        page_no: Optional[int] = None,
        page_role: Optional[str] = None,
        publish_date: Optional[str] = None,
        event_type: Optional[str] = None,
        knowledge_points: Optional[List[Any]] = None,
        tags: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        forced_text_cleanup: bool = True,
    ) -> VectorChunkRecord:
        cleaned = self._normalize_text(chunk_text) if forced_text_cleanup else _to_str(chunk_text)
        return VectorChunkRecord(
            doc_id=base["doc_id"],
            layer=base["layer"],
            chunk_id=chunk_id,
            source_file=base["source_file"],
            title=base["title"],
            subject=base["subject"],
            chunk_text=cleaned,
            course_name=base.get("course_name") or None,
            course_code=base.get("course_code") or None,
            textbook_role=base.get("textbook_role") or None,
            is_primary=base.get("is_primary"),
            page_no=page_no,
            chapter=_to_str(chapter) or None,
            section=_to_str(section) or None,
            page_role=_to_str(page_role) or None,
            publish_date=_to_str(publish_date) or None,
            event_type=_to_str(event_type) or None,
            knowledge_points=[_to_str(item) for item in (knowledge_points or []) if _to_str(item)],
            tags=[_to_str(item) for item in (tags or []) if _to_str(item)],
            metadata=metadata or {},
        )

    def _dedupe_and_limit(self, chunks: List[VectorChunkRecord], max_chunks: int) -> List[VectorChunkRecord]:
        deduped: List[VectorChunkRecord] = []
        seen = set()
        for item in chunks:
            text = self._normalize_text(item.chunk_text)
            if self._is_noisy(text):
                continue
            dedup_key = f"{item.layer}|{item.doc_id}|{item.section or ''}|{self._fingerprint(text)}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            item.chunk_text = text
            deduped.append(item)
            if len(deduped) >= max_chunks:
                break
        return deduped

    def validate_required_metadata(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        missing_samples: List[Dict[str, Any]] = []
        missing_count = 0
        for item in chunks:
            missing = [field for field in self.REQUIRED_FIELDS if field not in item]
            if missing:
                missing_count += 1
                if len(missing_samples) < 20:
                    missing_samples.append(
                        {
                            "doc_id": item.get("doc_id"),
                            "layer": item.get("layer"),
                            "chunk_id": item.get("chunk_id"),
                            "missing_fields": missing,
                        }
                    )
        return {
            "total_chunks": len(chunks),
            "missing_required_count": missing_count,
            "all_required_present": missing_count == 0,
            "missing_samples": missing_samples,
        }

    def _split_text(self, text: str) -> List[str]:
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        if len(normalized) <= self.fallback_chunk_size:
            return [normalized]

        chunks: List[str] = []
        start = 0
        max_len = self.fallback_chunk_size
        overlap = max(0, min(self.fallback_chunk_overlap, max_len // 3))
        punctuation = "。！？；.!?;\n"

        while start < len(normalized):
            end = min(start + max_len, len(normalized))
            if end < len(normalized):
                window = normalized[start:end]
                best_break = -1
                for token in punctuation:
                    pos = window.rfind(token)
                    if pos > best_break:
                        best_break = pos
                if best_break >= int(max_len * 0.45):
                    end = start + best_break + 1

            piece = normalized[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(normalized):
                break
            next_start = end - overlap
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks

    def _normalize_text(self, text: Any) -> str:
        value = _to_str(text)
        if not value:
            return ""
        value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", value)
        value = value.replace("\uFFFD", " ")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    def _is_noisy(self, text: str) -> bool:
        value = _to_str(text)
        if not value:
            return True
        if len(value) < self.min_chunk_chars:
            return True
        compact = value.lower().replace(" ", "")
        if re.fullmatch(r"(目录|目\s*录|contents?)", compact):
            return True
        if re.search(r"(copyright|all rights reserved|版权所有|未经许可|isbn)", compact) and len(compact) < 240:
            return True
        if re.search(r"[=+\-/*]{8,}", value):
            return True
        if re.search(r"[.。·]{10,}", value):
            return True
        if re.fullmatch(r"[\W_]+", value):
            return True

        useful_chars = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", value)
        if len(useful_chars) / max(1, len(value)) < 0.35:
            return True
        return False

    @staticmethod
    def _doc_seed(layer: str, doc_id: str, source_file: str) -> str:
        token = f"{layer}:{doc_id}:{source_file}"
        return hashlib.md5(token.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _make_chunk_id(seed: str, scope: str, index: int) -> str:
        safe_scope = re.sub(r"[^a-zA-Z0-9_\-]+", "-", _to_str(scope)).strip("-") or "part"
        return f"{safe_scope}-{seed}-{index:04d}"

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _group_items(values: List[str], group_size: int) -> Iterable[List[str]]:
        step = max(1, group_size)
        for i in range(0, len(values), step):
            yield values[i : i + step]

    @staticmethod
    def _join_lines(lines: Iterable[str]) -> str:
        return "\n".join([_to_str(line) for line in lines if _to_str(line)]).strip()

    def _render_list(self, label: str, values: Optional[Iterable[Any]]) -> str:
        items = [_to_str(item) for item in (values or []) if _to_str(item)]
        if not items:
            return ""
        return self._join_lines([f"{label}：", *[f"- {item}" for item in items]])

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except Exception:
            return None
