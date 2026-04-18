import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from services.mongo_kb_service import MongoKBService


class CrossLayerRetrievalService:
    # 最小跨层检索服务：按层检索并返回轻量摘要，不做生成

    DEFAULT_LAYERS = ["syllabus", "textbook", "resource", "hotspot"]
    HOTSPOT_BLOCKED_DOMAINS = {"caip.org.cn", "www.caip.org.cn"}
    SUBJECT_FILTER_LAYERS = {"textbook", "resource"}
    MAIN_TEXTBOOK_HINTS = {"主教材", "指定教材", "core", "main", "primary"}
    SUPPLEMENTARY_TEXTBOOK_HINTS = {"辅", "参考", "习题", "练习", "习题集", "参考书"}
    MAIN_RESOURCE_HINTS = {"课件", "讲义", "lecture", "chap", "章节", "导论", "主讲"}
    AUX_RESOURCE_HINTS = {"习题", "作业", "练习", "总结", "复习", "实验"}
    RESOURCE_ROLE_WEIGHTS = {
        "definition_page": 4,
        "principle_page": 4,
        "formula_page": 3,
        "summary_page": 3,
        "comparison_page": 2,
        "history_page": 2,
        "application_page": 2,
        "case_page": 2,
        "exercise_page": 1,
    }
    QUERY_SYNONYMS = {
        "梯度下降法": ["梯度下降", "gradient descent", "sgd", "随机梯度下降"],
        "梯度下降": ["梯度下降法", "gradient descent", "sgd", "随机梯度下降"],
        "假设空间": ["hypothesis space", "归纳偏好", "版本空间"],
        "文献筛选": ["文献筛查", "摘要筛选", "systematic review", "literature screening"],
        "大模型应用": ["大模型", "llm", "大型语言模型", "生成式ai"],
        "机器学习": ["machine learning", "ml", "监督学习", "非监督学习"],
    }

    def __init__(self, kb_service: MongoKBService) -> None:
        self.kb_service = kb_service
        self._query_variant_cache: Dict[str, List[str]] = {}

    def retrieve_across_layers(
        self,
        query: str,
        subject: Optional[str] = None,
        layers: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        # 跨层检索主入口：返回按 layer 分组的摘要结果
        q = (query or "").strip()
        if not q:
            return {layer: [] for layer in (layers or self.DEFAULT_LAYERS)}

        target_layers = self._normalize_layers(layers)
        results: Dict[str, List[Dict[str, Any]]] = {}

        syllabus_anchor: Optional[Dict[str, Any]] = None
        need_syllabus_anchor = "textbook" in target_layers
        if "syllabus" in target_layers or need_syllabus_anchor:
            syllabus_docs = self._list_docs(layer="syllabus", subject=subject)
            syllabus_limit = max(top_k, 8) if need_syllabus_anchor else top_k
            syllabus_results = self._retrieve_for_layer(
                layer="syllabus",
                docs=syllabus_docs,
                query=q,
                top_k=syllabus_limit,
                subject=subject,
            )
            if "syllabus" in target_layers:
                results["syllabus"] = syllabus_results[: max(1, min(top_k, 20))]
            syllabus_anchor = syllabus_results[0] if syllabus_results else None

        for layer in target_layers:
            if layer == "syllabus":
                continue
            docs = self._list_docs(layer=layer, subject=subject)
            matched = self._retrieve_for_layer(
                layer=layer,
                docs=docs,
                query=q,
                top_k=top_k,
                subject=subject,
                syllabus_anchor=syllabus_anchor,
            )
            results[layer] = matched
        return results

    def _normalize_layers(self, layers: Optional[List[str]]) -> List[str]:
        if not layers:
            return list(self.DEFAULT_LAYERS)
        normalized: List[str] = []
        seen = set()
        for item in layers:
            layer = (item or "").strip().lower()
            if layer not in self.DEFAULT_LAYERS:
                continue
            if layer in seen:
                continue
            seen.add(layer)
            normalized.append(layer)
        return normalized or list(self.DEFAULT_LAYERS)

    def _list_docs(self, layer: str, subject: Optional[str]) -> List[Dict[str, Any]]:
        # 从 Mongo 读取目标层文档，保留 data 供细粒度匹配
        self.kb_service._ensure_available()
        collection_name = self.kb_service.COLLECTION_MAP[layer]
        collection = self.kb_service.db[collection_name]

        mongo_query: Dict[str, Any] = {
            "layer": layer,
            "$or": [{"status": "active"}, {"status": {"$exists": False}}],
        }
        subject_text = (subject or "").strip()
        if subject_text and layer in self.SUBJECT_FILTER_LAYERS:
            mongo_query["subject"] = {"$regex": re.escape(subject), "$options": "i"}

        docs = list(collection.find(mongo_query).sort("updated_at", -1))
        if subject_text and layer in self.SUBJECT_FILTER_LAYERS and not docs:
            # 主题字段有时缺失或不规范，兜底回退到全量本层再做内容打分
            docs = list(
                collection.find(
                    {
                        "layer": layer,
                        "$or": [{"status": "active"}, {"status": {"$exists": False}}],
                    }
                ).sort("updated_at", -1)
            )
        return docs

    def _retrieve_for_layer(
        self,
        layer: str,
        docs: List[Dict[str, Any]],
        query: str,
        top_k: int,
        subject: Optional[str] = None,
        syllabus_anchor: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        scored_items: List[Dict[str, Any]] = []
        for doc in docs:
            if layer == "hotspot" and not self._is_static_hotspot_doc(doc):
                continue

            if layer == "syllabus":
                item = self._match_syllabus(doc, query)
            elif layer == "textbook":
                item = self._match_textbook(doc, query, syllabus_anchor=syllabus_anchor)
            elif layer == "resource":
                item = self._match_resource(doc, query)
            elif layer == "hotspot":
                item = self._match_hotspot(doc, query)
            else:
                continue

            if item is None:
                continue
            subject_boost = self._subject_boost(doc=doc, layer=layer, subject=subject)
            layer_priority = self._layer_priority_boost(
                layer=layer,
                doc=doc,
                item=item,
                query=query,
                subject=subject,
                syllabus_anchor=syllabus_anchor,
            )
            if subject_boost > 0:
                item["_score"] = item.get("_score", 0) + subject_boost
            if layer_priority != 0:
                item["_score"] = item.get("_score", 0) + layer_priority
            item["_layer_priority"] = layer_priority
            item["_subject_boost"] = subject_boost
            item["_evidence_score"] = self._estimate_evidence(layer=layer, item=item)
            scored_items.append(item)

        scored_items.sort(
            key=lambda item: (
                item.get("_score", 0),
                item.get("_layer_priority", 0),
                item.get("_subject_boost", 0),
                item.get("_evidence_score", 0),
                item.get("_match_count", 0),
                item.get("_updated_at", ""),
            ),
            reverse=True,
        )
        if layer == "hotspot":
            scored_items = self._deduplicate_hotspot_items(scored_items)

        output: List[Dict[str, Any]] = []
        for item in scored_items[: max(1, min(top_k, 20))]:
            item.pop("_score", None)
            item.pop("_layer_priority", None)
            item.pop("_subject_boost", None)
            item.pop("_evidence_score", None)
            item.pop("_match_count", None)
            item.pop("_updated_at", None)
            output.append(item)
        return output

    def _match_syllabus(self, doc: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        data = doc.get("data") or {}
        modules = data.get("course_modules") or []
        schedules = data.get("teaching_schedule") or []
        global_knowledge_points = data.get("knowledge_points") or []
        global_key_points = data.get("teaching_key_points") or []
        global_difficult_points = data.get("teaching_difficult_points") or []
        teaching_goals = data.get("teaching_goals") or []

        score = 0
        match_count = 0
        matched_modules: List[str] = []
        matched_key_points: List[str] = []
        matched_difficult_points: List[str] = []
        matched_knowledge_points: List[str] = []
        matched_schedule_topics: List[str] = []
        matched_goals: List[str] = []

        title = doc.get("title") or ""
        subject = doc.get("subject") or ""
        if self._contains(title, query):
            score += 16
            match_count += 1
        if self._contains(subject, query):
            score += 12
            match_count += 1

        for module in modules:
            module_name = str(module.get("module_name") or module.get("module_title") or "").strip()
            if self._contains(module_name, query):
                matched_modules.append(module_name)
                score += 14
                match_count += 1

            for item in module.get("key_points") or []:
                if self._contains(item, query):
                    matched_key_points.append(str(item).strip())
                    score += 20
                    match_count += 1

            for item in module.get("difficult_points") or []:
                if self._contains(item, query):
                    matched_difficult_points.append(str(item).strip())
                    score += 18
                    match_count += 1

        for schedule in schedules:
            topic = str(schedule.get("topic") or "").strip()
            if self._contains(topic, query):
                matched_schedule_topics.append(topic)
                score += 8
                match_count += 1

        for item in global_knowledge_points:
            value = str(item or "").strip()
            if self._contains(value, query):
                matched_knowledge_points.append(value)
                score += 14
                match_count += 1

        for item in global_key_points:
            value = str(item or "").strip()
            if self._contains(value, query):
                matched_key_points.append(value)
                score += 12
                match_count += 1

        for item in global_difficult_points:
            value = str(item or "").strip()
            if self._contains(value, query):
                matched_difficult_points.append(value)
                score += 10
                match_count += 1

        for goal in teaching_goals:
            value = str(goal or "").strip()
            if self._contains(value, query):
                matched_goals.append(self._truncate(value, 140))
                score += 8
                match_count += 1

        if score <= 0:
            return None
        syllabus_meta = self._normalize_syllabus_meta(doc.get("syllabus_meta") or {})
        main_textbooks, reference_textbooks = self._extract_syllabus_materials(data)
        return {
            "doc_id": doc.get("doc_id"),
            "source_file": doc.get("source_file"),
            "title": title,
            "matched_modules": self._deduplicate(matched_modules)[:5],
            "matched_key_points": self._deduplicate(matched_key_points)[:8],
            "matched_difficult_points": self._deduplicate(matched_difficult_points)[:8],
            "matched_knowledge_points": self._deduplicate(matched_knowledge_points)[:10],
            "matched_goals": self._deduplicate(matched_goals)[:4],
            "matched_schedule_topics": self._deduplicate(matched_schedule_topics)[:5],
            "is_primary": bool(syllabus_meta.get("is_primary")),
            "course_name": syllabus_meta.get("course_name"),
            "course_code": syllabus_meta.get("course_code"),
            "school": syllabus_meta.get("school"),
            "department": syllabus_meta.get("department"),
            "major": syllabus_meta.get("major"),
            "academic_year": syllabus_meta.get("academic_year"),
            "semester": syllabus_meta.get("semester"),
            "version": syllabus_meta.get("version"),
            "teacher": syllabus_meta.get("teacher"),
            "effective_date": syllabus_meta.get("effective_date"),
            "priority_score": float(syllabus_meta.get("priority_score") or 0.0),
            "main_textbooks": main_textbooks,
            "reference_textbooks": reference_textbooks,
            "detail_api": f"/kb/doc/syllabus/{doc.get('doc_id')}",
            "_score": score,
            "_match_count": match_count,
            "_updated_at": doc.get("updated_at") or "",
        }

    def _match_textbook(
        self,
        doc: Dict[str, Any],
        query: str,
        syllabus_anchor: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        data = doc.get("data") or {}
        textbook_info = data.get("textbook_info") or {}
        textbook_meta = self._normalize_textbook_meta(doc.get("textbook_meta") or {})
        sections = data.get("sections") or []
        kp_list = data.get("knowledge_points") or []
        chunks = data.get("chunks") or []

        score = 0
        match_count = 0
        matched_sections: List[str] = []
        matched_knowledge_points: List[str] = []
        matched_chunks_preview: List[str] = []

        title = doc.get("title") or ""
        subject = doc.get("subject") or ""
        if self._contains(title, query):
            score += 14
            match_count += 1
        if self._contains(subject, query):
            score += 12
            match_count += 1

        for kp in kp_list:
            name = str(kp.get("name") or "").strip()
            if self._contains(name, query):
                matched_knowledge_points.append(name)
                score += 22
                match_count += 1

        for section in sections:
            section_title = str(section.get("section_title") or "").strip()
            if self._contains(section_title, query):
                matched_sections.append(section_title)
                score += 16
                match_count += 1

        for chunk in chunks:
            text = str(chunk.get("text") or "").strip()
            if self._contains(text, query):
                snippet = self._extract_relevant_snippet(text=text, query=query, size=140)
                if snippet and not self._looks_noisy_chunk(snippet):
                    matched_chunks_preview.append(snippet)
                score += 6
                match_count += 1

        if score <= 0:
            return None
        role, role_source, matched_material = self._resolve_textbook_role(
            doc=doc,
            syllabus_anchor=syllabus_anchor,
            fallback_role=textbook_meta.get("textbook_role") or textbook_info.get("textbook_role"),
        )
        return {
            "doc_id": doc.get("doc_id"),
            "source_file": doc.get("source_file"),
            "title": title,
            "subject": subject,
            "textbook_role": role,
            "textbook_role_source": role_source,
            "matched_syllabus_material": matched_material,
            "syllabus_anchor_doc_id": syllabus_anchor.get("doc_id") if isinstance(syllabus_anchor, dict) else None,
            "priority_score": float(textbook_meta.get("priority_score") or 0.0),
            "edition": textbook_meta.get("edition") or textbook_info.get("edition"),
            "academic_year": textbook_meta.get("academic_year") or textbook_info.get("academic_year"),
            "author": textbook_meta.get("author"),
            "authors": textbook_meta.get("authors") or textbook_info.get("authors") or [],
            "matched_sections": self._deduplicate(matched_sections)[:6],
            "matched_knowledge_points": self._deduplicate(matched_knowledge_points)[:10],
            "matched_chunks_preview": self._deduplicate(matched_chunks_preview)[:5],
            "detail_api": f"/kb/doc/textbook/{doc.get('doc_id')}",
            "_score": score,
            "_match_count": match_count,
            "_updated_at": doc.get("updated_at") or "",
        }

    def _match_resource(self, doc: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        data = doc.get("data") or {}
        pages = data.get("pages") or []
        units = data.get("reusable_units") or []

        score = 0
        match_count = 0
        matched_pages: List[Dict[str, Any]] = []
        matched_page_roles: List[str] = []
        matched_units: List[str] = []

        title = doc.get("title") or ""
        subject = doc.get("subject") or ""
        if self._contains(title, query):
            score += 14
            match_count += 1
        if self._contains(subject, query):
            score += 12
            match_count += 1

        for role in doc.get("page_roles") or []:
            if self._contains(role, query):
                matched_page_roles.append(str(role))
                score += 10
                match_count += 1

        for page in pages:
            page_title = str(page.get("page_title") or "").strip()
            page_summary = str(page.get("page_summary") or "").strip()
            if self._contains(page_title, query) or self._contains(page_summary, query):
                matched_pages.append(
                    {
                        "page_no": page.get("page_no"),
                        "page_title": page_title,
                        "page_role": page.get("page_role"),
                    }
                )
                score += 14
                match_count += 1
                role = str(page.get("page_role") or "").strip()
                if role:
                    matched_page_roles.append(role)

        for unit in units:
            unit_title = str(unit.get("unit_title") or "").strip()
            if self._contains(unit_title, query):
                matched_units.append(unit_title)
                score += 10
                match_count += 1

        if score <= 0:
            return None
        dedup_pages = self._deduplicate_page_hits(matched_pages)
        return {
            "doc_id": doc.get("doc_id"),
            "source_file": doc.get("source_file"),
            "title": title,
            "matched_pages": dedup_pages[:6],
            "matched_page_roles": self._deduplicate(matched_page_roles)[:6],
            "matched_units": self._deduplicate(matched_units)[:6],
            "detail_api": f"/kb/doc/resource/{doc.get('doc_id')}",
            "_score": score,
            "_match_count": match_count,
            "_updated_at": doc.get("updated_at") or "",
        }

    def _match_hotspot(self, doc: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        data = doc.get("data") or {}
        hotspot_items = data.get("hotspot_item") or []
        chunks = data.get("chunks") or []

        score = 0
        match_count = 0
        matched_knowledge_points: List[str] = []
        matched_keywords: List[str] = []
        event_type = None
        summary = ""
        teaching_usage: List[str] = []

        title = doc.get("title") or ""
        if self._contains(title, query):
            score += 15
            match_count += 1

        for evt in doc.get("event_types") or []:
            if self._contains(evt, query):
                score += 12
                match_count += 1
            if event_type is None:
                event_type = str(evt)

        for item in hotspot_items:
            if not summary:
                summary = str(item.get("summary") or "").strip()
            if not teaching_usage:
                teaching_usage = [str(v) for v in (item.get("teaching_usage") or []) if str(v).strip()]
            if event_type is None:
                event_type = str(item.get("event_type") or "")

            for kp in item.get("related_knowledge_points") or []:
                if self._contains(kp, query):
                    matched_knowledge_points.append(str(kp).strip())
                    score += 20
                    match_count += 1

            for kw in item.get("keywords") or []:
                if self._contains(kw, query):
                    matched_keywords.append(str(kw).strip())
                    score += 10
                    match_count += 1

            if self._contains(item.get("summary") or "", query):
                score += 8
                match_count += 1

        for chunk in chunks:
            if self._contains(chunk.get("text") or "", query):
                score += 5
                match_count += 1
                break

        if score <= 0:
            return None
        return {
            "doc_id": doc.get("doc_id"),
            "source_file": doc.get("source_file"),
            "title": title,
            "event_type": event_type,
            "summary": self._truncate(summary, 220),
            "matched_knowledge_points": self._deduplicate(matched_knowledge_points)[:8],
            "matched_keywords": self._deduplicate(matched_keywords)[:8],
            "teaching_usage": self._deduplicate(teaching_usage)[:6],
            "detail_api": f"/kb/doc/hotspot/{doc.get('doc_id')}",
            "_score": score,
            "_match_count": match_count,
            "_updated_at": doc.get("updated_at") or "",
        }

    def _contains(self, text: Any, query: str) -> bool:
        if text is None:
            return False
        value = str(text).strip().lower()
        if not value:
            return False
        variants = self._query_variants(query)
        if not variants:
            return False

        normalized_value = re.sub(r"\s+", "", value)
        for term in variants:
            if not term:
                continue
            if term.isascii() and len(term) < 3:
                continue
            if len(term) < 2:
                continue
            if term in value:
                return True
            compact_term = re.sub(r"\s+", "", term)
            if compact_term and compact_term in normalized_value:
                return True
        return False

    def _truncate(self, text: str, size: int) -> str:
        value = str(text or "").strip()
        if len(value) <= size:
            return value
        return value[:size].rstrip("，,。.;； ") + "..."

    def _extract_relevant_snippet(self, text: str, query: str, size: int = 120) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if not value:
            return ""

        variants = self._query_variants(query)
        lower_value = value.lower()
        best_pos = -1
        best_term = ""
        for term in variants:
            if not term:
                continue
            pos = lower_value.find(term)
            if pos >= 0 and (best_pos < 0 or pos < best_pos):
                best_pos = pos
                best_term = term

        if best_pos < 0:
            return self._truncate(value, size)

        window = max(50, int(size * 0.85))
        start = max(0, best_pos - window // 2)
        end = min(len(value), start + window)
        snippet = value[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(value):
            snippet = snippet + "..."
        if best_term and best_term not in snippet.lower():
            return self._truncate(value, size)
        return self._truncate(snippet, size)

    def _looks_noisy_chunk(self, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        if re.search(r"[.。]{8,}", value):
            return True
        if re.search(r"[=+\-/*]{5,}", value):
            return True
        useful = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", value)
        return len(useful) / max(1, len(value)) < 0.45

    def _deduplicate(self, values: List[Any]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in values:
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result

    def _query_variants(self, query: str) -> List[str]:
        raw = str(query or "").strip().lower()
        if not raw:
            return []
        if raw in self._query_variant_cache:
            return self._query_variant_cache[raw]

        candidates: List[str] = [raw, re.sub(r"\s+", "", raw)]
        if raw.endswith("法") and len(raw) > 2:
            candidates.append(raw[:-1])
        if raw.endswith("算法") and len(raw) > 3:
            candidates.append(raw[:-2])

        for trigger, synonyms in self.QUERY_SYNONYMS.items():
            if trigger in raw:
                candidates.extend([str(item).strip().lower() for item in synonyms])

        unique: List[str] = []
        seen = set()
        for item in candidates:
            value = str(item or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            unique.append(value)

        self._query_variant_cache[raw] = unique
        return unique

    def _subject_boost(self, doc: Dict[str, Any], layer: str, subject: Optional[str]) -> int:
        subject_text = str(subject or "").strip()
        if not subject_text:
            return 0

        score = 0
        if self._contains(doc.get("subject") or "", subject_text):
            score += 8
        if self._contains(doc.get("title") or "", subject_text):
            score += 6

        data = doc.get("data") or {}
        if layer == "syllabus":
            course_info = data.get("course_info") or {}
            if self._contains(course_info.get("course_name") or "", subject_text):
                score += 10
        elif layer == "hotspot":
            for item in data.get("hotspot_item") or []:
                if self._contains(item.get("summary") or "", subject_text):
                    score += 4
                    break
        return score

    def _estimate_evidence(self, layer: str, item: Dict[str, Any]) -> int:
        if layer == "syllabus":
            return (
                len(item.get("matched_modules") or [])
                + len(item.get("matched_key_points") or [])
                + len(item.get("matched_difficult_points") or [])
                + len(item.get("matched_knowledge_points") or [])
                + len(item.get("matched_goals") or [])
                + len(item.get("matched_schedule_topics") or [])
            )
        if layer == "textbook":
            return (
                len(item.get("matched_sections") or [])
                + len(item.get("matched_knowledge_points") or [])
                + len(item.get("matched_chunks_preview") or [])
            )
        if layer == "resource":
            return (
                len(item.get("matched_pages") or []) * 2
                + len(item.get("matched_page_roles") or [])
                + len(item.get("matched_units") or [])
            )
        if layer == "hotspot":
            return (
                len(item.get("matched_knowledge_points") or [])
                + len(item.get("matched_keywords") or [])
                + len(item.get("teaching_usage") or [])
            )
        return 0

    def _layer_priority_boost(
        self,
        layer: str,
        doc: Dict[str, Any],
        item: Dict[str, Any],
        query: str,
        subject: Optional[str] = None,
        syllabus_anchor: Optional[Dict[str, Any]] = None,
    ) -> int:
        if layer == "syllabus":
            return self._syllabus_priority_boost(doc=doc, item=item, query=query, subject=subject)
        if layer == "textbook":
            return self._textbook_priority_boost(
                doc=doc,
                item=item,
                query=query,
                subject=subject,
                syllabus_anchor=syllabus_anchor,
            )
        if layer == "resource":
            return self._resource_priority_boost(doc=doc, item=item, query=query, subject=subject)
        if layer == "hotspot":
            return self._hotspot_priority_boost(doc=doc, item=item, query=query)
        return 0

    def _syllabus_priority_boost(
        self,
        doc: Dict[str, Any],
        item: Dict[str, Any],
        query: str,
        subject: Optional[str] = None,
    ) -> int:
        meta = self._normalize_syllabus_meta(doc.get("syllabus_meta") or {})
        score = 0.0

        if meta.get("is_primary"):
            score += 120.0
        score += float(meta.get("priority_score") or 0.0)

        query_text = str(query or "").strip()
        if query_text:
            course_name = str(meta.get("course_name") or item.get("title") or "").strip()
            course_code = str(meta.get("course_code") or "").strip()
            if course_code and self._contains(course_code, query_text):
                score += 24.0
            if course_name:
                if self._is_exact_text_match(course_name, query_text):
                    score += 20.0
                elif self._contains(course_name, query_text) or self._contains(query_text, course_name):
                    score += 12.0

        subject_text = str(subject or "").strip()
        if subject_text:
            for field_name in ["school", "department", "major", "course_name", "course_code"]:
                if self._contains(meta.get(field_name) or "", subject_text):
                    score += 8.0

        score += self._year_score(meta.get("academic_year"), cap=6.0)
        score += self._year_score(meta.get("effective_date"), cap=4.0)
        score += self._version_score(meta.get("version"))
        return int(round(score))

    def _normalize_syllabus_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "is_primary": self._to_bool(meta.get("is_primary"), False),
            "course_name": self._clean_text(meta.get("course_name")),
            "course_code": self._clean_text(meta.get("course_code")),
            "school": self._clean_text(meta.get("school")),
            "department": self._clean_text(meta.get("department")),
            "major": self._clean_text(meta.get("major")),
            "academic_year": self._clean_text(meta.get("academic_year")),
            "semester": self._clean_text(meta.get("semester")),
            "version": self._clean_text(meta.get("version")),
            "teacher": self._clean_text(meta.get("teacher")),
            "effective_date": self._clean_text(meta.get("effective_date")),
            "priority_score": self._to_float(meta.get("priority_score"), 0.0),
        }

    def _normalize_textbook_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(meta, dict):
            meta = {}
        role = str(meta.get("textbook_role") or "").strip().lower()
        authors = meta.get("authors")
        if not isinstance(authors, list):
            authors = []
        author = self._clean_text(meta.get("author"))
        if not author and authors:
            author = "、".join([str(name).strip() for name in authors if str(name).strip()]) or None
        return {
            "textbook_role": role if role in {"main", "supplementary"} else None,
            "subject": self._clean_text(meta.get("subject")),
            "title": self._clean_text(meta.get("title")),
            "edition": self._clean_text(meta.get("edition")),
            "academic_year": self._clean_text(meta.get("academic_year")),
            "author": author,
            "authors": [str(name).strip() for name in authors if str(name).strip()],
            "source_file": self._clean_text(meta.get("source_file")),
            "priority_score": self._to_float(meta.get("priority_score"), 0.0),
        }

    def _extract_syllabus_materials(self, data: Dict[str, Any]) -> tuple[List[str], List[str]]:
        teaching_materials = data.get("teaching_materials") or {}
        main_textbooks = self._deduplicate(teaching_materials.get("main_textbooks") or [])
        reference_textbooks = self._deduplicate(teaching_materials.get("reference_textbooks") or [])

        if main_textbooks or reference_textbooks:
            return main_textbooks[:12], reference_textbooks[:12]

        # 兼容旧结构：只有平铺 textbooks。
        fallback_main: List[str] = []
        fallback_reference: List[str] = []
        for item in data.get("textbooks") or []:
            text = str(item or "").strip()
            if not text:
                continue
            if self._looks_like_reference_textbook(text):
                fallback_reference.append(text)
            else:
                fallback_main.append(text)
        return self._deduplicate(fallback_main)[:12], self._deduplicate(fallback_reference)[:12]

    def _looks_like_reference_textbook(self, value: str) -> bool:
        text = str(value or "").strip().lower()
        if not text:
            return False
        tokens = ["参考", "推荐", "辅导", "习题", "题解", "workbook", "supplementary"]
        return any(token in text for token in tokens)

    def _resolve_textbook_role(
        self,
        doc: Dict[str, Any],
        syllabus_anchor: Optional[Dict[str, Any]],
        fallback_role: Optional[str],
    ) -> tuple[Optional[str], str, Optional[str]]:
        data = doc.get("data") or {}
        info = data.get("textbook_info") or {}
        title_candidates = self._deduplicate(
            [
                str(doc.get("title") or ""),
                str(doc.get("source_file") or ""),
                str(info.get("book_title") or ""),
            ]
        )

        if isinstance(syllabus_anchor, dict):
            main_materials = syllabus_anchor.get("main_textbooks") or []
            reference_materials = syllabus_anchor.get("reference_textbooks") or []
            main_score, main_hit = self._best_material_match(main_materials, title_candidates)
            ref_score, ref_hit = self._best_material_match(reference_materials, title_candidates)
            if main_score >= max(0.72, ref_score):
                return "main", "syllabus_main", main_hit
            if ref_score >= 0.72:
                return "supplementary", "syllabus_reference", ref_hit

        explicit_role = str(fallback_role or "").strip().lower()
        if explicit_role in {"main", "supplementary"}:
            return explicit_role, "explicit_meta", None

        hint_text = " ".join(title_candidates).lower()
        if any(token in hint_text for token in ["习题", "练习", "辅导", "参考", "题解", "workbook", "supplementary"]):
            return "supplementary", "heuristic_title", None
        if any(token in hint_text for token in ["主教材", "指定教材", "core", "main", "primary"]):
            return "main", "heuristic_title", None
        return None, "unresolved", None

    def _best_material_match(self, materials: List[str], title_candidates: List[str]) -> tuple[float, Optional[str]]:
        best_score = 0.0
        best_hit: Optional[str] = None
        for material in materials or []:
            material_text = str(material or "").strip()
            material_norm = self._normalize_compare_text(material_text)
            if len(material_norm) < 2:
                continue
            for candidate in title_candidates:
                candidate_norm = self._normalize_compare_text(candidate)
                if len(candidate_norm) < 2:
                    continue
                score = 0.0
                if material_norm == candidate_norm:
                    score = 1.0
                elif material_norm in candidate_norm or candidate_norm in material_norm:
                    score = 0.92
                else:
                    ratio = SequenceMatcher(None, material_norm[:120], candidate_norm[:120]).ratio()
                    score = ratio
                if score > best_score:
                    best_score = score
                    best_hit = material_text
        return best_score, best_hit

    def _normalize_compare_text(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\.(pdf|docx|pptx|txt)$", "", text)
        text = re.sub(r"[《》“”\"'‘’\[\]()（）\-_:：，,。；;、\s]+", "", text)
        return text

    def _is_exact_text_match(self, left: str, right: str) -> bool:
        l = re.sub(r"\s+", "", str(left or "").strip().lower())
        r = re.sub(r"\s+", "", str(right or "").strip().lower())
        return bool(l and r and l == r)

    def _year_score(self, value: Optional[str], cap: float) -> float:
        text = str(value or "").strip()
        if not text:
            return 0.0
        years = [int(item) for item in re.findall(r"(20\d{2}|19\d{2})", text)]
        years = [year for year in years if 1990 <= year <= 2100]
        if not years:
            return 0.0
        latest = max(years)
        now_year = datetime.now(timezone.utc).year
        recency = max(0, latest - (now_year - 10))
        return min(cap, recency * 0.6)

    def _version_score(self, value: Optional[str]) -> float:
        text = str(value or "").strip().lower()
        if not text:
            return 0.0
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if nums:
            return min(3.0, float(nums[-1]) * 0.4)
        if "new" in text or "latest" in text or "新版" in text:
            return 2.0
        if "old" in text or "旧版" in text:
            return -1.0
        return 0.0

    def _clean_text(self, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        return text or None

    def _to_bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _to_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _textbook_priority_boost(
        self,
        doc: Dict[str, Any],
        item: Dict[str, Any],
        query: str,
        subject: Optional[str] = None,
        syllabus_anchor: Optional[Dict[str, Any]] = None,
    ) -> int:
        data = doc.get("data") or {}
        info = data.get("textbook_info") or {}
        meta = self._normalize_textbook_meta(doc.get("textbook_meta") or {})
        title = str(doc.get("title") or "").strip()
        source_file = str(doc.get("source_file") or "").strip()
        hint_text = " ".join(
            [
                title,
                source_file,
                str(info.get("book_title") or ""),
                str(info.get("edition") or ""),
                " ".join([str(name) for name in (info.get("authors") or [])]),
                str(meta.get("textbook_role") or ""),
                str(meta.get("edition") or ""),
                str(meta.get("academic_year") or ""),
                str(meta.get("author") or ""),
                str(info.get("role") or ""),
                str(info.get("textbook_role") or ""),
            ]
        ).lower()

        score = 0
        explicit_role = str(meta.get("textbook_role") or item.get("textbook_role") or "").strip().lower()
        role_source = str(item.get("textbook_role_source") or "").strip().lower()
        if role_source == "syllabus_main":
            score += 52
        elif role_source == "syllabus_reference":
            score -= 14
        elif explicit_role == "main":
            score += 34
        elif explicit_role == "supplementary":
            score -= 8
        else:
            if any(hint in hint_text for hint in self.MAIN_TEXTBOOK_HINTS):
                score += 8
            if any(hint in hint_text for hint in self.SUPPLEMENTARY_TEXTBOOK_HINTS):
                score -= 3
        score += min(12, int(float(meta.get("priority_score") or item.get("priority_score") or 0.0) * 0.08))

        if syllabus_anchor:
            course_name = str(syllabus_anchor.get("course_name") or syllabus_anchor.get("title") or "").strip()
            course_code = str(syllabus_anchor.get("course_code") or "").strip()
            if course_name and (self._contains(title, course_name) or self._contains(course_name, title)):
                score += 8
            if course_code and self._contains(hint_text, course_code):
                score += 6

        query_compact = re.sub(r"\s+", "", str(query or "").lower())
        title_compact = re.sub(r"\s+", "", title.lower())
        if query_compact and query_compact in title_compact:
            score += 3

        subject_text = str(subject or "").strip()
        if subject_text:
            if self._contains(meta.get("subject") or "", subject_text):
                score += 6
            if self._contains(info.get("subject") or "", subject_text):
                score += 4
            if self._contains(title, subject_text):
                score += 2

        evidence = (
            len(item.get("matched_knowledge_points") or []) * 2
            + len(item.get("matched_sections") or [])
            + len(item.get("matched_chunks_preview") or [])
        )
        score += min(8, evidence)

        recent_year = self._extract_latest_year(" ".join([hint_text, str(doc.get("updated_at") or "")]))
        if recent_year:
            now_year = datetime.now(timezone.utc).year
            if recent_year >= now_year - 1:
                score += 3
            elif recent_year >= now_year - 3:
                score += 2
            elif recent_year >= now_year - 6:
                score += 1
        return score

    def _resource_priority_boost(
        self,
        doc: Dict[str, Any],
        item: Dict[str, Any],
        query: str,
        subject: Optional[str] = None,
    ) -> int:
        title = str(doc.get("title") or "").strip()
        source_file = str(doc.get("source_file") or "").strip()
        hint_text = f"{title} {source_file}".lower()

        score = 0
        if any(hint in hint_text for hint in self.MAIN_RESOURCE_HINTS):
            score += 5
        if any(hint in hint_text for hint in self.AUX_RESOURCE_HINTS):
            score -= 2

        role_set = {
            str(role or "").strip().lower()
            for role in (item.get("matched_page_roles") or [])
            if str(role).strip()
        }
        role_score = sum(self.RESOURCE_ROLE_WEIGHTS.get(role, 0) for role in role_set)
        score += min(8, role_score)

        matched_pages = item.get("matched_pages") or []
        score += min(6, len(matched_pages) * 2)

        subject_text = str(subject or "").strip()
        if subject_text and self._contains(doc.get("subject") or "", subject_text):
            score += 4

        recent_year = self._extract_latest_year(" ".join([hint_text, str(doc.get("updated_at") or "")]))
        if recent_year:
            now_year = datetime.now(timezone.utc).year
            if recent_year >= now_year - 1:
                score += 2
            elif recent_year >= now_year - 3:
                score += 1
        return score

    def _hotspot_priority_boost(self, doc: Dict[str, Any], item: Dict[str, Any], query: str) -> int:
        data = doc.get("data") or {}
        info = data.get("hotspot_info") or {}
        publish_date = str(info.get("publish_date") or "").strip()
        days_since = self._days_since(publish_date) if publish_date else None

        score = 0
        if days_since is not None:
            if days_since <= 30:
                score += 8
            elif days_since <= 180:
                score += 5
            elif days_since <= 365:
                score += 3
            elif days_since <= 730:
                score += 1

        event_type = str(item.get("event_type") or "").strip().lower()
        if event_type in {"product_release", "industry_application", "research_breakthrough"}:
            score += 2
        if event_type in {"policy_event", "other_event"}:
            score -= 1

        matched_kp_count = len(item.get("matched_knowledge_points") or [])
        if matched_kp_count:
            score += min(6, matched_kp_count * 2)
        else:
            score -= 1

        summary = str(item.get("summary") or "").strip()
        if summary and self._contains(summary, query):
            score += 2
        if len(summary) < 24:
            score -= 1

        source_file = str(doc.get("source_file") or "").lower()
        title = str(doc.get("title") or "").lower()
        if any(token in source_file or token in title for token in ["portal", "导航", "合集", "index"]):
            score -= 2
        return score

    def _deduplicate_page_hits(self, matched_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        seen = set()
        for page in matched_pages:
            page_no = page.get("page_no")
            page_title = str(page.get("page_title") or "").strip().lower()
            page_role = str(page.get("page_role") or "").strip().lower()
            key = (page_no, page_title, page_role)
            if key in seen:
                continue
            seen.add(key)
            output.append(page)
        return output

    def _deduplicate_hotspot_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 同一事件多源报道时优先保留最优一条，避免热点层重复占位。
        unique: List[Dict[str, Any]] = []
        seen_signatures = set()
        for item in items:
            signature = self._hotspot_event_signature(item=item)
            if signature and signature in seen_signatures:
                continue
            if signature:
                seen_signatures.add(signature)
            unique.append(item)
        return unique

    def _hotspot_event_signature(self, item: Dict[str, Any]) -> str:
        title = str(item.get("title") or "").strip().lower()
        summary = str(item.get("summary") or "").strip().lower()
        base = f"{title} {summary}".strip()
        if not base:
            return ""
        base = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " ", base)
        base = re.sub(r"\b\d+(\.\d+)?[kmbw万亿]?\b", " ", base)
        base = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", base)
        return base[:120]

    def _extract_latest_year(self, value: str) -> Optional[int]:
        years = [int(item) for item in re.findall(r"(20\d{2}|19\d{2})", str(value or ""))]
        valid = [year for year in years if 1990 <= year <= 2100]
        if not valid:
            return None
        return max(valid)

    def _days_since(self, date_text: str) -> Optional[int]:
        value = str(date_text or "").strip()
        if not value:
            return None
        normalized = value.replace("/", "-").replace(".", "-")
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y%m%d", "%Y%m"):
            try:
                dt = datetime.strptime(normalized, fmt).replace(tzinfo=timezone.utc)
                return max(0, (datetime.now(timezone.utc) - dt).days)
            except ValueError:
                continue
        return None

    def _is_static_hotspot_doc(self, doc: Dict[str, Any]) -> bool:
        # 热点层当前仅保留静态新闻来源（V1 策略）
        data = doc.get("data") or {}
        info = data.get("hotspot_info") or {}
        url = str(info.get("url") or "").strip().lower()
        source_file = str(doc.get("source_file") or "").strip().lower()
        source_type = str(doc.get("source_type") or "").strip().lower()

        if url:
            domain = MongoKBService.extract_domain(url)
            if domain in self.HOTSPOT_BLOCKED_DOMAINS:
                return False

        if source_type in {"dynamic_web", "dynamic_page", "dynamic_url"}:
            return False
        if "caip" in source_file:
            return False
        if source_file.endswith((".html", ".htm", ".pdf", ".md", ".docx")):
            return True
        return source_type in {"local_file", "web_url", "upload"}
