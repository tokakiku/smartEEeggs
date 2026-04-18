import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

"""图谱三元组规则抽取器。"""

from schema.graph_schema import KnowledgeTriple
from utils.text_cleaner import clean_text


class BaseSemanticTripleExtractor(ABC):
    """语义抽取扩展点（预留给未来 LLM 抽取能力）。"""

    @abstractmethod
    def extract(self, layer: str, doc: Dict[str, Any]) -> List[KnowledgeTriple]:
        raise NotImplementedError


class NoopSemanticTripleExtractor(BaseSemanticTripleExtractor):
    """默认语义抽取器：未启用外部模型时返回空结果。"""

    def extract(self, layer: str, doc: Dict[str, Any]) -> List[KnowledgeTriple]:
        _ = layer
        _ = doc
        return []


class GraphTripleExtractor:
    """四层 Mongo 文档的规则优先三元组抽取器。"""

    INVALID_TEXT_VALUES = {
        "",
        "none",
        "null",
        "nan",
        "n/a",
        "unknown",
        "目录",
        "目 录",
        "contents",
    }

    METHOD_HINTS = [
        "算法",
        "method",
        "model",
        "模型",
        "network",
        "回归",
        "分类器",
        "树",
        "bayes",
        "svm",
        "transformer",
    ]

    HYPERPARAMETER_HINTS = [
        "学习率",
        "正则化",
        "lambda",
        "alpha",
        "beta",
        "gamma",
        "batch",
        "epoch",
        "dropout",
        "温度",
        "k值",
        "k-",
        "阈值",
        "步长",
    ]

    HOTSPOT_TOPIC_HINTS = [
        "开源",
        "社区",
        "github",
        "agent",
        "平台",
        "生态",
        "应用",
        "部署",
        "场景",
        "产品",
        "政策",
        "监管",
        "research",
        "breakthrough",
    ]

    def __init__(
        self,
        semantic_extractor: Optional[BaseSemanticTripleExtractor] = None,
        enable_semantic: bool = False,
    ) -> None:
        self.semantic_extractor = semantic_extractor or NoopSemanticTripleExtractor()
        self.enable_semantic = bool(enable_semantic)

    def extract_document(self, doc: Dict[str, Any]) -> List[KnowledgeTriple]:
        layer = str(doc.get("layer") or "").strip().lower()
        if layer not in {"syllabus", "textbook", "resource", "hotspot"}:
            return []

        data = doc.get("data") or {}
        triples: List[KnowledgeTriple] = []
        if layer == "syllabus":
            triples.extend(self._extract_syllabus(doc, data))
        elif layer == "textbook":
            triples.extend(self._extract_textbook(doc, data))
        elif layer == "resource":
            triples.extend(self._extract_resource(doc, data))
        elif layer == "hotspot":
            triples.extend(self._extract_hotspot(doc, data))

        if self.enable_semantic:
            try:
                triples.extend(self.semantic_extractor.extract(layer=layer, doc=doc))
            except Exception:
                # 语义抽取失败时仍保持规则链路可用，避免整批中断。
                pass

        return self._deduplicate_triples(triples)

    def _extract_syllabus(self, doc: Dict[str, Any], data: Dict[str, Any]) -> List[KnowledgeTriple]:
        triples: List[KnowledgeTriple] = []
        course_info = data.get("course_info") or {}
        course_name = self._pick_first(
            [
                data.get("course_name"),
                course_info.get("course_name"),
                doc.get("title"),
                doc.get("subject"),
                doc.get("source_file"),
            ]
        )
        course_code = self._pick_first([data.get("course_code"), course_info.get("course_code")])

        if course_name and course_code:
            triple = self._build_triple(
                doc=doc,
                head=course_name,
                relation="has_course_code",
                tail=course_code,
                head_type="course",
                tail_type="course_code",
                source_field="data.course_code",
                confidence=0.99,
            )
            if triple:
                triples.append(triple)

        for prerequisite in data.get("prerequisites") or course_info.get("prerequisite_courses") or []:
            triple = self._build_triple(
                doc=doc,
                head=course_name,
                relation="requires_course",
                tail=prerequisite,
                head_type="course",
                tail_type="course",
                source_field="data.prerequisites",
                confidence=0.97,
            )
            if triple:
                triples.append(triple)

        for goal in data.get("teaching_goals") or []:
            triple = self._build_triple(
                doc=doc,
                head=course_name,
                relation="has_teaching_goal",
                tail=goal,
                head_type="course",
                tail_type="teaching_goal",
                source_field="data.teaching_goals",
                evidence_text=goal,
                confidence=0.95,
            )
            if triple:
                triples.append(triple)

        for kp in data.get("teaching_key_points") or []:
            triple = self._build_triple(
                doc=doc,
                head=course_name,
                relation="has_key_point",
                tail=kp,
                head_type="course",
                tail_type="knowledge_point",
                source_field="data.teaching_key_points",
                confidence=0.95,
            )
            if triple:
                triples.append(triple)

        for dp in data.get("teaching_difficult_points") or []:
            triple = self._build_triple(
                doc=doc,
                head=course_name,
                relation="has_difficult_point",
                tail=dp,
                head_type="course",
                tail_type="knowledge_point",
                source_field="data.teaching_difficult_points",
                confidence=0.95,
            )
            if triple:
                triples.append(triple)

        for kp in data.get("knowledge_points") or []:
            triple = self._build_triple(
                doc=doc,
                head=course_name,
                relation="covers_knowledge_point",
                tail=kp,
                head_type="course",
                tail_type="knowledge_point",
                source_field="data.knowledge_points",
                confidence=0.93,
            )
            if triple:
                triples.append(triple)

        for module in data.get("course_modules") or []:
            module_name = self._pick_first([module.get("module_name"), module.get("module_title")])
            module_desc = self._pick_first([module.get("description")])
            if module_name:
                triple = self._build_triple(
                    doc=doc,
                    head=course_name,
                    relation="contains_module",
                    tail=module_name,
                    head_type="course",
                    tail_type="module",
                    source_field="data.course_modules[].module_name",
                    evidence_text=module_desc,
                    confidence=0.98,
                )
                if triple:
                    triples.append(triple)

                reverse_triple = self._build_triple(
                    doc=doc,
                    head=module_name,
                    relation="belongs_to_course",
                    tail=course_name,
                    head_type="module",
                    tail_type="course",
                    source_field="data.course_modules[].module_name",
                    confidence=0.98,
                )
                if reverse_triple:
                    triples.append(reverse_triple)

            for key_point in module.get("key_points") or []:
                triple = self._build_triple(
                    doc=doc,
                    head=module_name,
                    relation="contains_knowledge_point",
                    tail=key_point,
                    head_type="module",
                    tail_type="knowledge_point",
                    source_field="data.course_modules[].key_points",
                    evidence_text=module_desc,
                    confidence=0.97,
                )
                if triple:
                    triples.append(triple)

                reverse_triple = self._build_triple(
                    doc=doc,
                    head=key_point,
                    relation="belongs_to_course",
                    tail=course_name,
                    head_type="knowledge_point",
                    tail_type="course",
                    source_field="data.course_modules[].key_points",
                    confidence=0.95,
                )
                if reverse_triple:
                    triples.append(reverse_triple)

            for difficult_point in module.get("difficult_points") or []:
                triple = self._build_triple(
                    doc=doc,
                    head=module_name,
                    relation="contains_difficult_point",
                    tail=difficult_point,
                    head_type="module",
                    tail_type="knowledge_point",
                    source_field="data.course_modules[].difficult_points",
                    evidence_text=module_desc,
                    confidence=0.97,
                )
                if triple:
                    triples.append(triple)

                reverse_triple = self._build_triple(
                    doc=doc,
                    head=difficult_point,
                    relation="is_difficult_point_of",
                    tail=course_name,
                    head_type="knowledge_point",
                    tail_type="course",
                    source_field="data.course_modules[].difficult_points",
                    confidence=0.95,
                )
                if reverse_triple:
                    triples.append(reverse_triple)

            for learning_requirement in module.get("learning_requirements") or []:
                triple = self._build_triple(
                    doc=doc,
                    head=module_name,
                    relation="has_learning_requirement",
                    tail=learning_requirement,
                    head_type="module",
                    tail_type="learning_requirement",
                    source_field="data.course_modules[].learning_requirements",
                    evidence_text=learning_requirement,
                    confidence=0.9,
                )
                if triple:
                    triples.append(triple)

        return triples

    def _extract_textbook(self, doc: Dict[str, Any], data: Dict[str, Any]) -> List[KnowledgeTriple]:
        triples: List[KnowledgeTriple] = []
        textbook_info = data.get("textbook_info") or {}
        textbook_title = self._pick_textbook_title(doc, textbook_info)
        subject = self._pick_first([textbook_info.get("subject"), doc.get("subject")])

        if textbook_title and subject:
            triple = self._build_triple(
                doc=doc,
                head=textbook_title,
                relation="belongs_to_subject",
                tail=subject,
                head_type="textbook",
                tail_type="subject",
                source_field="data.textbook_info.subject",
                confidence=0.98,
            )
            if triple:
                triples.append(triple)

        chapter_title_by_id: Dict[str, str] = {}
        for chapter in data.get("chapters") or []:
            chapter_id = str(chapter.get("chapter_id") or "").strip()
            chapter_title = self._pick_first([chapter.get("chapter_title"), chapter.get("chapter_index")])
            if chapter_id and chapter_title:
                chapter_title_by_id[chapter_id] = chapter_title

            chapter_triple = self._build_triple(
                doc=doc,
                head=textbook_title,
                relation="contains_chapter",
                tail=chapter_title,
                head_type="textbook",
                tail_type="chapter",
                source_field="data.chapters[].chapter_title",
                evidence_text=self._pick_first([chapter.get("raw_text")]),
                confidence=0.98,
            )
            if chapter_triple:
                triples.append(chapter_triple)

            reverse_triple = self._build_triple(
                doc=doc,
                head=chapter_title,
                relation="belongs_to_textbook",
                tail=textbook_title,
                head_type="chapter",
                tail_type="textbook",
                source_field="data.chapters[].chapter_title",
                confidence=0.98,
            )
            if reverse_triple:
                triples.append(reverse_triple)

            for section in chapter.get("sections") or []:
                triples.extend(
                    self._extract_textbook_section(
                        doc=doc,
                        textbook_title=textbook_title,
                        chapter_title=chapter_title,
                        section=section,
                    )
                )

        for section in data.get("sections") or []:
            chapter_title = self._pick_first(
                [chapter_title_by_id.get(str(section.get("chapter_id") or "").strip()), section.get("chapter")]
            )
            triples.extend(
                self._extract_textbook_section(
                    doc=doc,
                    textbook_title=textbook_title,
                    chapter_title=chapter_title,
                    section=section,
                )
            )

        for kp in data.get("knowledge_points") or []:
            if isinstance(kp, dict):
                kp_name = self._pick_first([kp.get("name"), kp.get("kp_id")])
                kp_type = self._infer_kp_type(kp_name)
                kp_source_field = "data.knowledge_points[].name"
                kp_description = self._pick_first([kp.get("description"), kp.get("source_text")])
            else:
                kp_name = self._pick_first([kp])
                kp_type = self._infer_kp_type(kp_name)
                kp_source_field = "data.knowledge_points[]"
                kp_description = None

            kp_triple = self._build_triple(
                doc=doc,
                head=textbook_title,
                relation="covers_knowledge_point",
                tail=kp_name,
                head_type="textbook",
                tail_type=kp_type,
                source_field=kp_source_field,
                evidence_text=kp_description,
                confidence=0.95,
            )
            if kp_triple:
                triples.append(kp_triple)

            if isinstance(kp, dict):
                chapter_title = chapter_title_by_id.get(str(kp.get("chapter_id") or "").strip())
                chapter_link = self._build_triple(
                    doc=doc,
                    head=kp_name,
                    relation="located_in_chapter",
                    tail=chapter_title,
                    head_type=kp_type,
                    tail_type="chapter",
                    source_field="data.knowledge_points[].chapter_id",
                    confidence=0.9,
                )
                if chapter_link:
                    triples.append(chapter_link)

                for alias in kp.get("aliases") or []:
                    alias_triple = self._build_triple(
                        doc=doc,
                        head=alias,
                        relation="alias_of",
                        tail=kp_name,
                        head_type=kp_type,
                        tail_type=kp_type,
                        source_field="data.knowledge_points[].aliases",
                        confidence=0.9,
                    )
                    if alias_triple:
                        triples.append(alias_triple)

        for relation in data.get("relations") or []:
            rel_name = self._normalize_relation_name(relation.get("relation"))
            rel_triple = self._build_triple(
                doc=doc,
                head=relation.get("source"),
                relation=rel_name,
                tail=relation.get("target"),
                source_field="data.relations[]",
                confidence=self._to_float(relation.get("confidence"), 0.85),
            )
            if rel_triple:
                triples.append(rel_triple)

        return triples

    def _extract_textbook_section(
        self,
        doc: Dict[str, Any],
        textbook_title: str,
        chapter_title: Optional[str],
        section: Dict[str, Any],
    ) -> List[KnowledgeTriple]:
        triples: List[KnowledgeTriple] = []
        section_title = self._pick_first([section.get("section_title"), section.get("section_index")])
        section_text = self._pick_first([section.get("raw_text"), section.get("text")])

        if chapter_title:
            relation = self._build_triple(
                doc=doc,
                head=chapter_title,
                relation="contains_section",
                tail=section_title,
                head_type="chapter",
                tail_type="section",
                source_field="data.sections[].section_title",
                evidence_text=section_text,
                confidence=0.97,
            )
            if relation:
                triples.append(relation)

        section_owner = self._build_triple(
            doc=doc,
            head=section_title,
            relation="belongs_to_textbook",
            tail=textbook_title,
            head_type="section",
            tail_type="textbook",
            source_field="data.sections[].section_title",
            confidence=0.95,
        )
        if section_owner:
            triples.append(section_owner)

        for kp in section.get("knowledge_points") or []:
            kp_name = self._pick_first([kp.get("name") if isinstance(kp, dict) else kp])
            kp_type = self._infer_kp_type(kp_name)
            section_kp = self._build_triple(
                doc=doc,
                head=section_title,
                relation="contains_knowledge_point",
                tail=kp_name,
                head_type="section",
                tail_type=kp_type,
                source_field="data.sections[].knowledge_points",
                evidence_text=section_text,
                confidence=0.95,
            )
            if section_kp:
                triples.append(section_kp)

            textbook_kp = self._build_triple(
                doc=doc,
                head=kp_name,
                relation="belongs_to_textbook",
                tail=textbook_title,
                head_type=kp_type,
                tail_type="textbook",
                source_field="data.sections[].knowledge_points",
                confidence=0.9,
            )
            if textbook_kp:
                triples.append(textbook_kp)

        method_name = self._infer_method_name(section_title=section_title, section_text=section_text)
        if method_name:
            method_triple = self._build_triple(
                doc=doc,
                head=section_title,
                relation="describes_method",
                tail=method_name,
                head_type="section",
                tail_type="method",
                source_field="data.sections[].section_title",
                evidence_text=section_text,
                confidence=0.9,
            )
            if method_triple:
                triples.append(method_triple)

            for hp in self._extract_hyperparameters(section_text):
                hp_triple = self._build_triple(
                    doc=doc,
                    head=method_name,
                    relation="has_hyperparameter",
                    tail=hp,
                    head_type="method",
                    tail_type="hyperparameter",
                    source_field="data.sections[].raw_text",
                    evidence_text=section_text,
                    confidence=0.82,
                )
                if hp_triple:
                    triples.append(hp_triple)

        return triples

    def _extract_resource(self, doc: Dict[str, Any], data: Dict[str, Any]) -> List[KnowledgeTriple]:
        triples: List[KnowledgeTriple] = []
        resource_info = data.get("resource_info") or {}
        topic = self._pick_first(
            [
                resource_info.get("course_topic"),
                resource_info.get("title"),
                doc.get("title"),
                doc.get("subject"),
                doc.get("source_file"),
            ]
        )
        subject = self._pick_first([resource_info.get("subject"), doc.get("subject")])

        if topic and subject and topic != subject:
            topic_triple = self._build_triple(
                doc=doc,
                head=topic,
                relation="belongs_to_subject",
                tail=subject,
                head_type="resource_topic",
                tail_type="subject",
                source_field="data.resource_info.subject",
                confidence=0.97,
            )
            if topic_triple:
                triples.append(topic_triple)

        page_title_by_id: Dict[str, str] = {}
        for page in data.get("pages") or []:
            page_id = str(page.get("page_id") or "").strip()
            page_title = self._pick_first([page.get("page_title"), page_id])
            page_title_by_id[page_id] = page_title
            page_summary = self._pick_first([page.get("page_summary"), page.get("page_text")])
            page_role = self._pick_first([page.get("page_role")])

            page_link = self._build_triple(
                doc=doc,
                head=topic,
                relation="contains_resource_page",
                tail=page_title,
                head_type="resource_topic",
                tail_type="resource_page",
                source_field="data.pages[].page_title",
                chunk_id=page_id or None,
                evidence_text=page_summary,
                confidence=0.98,
            )
            if page_link:
                triples.append(page_link)

            reverse_page = self._build_triple(
                doc=doc,
                head=page_title,
                relation="belongs_to_topic",
                tail=topic,
                head_type="resource_page",
                tail_type="resource_topic",
                source_field="data.pages[].page_title",
                chunk_id=page_id or None,
                confidence=0.97,
            )
            if reverse_page:
                triples.append(reverse_page)

            role_triple = self._build_triple(
                doc=doc,
                head=page_title,
                relation="has_page_role",
                tail=page_role,
                head_type="resource_page",
                tail_type="resource_page_role",
                source_field="data.pages[].page_role",
                chunk_id=page_id or None,
                confidence=0.96,
            )
            if role_triple:
                triples.append(role_triple)

            for kp in page.get("knowledge_points") or []:
                kp_triple = self._build_triple(
                    doc=doc,
                    head=page_title,
                    relation="explains",
                    tail=kp,
                    head_type="resource_page",
                    tail_type=self._infer_kp_type(kp),
                    source_field="data.pages[].knowledge_points",
                    chunk_id=page_id or None,
                    evidence_text=page_summary,
                    confidence=0.94,
                )
                if kp_triple:
                    triples.append(kp_triple)

        for unit in data.get("reusable_units") or []:
            unit_id = str(unit.get("unit_id") or "").strip()
            unit_title = self._pick_first([unit.get("unit_title"), unit_id])
            page_title = page_title_by_id.get(str(unit.get("page_id") or "").strip())
            unit_summary = self._pick_first([unit.get("unit_summary")])

            unit_topic = self._build_triple(
                doc=doc,
                head=topic,
                relation="contains_resource_unit",
                tail=unit_title,
                head_type="resource_topic",
                tail_type="resource_unit",
                source_field="data.reusable_units[].unit_title",
                chunk_id=unit_id or None,
                evidence_text=unit_summary,
                confidence=0.95,
            )
            if unit_topic:
                triples.append(unit_topic)

            page_relation = self._build_triple(
                doc=doc,
                head=unit_title,
                relation="belongs_to_resource_page",
                tail=page_title,
                head_type="resource_unit",
                tail_type="resource_page",
                source_field="data.reusable_units[].page_id",
                chunk_id=unit_id or None,
                confidence=0.95,
            )
            if page_relation:
                triples.append(page_relation)

            for kp in unit.get("knowledge_points") or []:
                kp_relation = self._build_triple(
                    doc=doc,
                    head=unit_title,
                    relation="explains",
                    tail=kp,
                    head_type="resource_unit",
                    tail_type=self._infer_kp_type(kp),
                    source_field="data.reusable_units[].knowledge_points",
                    chunk_id=unit_id or None,
                    evidence_text=unit_summary,
                    confidence=0.93,
                )
                if kp_relation:
                    triples.append(kp_relation)

            for use in unit.get("recommended_use") or []:
                use_relation = self._build_triple(
                    doc=doc,
                    head=unit_title,
                    relation="supports",
                    tail=use,
                    head_type="resource_unit",
                    tail_type="teaching_usage",
                    source_field="data.reusable_units[].recommended_use",
                    chunk_id=unit_id or None,
                    confidence=0.9,
                )
                if use_relation:
                    triples.append(use_relation)

        for relation in data.get("relations") or []:
            rel_name = self._normalize_relation_name(relation.get("relation"))
            rel_triple = self._build_triple(
                doc=doc,
                head=relation.get("source"),
                relation=rel_name,
                tail=relation.get("target"),
                source_field="data.relations[]",
                confidence=self._to_float(relation.get("confidence"), 0.85),
            )
            if rel_triple:
                triples.append(rel_triple)

        return triples

    def _extract_hotspot(self, doc: Dict[str, Any], data: Dict[str, Any]) -> List[KnowledgeTriple]:
        triples: List[KnowledgeTriple] = []
        hotspot_info = data.get("hotspot_info") or {}
        default_title = self._pick_first([hotspot_info.get("title"), doc.get("title"), doc.get("source_file")])
        publish_date = self._pick_first([hotspot_info.get("publish_date")])

        for item in data.get("hotspot_item") or [{}]:
            event_title = self._pick_first([item.get("title"), default_title])
            event_type = self._pick_first([item.get("event_type"), "other_event"])
            event_summary = self._pick_first([item.get("summary")])

            event_type_triple = self._build_triple(
                doc=doc,
                head=event_title,
                relation="has_event_type",
                tail=event_type,
                head_type="event",
                tail_type="event_type",
                source_field="data.hotspot_item[].event_type",
                evidence_text=event_summary,
                confidence=0.98,
            )
            if event_type_triple:
                triples.append(event_type_triple)

            published_triple = self._build_triple(
                doc=doc,
                head=event_title,
                relation="published_on",
                tail=publish_date,
                head_type="event",
                tail_type="date",
                source_field="data.hotspot_info.publish_date",
                confidence=0.95,
            )
            if published_triple:
                triples.append(published_triple)

            for kp in item.get("related_knowledge_points") or []:
                kp_triple = self._build_triple(
                    doc=doc,
                    head=event_title,
                    relation="applies",
                    tail=kp,
                    head_type="event",
                    tail_type=self._infer_kp_type(kp),
                    source_field="data.hotspot_item[].related_knowledge_points",
                    evidence_text=event_summary,
                    confidence=0.93,
                )
                if kp_triple:
                    triples.append(kp_triple)

            for keyword in item.get("keywords") or []:
                keyword_triple = self._build_triple(
                    doc=doc,
                    head=event_title,
                    relation="involves_topic",
                    tail=keyword,
                    head_type="event",
                    tail_type="technology_topic",
                    source_field="data.hotspot_item[].keywords",
                    evidence_text=event_summary,
                    confidence=0.85,
                )
                if keyword_triple:
                    triples.append(keyword_triple)

            for usage in item.get("teaching_usage") or []:
                usage_triple = self._build_triple(
                    doc=doc,
                    head=event_title,
                    relation="supports",
                    tail=usage,
                    head_type="event",
                    tail_type="teaching_usage",
                    source_field="data.hotspot_item[].teaching_usage",
                    evidence_text=event_summary,
                    confidence=0.88,
                )
                if usage_triple:
                    triples.append(usage_triple)

            for topic in self._extract_hotspot_topics(event_summary):
                topic_triple = self._build_triple(
                    doc=doc,
                    head=event_title,
                    relation="involves_topic",
                    tail=topic,
                    head_type="event",
                    tail_type="technology_topic",
                    source_field="data.hotspot_item[].summary",
                    evidence_text=event_summary,
                    confidence=0.82,
                )
                if topic_triple:
                    triples.append(topic_triple)

            news_role = self._pick_first([item.get("news_role")])
            role_triple = self._build_triple(
                doc=doc,
                head=event_title,
                relation="has_news_role",
                tail=news_role,
                head_type="event",
                tail_type="news_role",
                source_field="data.hotspot_item[].news_role",
                confidence=0.86,
            )
            if role_triple:
                triples.append(role_triple)

        for relation in data.get("relations") or []:
            rel_name = self._normalize_relation_name(relation.get("relation"))
            rel_triple = self._build_triple(
                doc=doc,
                head=relation.get("source"),
                relation=rel_name,
                tail=relation.get("target"),
                source_field="data.relations[]",
                confidence=self._to_float(relation.get("confidence"), 0.85),
            )
            if rel_triple:
                triples.append(rel_triple)

        return triples

    def _build_triple(
        self,
        doc: Dict[str, Any],
        head: Any,
        relation: Any,
        tail: Any,
        head_type: Optional[str] = None,
        tail_type: Optional[str] = None,
        evidence_text: Optional[str] = None,
        source_field: Optional[str] = None,
        chunk_id: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Optional[KnowledgeTriple]:
        head_text = self._normalize_entity_text(head)
        relation_text = self._normalize_relation_name(relation)
        tail_text = self._normalize_entity_text(tail)

        if not head_text or not relation_text or not tail_text:
            return None
        if head_text == tail_text:
            return None
        if not self._is_valid_entity(head_text) or not self._is_valid_entity(tail_text):
            return None

        doc_id = str(doc.get("doc_id") or "").strip()
        layer = str(doc.get("layer") or "").strip().lower()
        source_file = str(doc.get("source_file") or "").strip()
        if not doc_id or not layer:
            return None

        return KnowledgeTriple(
            head=head_text,
            relation=relation_text,
            tail=tail_text,
            doc_id=doc_id,
            layer=layer,
            source_file=source_file,
            head_type=self._normalize_entity_type(head_type),
            tail_type=self._normalize_entity_type(tail_type),
            evidence_text=self._normalize_evidence(evidence_text),
            source_field=source_field,
            chunk_id=self._normalize_chunk_id(chunk_id),
            confidence=max(0.0, min(1.0, self._to_float(confidence, 1.0))),
            metadata={},
        )

    def _pick_textbook_title(self, doc: Dict[str, Any], textbook_info: Dict[str, Any]) -> str:
        candidates = [
            textbook_info.get("book_title"),
            doc.get("title"),
            textbook_info.get("source_file"),
            doc.get("source_file"),
        ]
        for item in candidates:
            value = self._normalize_entity_text(item)
            if not value:
                continue
            if value.lower() in self.INVALID_TEXT_VALUES:
                continue
            return value
        return "textbook"

    def _infer_method_name(self, section_title: str, section_text: Optional[str]) -> Optional[str]:
        title = self._normalize_entity_text(section_title)
        text = clean_text(section_text or "")
        content = f"{title} {text}".lower()
        if not title:
            return None
        if any(hint in content for hint in self.METHOD_HINTS):
            return title
        return None

    def _extract_hyperparameters(self, text: Optional[str]) -> List[str]:
        content = clean_text(text or "")
        if not content:
            return []
        found: List[str] = []
        for hint in self.HYPERPARAMETER_HINTS:
            if hint.lower() in content.lower():
                found.append(hint)
        return self._deduplicate_values(found)[:8]

    def _extract_hotspot_topics(self, summary: Optional[str]) -> List[str]:
        content = clean_text(summary or "")
        if not content:
            return []
        tokens: List[str] = []
        for hint in self.HOTSPOT_TOPIC_HINTS:
            if hint.lower() in content.lower():
                tokens.append(hint)
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9\-\+]{3,30}\b", content):
            if token.lower() in {"with", "that", "from", "this", "have", "will"}:
                continue
            if token.lower() in {"github", "openclaw", "clawd", "agent", "aibase"}:
                tokens.append(token)
        return self._deduplicate_values(tokens)[:8]

    def _infer_kp_type(self, value: Optional[str]) -> str:
        text = self._normalize_entity_text(value).lower()
        if not text:
            return "knowledge_point"
        if any(hint in text for hint in ["算法", "method", "model", "模型", "network", "svm", "tree", "回归"]):
            return "method"
        if any(hint in text for hint in ["场景", "应用", "case", "实践"]):
            return "application_scenario"
        if any(hint in text for hint in ["概念", "theory", "定理"]):
            return "concept"
        return "knowledge_point"

    def _normalize_entity_text(self, value: Any) -> str:
        text = clean_text(str(value or ""))
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip("`'\"“”‘’.,;:，。；：|/\\")
        text = re.sub(r"^\d+(?:\.\d+){0,3}\s*", "", text)
        if text.lower() in self.INVALID_TEXT_VALUES:
            return ""
        if len(text) > 180:
            text = text[:180].strip()
        return text

    def _normalize_evidence(self, value: Optional[str]) -> Optional[str]:
        text = clean_text(str(value or ""))
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None
        if len(text) > 300:
            return text[:300].rstrip(".,;:，。；：") + "..."
        return text

    def _normalize_chunk_id(self, value: Optional[str]) -> Optional[str]:
        text = str(value or "").strip()
        return text or None

    def _normalize_relation_name(self, value: Any) -> str:
        text = self._normalize_entity_text(value).lower()
        if not text:
            return ""
        mapping = {
            "has_type": "has_event_type",
            "recommended_for": "supports",
            "has_role": "has_page_role",
            "related_to": "related_to",
            "contains": "contains",
        }
        if text in mapping:
            return mapping[text]
        text = re.sub(r"[^a-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "related_to"

    def _normalize_entity_type(self, value: Optional[str]) -> Optional[str]:
        text = self._normalize_entity_text(value).lower()
        if not text:
            return None
        text = re.sub(r"[^a-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or None

    def _is_valid_entity(self, text: str) -> bool:
        if not text:
            return False
        if len(text) < 2:
            return False
        if text.lower() in self.INVALID_TEXT_VALUES:
            return False
        if re.fullmatch(r"[\W_]+", text):
            return False
        if re.fullmatch(r"\d+(?:\.\d+)*", text):
            return False
        return True

    def _pick_first(self, values: Sequence[Any]) -> str:
        for item in values:
            value = self._normalize_entity_text(item)
            if value:
                return value
        return ""

    def _to_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _deduplicate_values(self, values: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for value in values:
            normalized = self._normalize_entity_text(value)
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return result

    def _deduplicate_triples(self, triples: Iterable[KnowledgeTriple]) -> List[KnowledgeTriple]:
        result: List[KnowledgeTriple] = []
        seen: Set[Tuple[str, str, str, str, str, str, str]] = set()
        for triple in triples:
            key = (
                triple.head.lower(),
                triple.relation,
                triple.tail.lower(),
                triple.doc_id,
                triple.layer,
                str(triple.source_field or ""),
                str(triple.chunk_id or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(triple)
        return result
