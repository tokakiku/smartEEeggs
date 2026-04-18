import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from schema.parsed_document_schema import ParsedDocument
from schema.syllabus_schema import (
    CourseInfo,
    CourseModule,
    SyllabusExtractionResult,
    TeachingMaterials,
    TeachingScheduleItem,
)
from utils.text_cleaner import clean_text, split_paragraphs


def _garble_utf8_as_gbk(value: str) -> str:
    # 兼容历史乱码文本（UTF-8 被 GBK 误解码）。
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return text.encode("utf-8").decode("gbk")
    except Exception:
        return text


class SyllabusExtractor:
    """基于规则的教学大纲抽取器，供主入库链路使用。"""

    COURSE_INFO_LABELS = {
        "course_name": ["课程名称", "课程名"],
        "course_code": ["课程编号", "课程代码", "课程号"],
        "credit_hours": ["学分/学时", "学时/学分", "学分与学时", "学分学时"],
        "course_type": ["课程性质", "课程类型"],
        "applicable_major": ["适用专业", "面向专业", "适用对象"],
        "suggested_term": ["建议开设学期", "开设学期", "建议学期"],
        "prerequisite_courses": ["先修课程", "课程先导课", "建议先修课程", "先修课"],
        "offering_institute": ["开课单位", "开课院系", "授课单位"],
    }

    GOAL_LABELS = ["课程目标", "教学目标", "培养目标", "教学目标与任务", "课程的教学目标与任务"]
    GOAL_HINT_VERBS = ["掌握", "理解", "了解", "能够", "熟悉", "学会", "培养", "具备", "认识", "说明"]
    MODULE_SECTION_LABELS = [
        "课程具体内容及基本要求",
        "课程具体内容",
        "教学内容及基本要求",
        "教学基本内容",
        "课程内容与学时分配",
        "教学内容与学时分配",
        "课程教学内容与学时分配",
        "理论教学内容与学时安排",
        "课程内容与教学安排",
        "教学内容与安排",
        "教学内容和要求",
        "课程内容与要求",
        "各章节教学内容",
        "教学章节与学时分配",
    ]
    SCHEDULE_SECTION_LABELS = ["教学安排及方式", "教学安排", "教学进度安排", "学时分配", "教学计划"]
    TEXTBOOK_SECTION_LABELS = ["教材及参考书目", "教材与参考书目", "教材及参考资料"]
    MAIN_TEXTBOOK_LABELS = ["主课教材", "主教材", "指定教材", "教材"]
    REFERENCE_TEXTBOOK_LABELS = ["参考教材", "参考书目", "推荐阅读", "辅导教材", "参考资料", "参考书"]
    MODULE_END_LABELS = SCHEDULE_SECTION_LABELS + TEXTBOOK_SECTION_LABELS

    REQUIREMENT_LABELS = ["基本要求", "教学要求", "学习要求"]
    KEY_DIFF_SECTION_LABELS = ["重点、难点", "教学重点", "教学难点"]
    ASSIGNMENT_LABELS = ["作业及课外学习要求", "课外学习要求", "作业要求", "作业"]

    KEY_LABELS = ["重点", "教学重点"]
    DIFF_LABELS = ["难点", "教学难点"]
    HOUR_WORDS = ["学时", "课时"]
    METHOD_WORDS = ["讲授", "上机", "实验", "讨论", "案例教学", "翻转课堂", "线上教学", "线下教学", "实训"]

    MODULE_TITLE_HINTS = [
        "绪论",
        "导论",
        "概述",
        "概念",
        "方法",
        "模型",
        "回归",
        "分类",
        "聚类",
        "网络",
        "优化",
        "算法",
        "学习",
        "神经",
        "评估",
        "实验",
        "支持向量",
        "降维",
        "集成学习",
        "概率",
        "贝叶斯",
    ]
    EXCLUDED_MODULE_TITLES = [
        "课程名称",
        "课程性质",
        "学时与学分",
        "课程先导课",
        "课程介绍",
        "课程目标",
        "课程目标对毕业要求的支撑关系",
        "教学设计及对课程目标的支撑",
        "教学基本内容",
        "教学基本要求",
        "教学要求",
        "学习要求",
        "重点、难点",
    ]
    KP_HINTS = [
        "机器学习",
        "假设空间",
        "梯度下降",
        "随机梯度下降",
        "小批量梯度下降",
        "损失函数",
        "线性回归",
        "逻辑回归",
        "神经网络",
        "参数优化",
        "SGD",
    ]
    KP_STOP = {"课程目标", "教学目标", "教学重点", "教学难点", "本章", "本节", "小结", "总结"}

    def extract(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        return self.extract_from_parsed_document(parsed_doc).model_dump()

    def extract_from_parsed_document(self, parsed_doc: ParsedDocument) -> SyllabusExtractionResult:
        text = parsed_doc.raw_text or "\n".join([item.text for item in parsed_doc.elements if item.text])
        return self.extract_from_text(text)

    def extract_from_text(self, text: str) -> SyllabusExtractionResult:
        cleaned = self._clean_text(text)
        if not cleaned:
            return SyllabusExtractionResult()

        info = self._extract_course_info(cleaned)
        goals = self._extract_goals(cleaned)
        modules = self._extract_modules(cleaned)
        schedule = self._extract_schedule(cleaned, modules)
        teaching_materials = self._extract_teaching_materials(cleaned)

        key_points = self._dedup([x for m in modules for x in (m.key_points or [])])
        difficult_points = self._dedup([x for m in modules for x in (m.difficult_points or [])])
        knowledge_points = self._extract_knowledge_points(cleaned, modules, goals, key_points, difficult_points)
        textbooks = self._dedup(
            (teaching_materials.main_textbooks or [])
            + (teaching_materials.reference_textbooks or [])
            + (teaching_materials.other_materials or [])
        )

        return SyllabusExtractionResult(
            course_info=info,
            course_name=info.course_name,
            course_code=info.course_code,
            prerequisites=info.prerequisite_courses or [],
            teaching_goals=goals,
            course_modules=modules,
            teaching_key_points=key_points,
            teaching_difficult_points=difficult_points,
            knowledge_points=knowledge_points,
            teaching_schedule=schedule,
            textbooks=textbooks,
            teaching_materials=teaching_materials,
            raw_sections={},
        )

    def _expand_labels(self, labels: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for label in labels:
            for variant in [str(label or "").strip(), _garble_utf8_as_gbk(label)]:
                value = str(variant or "").strip()
                if not value:
                    continue
                key = value.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(value)
        return out

    def _clean_text(self, text: str) -> str:
        value = clean_text(text or "")
        if not value:
            return ""
        lines: List[str] = []
        for line in value.split("\n"):
            s = line.strip()
            if not s:
                continue
            if re.match(r"^\d+\s*/\s*\d+$", s):
                continue
            if re.match(r"^第?\s*\d+\s*页$", s):
                continue
            lines.append(s)
        return "\n".join(lines)

    def _extract_course_info(self, text: str) -> CourseInfo:
        block = self._prefix_before(text, self.GOAL_LABELS + self.MODULE_SECTION_LABELS)
        return CourseInfo(
            course_name=self._extract_single_field(block, self.COURSE_INFO_LABELS["course_name"]) or None,
            course_code=self._extract_single_field(block, self.COURSE_INFO_LABELS["course_code"]) or None,
            credit_hours=self._extract_single_field(block, self.COURSE_INFO_LABELS["credit_hours"]) or None,
            course_type=self._extract_single_field(block, self.COURSE_INFO_LABELS["course_type"]) or None,
            applicable_major=self._extract_list_field(block, self.COURSE_INFO_LABELS["applicable_major"]),
            suggested_term=self._extract_single_field(block, self.COURSE_INFO_LABELS["suggested_term"]) or None,
            prerequisite_courses=self._extract_list_field(block, self.COURSE_INFO_LABELS["prerequisite_courses"]),
            offering_institute=self._extract_single_field(block, self.COURSE_INFO_LABELS["offering_institute"]) or None,
        )

    def _extract_goals(self, text: str) -> List[str]:
        section = self._section(text, self.GOAL_LABELS, self.MODULE_SECTION_LABELS + self.SCHEDULE_SECTION_LABELS)
        goals: List[str] = []
        for line in split_paragraphs(section or ""):
            s = self._strip_item_prefix(clean_text(line))
            s = self._strip_label_prefix(s, self.GOAL_LABELS)
            if not s:
                continue
            if 6 <= len(s) <= 160 and self._is_goal_like_sentence(s):
                goals.append(s)

        if goals:
            return self._dedup(goals)[:24]

        for line in split_paragraphs(text):
            s = clean_text(line)
            if not (6 <= len(s) <= 160):
                continue
            if self._contains_any_label(s, self.GOAL_LABELS):
                goals.append(self._strip_label_prefix(s, self.GOAL_LABELS))
            if len(goals) >= 6:
                break
        return self._dedup(goals)[:24]

    def _is_goal_like_sentence(self, text: str) -> bool:
        value = clean_text(text or "")
        if not value:
            return False
        if self._contains_any_label(value, self.GOAL_LABELS):
            return True
        return any(token in value for token in self._expand_labels(self.GOAL_HINT_VERBS))

    def _extract_modules(self, text: str) -> List[CourseModule]:
        body = self._section(text, self.MODULE_SECTION_LABELS, self.MODULE_END_LABELS) or text
        lines = split_paragraphs(body)
        if not lines:
            return []

        heading_indexes = [idx for idx, line in enumerate(lines) if self._is_module_heading(line)]
        if not heading_indexes:
            return []

        modules: List[CourseModule] = []
        for pos, start_idx in enumerate(heading_indexes):
            end_idx = heading_indexes[pos + 1] if (pos + 1) < len(heading_indexes) else len(lines)
            block_lines = lines[start_idx:end_idx]
            heading = clean_text(block_lines[0])
            block_text = "\n".join(block_lines)

            module_name, hours = self._parse_heading(heading)
            requirements = self._extract_labeled_items(block_text, self.REQUIREMENT_LABELS)
            key_points, difficult_points = self._split_key_diff(block_text)
            assignments = self._extract_labeled_items(block_text, self.ASSIGNMENT_LABELS)
            description = self._module_description(block_lines)

            modules.append(
                CourseModule(
                    module_index=len(modules) + 1,
                    module_name=module_name or f"模块{len(modules) + 1}",
                    hours=hours or None,
                    description=description or None,
                    learning_requirements=requirements,
                    key_points=key_points,
                    difficult_points=difficult_points,
                    assignments=assignments,
                )
            )
        return modules

    def _module_description(self, block_lines: List[str]) -> str:
        desc: List[str] = []
        for line in block_lines[1:]:
            s = clean_text(line)
            if not s:
                continue
            if self._contains_any_label(
                s,
                self.REQUIREMENT_LABELS + self.KEY_DIFF_SECTION_LABELS + self.ASSIGNMENT_LABELS,
            ):
                continue
            desc.append(self._strip_item_prefix(s))
            if len(" ".join(desc)) >= 220:
                break
        return self._truncate(" ".join(desc), 260)

    def _extract_schedule(self, text: str, modules: List[CourseModule]) -> List[TeachingScheduleItem]:
        section = self._section(text, self.SCHEDULE_SECTION_LABELS, self.TEXTBOOK_SECTION_LABELS)
        items: List[TeachingScheduleItem] = []
        for line in split_paragraphs(section or ""):
            s = self._strip_item_prefix(clean_text(line))
            if not s:
                continue
            hours = self._extract_hours(s)
            topic = s.replace(hours, " ").strip() if hours else s
            topic = topic.strip(" ：:;；,.，")
            if len(topic) < 2:
                continue
            items.append(
                TeachingScheduleItem(
                    order=len(items) + 1,
                    topic=self._truncate(topic, 80),
                    hours=hours or None,
                    teaching_method=self._method(s),
                )
            )
        if items:
            return items

        for idx, module in enumerate(modules, start=1):
            items.append(
                TeachingScheduleItem(
                    order=idx,
                    topic=module.module_name,
                    hours=module.hours,
                    teaching_method="讲授",
                )
            )
        return items

    def _extract_teaching_materials(self, text: str) -> TeachingMaterials:
        section = self._section(text, self.TEXTBOOK_SECTION_LABELS, [])
        lines = split_paragraphs(section or "")
        if not lines:
            return TeachingMaterials()

        main_items: List[str] = []
        reference_items: List[str] = []
        other_items: List[str] = []
        current_bucket: Optional[str] = None

        for raw_line in lines:
            line = self._strip_item_prefix(clean_text(raw_line))
            if not line:
                continue
            line = self._strip_label_prefix(line, self.TEXTBOOK_SECTION_LABELS)
            if not line:
                continue

            if self._contains_any_label(line, self.REFERENCE_TEXTBOOK_LABELS) or self._looks_like_reference_material(line):
                payload = self._strip_label_prefix(line, self.REFERENCE_TEXTBOOK_LABELS)
                reference_items.extend(self._split_material_items(payload or line))
                current_bucket = "reference"
                continue

            if self._contains_any_label(line, self.MAIN_TEXTBOOK_LABELS) or self._looks_like_main_material(line):
                payload = self._strip_label_prefix(line, self.MAIN_TEXTBOOK_LABELS)
                main_items.extend(self._split_material_items(payload or line))
                current_bucket = "main"
                continue

            values = self._split_material_items(line)
            for value in values:
                if self._looks_like_reference_material(value):
                    reference_items.append(value)
                elif self._looks_like_main_material(value):
                    main_items.append(value)
                elif current_bucket == "reference":
                    reference_items.append(value)
                elif current_bucket == "main":
                    main_items.append(value)
                else:
                    other_items.append(value)

        if not main_items and not reference_items:
            fallback_items = self._items(section)
            for item in fallback_items:
                if self._looks_like_reference_material(item):
                    reference_items.append(item)
                else:
                    main_items.append(item)

        return TeachingMaterials(
            main_textbooks=self._dedup(main_items)[:12],
            reference_textbooks=self._dedup(reference_items)[:12],
            other_materials=self._dedup(other_items)[:12],
        )

    def _split_material_items(self, text: str) -> List[str]:
        value = clean_text(text or "")
        if not value:
            return []
        parts: List[str] = []
        all_labels = self.MAIN_TEXTBOOK_LABELS + self.REFERENCE_TEXTBOOK_LABELS
        for chunk in re.split(r"[\n;；]", value):
            chunk = clean_text(chunk)
            if not chunk:
                continue
            for item in re.split(r"[、，,/]+", chunk):
                s = clean_text(item).strip(" ：:;；,.，")
                if len(s) < 2:
                    continue
                s = self._strip_label_prefix(s, all_labels)
                s = s.strip(" ：:;；,.，")
                if len(s) < 2:
                    continue
                parts.append(s)
        return self._dedup(parts)

    def _looks_like_reference_material(self, text: str) -> bool:
        value = clean_text(text or "").lower()
        if not value:
            return False
        tokens = ["参考", "推荐", "辅导", "习题", "题解", "workbook", "supplementary", "reference", "鍙傝", "鎺ㄨ崘"]
        return any(token in value for token in tokens)

    def _looks_like_main_material(self, text: str) -> bool:
        value = clean_text(text or "").lower()
        if not value:
            return False
        tokens = ["主教材", "主课教材", "指定教材", "核心教材", "core", "primary", "main", "涓昏", "鎸囧畾"]
        return any(token in value for token in tokens)

    def _extract_knowledge_points(
        self,
        text: str,
        modules: List[CourseModule],
        goals: List[str],
        key_points: List[str],
        difficult_points: List[str],
    ) -> List[str]:
        points: List[str] = []
        points.extend(key_points)
        points.extend(difficult_points)

        for module in modules:
            points.append(module.module_name)
            points.extend(module.learning_requirements or [])

        for goal in goals:
            points.extend(self._term_candidates(goal))

        lowered_text = text.lower()
        for hint in self._expand_labels(self.KP_HINTS):
            if hint and hint.lower() in lowered_text:
                points.append(hint)

        points.extend(self._term_candidates(text[:3600]))
        stop_words = {item.lower() for item in self._expand_labels(self.KP_STOP)}

        out: List[str] = []
        for point in points:
            term = self._norm_term(point)
            if not term:
                continue
            if term.lower() in stop_words:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", term):
                continue
            if 2 <= len(term) <= 30:
                out.append(term)
        return self._dedup(out)[:80]

    def _term_candidates(self, text: str) -> List[str]:
        value = clean_text(text or "")
        if not value:
            return []
        patterns = [
            r"([\u4e00-\u9fffA-Za-z]{2,20}(?:方法|模型|算法|理论|回归|学习|优化|空间|网络|函数|策略))",
            r"([A-Za-z][A-Za-z\-]{2,20})",
        ]
        out: List[str] = []
        for pattern in patterns:
            out.extend([str(item).strip() for item in re.findall(pattern, value)])
        return out

    def _extract_single_field(self, text: str, labels: Sequence[str]) -> str:
        lines = split_paragraphs(text or "")
        variants = self._expand_labels(labels)

        for line in lines:
            raw = clean_text(line)
            for label in variants:
                m = re.match(rf"^\s*{re.escape(label)}\s*[：:]\s*(.+)$", raw, flags=re.I)
                if m:
                    return self._tail_trim(clean_text(m.group(1)))

        for line in lines:
            raw = clean_text(line)
            for label in variants:
                if raw.startswith(label):
                    remain = raw[len(label) :].lstrip(" ：:")
                    if remain:
                        return self._tail_trim(remain)
        return ""

    def _extract_list_field(self, text: str, labels: Sequence[str]) -> List[str]:
        value = self._extract_single_field(text, labels)
        if not value:
            return []
        out: List[str] = []
        for item in re.split(r"[、，,;/；]+", value):
            s = clean_text(item)
            if not s:
                continue
            if s in {"无", "暂无", "无要求"}:
                continue
            out.append(s)
        return self._dedup(out)

    def _section(self, text: str, starts: Sequence[str], ends: Sequence[str]) -> str:
        if not text:
            return ""
        start_variants = self._expand_labels(starts)
        end_variants = self._expand_labels(ends)

        start_positions = [text.find(mark) for mark in start_variants if mark and text.find(mark) >= 0]
        if not start_positions:
            return ""
        start = min(start_positions)
        end = len(text)
        end_positions = [text.find(mark, start + 1) for mark in end_variants if mark and text.find(mark, start + 1) >= 0]
        if end_positions:
            end = min(end_positions)
        return text[start:end].strip()

    def _prefix_before(self, text: str, marks: Sequence[str]) -> str:
        variants = self._expand_labels(marks)
        positions = [text.find(mark) for mark in variants if mark and text.find(mark) >= 0]
        return text[: min(positions)].strip() if positions else text

    def _extract_labeled_items(self, text: str, labels: Sequence[str]) -> List[str]:
        lines = split_paragraphs(text or "")
        output: List[str] = []
        for line in lines:
            s = clean_text(line)
            if not s:
                continue
            if not self._contains_any_label(s, labels):
                continue
            s = self._strip_label_prefix(s, labels)
            output.extend(self._items(s))
        return self._dedup(output)

    def _items(self, text: str) -> List[str]:
        value = clean_text(text or "")
        if not value:
            return []
        out: List[str] = []
        for raw in re.split(r"[\n;；]+", value):
            s = self._strip_item_prefix(clean_text(raw))
            s = s.strip(" ：:;；,.，")
            if len(s) >= 2:
                out.append(s)
        return self._dedup(out)

    def _split_key_diff(self, text: str) -> Tuple[List[str], List[str]]:
        key_points: List[str] = []
        difficult_points: List[str] = []
        for line in split_paragraphs(text or ""):
            s = clean_text(line)
            if not s:
                continue
            if self._contains_any_label(s, self.KEY_LABELS):
                key_points.extend(self._items(self._strip_label_prefix(s, self.KEY_LABELS)))
                continue
            if self._contains_any_label(s, self.DIFF_LABELS):
                difficult_points.extend(self._items(self._strip_label_prefix(s, self.DIFF_LABELS)))
        return self._dedup(key_points), self._dedup(difficult_points)

    def _parse_heading(self, heading: str) -> Tuple[str, Optional[str]]:
        s = clean_text(heading or "")
        if not s:
            return "", None
        hours = self._extract_hours(s)
        if hours:
            s = s.replace(hours, " ")
        s = re.sub(r"^\s*[一二三四五六七八九十IVX\d]+\s*[、\.\-]\s*", "", s, flags=re.I)
        s = re.sub(r"^\s*(?:第\s*[一二三四五六七八九十\d]+\s*[章节]|chapter\s*\d+)\s*", "", s, flags=re.I)
        s = re.sub(r"^\s*[\(（][^()（）]{1,10}[\)）]\s*", "", s)
        return s.strip(" ：:;；,.，()（）"), hours

    def _is_module_heading(self, line: str) -> bool:
        s = clean_text(line or "")
        if not s or len(s) < 3 or len(s) > 80:
            return False
        lowered = s.lower()
        if any(token in lowered for token in ["基本要求", "教学要求", "学习要求", "重点", "难点", "作业", "瑕佹眰", "閲嶇偣", "闅剧偣", "浣滀笟"]):
            return False

        if self._contains_any_label(
            s,
            self.REQUIREMENT_LABELS
            + self.KEY_DIFF_SECTION_LABELS
            + self.ASSIGNMENT_LABELS
            + self.GOAL_LABELS
            + self.SCHEDULE_SECTION_LABELS
            + self.TEXTBOOK_SECTION_LABELS,
        ):
            return False
        if any(token == s for token in self._expand_labels(self.EXCLUDED_MODULE_TITLES)):
            return False
        if re.match(r"^\s*(?:第\s*[一二三四五六七八九十\d]+\s*[章节]|chapter\s*\d+)", s, flags=re.I):
            return True
        if re.match(r"^\s*[一二三四五六七八九十IVX\d]+\s*[、\.\-]\s*", s, flags=re.I):
            trimmed = re.sub(r"^\s*[一二三四五六七八九十IVX\d]+\s*[、\.\-]\s*", "", s, flags=re.I)
            if self._extract_hours(s):
                return True
            if any(h in trimmed for h in self._expand_labels(self.MODULE_TITLE_HINTS)):
                return True
            return False
        if self._extract_hours(s):
            if any(h in s for h in self._expand_labels(self.MODULE_TITLE_HINTS)):
                return True
            return len(s) <= 40
        return False

    def _extract_hours(self, text: str) -> str:
        value = clean_text(text or "")
        if not value:
            return ""
        for word in self._expand_labels(self.HOUR_WORDS):
            m = re.search(rf"(\d+(?:\.\d+)?\s*{re.escape(word)})", value)
            if m:
                return clean_text(m.group(1))
        return ""

    def _method(self, text: str) -> Optional[str]:
        value = clean_text(text or "")
        methods: List[str] = []
        for token in self._expand_labels(self.METHOD_WORDS):
            if token and token in value:
                methods.append(token)
        merged = self._dedup(methods)
        return "、".join(merged) if merged else None

    def _tail_trim(self, value: str) -> str:
        text = clean_text(value or "")
        labels = []
        for names in self.COURSE_INFO_LABELS.values():
            labels.extend(names)
        labels.extend(self.GOAL_LABELS)
        labels.extend(self.MODULE_SECTION_LABELS)
        for marker in self._expand_labels(labels):
            for sep in ["：", ":"]:
                token = f"{marker}{sep}"
                idx = text.find(token)
                if idx > 0:
                    text = text[:idx].strip()
        return text

    def _strip_item_prefix(self, text: str) -> str:
        s = str(text or "").strip()
        s = re.sub(r"^(?:\d+\s*[\.、]\s*)", "", s)
        s = re.sub(r"^(?:[（(]\d+[)）]\s*)", "", s)
        s = re.sub(r"^(?:[一二三四五六七八九十IVX\d]+\s*[、\.\-]\s*)", "", s, flags=re.I)
        return s.strip()

    def _strip_label_prefix(self, text: str, labels: Sequence[str]) -> str:
        s = clean_text(text or "")
        for label in self._expand_labels(labels):
            s = re.sub(rf"^\s*{re.escape(label)}\s*[：:]?\s*", "", s, flags=re.I)
        return s.strip()

    def _contains_any_label(self, text: str, labels: Sequence[str]) -> bool:
        value = str(text or "")
        for label in self._expand_labels(labels):
            if label and label in value:
                return True
        return False

    def _norm_term(self, text: Any) -> str:
        s = clean_text(str(text or ""))
        s = re.sub(r"^(?:重点|难点|目标|要求)\s*[：:]?\s*", "", s)
        s = re.sub(r"^(?:\d+[\.、]\s*|[（(]\d+[)）]\s*)", "", s)
        s = re.split(r"[；;。]", s, maxsplit=1)[0].strip(" ：:;；,.，()（）")
        return s[:40]

    def _truncate(self, text: str, size: int) -> str:
        value = str(text or "").strip()
        if len(value) <= size:
            return value
        return value[:size].rstrip("，,。.;； ") + "..."

    def _dedup(self, values: Sequence[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            s = clean_text(str(value or ""))
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out


class CurriculumSyllabusExtractor:
    """面向 build_curriculum_layer.py 产物的抽取器。"""

    TARGET_TOPIC = "梯度下降法"

    def __init__(self) -> None:
        self.base = SyllabusExtractor()

    def extract_from_text(self, text: str, source_file: str = "", source_type: str = "pdf") -> Dict[str, Any]:
        result = self.base.extract_from_text(text)
        info = result.course_info
        credits, total_hours, theory_hours, lab_hours = self._parse_hours(info.credit_hours or "")
        chapters = [self._to_chapter(module) for module in (result.course_modules or [])]
        textbooks = [str(x).strip() for x in (result.textbooks or []) if str(x).strip()]

        return {
            "source_file": source_file,
            "source_type": source_type or "pdf",
            "course_name": info.course_name or "",
            "course_code": info.course_code or "",
            "course_type": info.course_type or "",
            "target_major": info.applicable_major or [],
            "credits": credits,
            "total_hours": total_hours,
            "theory_hours": theory_hours,
            "lab_hours": lab_hours,
            "prerequisites": info.prerequisite_courses or [],
            "course_intro": clean_text(text)[:220],
            "course_objectives": result.teaching_goals or [],
            "chapters": chapters,
            "teaching_methods": self._methods(text),
            "assessment": self._assessment(text),
            "textbooks": textbooks[:1],
            "references": textbooks[1:] if len(textbooks) > 1 else [],
            "target_topic": self.TARGET_TOPIC,
            "target_topic_hits": [],
            "target_topic_summary": "",
        }

    def _parse_hours(self, text: str) -> Tuple[str, str, str, str]:
        nums = re.findall(r"\d+(?:\.\d+)?", clean_text(text))
        return (
            nums[0] if len(nums) >= 1 else "",
            nums[1] if len(nums) >= 2 else "",
            nums[2] if len(nums) >= 3 else "",
            nums[3] if len(nums) >= 4 else "",
        )

    def _to_chapter(self, module: CourseModule) -> Dict[str, Any]:
        parts: List[str] = []
        if module.description:
            parts.append(clean_text(module.description))
        if module.learning_requirements:
            parts.append("、".join(module.learning_requirements[:4]))
        return {
            "chapter_title": module.module_name,
            "chapter_content": " ".join(parts)[:320],
            "objectives": module.learning_requirements or [],
            "key_points": module.key_points or [],
            "difficulties": module.difficult_points or [],
            "hours": module.hours or "",
        }

    def _methods(self, text: str) -> List[str]:
        vals = re.findall(r"(?:讲授|上机|实验|讨论|案例教学|翻转课堂|线上教学|线下教学|实训)", str(text or ""))
        return self._dedup(vals)

    def _assessment(self, text: str) -> List[str]:
        vals = re.findall(r"(?:平时成绩|期末考试|期末成绩|实验成绩|作业成绩)\s*[\d.%％ ]+", str(text or ""))
        return self._dedup(vals)

    def _dedup(self, values: Sequence[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            s = clean_text(value)
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out
