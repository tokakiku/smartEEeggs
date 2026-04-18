from __future__ import annotations

import re
from typing import Any, Dict, List

from schemas.syllabus_schema import ChapterItem, SyllabusRecord


class SyllabusNormalizer:
    """将教学大纲抽取字段规范化为稳定结构。"""

    _COURSE_TYPE_RULES = [
        ("专业必修课", ["专业必修", "必修", "core"]),
        ("专业选修课", ["专业选修", "选修", "elective"]),
        ("公共基础课", ["公共基础", "通识", "general"]),
    ]

    _PREREQ_REMAP = {
        "高数": "高等数学",
        "线代": "线性代数",
        "概率论": "概率统计",
        "概率与数理统计": "概率统计",
        "程序设计基础": "程序设计",
        "python程序设计": "Python程序设计",
    }

    def normalize(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        record = SyllabusRecord(
            source_file=self._to_text(raw.get("source_file")),
            source_type="pdf",
            course_name=self._to_text(raw.get("course_name")),
            course_code=self._to_text(raw.get("course_code")),
            course_type=self._normalize_course_type(self._to_text(raw.get("course_type"))),
            target_major=self._normalize_list(raw.get("target_major")),
            credits=self._normalize_numeric_text(raw.get("credits")),
            total_hours=self._normalize_numeric_text(raw.get("total_hours")),
            theory_hours=self._normalize_numeric_text(raw.get("theory_hours")),
            lab_hours=self._normalize_numeric_text(raw.get("lab_hours")),
            prerequisites=self._normalize_prerequisites(raw.get("prerequisites")),
            course_intro=self._clean_text(raw.get("course_intro")),
            course_objectives=self._normalize_list(raw.get("course_objectives")),
            chapters=self._normalize_chapters(raw.get("chapters")),
            teaching_methods=self._normalize_list(raw.get("teaching_methods")),
            assessment=self._normalize_list(raw.get("assessment")),
            textbooks=self._normalize_list(raw.get("textbooks")),
            references=self._normalize_list(raw.get("references")),
            target_topic="梯度下降法",
            target_topic_hits=self._normalize_hits(raw.get("target_topic_hits")),
            target_topic_summary=self._clean_text(raw.get("target_topic_summary")),
        )
        return record.to_dict()

    def _normalize_course_type(self, value: str) -> str:
        lowered = value.lower()
        for normalized, aliases in self._COURSE_TYPE_RULES:
            if any(alias in lowered for alias in aliases):
                return normalized
        return value

    def _normalize_prerequisites(self, value: Any) -> List[str]:
        items = self._normalize_list(value)
        normalized: List[str] = []
        for item in items:
            replaced = self._PREREQ_REMAP.get(item, item)
            normalized.append(replaced)
        return self._dedup(normalized)

    def _normalize_chapters(self, value: Any) -> List[ChapterItem]:
        chapters: List[ChapterItem] = []
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, dict):
            raw_items = [value]
        else:
            raw_items = []

        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            chapter = ChapterItem(
                chapter_title=self._to_text(raw_item.get("chapter_title")),
                chapter_content=self._clean_text(raw_item.get("chapter_content")),
                objectives=self._normalize_list(raw_item.get("objectives")),
                key_points=self._normalize_list(raw_item.get("key_points")),
                difficulties=self._normalize_list(raw_item.get("difficulties")),
                hours=self._normalize_numeric_text(raw_item.get("hours")),
            )
            if not any(
                [
                    chapter.chapter_title,
                    chapter.chapter_content,
                    chapter.objectives,
                    chapter.key_points,
                    chapter.difficulties,
                    chapter.hours,
                ]
            ):
                continue
            chapters.append(chapter)
        return chapters

    def _normalize_hits(self, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        output: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            cleaned: Dict[str, Any] = {}
            for key, item_value in item.items():
                if isinstance(item_value, str):
                    v = self._clean_text(item_value)
                    if v:
                        cleaned[key] = v
                elif isinstance(item_value, list):
                    cleaned_list = self._normalize_list(item_value)
                    if cleaned_list:
                        cleaned[key] = cleaned_list
                elif item_value is not None:
                    cleaned[key] = item_value
            if cleaned:
                output.append(cleaned)
        return output

    def _normalize_numeric_text(self, value: Any) -> str:
        text = self._to_text(value)
        if not text:
            return ""
        match = re.search(r"\d+(?:\.\d+)?", text)
        return match.group(0) if match else text

    def _normalize_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            chunks = re.split(r"[，,；;、\n]+", value)
        elif isinstance(value, list):
            chunks = []
            for item in value:
                if isinstance(item, str):
                    chunks.extend(re.split(r"[，,；;、\n]+", item))
                elif item is not None:
                    chunks.append(str(item))
        else:
            chunks = [str(value)]

        cleaned = [self._clean_text(chunk) for chunk in chunks]
        cleaned = [item for item in cleaned if item and item not in {"无", "暂无", "N/A", "NA"}]
        return self._dedup(cleaned)

    def _clean_text(self, value: Any) -> str:
        text = self._to_text(value)
        text = text.replace("\u3000", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip(" \t\r\n；;，,。")

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _dedup(self, items: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in items:
            key = item.lower()
            if not item or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

