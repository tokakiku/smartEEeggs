from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ChapterItem:
    chapter_title: str = ""
    chapter_content: str = ""
    objectives: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    difficulties: List[str] = field(default_factory=list)
    hours: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SyllabusRecord:
    source_file: str = ""
    source_type: str = "pdf"
    course_name: str = ""
    course_code: str = ""
    course_type: str = ""
    target_major: List[str] = field(default_factory=list)
    credits: str = ""
    total_hours: str = ""
    theory_hours: str = ""
    lab_hours: str = ""
    prerequisites: List[str] = field(default_factory=list)
    course_intro: str = ""
    course_objectives: List[str] = field(default_factory=list)
    chapters: List[ChapterItem] = field(default_factory=list)
    teaching_methods: List[str] = field(default_factory=list)
    assessment: List[str] = field(default_factory=list)
    textbooks: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    target_topic: str = "梯度下降法"
    target_topic_hits: List[Dict[str, Any]] = field(default_factory=list)
    target_topic_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # 保证固定值语义。
        payload["source_type"] = "pdf"
        payload["target_topic"] = "梯度下降法"
        return payload
