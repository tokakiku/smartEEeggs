from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


class TopicLocator:
    """在规范化教学大纲中定位主题证据。"""

    TOPIC = "梯度下降法"
    KEYWORDS = [
        "梯度下降",
        "随机梯度下降",
        "SGD",
        "小批量梯度下降",
        "成本函数",
        "线性回归",
        "逻辑回归",
        "参数优化",
    ]

    PREREQ_HINTS = {
        "高等数学": ["高等数学", "高数", "微积分", "导数", "偏导"],
        "线性代数": ["线性代数", "线代", "矩阵", "向量"],
        "概率统计": ["概率统计", "概率论", "数理统计", "分布"],
    }

    FOLLOWUP_HINTS = {
        "线性回归": ["线性回归"],
        "逻辑回归": ["逻辑回归"],
        "神经网络": ["神经网络", "反向传播", "BP算法"],
        "随机梯度下降": ["随机梯度下降", "SGD"],
        "小批量梯度下降": ["小批量梯度下降", "Mini-Batch"],
    }

    def locate(self, normalized_syllabus: Dict[str, Any]) -> Dict[str, Any]:
        hits: List[Dict[str, Any]] = []
        all_texts: List[Tuple[str, str]] = []

        intro = self._to_text(normalized_syllabus.get("course_intro"))
        if intro:
            all_texts.append(("course_intro", intro))

        objectives = normalized_syllabus.get("course_objectives") or []
        if isinstance(objectives, list):
            for idx, obj in enumerate(objectives, start=1):
                text = self._to_text(obj)
                if text:
                    all_texts.append((f"course_objective_{idx}", text))

        chapters = normalized_syllabus.get("chapters") or []
        if isinstance(chapters, list):
            for chapter in chapters:
                if not isinstance(chapter, dict):
                    continue
                chapter_title = self._to_text(chapter.get("chapter_title")) or "未命名章节"
                chapter_parts = [
                    self._to_text(chapter.get("chapter_content")),
                    "；".join(chapter.get("objectives") or []),
                    "；".join(chapter.get("key_points") or []),
                    "；".join(chapter.get("difficulties") or []),
                ]
                chapter_text = " ".join([part for part in chapter_parts if part]).strip()
                if chapter_text:
                    all_texts.append((f"chapter::{chapter_title}", chapter_text))

        for location, text in all_texts:
            matched = self._match_keywords(text)
            if not matched:
                continue
            hit = {
                "location": location,
                "matched_keywords": matched,
                "context": self._build_context(text, matched),
            }
            hits.append(hit)

        prereq = self._infer_prerequisites(normalized_syllabus, all_texts)
        followups = self._infer_followups(all_texts)
        summary = self._build_summary(
            source_file=self._to_text(normalized_syllabus.get("source_file")),
            hit_count=len(hits),
            hit_locations=[hit.get("location", "") for hit in hits],
            prereq=prereq,
            followups=followups,
        )

        return {
            "target_topic": self.TOPIC,
            "target_topic_hits": hits,
            "target_topic_summary": summary,
            "target_topic_prerequisites": prereq,
            "target_topic_followups": followups,
        }

    def _match_keywords(self, text: str) -> List[str]:
        lowered = text.lower()
        matched: List[str] = []
        for keyword in self.KEYWORDS:
            if keyword.lower() in lowered:
                matched.append(keyword)
        return self._dedup(matched)

    def _build_context(self, text: str, keywords: List[str], max_len: int = 180) -> str:
        if not text:
            return ""
        lower_text = text.lower()
        first_pos = len(text)
        for keyword in keywords:
            pos = lower_text.find(keyword.lower())
            if 0 <= pos < first_pos:
                first_pos = pos
        if first_pos == len(text):
            snippet = text[:max_len]
        else:
            left = max(0, first_pos - 40)
            right = min(len(text), first_pos + max_len)
            snippet = text[left:right]
        snippet = re.sub(r"\s+", " ", snippet).strip()
        return snippet

    def _infer_prerequisites(self, syllabus: Dict[str, Any], all_texts: List[Tuple[str, str]]) -> List[str]:
        found: Set[str] = set()

        raw_prereq = syllabus.get("prerequisites") or []
        if isinstance(raw_prereq, list):
            for item in raw_prereq:
                text = self._to_text(item)
                for canonical, aliases in self.PREREQ_HINTS.items():
                    if any(alias.lower() in text.lower() for alias in aliases):
                        found.add(canonical)

        corpus = " ".join(text for _, text in all_texts).lower()
        for canonical, aliases in self.PREREQ_HINTS.items():
            if any(alias.lower() in corpus for alias in aliases):
                found.add(canonical)

        return sorted(found)

    def _infer_followups(self, all_texts: List[Tuple[str, str]]) -> List[str]:
        found: Set[str] = set()
        corpus = " ".join(text for _, text in all_texts).lower()
        for canonical, aliases in self.FOLLOWUP_HINTS.items():
            if any(alias.lower() in corpus for alias in aliases):
                found.add(canonical)
        return sorted(found)

    def _build_summary(
        self,
        source_file: str,
        hit_count: int,
        hit_locations: List[str],
        prereq: List[str],
        followups: List[str],
    ) -> str:
        loc_preview = "、".join([loc for loc in hit_locations[:4] if loc]) or "未定位到明确章节"
        prereq_preview = "、".join(prereq) if prereq else "未显式给出"
        followup_preview = "、".join(followups) if followups else "未显式给出"
        return (
            f"{source_file or '该大纲'}中与“{self.TOPIC}”相关命中 {hit_count} 处，主要位置：{loc_preview}。"
            f" 关联前置知识：{prereq_preview}。关联后续内容：{followup_preview}。"
        )

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _dedup(self, items: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for item in items:
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

