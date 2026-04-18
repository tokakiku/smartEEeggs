from __future__ import annotations

"""检索证据整理器（Context Consolidator）。

职责：
- 对混合检索结果做规则清洗、去重与主次排序；
- 可选调用 LLM 再整理为可喂给生成层的教学简报；
- 在 LLM 不可用时提供稳定的规则兜底文本。
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _to_str(value: Any) -> str:
    return str(value or "").strip()


class ContextConsolidator:
    """将混合检索结果整理为高质量教学上下文。"""
    PRIMARY_LAYERS = {"syllabus", "textbook"}
    SUPPORT_LAYERS = {"resource", "hotspot"}

    STRONG_RELATIONS = {"supports", "explains", "applies", "implements"}
    WEAK_RELATIONS = {
        "next",
        "belongs_to",
        "belongs_to_textbook",
        "contains_chapter",
        "contains_section",
        "has_page_role",
        "has_event_type",
        "related_to",
    }

    LAYER_WEIGHT = {
        "syllabus": 1.20,
        "textbook": 1.10,
        "resource": 0.90,
        "hotspot": 0.80,
    }

    def __init__(
        self,
        max_vector_keep: int = 16,
        max_per_source_file: int = 2,
        max_per_doc: int = 3,
    ) -> None:
        self.max_vector_keep = max(6, int(max_vector_keep))
        self.max_per_source_file = max(1, int(max_per_source_file))
        self.max_per_doc = max(1, int(max_per_doc))

    def consolidate(
        self,
        retrieval_payload: Dict[str, Any],
        course_topic: Optional[str] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """执行证据整理主流程并返回 clean_lesson_brief。"""
        query = _to_str((retrieval_payload or {}).get("query")) or _to_str(course_topic)
        topic = _to_str(course_topic) or query or "未命名主题"

        vector_hits = (retrieval_payload or {}).get("vector_hits") or []
        graph_hits = (retrieval_payload or {}).get("graph_hits") or {}
        graph_edges = graph_hits.get("edges") or []
        mongo_docs = (retrieval_payload or {}).get("mongo_docs") or []

        primary_vectors, support_vectors, vector_stats = self._filter_vector_hits(vector_hits)
        strong_edges, weak_edges, graph_stats = self._filter_graph_edges(graph_edges)
        prioritized_docs = self._prioritize_docs(mongo_docs)

        rule_material = self._build_rule_material(
            topic=topic,
            primary_vectors=primary_vectors,
            support_vectors=support_vectors,
            strong_edges=strong_edges,
            weak_edges=weak_edges,
            docs=prioritized_docs,
        )
        fallback_brief = self._build_rule_fallback_brief(
            topic=topic,
            primary_vectors=primary_vectors,
            support_vectors=support_vectors,
            strong_edges=strong_edges,
            docs=prioritized_docs,
        )

        llm_brief = ""
        llm_meta: Dict[str, Any] = {"is_fallback": True, "reason": "llm_not_requested"}
        if use_llm:
            llm_brief, llm_meta = self._refine_with_llm(topic=topic, filtered_material=rule_material)

        clean_lesson_brief = llm_brief or fallback_brief

        return {
            "query": query or topic,
            "course_topic": topic,
            "clean_lesson_brief": clean_lesson_brief,
            "rule_filtered_material": rule_material,
            "filtered_evidence": {
                "primary_vector_hits": primary_vectors,
                "support_vector_hits": support_vectors,
                "strong_graph_edges": strong_edges,
                "weak_graph_edges": weak_edges,
                "prioritized_mongo_docs": prioritized_docs,
            },
            "debug": {
                "vector_stats": vector_stats,
                "graph_stats": graph_stats,
                "mongo_doc_count": len(prioritized_docs),
                "llm_used": bool(llm_brief),
                "llm_provider": _to_str(llm_meta.get("provider")),
                "llm_model": _to_str(llm_meta.get("model")),
                "llm_reason": _to_str(llm_meta.get("reason")),
            },
        }

    def _filter_vector_hits(
        self, vector_hits: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """过滤向量命中：去噪、去重、限流并按层级权重重排。"""
        scored: List[Dict[str, Any]] = []
        dropped_noise = 0

        for hit in vector_hits:
            layer = _to_str(hit.get("layer")).lower()
            snippet = self._clean_text(hit.get("chunk_text"))
            if self._is_noisy_text(snippet):
                dropped_noise += 1
                continue

            score = float(hit.get("score") or 0.0) + self.LAYER_WEIGHT.get(layer, 0.75)
            scored.append(
                {
                    "score": float(hit.get("score") or 0.0),
                    "rank_score": score,
                    "layer": layer,
                    "doc_id": _to_str(hit.get("doc_id")),
                    "chunk_id": _to_str(hit.get("chunk_id")),
                    "source_file": _to_str(hit.get("source_file")),
                    "title": _to_str(hit.get("title")),
                    "subject": _to_str(hit.get("subject")),
                    "chunk_text": snippet,
                }
            )

        scored.sort(key=lambda row: float(row.get("rank_score") or 0.0), reverse=True)

        selected: List[Dict[str, Any]] = []
        selected_norm: List[str] = []
        per_source: Dict[str, int] = {}
        per_doc: Dict[str, int] = {}
        dedup_skipped = 0

        for row in scored:
            if len(selected) >= self.max_vector_keep:
                break
            source_file = _to_str(row.get("source_file")) or "_missing_source"
            doc_id = _to_str(row.get("doc_id")) or "_missing_doc"
            if per_source.get(source_file, 0) >= self.max_per_source_file:
                continue
            if per_doc.get(doc_id, 0) >= self.max_per_doc:
                continue

            norm = self._normalize_text(row.get("chunk_text"))
            if not norm or self._is_near_duplicate(norm, selected_norm):
                dedup_skipped += 1
                continue

            selected_norm.append(norm)
            per_source[source_file] = per_source.get(source_file, 0) + 1
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            selected.append(row)

        primary = [row for row in selected if row.get("layer") in self.PRIMARY_LAYERS]
        support = [row for row in selected if row.get("layer") in self.SUPPORT_LAYERS]

        stats = {
            "input_count": len(vector_hits),
            "selected_count": len(selected),
            "primary_count": len(primary),
            "support_count": len(support),
            "dropped_noise_count": dropped_noise,
            "dedup_skipped_count": dedup_skipped,
        }
        return primary, support, stats

    def _filter_graph_edges(
        self, graph_edges: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """过滤图谱边：优先强关系，弱关系仅保留少量参考。"""
        strong: List[Dict[str, Any]] = []
        weak: List[Dict[str, Any]] = []
        seen = set()
        weak_skipped = 0

        for edge in graph_edges:
            relation = _to_str(edge.get("relation")).lower()
            source = edge.get("source_node") or {}
            target = edge.get("target_node") or {}
            source_label = self._clean_text(source.get("label") or source.get("node_id"))
            target_label = self._clean_text(target.get("label") or target.get("node_id"))
            if self._is_noisy_label(source_label) or self._is_noisy_label(target_label):
                continue

            source_layer = _to_str(source.get("layer")).lower()
            target_layer = _to_str(target.get("layer")).lower()
            cross_layer = bool(edge.get("cross_layer"))
            key = (
                relation,
                source_layer,
                source_label[:80],
                target_layer,
                target_label[:80],
            )
            if key in seen:
                continue
            seen.add(key)

            payload = {
                "relation": relation,
                "cross_layer": cross_layer,
                "source_layer": source_layer,
                "source_label": source_label,
                "target_layer": target_layer,
                "target_label": target_label,
                "doc_id": _to_str(edge.get("doc_id")),
                "confidence": float(edge.get("confidence") or 0.0),
                "evidence_text": self._clean_text(edge.get("evidence_text")),
            }

            if relation in self.STRONG_RELATIONS:
                strong.append(payload)
                continue

            if relation in self.WEAK_RELATIONS:
                weak_skipped += 1
                weak.append(payload)
                continue

            if cross_layer:
                strong.append(payload)

        strong.sort(
            key=lambda row: (
                bool(row.get("cross_layer")),
                float(row.get("confidence") or 0.0),
                row.get("relation") in self.STRONG_RELATIONS,
            ),
            reverse=True,
        )
        weak.sort(key=lambda row: float(row.get("confidence") or 0.0), reverse=True)

        strong = strong[:18]
        weak = weak[:6] if not strong else weak[:2]

        stats = {
            "input_count": len(graph_edges),
            "strong_selected_count": len(strong),
            "weak_selected_count": len(weak),
            "weak_skipped_count": weak_skipped,
            "cross_layer_strong_count": sum(1 for row in strong if row.get("cross_layer")),
        }
        return strong, weak, stats

    def _prioritize_docs(self, docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按层级权重整理 Mongo 文档摘要，保留可追溯关键字段。"""
        output: List[Dict[str, Any]] = []
        seen = set()

        for doc in docs:
            layer = _to_str(doc.get("layer")).lower()
            doc_id = _to_str(doc.get("doc_id"))
            key = f"{layer}:{doc_id}"
            if not layer or not doc_id or key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "doc_id": doc_id,
                    "layer": layer,
                    "title": _to_str(doc.get("title")),
                    "subject": _to_str(doc.get("subject")),
                    "source_file": _to_str(doc.get("source_file")),
                    "structured_summary": self._clean_text(doc.get("structured_summary")),
                }
            )

        output.sort(key=lambda row: (-int(self.LAYER_WEIGHT.get(row.get("layer"), 0) * 100), row.get("doc_id")))
        return output[:14]

    def _build_rule_material(
        self,
        topic: str,
        primary_vectors: Sequence[Dict[str, Any]],
        support_vectors: Sequence[Dict[str, Any]],
        strong_edges: Sequence[Dict[str, Any]],
        weak_edges: Sequence[Dict[str, Any]],
        docs: Sequence[Dict[str, Any]],
    ) -> str:
        """组装规则过滤后的证据文本，供 LLM 二次整理。"""
        blocks: List[str] = [f"【检索主题】{topic}", "", "【主证据片段（syllabus/textbook）】"]

        if primary_vectors:
            for row in primary_vectors[:8]:
                blocks.append(self._render_vector_line(row))
        else:
            blocks.append("- (无主证据片段)")

        blocks.append("")
        blocks.append("【辅助证据片段（resource/hotspot）】")
        if support_vectors:
            for row in support_vectors[:6]:
                blocks.append(self._render_vector_line(row))
        else:
            blocks.append("- (无辅助证据片段)")

        blocks.append("")
        blocks.append("【跨层图谱关系（强关系优先）】")
        if strong_edges:
            for edge in strong_edges[:10]:
                blocks.append(self._render_edge_line(edge))
        else:
            blocks.append("- (无强关系图谱边)")

        if weak_edges:
            blocks.append("")
            blocks.append("【图谱弱关系（仅供参考）】")
            for edge in weak_edges[:2]:
                blocks.append(self._render_edge_line(edge))

        blocks.append("")
        blocks.append("【结构化文档溯源】")
        if docs:
            for doc in docs[:10]:
                blocks.append(self._render_doc_line(doc))
        else:
            blocks.append("- (无文档溯源信息)")

        return "\n".join(blocks).strip()

    def _build_rule_fallback_brief(
        self,
        topic: str,
        primary_vectors: Sequence[Dict[str, Any]],
        support_vectors: Sequence[Dict[str, Any]],
        strong_edges: Sequence[Dict[str, Any]],
        docs: Sequence[Dict[str, Any]],
    ) -> str:
        """在 LLM 不可用时，构建规则版教学简报。"""
        core_concepts = self._collect_unique(
            [self._shorten(row.get("chunk_text"), 110) for row in primary_vectors[:4]]
        )
        key_points = self._collect_unique(
            [self._shorten(row.get("chunk_text"), 95) for row in primary_vectors[1:6]]
        )
        difficult_points = self._extract_difficult_points(primary_vectors, docs)
        cross_links = self._collect_unique([self._render_edge_line(row) for row in strong_edges[:6]])
        practice_cases = self._collect_unique(
            [self._shorten(row.get("chunk_text"), 100) for row in support_vectors[:4]]
        )

        if not core_concepts:
            core_concepts = [f"{topic}的概念与适用场景"]
        if not key_points:
            key_points = [f"{topic}相关方法的关键流程与评估方式"]
        if not difficult_points:
            difficult_points = [f"{topic}中参数选择、误差分析与迁移应用的难点"]
        if not cross_links:
            cross_links = [f"- [Textbook] {topic} -> supports -> [Syllabus] {topic}"]
        if not practice_cases:
            practice_cases = [f"{topic}在实际项目中的应用案例与效果对比"]

        return "\n".join(
            [
                f"【课程主题】{topic}",
                "【核心概念定义】",
                *[f"- {item}" for item in core_concepts[:3]],
                "【教学重点】",
                *[f"- {item}" for item in key_points[:4]],
                "【教学难点】",
                *[f"- {item}" for item in difficult_points[:3]],
                "【跨层关联（教材↔资源↔热点）】",
                *cross_links[:6],
                "【前沿/实践案例】",
                *[f"- {item}" for item in practice_cases[:4]],
                "【课堂引入与互动建议】",
                f"- 用“{topic}”真实问题导入，先让学生预测方案，再对照证据讲解。",
                f"- 通过教材概念 + 资源页示例 + 热点案例串联，形成从原理到应用的闭环。",
            ]
        )

    def _extract_difficult_points(
        self, primary_vectors: Sequence[Dict[str, Any]], docs: Sequence[Dict[str, Any]]
    ) -> List[str]:
        """提取教学难点相关线索。"""
        candidates: List[str] = []
        hard_markers = ("难点", "困难", "误区", "挑战", "收敛", "偏差", "方差", "复杂")

        for row in primary_vectors[:8]:
            text = _to_str(row.get("chunk_text"))
            if any(marker in text for marker in hard_markers):
                candidates.append(self._shorten(text, 100))

        for doc in docs:
            summary = _to_str(doc.get("structured_summary"))
            if "key_points" in summary or "difficult" in summary.lower():
                candidates.append(self._shorten(summary, 100))

        return self._collect_unique(candidates)

    def _refine_with_llm(self, topic: str, filtered_material: str) -> Tuple[str, Dict[str, Any]]:
        """调用 LLM 做最终证据整编，失败时返回兜底元信息。"""
        self._ensure_project_root_on_path()

        try:
            from app.services.llm_client import consolidate_lesson_brief  # type: ignore
        except Exception as exc:
            return "", {"is_fallback": True, "reason": f"llm_client_import_failed: {exc}"}

        try:
            payload = consolidate_lesson_brief(course_topic=topic, retrieved_evidence=filtered_material)
            if not isinstance(payload, dict):
                return "", {"is_fallback": True, "reason": "llm_payload_not_dict"}
            text = _to_str(payload.get("text"))
            if not text:
                return "", {"is_fallback": True, "reason": _to_str(payload.get("reason")) or "llm_empty_text"}
            return text, payload
        except Exception as exc:
            return "", {"is_fallback": True, "reason": f"llm_refine_failed: {exc}"}

    def _ensure_project_root_on_path(self) -> None:
        """确保可导入 app 侧 LLM 客户端。"""
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)

    def _render_vector_line(self, row: Dict[str, Any]) -> str:
        """将向量命中渲染为可读证据行。"""
        return (
            f"- [{_to_str(row.get('layer'))}] {_to_str(row.get('title')) or _to_str(row.get('doc_id'))}"
            f" (doc_id={_to_str(row.get('doc_id'))}, chunk_id={_to_str(row.get('chunk_id'))}): "
            f"{self._shorten(row.get('chunk_text'), 150)}"
        )

    def _render_edge_line(self, edge: Dict[str, Any]) -> str:
        """将图谱边渲染为可读跨层关系行。"""
        marker = " [cross-layer]" if edge.get("cross_layer") else ""
        relation = _to_str(edge.get("relation")) or "related_to"
        source = _to_str(edge.get("source_label"))
        target = _to_str(edge.get("target_label"))
        source_layer = _to_str(edge.get("source_layer"))
        target_layer = _to_str(edge.get("target_layer"))
        line = f"- [{source_layer}: {source}] -> {relation} -> [{target_layer}: {target}]{marker}"
        evidence = _to_str(edge.get("evidence_text"))
        if evidence and not self._is_noisy_text(evidence):
            line += f" | evidence: {self._shorten(evidence, 70)}"
        return line

    def _render_doc_line(self, doc: Dict[str, Any]) -> str:
        """将 Mongo 文档摘要渲染为可读溯源行。"""
        summary = _to_str(doc.get("structured_summary"))
        line = (
            f"- [{_to_str(doc.get('layer'))}] {_to_str(doc.get('title')) or _to_str(doc.get('doc_id'))}"
            f" (doc_id={_to_str(doc.get('doc_id'))}, source_file={_to_str(doc.get('source_file'))})"
        )
        if summary and not self._is_noisy_text(summary):
            line += f": {self._shorten(summary, 120)}"
        return line

    def _clean_text(self, value: Any) -> str:
        """统一清洗文本中的控制字符与异常空白。"""
        text = _to_str(value)
        text = text.replace("\uFFFD", " ")
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_noisy_text(self, value: Any) -> bool:
        """识别明显噪声文本（目录、参考文献、乱码、数字表格串等）。"""
        text = self._clean_text(value)
        if len(text) < 8:
            return True
        lower = text.lower()

        if "参考文献" in text or "bibliography" in lower or "版权" in text:
            return True
        if re.search(r"\b(references?|proceedings?|doi|icml|isbn)\b", lower):
            return True
        if re.search(r"\b(目录|contents?)\b", text, flags=re.I):
            return True
        if re.search(r"[\uFFFD]|æ|ð|皿OO|:::{2,}|阳工呻", text):
            return True
        if re.fullmatch(r"[\W_]+", text):
            return True

        digit_ratio = len(re.findall(r"\d", text)) / max(1, len(text))
        if len(text) > 40 and digit_ratio > 0.33 and re.search(r"(?:\d+\s+){8,}\d+", text):
            return True

        weird = len(re.findall(r"[^A-Za-z0-9\u4e00-\u9fff，。！？；：、,.!?;:()（）\-\s]", text))
        if weird / max(1, len(text)) > 0.20:
            return True

        return False

    def _normalize_text(self, value: Any) -> str:
        """文本归一化，用于去重比较。"""
        text = self._clean_text(value).lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text)
        return text[:320]

    def _is_noisy_label(self, value: Any) -> bool:
        """识别图节点标签噪声。"""
        text = self._clean_text(value)
        if len(text) < 2:
            return True
        if re.search(r"[\uFFFD]|æ|ð|:::{2,}|阳工呻", text):
            return True
        if re.fullmatch(r"[\W_]+", text):
            return True
        weird = len(re.findall(r"[^A-Za-z0-9\u4e00-\u9fff\-\s]", text))
        if weird / max(1, len(text)) > 0.35:
            return True
        return False

    def _is_near_duplicate(self, text: str, existing: Sequence[str]) -> bool:
        """近似重复判断，避免证据堆叠。"""
        for row in existing:
            if not row:
                continue
            if text == row:
                return True
            if min(len(text), len(row)) >= 28 and (text in row or row in text):
                return True
            if SequenceMatcher(None, text[:240], row[:240]).ratio() >= 0.9:
                return True
        return False

    def _shorten(self, value: Any, max_len: int) -> str:
        """安全截断文本，保留可读性。"""
        text = self._clean_text(value)
        if len(text) <= max_len:
            return text
        return text[: max(12, int(max_len))].rstrip(" ,.;，。；") + "..."

    def _collect_unique(self, values: Sequence[str]) -> List[str]:
        """按归一化键做保序去重。"""
        output: List[str] = []
        seen = set()
        for item in values:
            text = self._clean_text(item)
            norm = self._normalize_text(text)
            if not text or not norm or norm in seen:
                continue
            seen.add(norm)
            output.append(text)
        return output
