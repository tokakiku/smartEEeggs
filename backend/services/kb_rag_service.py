import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from services.kb_rag_adapter import KBRAGAdapter
from services.mongo_kb_service import MongoKBService


class KBRAGService:
    # 知识库驱动 RAG v2：混合检索 + 更干净的解释性生成

    LAYER_LABELS = {
        "syllabus": "教学大纲层",
        "textbook": "教材层",
        "resource": "教学资源层",
        "hotspot": "热点层",
    }

    def __init__(self, kb_service: MongoKBService) -> None:
        self.adapter = KBRAGAdapter(kb_service=kb_service)

    def rag_search(
        self,
        query: str,
        subject: Optional[str] = None,
        top_k: int = 5,
        layers: Optional[List[str]] = None,
        max_contexts: int = 12,
    ) -> Dict[str, Any]:
        bundle = self.adapter.build_contexts(
            query=query,
            subject=subject,
            top_k=top_k,
            layers=layers,
            max_contexts=max_contexts,
        )
        contexts = bundle["contexts"]
        answer = self._synthesize_answer(query=query, contexts=contexts, counts=bundle["counts"])
        debug = dict(bundle.get("debug", {}))
        return {
            "query": query,
            "subject": subject,
            "top_k": top_k,
            "layers": bundle["layers"],
            "counts": bundle["counts"],
            "answer": answer,
            "context_count": len(contexts),
            "engine": "kb_rag",
            "mode": "kb_mainline",
            "retrieval_mode": debug.get("retrieval_mode"),
            "contexts_used": [
                {
                    "layer": item.get("layer"),
                    "doc_id": item.get("doc_id"),
                    "source_file": item.get("source_file"),
                    "title": item.get("title"),
                    "retrieval_mode": item.get("retrieval_mode"),
                    "vector_score": item.get("vector_score"),
                    "is_primary": item.get("is_primary"),
                    "priority_score": item.get("priority_score"),
                    "course_code": item.get("course_code"),
                    "textbook_role": item.get("textbook_role"),
                    "textbook_role_source": item.get("textbook_role_source"),
                    "matched_syllabus_material": item.get("matched_syllabus_material"),
                    "syllabus_anchor_doc_id": item.get("syllabus_anchor_doc_id"),
                    "edition": item.get("edition"),
                }
                for item in contexts
            ],
            "debug": debug,
            "contexts": contexts,
            "retrieval": bundle["results"],
        }

    def _synthesize_answer(self, query: str, contexts: List[Dict[str, Any]], counts: Dict[str, int]) -> str:
        if not contexts:
            return (
                f"在当前知识库中未检索到与“{query}”直接相关的内容。"
                "建议补充该主题的教材、课件或教学大纲后再检索。"
            )

        lines: List[str] = []
        lines.append(f"问题：{query}")
        lines.append(f"概念定义：{self._topic_brief(query)}")
        lines.append(f"在机器学习中的作用：{self._topic_role(query)}")
        lines.append(f"知识库命中位置：{self._render_layer_summary(counts)}")
        lines.append("支撑资料：")
        for idx, evidence in enumerate(self._build_evidence_lines(contexts, query=query, max_items=4), start=1):
            lines.append(f"{idx}. {evidence}")
        lines.append(f"来源分布：{self._render_source_summary(contexts)}")
        lines.append("结论：当前检索结果已可用于课堂讲解与备课，可在详情页进一步补充公式推导与例题。")
        return "\n".join(lines)

    def _topic_brief(self, query: str) -> str:
        q = str(query or "").strip().lower()
        if "梯度下降" in q:
            return "梯度下降法通过沿损失函数的负梯度方向迭代更新参数，以逐步逼近最优解。"
        if "假设空间" in q:
            return "假设空间是模型可选函数集合，决定了学习器能够表达的模式范围。"
        if "文献筛选" in q:
            return "文献筛选用于判断研究资料是否与目标问题相关，帮助建立高质量证据集。"
        if "大模型应用" in q:
            return "大模型应用关注模型在实际场景中的落地能力、效果边界和成本收益。"
        if "机器学习" in q:
            return "机器学习通过数据驱动方式学习规律，并用于预测、分类或决策任务。"
        return "以下回答基于知识库检索到的上下文证据进行整理。"

    def _topic_role(self, query: str) -> str:
        q = str(query or "").strip().lower()
        if "梯度下降" in q:
            return "它是回归、分类与神经网络训练中的核心优化手段，直接影响训练收敛速度与稳定性。"
        if "假设空间" in q:
            return "它影响模型复杂度、泛化能力和归纳偏好，是理解过拟合与模型选择的基础。"
        if "文献筛选" in q:
            return "它可以显著降低人工阅读成本，并提高研究综述与选题阶段的效率。"
        if "大模型应用" in q:
            return "它帮助把模型能力转化为业务价值，也可作为教学案例设计的实践入口。"
        if "机器学习" in q:
            return "它是课程主线知识，连接算法原理、工程实现和应用评估三个层面。"
        return "该主题可作为课程讲解与项目实践的关键知识点。"

    def _render_layer_summary(self, counts: Dict[str, int]) -> str:
        parts = [
            f"{self.LAYER_LABELS[layer]}{counts[layer]}条"
            for layer in ["syllabus", "textbook", "resource", "hotspot"]
            if counts.get(layer, 0) > 0
        ]
        if not parts:
            return "无有效命中"
        return "，".join(parts)

    def _render_source_summary(self, contexts: List[Dict[str, Any]]) -> str:
        source_parts: List[str] = []
        seen = set()
        for context in contexts:
            layer = str(context.get("layer") or "unknown")
            layer_label = self.LAYER_LABELS.get(layer, layer)
            title = str(context.get("title") or context.get("source_file") or "").strip()
            if not title:
                continue
            key = f"{layer}:{title.lower()}"
            if key in seen:
                continue
            seen.add(key)
            source_parts.append(f"[{layer_label}] {title}")
            if len(source_parts) >= 6:
                break
        return "；".join(source_parts) if source_parts else "无可用来源"

    def _build_evidence_lines(self, contexts: List[Dict[str, Any]], query: str, max_items: int = 4) -> List[str]:
        lines: List[str] = []
        ranked_contexts = sorted(contexts, key=self._context_priority, reverse=True)
        selected_norms: List[str] = []
        selected_sources: set = set()

        def _append_context(context: Dict[str, Any]) -> bool:
            snippet = self._extract_snippet(str(context.get("text") or ""), query=query)
            if self._is_noisy_snippet(snippet):
                return False
            norm = self._normalize_for_dedup(snippet)
            if not norm:
                return False
            if self._is_near_duplicate(norm, selected_norms):
                return False

            layer = str(context.get("layer") or "unknown")
            layer_label = self.LAYER_LABELS.get(layer, layer)
            source = str(context.get("title") or context.get("source_file") or "未命名文档").strip()
            source_key = f"{layer}:{source.lower()}"
            if source_key in selected_sources and len(lines) >= 2:
                return False

            primary_tag = " [主大纲]" if layer == "syllabus" and context.get("is_primary") else ""
            textbook_role = str(context.get("textbook_role") or "").strip().lower()
            textbook_role_source = str(context.get("textbook_role_source") or "").strip().lower()
            textbook_tag = ""
            if layer == "textbook" and textbook_role == "main":
                textbook_tag = " [主教材]"
            elif layer == "textbook" and textbook_role == "supplementary":
                textbook_tag = " [辅教材]"
            syllabus_hint_tag = ""
            if layer == "textbook" and textbook_role_source == "syllabus_main":
                syllabus_hint_tag = " [课标驱动]"
            elif layer == "textbook" and textbook_role_source == "syllabus_reference":
                syllabus_hint_tag = " [课标参考]"
            lines.append(f"[{layer_label}]{primary_tag}{textbook_tag}{syllabus_hint_tag} {source}: {snippet}")
            selected_norms.append(norm)
            selected_sources.add(source_key)
            return True

        # 第零轮：主教材优先，先尝试加入一条主教材证据。
        for context in ranked_contexts:
            if len(lines) >= max_items:
                break
            if str(context.get("layer") or "") != "textbook":
                continue
            if str(context.get("textbook_role") or "").strip().lower() != "main":
                continue
            if _append_context(context):
                break

        # 第一轮：尽量保证证据的层次多样性。
        used_layers = set()
        for line in lines:
            if "[教材层]" in line:
                used_layers.add("textbook")
        for context in ranked_contexts:
            if len(lines) >= max_items:
                break
            layer = str(context.get("layer") or "unknown")
            if layer in used_layers:
                continue
            if _append_context(context):
                used_layers.add(layer)

        # 第二轮：从全量排序上下文中补齐剩余名额。
        if len(lines) < max_items:
            for context in ranked_contexts:
                if len(lines) >= max_items:
                    break
                _append_context(context)

        return lines

    def _context_priority(self, context: Dict[str, Any]) -> float:
        text = str(context.get("text") or "")
        mode = str(context.get("retrieval_mode") or "lexical")
        vector_score = float(context.get("vector_score") or 0.0)

        score = 0.0
        for key in ["重点命中", "难点命中", "章节命中", "知识点命中", "模块命中", "页面命中"]:
            if key in text:
                score += 2.0
        if "片段命中" in text:
            score -= 0.6

        if mode == "both":
            score += 2.0
        elif mode == "lexical":
            score += 1.0
        elif mode == "vector":
            score += 0.8
        score += vector_score

        layer = str(context.get("layer") or "")
        if layer == "syllabus":
            score += 1.2
            if context.get("is_primary"):
                score += 4.0
            score += min(3.0, float(context.get("priority_score") or 0.0) * 0.05)
        elif layer == "textbook":
            score += 1.0
            textbook_role = str(context.get("textbook_role") or "").strip().lower()
            role_source = str(context.get("textbook_role_source") or "").strip().lower()
            if textbook_role == "main":
                score += 3.2
            elif textbook_role == "supplementary":
                score -= 0.8
            if role_source == "syllabus_main":
                score += 2.5
            elif role_source == "syllabus_reference":
                score -= 1.2
            score += min(2.0, float(context.get("priority_score") or 0.0) * 0.02)
        elif layer == "resource":
            score += 0.7

        source = str(context.get("title") or context.get("source_file") or "").strip().lower()
        if "目录" in source or source == "contents":
            score -= 1.2
        return score

    def _extract_snippet(self, text: str, query: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return "（无可用片段）"

        candidates = [seg.strip() for seg in re.split(r"[\n;；。]", cleaned) if seg.strip()]
        preferred = ""
        query_terms = self._query_terms(query)
        for seg in candidates:
            if any(term and term in seg.lower() for term in query_terms):
                preferred = seg
                break
        for seg in candidates:
            if preferred:
                break
            if any(key in seg for key in ["命中", "定义", "作用", "梯度下降", "假设空间", "优化", "模型"]):
                preferred = seg
                break
        if not preferred and candidates:
            preferred = candidates[0]

        if ":" in preferred:
            preferred = preferred.split(":", 1)[1].strip()
        if "：" in preferred:
            preferred = preferred.split("：", 1)[1].strip()
        preferred = re.sub(r"(?:\s*[.。]\s*){3,}\d*", " ", preferred).strip()
        return self._truncate(preferred, 120)

    def _is_noisy_snippet(self, text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        if len(value) < 10:
            return True
        if re.search(r"[.。]{8,}", value):
            return True
        if re.search(r"[=+\-/*]{6,}", value):
            return True
        if re.search(r"(?:[\u4e00-\u9fffA-Za-z]\s+){8,}[\u4e00-\u9fffA-Za-z]?", value):
            return True
        tokens = value.split()
        if len(tokens) >= 8:
            single_char_ratio = len([t for t in tokens if len(t) == 1]) / len(tokens)
            if single_char_ratio > 0.6:
                return True
        if re.fullmatch(r"(目\s*录|contents?)", value, flags=re.I):
            return True
        useful = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", value)
        return len(useful) / max(1, len(value)) < 0.45

    def _query_terms(self, query: str) -> List[str]:
        q = str(query or "").strip().lower()
        if not q:
            return []
        terms = [q, re.sub(r"\s+", "", q)]
        if q.endswith("法") and len(q) > 2:
            terms.append(q[:-1])
        if q.endswith("算法") and len(q) > 3:
            terms.append(q[:-2])

        split_terms = [item for item in re.split(r"[\s,，;；/]+", q) if item and len(item) >= 2]
        terms.extend(split_terms)

        unique: List[str] = []
        seen = set()
        for term in terms:
            value = str(term or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return unique

    def _normalize_for_dedup(self, text: str) -> str:
        value = str(text or "").lower()
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value)
        return value[:260]

    def _is_near_duplicate(self, text: str, existing_texts: List[str]) -> bool:
        for existing in existing_texts:
            if not existing:
                continue
            if text == existing:
                return True
            if min(len(text), len(existing)) >= 24 and (text in existing or existing in text):
                return True
            if SequenceMatcher(None, text[:220], existing[:220]).ratio() >= 0.9:
                return True
        return False

    def _truncate(self, text: str, size: int) -> str:
        value = str(text or "").strip()
        if len(value) <= size:
            return value
        return value[:size].rstrip("，,。.;； ") + "..."
