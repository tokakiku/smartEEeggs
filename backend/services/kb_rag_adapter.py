import os
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from services.hybrid_retrieval_service import HybridRetrievalService
from services.mongo_kb_service import MongoKBService


class KBRAGAdapter:
    # 适配层：将检索结果整理为可直接用于答案生成的上下文。
    PREFERRED_LAYER_ORDER = ["syllabus", "textbook", "resource", "hotspot"]
    DEFAULT_MAX_PER_LAYER = max(1, int(os.getenv("KB_RAG_MAX_PER_LAYER", "4")))
    TEXTBOOK_CONTEXT_CAP = max(1, int(os.getenv("KB_RAG_TEXTBOOK_CAP", "4")))
    RESOURCE_CONTEXT_CAP = max(1, int(os.getenv("KB_RAG_RESOURCE_CAP", "2")))
    HOTSPOT_CONTEXT_CAP = max(1, int(os.getenv("KB_RAG_HOTSPOT_CAP", "1")))
    NEAR_DUP_THRESHOLD = float(os.getenv("KB_RAG_NEAR_DUP_THRESHOLD", "0.9"))
    MIN_CONTEXT_CHARS = max(10, int(os.getenv("KB_RAG_MIN_CONTEXT_CHARS", "18")))

    def __init__(self, kb_service: MongoKBService) -> None:
        self.kb_service = kb_service
        self.retrieval_service = HybridRetrievalService(kb_service=kb_service)

    def build_contexts(
        self,
        query: str,
        subject: Optional[str] = None,
        top_k: int = 5,
        layers: Optional[List[str]] = None,
        max_contexts: int = 12,
    ) -> Dict[str, Any]:
        retrieval_bundle = self.retrieval_service.retrieve_hybrid(
            query=query,
            subject=subject,
            layers=layers,
            top_k=top_k,
        )

        grouped = retrieval_bundle["results"]
        contexts = self._flatten_grouped_results(grouped=grouped, max_contexts=max_contexts)
        return {
            "query": query,
            "subject": subject,
            "top_k": top_k,
            "layers": list(grouped.keys()),
            "results": grouped,
            "counts": retrieval_bundle["counts"],
            "debug": retrieval_bundle.get("debug", {}),
            "contexts": contexts,
        }

    def _flatten_grouped_results(self, grouped: Dict[str, List[Dict[str, Any]]], max_contexts: int) -> List[Dict[str, Any]]:
        limit = max(1, min(int(max_contexts), 50))
        candidates_by_layer: Dict[str, List[Dict[str, Any]]] = {}
        seen_doc_keys = set()

        for layer, items in grouped.items():
            layer_candidates: List[Dict[str, Any]] = []
            for item in items:
                doc_id = str(item.get("doc_id") or "").strip()
                source_file = str(item.get("source_file") or "").strip()
                key = f"{layer}:{doc_id or source_file}"
                if not key or key in seen_doc_keys:
                    continue
                seen_doc_keys.add(key)

                raw_text = self._render_context_text(layer=layer, item=item)
                clean_text = self._truncate(self._clean_text(raw_text), 420)
                if not clean_text:
                    continue
                if len(clean_text) < self.MIN_CONTEXT_CHARS:
                    continue
                if self._is_noisy(clean_text):
                    continue

                layer_candidates.append(
                    {
                        "layer": layer,
                        "doc_id": doc_id or None,
                        "source_file": source_file or None,
                        "title": str(item.get("title") or "").strip() or None,
                        "detail_api": item.get("detail_api"),
                        "retrieval_mode": item.get("retrieval_mode", "lexical"),
                        "vector_score": item.get("vector_score"),
                        "is_primary": bool(item.get("is_primary")),
                        "priority_score": float(item.get("priority_score") or 0.0),
                        "course_code": item.get("course_code"),
                        "school": item.get("school"),
                        "department": item.get("department"),
                        "major": item.get("major"),
                        "academic_year": item.get("academic_year"),
                        "semester": item.get("semester"),
                        "version": item.get("version"),
                        "teacher": item.get("teacher"),
                        "effective_date": item.get("effective_date"),
                        "textbook_role": item.get("textbook_role"),
                        "edition": item.get("edition"),
                        "academic_year": item.get("academic_year"),
                        "author": item.get("author"),
                        "authors": item.get("authors") or [],
                        "textbook_role_source": item.get("textbook_role_source"),
                        "matched_syllabus_material": item.get("matched_syllabus_material"),
                        "syllabus_anchor_doc_id": item.get("syllabus_anchor_doc_id"),
                        "text": clean_text,
                        "_score": self._context_score(layer=layer, item=item, text=clean_text),
                        "_norm_text": self._normalize_for_dedup(clean_text),
                    }
                )

            if layer_candidates:
                layer_candidates.sort(key=lambda x: float(x.get("_score") or 0.0), reverse=True)
                candidates_by_layer[layer] = layer_candidates

        if not candidates_by_layer:
            return []

        active_layers: List[str] = [layer for layer in self.PREFERRED_LAYER_ORDER if layer in candidates_by_layer]
        for layer in grouped.keys():
            if layer in candidates_by_layer and layer not in active_layers:
                active_layers.append(layer)

        layer_caps = self._resolve_layer_caps(active_layers=active_layers, limit=limit)
        layer_cursor: Dict[str, int] = {layer: 0 for layer in active_layers}
        layer_counts: Dict[str, int] = defaultdict(int)

        selected: List[Dict[str, Any]] = []
        selected_norms: List[str] = []

        def _pick_one(layer: str, ignore_cap: bool = False) -> bool:
            if not ignore_cap and layer_counts[layer] >= layer_caps.get(layer, 1):
                return False
            candidates = candidates_by_layer.get(layer) or []
            index = layer_cursor.get(layer, 0)
            while index < len(candidates):
                candidate = candidates[index]
                index += 1
                layer_cursor[layer] = index
                norm_text = str(candidate.get("_norm_text") or "")
                if not norm_text:
                    continue
                if self._is_semantic_duplicate(norm_text, selected_norms):
                    continue

                selected.append(
                    {
                        "layer": candidate.get("layer"),
                        "doc_id": candidate.get("doc_id"),
                        "source_file": candidate.get("source_file"),
                        "title": candidate.get("title"),
                        "detail_api": candidate.get("detail_api"),
                        "retrieval_mode": candidate.get("retrieval_mode"),
                        "vector_score": candidate.get("vector_score"),
                        "is_primary": candidate.get("is_primary"),
                        "priority_score": candidate.get("priority_score"),
                        "course_code": candidate.get("course_code"),
                        "school": candidate.get("school"),
                        "department": candidate.get("department"),
                        "major": candidate.get("major"),
                        "academic_year": candidate.get("academic_year"),
                        "semester": candidate.get("semester"),
                        "version": candidate.get("version"),
                        "teacher": candidate.get("teacher"),
                        "effective_date": candidate.get("effective_date"),
                        "textbook_role": candidate.get("textbook_role"),
                        "edition": candidate.get("edition"),
                        "academic_year": candidate.get("academic_year"),
                        "author": candidate.get("author"),
                        "authors": candidate.get("authors") or [],
                        "textbook_role_source": candidate.get("textbook_role_source"),
                        "matched_syllabus_material": candidate.get("matched_syllabus_material"),
                        "syllabus_anchor_doc_id": candidate.get("syllabus_anchor_doc_id"),
                        "text": candidate.get("text"),
                    }
                )
                selected_norms.append(norm_text)
                layer_counts[layer] += 1
                return True
            return False

        # 第一轮：尽量保证跨层证据覆盖。
        for layer in active_layers:
            if len(selected) >= limit:
                break
            _pick_one(layer)

        # 第二轮：按层轮询补齐，并控制每层配额。
        progress = True
        while len(selected) < limit and progress:
            progress = False
            for layer in active_layers:
                if len(selected) >= limit:
                    break
                if _pick_one(layer):
                    progress = True

        # 第三轮：若配额导致数量不足，放宽配额但继续去重。
        if len(selected) < limit:
            progress = True
            while len(selected) < limit and progress:
                progress = False
                for layer in active_layers:
                    if len(selected) >= limit:
                        break
                    if _pick_one(layer, ignore_cap=True):
                        progress = True

        return selected[:limit]

    def _resolve_layer_caps(self, active_layers: List[str], limit: int) -> Dict[str, int]:
        if not active_layers:
            return {}
        if len(active_layers) == 1:
            return {active_layers[0]: min(limit, self.DEFAULT_MAX_PER_LAYER)}

        avg_quota = max(1, (limit + len(active_layers) - 1) // len(active_layers))
        cap = max(1, min(self.DEFAULT_MAX_PER_LAYER, avg_quota + 1))
        caps = {layer: cap for layer in active_layers}

        # 仅对 textbook/resource/hotspot 做层内优先级配额，不改 syllabus 逻辑。
        if "textbook" in caps:
            caps["textbook"] = min(limit, max(caps["textbook"], self.TEXTBOOK_CONTEXT_CAP))
        if "resource" in caps:
            caps["resource"] = min(limit, min(caps["resource"], self.RESOURCE_CONTEXT_CAP))
        if "hotspot" in caps:
            caps["hotspot"] = min(limit, min(caps["hotspot"], self.HOTSPOT_CONTEXT_CAP))
        return caps

    def _context_score(self, layer: str, item: Dict[str, Any], text: str) -> float:
        score = 0.0
        mode = str(item.get("retrieval_mode") or "lexical")
        vector_score = float(item.get("vector_score") or 0.0)

        if mode == "both":
            score += 2.4
        elif mode == "lexical":
            score += 1.2
        elif mode == "vector":
            score += 0.9
        score += vector_score

        if layer == "syllabus":
            score += 0.7 * len(item.get("matched_key_points") or [])
            score += 0.5 * len(item.get("matched_modules") or [])
            score += 0.5 * len(item.get("matched_knowledge_points") or [])
            score += 0.3 * len(item.get("matched_goals") or [])
            if item.get("is_primary"):
                score += 4.0
            score += min(4.0, float(item.get("priority_score") or 0.0) * 0.05)
        elif layer == "textbook":
            # 教材层作为概念主证据，提升权重。
            score += 1.0 * len(item.get("matched_knowledge_points") or [])
            score += 0.7 * len(item.get("matched_sections") or [])
            score += 0.5 * len(item.get("matched_chunks_preview") or [])
            textbook_role = str(item.get("textbook_role") or "").strip().lower()
            role_source = str(item.get("textbook_role_source") or "").strip().lower()
            if textbook_role == "main":
                score += 4.0
            elif textbook_role == "supplementary":
                score -= 0.8
            if role_source == "syllabus_main":
                score += 2.8
            elif role_source == "syllabus_reference":
                score -= 1.2
            score += min(3.0, float(item.get("priority_score") or 0.0) * 0.03)
        elif layer == "resource":
            # 资源层作为辅助教学证据，保留但低于教材层。
            score += 0.7 * len(item.get("matched_pages") or [])
            score += 0.6 * len(item.get("matched_units") or [])
        elif layer == "hotspot":
            # 热点层作为补充背景证据，默认权重最低。
            score += 0.8 * len(item.get("matched_knowledge_points") or [])
            score += 0.5 * len(item.get("matched_keywords") or [])
            if not item.get("matched_knowledge_points"):
                score -= 0.8

        score += min(len(text) / 160.0, 1.5)
        if self._looks_like_toc_or_heading(text):
            score -= 1.8
        return score

    def _normalize_for_dedup(self, text: str) -> str:
        value = str(text or "").lower()
        value = re.sub(r"\s+", "", value)
        value = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value)
        return value[:320]

    def _is_semantic_duplicate(self, norm_text: str, selected_norms: List[str]) -> bool:
        if not norm_text:
            return True
        for existing in selected_norms:
            if not existing:
                continue
            if norm_text == existing:
                return True
            short = min(len(norm_text), len(existing))
            if short >= 30 and (norm_text in existing or existing in norm_text):
                return True
            jaccard = self._char_bigram_jaccard(norm_text, existing)
            if jaccard >= 0.92:
                return True
            ratio = SequenceMatcher(None, norm_text[:260], existing[:260]).ratio()
            if ratio >= self.NEAR_DUP_THRESHOLD:
                return True
        return False

    def _char_bigram_jaccard(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0

        def _to_bigrams(value: str) -> set:
            if len(value) <= 1:
                return {value}
            return {value[idx : idx + 2] for idx in range(len(value) - 1)}

        left_set = _to_bigrams(left)
        right_set = _to_bigrams(right)
        union = left_set | right_set
        if not union:
            return 0.0
        return len(left_set & right_set) / len(union)

    def _render_context_text(self, layer: str, item: Dict[str, Any]) -> str:
        if layer == "syllabus":
            return self._render_syllabus_context(item)
        if layer == "textbook":
            return self._render_textbook_context(item)
        if layer == "resource":
            return self._render_resource_context(item)
        if layer == "hotspot":
            return self._render_hotspot_context(item)
        return self._render_generic_context(item)

    def _render_syllabus_context(self, item: Dict[str, Any]) -> str:
        parts: List[str] = []
        self._append_labeled(parts, "课程", item.get("title"))
        self._append_labeled(parts, "主大纲", "是" if item.get("is_primary") else "否")
        self._append_labeled(parts, "课程代码", item.get("course_code"))
        self._append_labeled(parts, "专业", item.get("major"))
        self._append_labeled(parts, "学院", item.get("department"))
        self._append_labeled(parts, "学校", item.get("school"))
        self._append_labeled(parts, "学年", item.get("academic_year"))
        self._append_labeled(parts, "学期", item.get("semester"))
        self._append_labeled(parts, "版本", item.get("version"))
        self._append_labeled(parts, "教师", item.get("teacher"))
        self._append_labeled(parts, "生效日期", item.get("effective_date"))
        self._append_list(parts, "模块命中", item.get("matched_modules"), max_items=4, max_chars=52)
        self._append_list(parts, "知识点命中", item.get("matched_knowledge_points"), max_items=4, max_chars=52)
        self._append_list(parts, "重点命中", item.get("matched_key_points"), max_items=4, max_chars=52)
        self._append_list(parts, "难点命中", item.get("matched_difficult_points"), max_items=4, max_chars=52)
        self._append_list(parts, "目标命中", item.get("matched_goals"), max_items=3, max_chars=60)
        self._append_list(parts, "进度命中", item.get("matched_schedule_topics"), max_items=3, max_chars=52)
        self._append_list(parts, "语义补充", item.get("vector_snippets"), max_items=2, max_chars=80)
        return "\n".join(parts).strip()

    def _render_textbook_context(self, item: Dict[str, Any]) -> str:
        parts: List[str] = []
        self._append_labeled(parts, "教材", item.get("title"))
        role = str(item.get("textbook_role") or "").strip().lower()
        if role == "main":
            self._append_labeled(parts, "教材角色", "主教材")
        elif role == "supplementary":
            self._append_labeled(parts, "教材角色", "辅教材")
        else:
            self._append_labeled(parts, "教材角色", "未标注")
        role_source = str(item.get("textbook_role_source") or "").strip().lower()
        if role_source == "syllabus_main":
            self._append_labeled(parts, "角色依据", "课标主教材匹配")
        elif role_source == "syllabus_reference":
            self._append_labeled(parts, "角色依据", "课标参考教材匹配")
        self._append_labeled(parts, "版本", item.get("edition"))
        self._append_labeled(parts, "作者", item.get("author"))
        self._append_labeled(parts, "学年", item.get("academic_year"))
        self._append_labeled(parts, "匹配教材条目", item.get("matched_syllabus_material"))
        self._append_list(parts, "章节命中", item.get("matched_sections"), max_items=4, max_chars=56)
        self._append_list(parts, "知识点命中", item.get("matched_knowledge_points"), max_items=5, max_chars=52)
        self._append_list(parts, "片段命中", item.get("matched_chunks_preview"), max_items=2, max_chars=120)
        self._append_list(parts, "语义补充", item.get("vector_snippets"), max_items=2, max_chars=96)
        return "\n".join(parts).strip()

    def _render_resource_context(self, item: Dict[str, Any]) -> str:
        parts: List[str] = []
        self._append_labeled(parts, "资源", item.get("title"))
        page_lines: List[str] = []
        for page in item.get("matched_pages") or []:
            page_no = page.get("page_no")
            page_title = str(page.get("page_title") or "").strip()
            page_role = str(page.get("page_role") or "").strip()
            line = f"p{page_no}: {page_title}".strip()
            if page_role:
                line = f"{line} ({page_role})"
            if line:
                page_lines.append(line)
        self._append_list(parts, "页面命中", page_lines, max_items=4, max_chars=60)
        self._append_list(parts, "页面角色命中", item.get("matched_page_roles"), max_items=4, max_chars=42)
        self._append_list(parts, "复用单元命中", item.get("matched_units"), max_items=4, max_chars=58)
        self._append_list(parts, "语义补充", item.get("vector_snippets"), max_items=2, max_chars=96)
        return "\n".join(parts).strip()

    def _render_hotspot_context(self, item: Dict[str, Any]) -> str:
        parts: List[str] = []
        self._append_labeled(parts, "热点", item.get("title"))
        self._append_labeled(parts, "事件类型", item.get("event_type"))
        self._append_labeled(parts, "摘要", self._truncate(str(item.get("summary") or ""), 180))
        self._append_list(parts, "知识点命中", item.get("matched_knowledge_points"), max_items=4, max_chars=45)
        self._append_list(parts, "关键词命中", item.get("matched_keywords"), max_items=4, max_chars=36)
        self._append_list(parts, "教学用途", item.get("teaching_usage"), max_items=4, max_chars=36)
        self._append_list(parts, "语义补充", item.get("vector_snippets"), max_items=2, max_chars=96)
        return "\n".join(parts).strip()

    def _render_generic_context(self, item: Dict[str, Any]) -> str:
        parts: List[str] = []
        self._append_labeled(parts, "标题", item.get("title"))
        self._append_list(parts, "片段", item.get("vector_snippets"), max_items=2, max_chars=100)
        return "\n".join(parts).strip()

    def _append_labeled(self, parts: List[str], label: str, value: Any) -> None:
        text = self._clean_text(value)
        if text:
            parts.append(f"{label}: {text}")

    def _append_list(
        self,
        parts: List[str],
        label: str,
        values: Any,
        max_items: int = 5,
        max_chars: int = 80,
    ) -> None:
        items: List[str] = []
        for value in values or []:
            text = self._clean_text(value)
            if not text:
                continue
            if self._is_noisy(text):
                continue
            if self._looks_like_toc_or_heading(text):
                continue
            items.append(self._truncate(text, max_chars))
            if len(items) >= max_items:
                break
        if items:
            parts.append(f"{label}: {'; '.join(items)}")

    def _clean_text(self, value: Any) -> str:
        text = str(value or "")
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
        text = text.replace("\uFFFD", " ")
        text = re.sub(r"[·•●■□◆►▸▪◦]+", " ", text)
        text = re.sub(r"(?:\s*[.。]\s*){3,}\d*", " ", text)
        text = re.sub(r"[=+\-/*]{6,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_noisy(self, text: str) -> bool:
        clean = str(text or "").strip()
        if not clean:
            return True
        if len(clean) < self.MIN_CONTEXT_CHARS:
            return True
        if self._looks_like_toc_or_heading(clean):
            return True
        if re.search(r"[.。]{8,}", clean):
            return True
        if re.search(r"[=+\-/*]{6,}", clean):
            return True
        if re.search(r"(?:[\u4e00-\u9fffA-Za-z]\s+){8,}[\u4e00-\u9fffA-Za-z]?", clean):
            return True
        tokens = clean.split()
        if len(tokens) >= 8:
            single_char_ratio = len([t for t in tokens if len(t) == 1]) / len(tokens)
            if single_char_ratio > 0.6:
                return True
        digit_ratio = len(re.findall(r"\d", clean)) / max(1, len(clean))
        if digit_ratio > 0.35 and len(clean) > 24:
            return True
        if re.fullmatch(r"[\W_]+", clean):
            return True
        allowed = re.sub(r"[A-Za-z0-9\u4e00-\u9fff，。！？；：、,.!?;:()（）\-_/\s]", "", clean)
        return len(allowed) / max(1, len(clean)) > 0.28

    def _looks_like_toc_or_heading(self, text: str) -> bool:
        value = str(text or "").strip().lower()
        if not value:
            return False
        if re.fullmatch(r"(目\s*录|contents?)", value):
            return True
        if re.search(r"\.{2,}\s*\d+$", value):
            return True
        if re.search(r"(第\s*\d+\s*页|\bpage\s*\d+\b)$", value):
            return True
        if re.fullmatch(r"\d+(\.\d+){0,4}", value):
            return True
        return False

    def _truncate(self, text: str, size: int) -> str:
        value = str(text or "").strip()
        if len(value) <= size:
            return value
        return value[:size].rstrip("，,。.;； ") + "..."
