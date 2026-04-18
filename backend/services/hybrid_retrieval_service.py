from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from services.cross_layer_retrieval_service import CrossLayerRetrievalService
from services.mongo_kb_service import MongoKBService
from services.vector_index_service import VectorIndexService, get_vector_index_service

LEXICAL_WEIGHT = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.6"))
VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.4"))
VECTOR_THRESHOLD = float(os.getenv("HYBRID_VECTOR_THRESHOLD", "0.45"))


class HybridRetrievalService:
    DEFAULT_LAYERS = CrossLayerRetrievalService.DEFAULT_LAYERS

    def __init__(
        self,
        kb_service: MongoKBService,
        vector_index: Optional[VectorIndexService] = None,
    ) -> None:
        self.kb_service = kb_service
        self.lexical_service = CrossLayerRetrievalService(kb_service=kb_service)
        self.vector_index = vector_index or get_vector_index_service()

    def retrieve_hybrid(
        self,
        query: str,
        subject: Optional[str] = None,
        layers: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        q = str(query or "").strip()
        target_layers = self.lexical_service._normalize_layers(layers)
        if not q:
            empty = {layer: [] for layer in target_layers}
            return {"results": empty, "counts": {k: 0 for k in empty}, "debug": {"retrieval_mode": "empty_query"}}

        lexical_grouped = self.lexical_service.retrieve_across_layers(
            query=q,
            subject=subject,
            layers=target_layers,
            top_k=top_k,
        )

        debug: Dict[str, Any] = {
            "retrieval_mode": "lexical_only",
            "lexical_hits": sum(len(items) for items in lexical_grouped.values()),
            "vector_hits": 0,
            "vector_hits_raw": 0,
            "hybrid_hits": 0,
            "lexical_weight": LEXICAL_WEIGHT,
            "vector_weight": VECTOR_WEIGHT,
            "vector_threshold": VECTOR_THRESHOLD,
            "vector_bootstrap": "skip",
        }

        vector_hits: List[Dict[str, Any]] = []
        try:
            vector_threshold = VECTOR_THRESHOLD
            if debug["lexical_hits"] == 0:
                vector_threshold = max(VECTOR_THRESHOLD, 0.72)
            debug["effective_vector_threshold"] = vector_threshold

            bootstrap_info = self._ensure_vector_index_ready()
            debug["vector_bootstrap"] = bootstrap_info
            raw_vector_hits = self.vector_index.search(
                query=q,
                top_k=max(top_k * 8, 24),
                layers=target_layers,
                min_score=vector_threshold,
            )
            debug["vector_hits_raw"] = len(raw_vector_hits)
            vector_hits, existence_debug = self._filter_vector_hits_by_mongo(raw_vector_hits, target_layers)
            debug.update(existence_debug)
            debug["vector_hits"] = len(vector_hits)
            debug["vector_index_size"] = self.vector_index.count_active()
        except Exception as exc:
            debug["vector_error"] = str(exc)
            debug["retrieval_mode"] = "lexical_fallback"

        merged: Dict[str, List[Dict[str, Any]]] = {}
        for layer in target_layers:
            lex_items = lexical_grouped.get(layer, [])
            layer_vector_hits = [item for item in vector_hits if item.get("layer") == layer]
            merged[layer] = self._merge_layer(
                layer=layer,
                lexical_items=lex_items,
                vector_items=layer_vector_hits,
                top_k=top_k,
            )

        hybrid_hits = sum(len(items) for items in merged.values())
        debug["hybrid_hits"] = hybrid_hits
        if debug.get("vector_hits", 0) > 0:
            debug["retrieval_mode"] = "hybrid"

        return {
            "results": merged,
            "counts": {layer: len(items) for layer, items in merged.items()},
            "debug": debug,
        }

    def _ensure_vector_index_ready(self) -> Dict[str, Any]:
        current_size = self.vector_index.count_active()
        if current_size > 0:
            return {"status": "ready", "active_entries": current_size}
        build_info = self.vector_index.build_from_mongo(self.kb_service, force=False)
        return {**build_info, "active_entries": self.vector_index.count_active()}

    def _filter_vector_hits_by_mongo(
        self,
        vector_hits: List[Dict[str, Any]],
        target_layers: List[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not vector_hits:
            return [], {
                "vector_hits_after_existence": 0,
                "vector_hits_dropped_missing_doc": 0,
                "vector_hits_dropped_sample": [],
            }

        self.kb_service._ensure_available()
        allowed_layers = set(target_layers)
        doc_ids_by_layer: Dict[str, set] = defaultdict(set)
        source_by_layer: Dict[str, set] = defaultdict(set)

        for hit in vector_hits:
            layer = str(hit.get("layer") or "").strip()
            if layer not in allowed_layers:
                continue
            doc_id = str(hit.get("doc_id") or "").strip()
            source_file = str(hit.get("source_file") or "").strip()
            if doc_id:
                doc_ids_by_layer[layer].add(doc_id)
            elif source_file:
                source_by_layer[layer].add(source_file)

        valid_doc_ids: Dict[str, set] = defaultdict(set)
        valid_source_files: Dict[str, set] = defaultdict(set)

        for layer, doc_ids in doc_ids_by_layer.items():
            if not doc_ids:
                continue
            collection = self.kb_service.db[self.kb_service.COLLECTION_MAP[layer]]
            docs = collection.find(
                {
                    "layer": layer,
                    "doc_id": {"$in": list(doc_ids)},
                    "$or": [{"status": "active"}, {"status": {"$exists": False}}],
                },
                {"doc_id": 1},
            )
            for doc in docs:
                doc_id = str(doc.get("doc_id") or "").strip()
                if doc_id:
                    valid_doc_ids[layer].add(doc_id)

        for layer, source_files in source_by_layer.items():
            if not source_files:
                continue
            collection = self.kb_service.db[self.kb_service.COLLECTION_MAP[layer]]
            docs = collection.find(
                {
                    "layer": layer,
                    "source_file": {"$in": list(source_files)},
                    "$or": [{"status": "active"}, {"status": {"$exists": False}}],
                },
                {"source_file": 1},
            )
            for doc in docs:
                source_file = str(doc.get("source_file") or "").strip()
                if source_file:
                    valid_source_files[layer].add(source_file)

        filtered: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        dropped_by_layer: Dict[str, int] = defaultdict(int)
        for hit in vector_hits:
            layer = str(hit.get("layer") or "").strip()
            doc_id = str(hit.get("doc_id") or "").strip()
            source_file = str(hit.get("source_file") or "").strip()

            if layer not in allowed_layers:
                dropped.append({"layer": layer, "doc_id": doc_id, "source_file": source_file})
                dropped_by_layer[layer or "unknown"] += 1
                continue

            if doc_id:
                exists = doc_id in valid_doc_ids[layer]
            else:
                exists = bool(source_file) and source_file in valid_source_files[layer]

            if exists:
                filtered.append(hit)
            else:
                dropped.append({"layer": layer, "doc_id": doc_id, "source_file": source_file})
                dropped_by_layer[layer or "unknown"] += 1

        return filtered, {
            "vector_hits_after_existence": len(filtered),
            "vector_hits_dropped_missing_doc": len(dropped),
            "vector_hits_dropped_by_layer": dict(dropped_by_layer),
            "vector_hits_dropped_sample": dropped[:5],
        }

    def _merge_layer(
        self,
        layer: str,
        lexical_items: List[Dict[str, Any]],
        vector_items: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        def _doc_key(item: Dict[str, Any]) -> str:
            return str(item.get("doc_id") or item.get("source_file") or "")

        vector_best: Dict[str, Dict[str, Any]] = {}
        vector_snippets: Dict[str, List[str]] = {}
        for hit in vector_items:
            key = _doc_key(hit)
            if not key:
                continue
            score = float(hit.get("vector_score") or 0.0)
            text = self._clean_snippet(str(hit.get("text") or ""))
            if key not in vector_best or score > float(vector_best[key].get("vector_score") or 0.0):
                vector_best[key] = hit
            if text and not self._is_noisy_snippet(text):
                vector_snippets.setdefault(key, []).append(text)

        merged: List[Dict[str, Any]] = []
        seen_keys = set()
        vector_only_min = 0.72 if not lexical_items else 0.58

        for lexical in lexical_items:
            key = _doc_key(lexical)
            if not key:
                continue
            seen_keys.add(key)

            vector_hit = vector_best.get(key)
            vector_score = float(vector_hit.get("vector_score") or 0.0) if vector_hit else 0.0
            snippets = self._dedup(vector_snippets.get(key, []))[:2]

            merged_item = dict(lexical)
            merged_item["retrieval_mode"] = "both" if vector_hit else "lexical"
            merged_item["vector_score"] = round(vector_score, 4)
            if snippets:
                merged_item["vector_snippets"] = snippets
                self._inject_vector_snippets(layer=layer, item=merged_item, snippets=snippets)

            lexical_rank = self._estimate_lexical_rank(layer=layer, item=lexical)
            hybrid_rank = (LEXICAL_WEIGHT * lexical_rank) + (VECTOR_WEIGHT * vector_score)
            merged_item["_hybrid_rank"] = hybrid_rank
            merged.append(merged_item)

        for key, vector_hit in vector_best.items():
            if key in seen_keys:
                continue

            snippets = self._dedup(vector_snippets.get(key, []))[:2]
            vector_score = float(vector_hit.get("vector_score") or 0.0)
            if vector_score < vector_only_min:
                continue
            doc_id = vector_hit.get("doc_id")
            extra_meta = self._fetch_vector_doc_meta(
                layer=layer,
                doc_id=str(doc_id or ""),
                source_file=str(vector_hit.get("source_file") or ""),
            )
            vector_item = {
                "doc_id": doc_id,
                "source_file": vector_hit.get("source_file"),
                "title": vector_hit.get("title"),
                "detail_api": f"/kb/doc/{layer}/{doc_id}" if doc_id else None,
                "retrieval_mode": "vector",
                "vector_score": round(vector_score, 4),
                "vector_snippets": snippets,
                "_hybrid_rank": VECTOR_WEIGHT * vector_score,
            }
            vector_item.update(extra_meta)
            self._inject_vector_snippets(layer=layer, item=vector_item, snippets=snippets)
            merged.append(vector_item)

        merged.sort(
            key=lambda item: (
                float(item.get("_hybrid_rank") or 0.0),
                float(item.get("vector_score") or 0.0),
            ),
            reverse=True,
        )

        output: List[Dict[str, Any]] = []
        for item in merged[: max(1, min(top_k, 20))]:
            clean = dict(item)
            clean.pop("_hybrid_rank", None)
            output.append(clean)
        return output

    def _fetch_vector_doc_meta(self, layer: str, doc_id: str, source_file: str) -> Dict[str, Any]:
        # 向量-only 命中补齐轻量元信息，避免丢失主/辅教材角色。
        try:
            self.kb_service._ensure_available()
            collection = self.kb_service.db[self.kb_service.COLLECTION_MAP[layer]]
            query: Dict[str, Any] = {
                "layer": layer,
                "$or": [{"status": "active"}, {"status": {"$exists": False}}],
            }
            if doc_id:
                query["doc_id"] = doc_id
            elif source_file:
                query["source_file"] = source_file
            else:
                return {}

            doc = collection.find_one(query, {"textbook_meta": 1, "subject": 1})
            if not doc or layer != "textbook":
                return {}
            meta = doc.get("textbook_meta") or {}
            authors = meta.get("authors")
            if not isinstance(authors, list):
                authors = []
            return {
                "textbook_role": meta.get("textbook_role"),
                "priority_score": float(meta.get("priority_score") or 0.0),
                "edition": meta.get("edition"),
                "academic_year": meta.get("academic_year"),
                "author": meta.get("author"),
                "authors": authors,
                "subject": meta.get("subject") or doc.get("subject"),
            }
        except Exception:
            return {}

    def _estimate_lexical_rank(self, layer: str, item: Dict[str, Any]) -> float:
        if layer == "syllabus":
            base = float(
                1
                + len(item.get("matched_modules") or [])
                + len(item.get("matched_key_points") or [])
                + len(item.get("matched_difficult_points") or [])
                + len(item.get("matched_schedule_topics") or [])
            )
            if item.get("is_primary"):
                base += 6.0
            base += min(3.0, float(item.get("priority_score") or 0.0) * 0.05)
            return base
        if layer == "textbook":
            base = float(
                1
                + len(item.get("matched_sections") or [])
                + len(item.get("matched_knowledge_points") or [])
                + len(item.get("matched_chunks_preview") or [])
            )
            textbook_role = str(item.get("textbook_role") or "").strip().lower()
            if textbook_role == "main":
                base += 4.0
            elif textbook_role == "supplementary":
                base -= 1.0
            base += min(3.0, float(item.get("priority_score") or 0.0) * 0.03)
            return base
        if layer == "resource":
            return float(
                1
                + len(item.get("matched_pages") or [])
                + len(item.get("matched_page_roles") or [])
                + len(item.get("matched_units") or [])
            )
        if layer == "hotspot":
            return float(
                1
                + len(item.get("matched_knowledge_points") or [])
                + len(item.get("matched_keywords") or [])
                + len(item.get("teaching_usage") or [])
            )
        return 1.0

    def _inject_vector_snippets(self, layer: str, item: Dict[str, Any], snippets: List[str]) -> None:
        if not snippets:
            return
        if layer == "textbook":
            merged = self._dedup((item.get("matched_chunks_preview") or []) + snippets)
            item["matched_chunks_preview"] = merged[:4]
            return
        if layer == "resource":
            merged = self._dedup((item.get("matched_units") or []) + snippets)
            item["matched_units"] = merged[:4]
            return
        if layer == "syllabus":
            merged = self._dedup((item.get("matched_key_points") or []) + snippets)
            item["matched_key_points"] = merged[:4]
            return
        if layer == "hotspot":
            if not item.get("summary"):
                item["summary"] = snippets[0][:220]

    @staticmethod
    def _clean_snippet(value: str) -> str:
        text = str(value or "")
        text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
        text = text.replace("\uFFFD", " ")
        text = re.sub(r"(?:\s*[.。]\s*){3,}\d*", " ", text)
        text = re.sub(r"[=+\-/*]{6,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _is_noisy_snippet(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        if len(value) < 12:
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

    @staticmethod
    def _dedup(values: List[str]) -> List[str]:
        output: List[str] = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(text)
        return output
