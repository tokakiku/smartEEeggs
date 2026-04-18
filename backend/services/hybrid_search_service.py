from __future__ import annotations

"""统一混合检索编排层。

职责：
- 并行组织向量召回与图谱召回；
- 汇总 doc_id/source_file 后回查 Mongo；
- 产出可直接供生成层使用的组装上下文与调试信息。
"""

import re
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from services.embedding_service import EmbeddingServiceError, get_embedding_service
from services.graph_extraction_service import GraphExtractionService
from services.graph_store_service import GraphStoreService
from services.milvus_service import MilvusService, MilvusServiceError, MilvusUnavailableError
from services.mongo_kb_service import MongoKBService, MongoKBUnavailableError


def _to_str(value: Any) -> str:
    return str(value or "").strip()


class HybridSearchServiceError(RuntimeError):
    pass


class HybridSearchService:
    """Mongo + 向量 + 图谱的统一检索服务。"""
    DEFAULT_LAYERS = ["syllabus", "textbook", "resource", "hotspot"]
    TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-+.]{1,}|[\u4e00-\u9fff]{2,}")

    STOPWORDS = {
        "what",
        "how",
        "about",
        "explain",
        "intro",
        "introduction",
        "overview",
        "please",
        "query",
        "search",
        "什么",
        "如何",
        "请问",
        "一下",
        "相关",
        "介绍",
        "解释",
        "内容",
        "知识",
        "课程",
    }

    LAYER_LABELS = {
        "syllabus": "Syllabus",
        "textbook": "Textbook",
        "resource": "Resource",
        "hotspot": "Hotspot",
    }

    def __init__(
        self,
        kb_service: Optional[MongoKBService] = None,
        embedding_service: Any = None,
        milvus_service: Optional[MilvusService] = None,
        graph_extraction_service: Optional[GraphExtractionService] = None,
    ) -> None:
        self.kb_service = kb_service or MongoKBService.from_env()
        self._embedding_service = embedding_service
        self.milvus_service = milvus_service or MilvusService()
        self.graph_extraction_service = graph_extraction_service or GraphExtractionService(kb_service=self.kb_service)

        self._graph_store: Optional[GraphStoreService] = None
        self._graph_signature: Optional[Tuple[Tuple[str, int], ...]] = None
        self._graph_summary: Dict[str, Any] = {}
        self._graph_lock = threading.Lock()

    def orchestrate_search(
        self,
        query: str,
        top_k: int = 5,
        layers: Optional[List[str]] = None,
        graph_hops: int = 1,
    ) -> Dict[str, Any]:
        """执行统一检索编排，返回标准化混合检索结果。"""
        q = _to_str(query)
        target_layers = self._normalize_layers(layers)
        limit = max(1, min(int(top_k), 20))
        hops = max(1, min(int(graph_hops), 2))

        parsed = self._parse_query(q)

        vector_payload = self._search_vector(
            query=parsed["normalized_query"] or q,
            top_k=limit,
            layers=target_layers,
        )

        graph_payload = self._search_graph(
            query_entities=parsed["query_entities"],
            query_keywords=parsed["query_keywords"],
            layers=target_layers,
            top_k=limit,
            graph_hops=hops,
        )

        merged_doc_refs, merged_source_refs, merged_doc_ids = self._merge_trace_refs(
            vector_hits=vector_payload["hits"],
            graph_doc_refs=graph_payload["doc_refs"],
            graph_source_refs=graph_payload["source_refs"],
        )

        mongo_docs = self._fetch_mongo_documents(
            doc_refs=merged_doc_refs,
            source_refs=merged_source_refs,
            layers=target_layers,
        )

        assembled_context = self._assemble_context(
            vector_hits=vector_payload["hits"],
            graph_payload=graph_payload,
            mongo_docs=mongo_docs,
            top_k=limit,
        )

        fallback_reasons: List[str] = []
        if not vector_payload["available"]:
            fallback_reasons.append(f"vector_fallback={_to_str(vector_payload.get('reason')) or 'unknown'}")
        if not graph_payload["available"]:
            fallback_reasons.append(f"graph_fallback={_to_str(graph_payload.get('reason')) or 'unknown'}")

        debug = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "vector_available": bool(vector_payload["available"]),
            "graph_available": bool(graph_payload["available"]),
            "vector_hit_count": len(vector_payload["hits"]),
            "graph_hit_count": len(graph_payload["edges"]),
            "graph_seed_node_count": len(graph_payload["matched_nodes"]),
            "graph_neighbor_node_count": len(graph_payload["neighbor_nodes"]),
            "merged_doc_count": len(merged_doc_ids),
            "mongo_doc_count": len(mongo_docs),
            "query_entities": parsed["query_entities"],
            "query_keywords": parsed["query_keywords"],
            "cross_layer_edges_used": int(graph_payload["cross_layer_edges_used"]),
            "vector_backend": vector_payload.get("backend"),
            "vector_backend_reason": vector_payload.get("reason"),
            "graph_backend_reason": graph_payload.get("reason"),
            "graph_build": dict(self._graph_summary or {}),
            "fallback_reasons": fallback_reasons,
        }

        return {
            "query": q,
            "normalized_query": parsed["normalized_query"],
            "query_entities": parsed["query_entities"],
            "query_keywords": parsed["query_keywords"],
            "vector_hits": vector_payload["hits"],
            "graph_hits": {
                "matched_nodes": graph_payload["matched_nodes"],
                "neighbor_nodes": graph_payload["neighbor_nodes"],
                "edges": graph_payload["edges"],
            },
            "merged_doc_ids": merged_doc_ids,
            "mongo_docs": mongo_docs,
            "assembled_context": assembled_context,
            "debug": debug,
        }

    def _normalize_layers(self, layers: Optional[Sequence[str]]) -> List[str]:
        """标准化层级参数，兜底为默认四层。"""
        if not layers:
            return list(self.DEFAULT_LAYERS)
        normalized: List[str] = []
        seen = set()
        for value in layers:
            layer = _to_str(value).lower()
            if layer not in self.DEFAULT_LAYERS:
                continue
            if layer in seen:
                continue
            seen.add(layer)
            normalized.append(layer)
        return normalized or list(self.DEFAULT_LAYERS)

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """做轻量 query 解析，提取实体与关键词。"""
        normalized = re.sub(r"\s+", " ", _to_str(query))
        candidates: List[str] = []

        for token in self.TOKEN_PATTERN.findall(normalized):
            value = _to_str(token)
            if not value:
                continue
            candidates.append(value)
            if value.lower() != value:
                candidates.append(value.lower())
            candidates.extend(self._expand_query_variants(value))

        compact_query = re.sub(r"\s+", "", normalized)
        if compact_query:
            candidates.append(compact_query)
            candidates.extend(self._expand_query_variants(compact_query))

        filtered: List[str] = []
        for token in candidates:
            value = _to_str(token)
            if len(value) <= 1:
                continue
            lower = value.lower()
            if lower in self.STOPWORDS:
                continue
            if value.isdigit():
                continue
            filtered.append(value)

        deduped = self._dedupe_preserve_order(filtered)
        entities = deduped[:8]
        keywords = deduped[:16]
        if not entities and normalized:
            entities = [normalized]
            keywords = [normalized]

        return {
            "query": query,
            "normalized_query": normalized,
            "query_entities": entities,
            "query_keywords": keywords,
        }

    def _expand_query_variants(self, token: str) -> List[str]:
        """扩展常见词形变体，提升召回覆盖率。"""
        value = _to_str(token)
        if not value:
            return []

        variants: List[str] = []
        for suffix in ["方法", "应用", "技术", "模型", "原理", "案例", "法"]:
            if len(value) > len(suffix) + 1 and value.endswith(suffix):
                variants.append(value[: -len(suffix)])
        if value.lower().endswith("method") and len(value) > len("method") + 2:
            variants.append(value[: -len("method")].strip())
        if value.lower().endswith("model") and len(value) > len("model") + 2:
            variants.append(value[: -len("model")].strip())
        return [item for item in variants if _to_str(item)]

    def _search_vector(self, query: str, top_k: int, layers: List[str]) -> Dict[str, Any]:
        """向量召回路径：query -> embedding -> Milvus search。"""
        result = {
            "available": False,
            "backend": "milvus",
            "reason": None,
            "hits": [],
        }
        q = _to_str(query)
        if not q:
            result["reason"] = "empty_query"
            return result

        try:
            if not self.milvus_service.has_collection():
                result["reason"] = "milvus_collection_not_found"
                return result

            embedding_service = self._get_embedding_service()
            query_vector = embedding_service.embed_query(q)
            if len(getattr(query_vector, "shape", [])) < 2 or int(query_vector.shape[0]) <= 0:
                result["reason"] = "query_embedding_empty"
                return result

            raw_hits: List[Dict[str, Any]] = []
            per_layer_limit = max(top_k * 3, 20)
            for layer in layers:
                layer_hits = self.milvus_service.search(
                    query_vector=query_vector[0],
                    top_k=per_layer_limit,
                    layer=layer,
                )
                raw_hits.extend(layer_hits)

            normalized_hits = self._normalize_vector_hits(raw_hits=raw_hits, layers=layers)
            result["hits"] = normalized_hits[:top_k]
            result["available"] = True
            return result
        except (MilvusUnavailableError, MilvusServiceError, EmbeddingServiceError, RuntimeError) as exc:
            result["reason"] = str(exc)
            return result
        except Exception as exc:  # pragma: no cover - defensive branch
            result["reason"] = str(exc)
            return result

    def _normalize_vector_hits(self, raw_hits: Iterable[Dict[str, Any]], layers: List[str]) -> List[Dict[str, Any]]:
        """归一化向量命中并做去重排序。"""
        allowed = set(layers)
        best_by_key: Dict[str, Dict[str, Any]] = {}

        for item in raw_hits:
            layer = _to_str(item.get("layer")).lower()
            if layer and allowed and layer not in allowed:
                continue
            doc_id = _to_str(item.get("doc_id"))
            source_file = _to_str(item.get("source_file"))
            chunk_id = _to_str(item.get("chunk_id"))
            title = _to_str(item.get("title"))
            subject = _to_str(item.get("subject"))
            chunk_text = _to_str(item.get("chunk_text") or item.get("text"))
            if not chunk_text:
                continue
            score = float(item.get("score") or item.get("distance") or item.get("vector_score") or 0.0)

            key = f"{layer}|{doc_id}|{chunk_id}|{source_file}"
            payload = {
                "score": score,
                "doc_id": doc_id,
                "layer": layer,
                "chunk_id": chunk_id,
                "source_file": source_file,
                "title": title,
                "subject": subject,
                "chunk_text": chunk_text,
                "metadata": item.get("metadata") or {},
            }
            if key not in best_by_key or score > float(best_by_key[key].get("score") or 0.0):
                best_by_key[key] = payload

        hits = list(best_by_key.values())
        hits.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return hits

    def _search_graph(
        self,
        query_entities: List[str],
        query_keywords: List[str],
        layers: List[str],
        top_k: int,
        graph_hops: int,
    ) -> Dict[str, Any]:
        """图谱召回路径：实体匹配 + 邻居扩展 + 边集合整理。"""
        output = {
            "available": False,
            "reason": None,
            "matched_nodes": [],
            "neighbor_nodes": [],
            "edges": [],
            "doc_refs": [],
            "source_refs": [],
            "cross_layer_edges_used": 0,
        }
        try:
            graph_store = self._ensure_graph_store()
        except (MongoKBUnavailableError, HybridSearchServiceError, RuntimeError) as exc:
            output["reason"] = str(exc)
            return output
        except Exception as exc:  # pragma: no cover - defensive branch
            output["reason"] = str(exc)
            return output

        terms = self._dedupe_preserve_order(list(query_entities) + list(query_keywords))
        if not terms:
            output["available"] = True
            output["reason"] = "empty_graph_terms"
            return output

        allowed_layers = set(layers)
        seed_nodes: Dict[str, Dict[str, Any]] = {}
        for term in terms[:12]:
            matches = graph_store.query_entity(entity=term, max_neighbors=max(40, top_k * 8))
            for node in matches.get("matches") or []:
                node_layer = _to_str(node.get("layer")).lower()
                if allowed_layers and node_layer not in allowed_layers:
                    continue
                node_id = _to_str(node.get("node_id"))
                if not node_id:
                    continue
                payload = self._normalize_graph_node(node)
                payload.setdefault("matched_by", [])
                if term not in payload["matched_by"]:
                    payload["matched_by"].append(term)
                if node_id not in seed_nodes:
                    seed_nodes[node_id] = payload
                else:
                    existing = seed_nodes[node_id]
                    existing["matched_by"] = self._dedupe_preserve_order(
                        list(existing.get("matched_by") or []) + list(payload.get("matched_by") or [])
                    )

        if not seed_nodes:
            output["available"] = True
            output["reason"] = "graph_no_seed_match"
            return output

        graph = graph_store.graph
        visited_depth: Dict[str, int] = {node_id: 0 for node_id in seed_nodes.keys()}
        frontier = set(seed_nodes.keys())
        neighbor_nodes: Dict[str, Dict[str, Any]] = {}
        edge_map: Dict[str, Dict[str, Any]] = {}

        for hop in range(1, max(1, graph_hops) + 1):
            if not frontier:
                break
            next_frontier: Set[str] = set()
            for node_id in frontier:
                for source, target, edge_key, attrs in graph.out_edges(node_id, keys=True, data=True):
                    self._collect_graph_edge(
                        edge_map=edge_map,
                        graph=graph,
                        source=source,
                        target=target,
                        edge_key=edge_key,
                        attrs=attrs,
                        hop=hop,
                        allowed_layers=allowed_layers,
                    )
                    if target not in visited_depth:
                        visited_depth[target] = hop
                        next_frontier.add(target)
                    if target not in seed_nodes:
                        neighbor_nodes[target] = self._serialize_graph_node(graph=graph, node_id=target)

                for source, target, edge_key, attrs in graph.in_edges(node_id, keys=True, data=True):
                    self._collect_graph_edge(
                        edge_map=edge_map,
                        graph=graph,
                        source=source,
                        target=target,
                        edge_key=edge_key,
                        attrs=attrs,
                        hop=hop,
                        allowed_layers=allowed_layers,
                    )
                    if source not in visited_depth:
                        visited_depth[source] = hop
                        next_frontier.add(source)
                    if source not in seed_nodes:
                        neighbor_nodes[source] = self._serialize_graph_node(graph=graph, node_id=source)
            frontier = next_frontier

        edges = list(edge_map.values())
        edges.sort(
            key=lambda row: (
                bool(row.get("cross_layer")),
                float(row.get("confidence") or 0.0),
                -int(row.get("hop") or 0),
            ),
            reverse=True,
        )
        edge_limit = max(top_k * 8, 30)
        edges = edges[:edge_limit]

        matched_nodes = list(seed_nodes.values())
        matched_nodes.sort(key=lambda row: len(row.get("matched_by") or []), reverse=True)
        neighbor_list = list(neighbor_nodes.values())
        neighbor_list.sort(key=lambda row: len(row.get("doc_ids") or []), reverse=True)

        doc_refs: Set[Tuple[str, str]] = set()
        source_refs: Set[Tuple[str, str]] = set()
        for node in list(seed_nodes.values()) + neighbor_list:
            layer = _to_str(node.get("layer")).lower()
            for doc_id in node.get("doc_ids") or []:
                if _to_str(doc_id) and _to_str(layer):
                    doc_refs.add((_to_str(doc_id), _to_str(layer)))
            for source_file in node.get("source_files") or []:
                if _to_str(source_file) and _to_str(layer):
                    source_refs.add((_to_str(source_file), _to_str(layer)))

        for edge in edges:
            edge_layer = _to_str(edge.get("layer")).lower()
            source_layer = _to_str((edge.get("source_node") or {}).get("layer")).lower()
            target_layer = _to_str((edge.get("target_node") or {}).get("layer")).lower()
            metadata = edge.get("metadata") or {}

            if edge.get("cross_layer") and isinstance(metadata, dict):
                for doc_id in metadata.get("source_doc_ids") or []:
                    if _to_str(doc_id):
                        doc_refs.add((_to_str(doc_id), source_layer or edge_layer))
                for doc_id in metadata.get("target_doc_ids") or []:
                    if _to_str(doc_id):
                        doc_refs.add((_to_str(doc_id), target_layer or edge_layer))

            for doc_id in edge.get("doc_ids") or []:
                if _to_str(doc_id):
                    doc_refs.add((_to_str(doc_id), edge_layer or source_layer or target_layer))
            if _to_str(edge.get("doc_id")):
                doc_refs.add((_to_str(edge.get("doc_id")), edge_layer or source_layer or target_layer))

            if _to_str(edge.get("source_file")):
                source_refs.add((_to_str(edge.get("source_file")), edge_layer or source_layer or target_layer))

        output["available"] = True
        output["matched_nodes"] = matched_nodes
        output["neighbor_nodes"] = neighbor_list[: max(top_k * 6, 24)]
        output["edges"] = edges
        output["doc_refs"] = [{"doc_id": doc_id, "layer": layer} for doc_id, layer in sorted(doc_refs)]
        output["source_refs"] = [{"source_file": source_file, "layer": layer} for source_file, layer in sorted(source_refs)]
        output["cross_layer_edges_used"] = sum(1 for edge in edges if edge.get("cross_layer"))
        return output

    def _ensure_graph_store(self) -> GraphStoreService:
        """确保图存储已构建；若 Mongo 文档计数变化则自动重建。"""
        if not self.kb_service.is_available:
            reason = self.kb_service.unavailable_reason or "mongodb_not_available"
            raise MongoKBUnavailableError(reason)

        signature = self._get_graph_signature()
        if self._graph_store is not None and signature == self._graph_signature:
            return self._graph_store

        with self._graph_lock:
            signature = self._get_graph_signature()
            if self._graph_store is not None and signature == self._graph_signature:
                return self._graph_store

            extraction = self.graph_extraction_service.extract_from_mongo(layers=self.DEFAULT_LAYERS)
            triples = extraction.get("triples") or []
            graph_store = GraphStoreService()
            graph_store.add_triples(triples)
            newly_built_cross_edges = graph_store.build_cross_layer_edges(min_score=0.9, max_matches_per_source=3)

            self._graph_store = graph_store
            self._graph_signature = signature
            self._graph_summary = {
                "docs_by_layer": extraction.get("docs_by_layer") or {},
                "triples_total": len(triples),
                "graph_stats": graph_store.get_stats(),
                "cross_layer_edges_newly_built": int(newly_built_cross_edges),
            }
            return graph_store

    def _get_graph_signature(self) -> Tuple[Tuple[str, int], ...]:
        """获取图构建签名（按层文档计数）。"""
        counts = self.kb_service.get_active_collection_counts()
        return tuple(sorted((layer, int(counts.get(layer, 0))) for layer in self.DEFAULT_LAYERS))

    def _collect_graph_edge(
        self,
        edge_map: Dict[str, Dict[str, Any]],
        graph: Any,
        source: str,
        target: str,
        edge_key: str,
        attrs: Dict[str, Any],
        hop: int,
        allowed_layers: Set[str],
    ) -> None:
        """收集并规范化图边，同时按置信度保留最佳版本。"""
        source_node = self._serialize_graph_node(graph=graph, node_id=source)
        target_node = self._serialize_graph_node(graph=graph, node_id=target)

        source_layer = _to_str(source_node.get("layer")).lower()
        target_layer = _to_str(target_node.get("layer")).lower()
        edge_layer = _to_str(attrs.get("layer")).lower()
        if allowed_layers and source_layer not in allowed_layers and target_layer not in allowed_layers:
            return

        doc_ids = self._normalize_str_list(attrs.get("doc_ids") or [])
        if not doc_ids and _to_str(attrs.get("doc_id")):
            doc_ids = [_to_str(attrs.get("doc_id"))]

        confidence = float(attrs.get("confidence") or 0.0)
        key = f"{source}|{target}|{edge_key}"
        edge_payload = {
            "edge_key": _to_str(edge_key),
            "hop": int(hop),
            "source_node": source_node,
            "target_node": target_node,
            "relation": _to_str(attrs.get("relation")),
            "layer": edge_layer,
            "doc_id": _to_str(attrs.get("doc_id")),
            "doc_ids": doc_ids,
            "source_file": _to_str(attrs.get("source_file")),
            "source_field": _to_str(attrs.get("source_field")),
            "chunk_id": _to_str(attrs.get("chunk_id")),
            "evidence_text": _to_str(attrs.get("evidence_text")),
            "confidence": confidence,
            "cross_layer": bool(attrs.get("cross_layer")),
            "metadata": attrs.get("metadata") or {},
        }
        if key not in edge_map:
            edge_map[key] = edge_payload
        else:
            current = edge_map[key]
            if confidence > float(current.get("confidence") or 0.0):
                edge_map[key] = edge_payload

    def _serialize_graph_node(self, graph: Any, node_id: str) -> Dict[str, Any]:
        """将图节点序列化为对外返回结构。"""
        attrs = graph.nodes[node_id]
        return {
            "node_id": _to_str(node_id),
            "label": _to_str(attrs.get("label")),
            "layer": _to_str(attrs.get("layer")),
            "normalized": _to_str(attrs.get("normalized")),
            "entity_types": self._normalize_str_list(attrs.get("entity_types") or []),
            "doc_ids": self._normalize_str_list(attrs.get("doc_ids") or []),
            "source_files": self._normalize_str_list(attrs.get("source_files") or []),
            "aliases": self._normalize_str_list(attrs.get("aliases") or []),
        }

    def _normalize_graph_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """归一化图查询返回的节点结构。"""
        return {
            "node_id": _to_str(node.get("node_id")),
            "label": _to_str(node.get("label")),
            "layer": _to_str(node.get("layer")),
            "normalized": _to_str(node.get("normalized")),
            "entity_types": self._normalize_str_list(node.get("entity_types") or []),
            "doc_ids": self._normalize_str_list(node.get("doc_ids") or []),
            "source_files": self._normalize_str_list(node.get("source_files") or []),
            "aliases": self._normalize_str_list(node.get("aliases") or []),
        }

    def _merge_trace_refs(
        self,
        vector_hits: List[Dict[str, Any]],
        graph_doc_refs: List[Dict[str, Any]],
        graph_source_refs: List[Dict[str, Any]],
    ) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], List[str]]:
        """合并向量/图谱来源的追溯键，生成去重 doc_id 列表。"""
        doc_refs: Set[Tuple[str, str]] = set()
        source_refs: Set[Tuple[str, str]] = set()

        for hit in vector_hits:
            doc_id = _to_str(hit.get("doc_id"))
            layer = _to_str(hit.get("layer")).lower()
            source_file = _to_str(hit.get("source_file"))
            if doc_id and layer:
                doc_refs.add((doc_id, layer))
            if source_file and layer:
                source_refs.add((source_file, layer))

        for item in graph_doc_refs:
            doc_id = _to_str(item.get("doc_id"))
            layer = _to_str(item.get("layer")).lower()
            if doc_id and layer:
                doc_refs.add((doc_id, layer))

        for item in graph_source_refs:
            source_file = _to_str(item.get("source_file"))
            layer = _to_str(item.get("layer")).lower()
            if source_file and layer:
                source_refs.add((source_file, layer))

        merged_doc_ids = sorted({doc_id for doc_id, _ in doc_refs if doc_id})
        return doc_refs, source_refs, merged_doc_ids

    def _fetch_mongo_documents(
        self,
        doc_refs: Set[Tuple[str, str]],
        source_refs: Set[Tuple[str, str]],
        layers: List[str],
    ) -> List[Dict[str, Any]]:
        """按 doc_id/source_file 回查 Mongo，并投影为轻量文档摘要。"""
        if not self.kb_service.is_available:
            return []

        output: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()
        ordered_layers = self._normalize_layers(layers)

        for doc_id, layer_hint in sorted(doc_refs):
            layer_candidates = self._candidate_layers(layer_hint=layer_hint, ordered_layers=ordered_layers)
            mongo_doc = None
            for layer in layer_candidates:
                mongo_doc = self.kb_service.get_document(layer=layer, doc_id=doc_id)
                if mongo_doc is not None:
                    break
            if mongo_doc is None:
                continue
            key = f"{_to_str(mongo_doc.get('layer'))}:{_to_str(mongo_doc.get('doc_id'))}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            output.append(self._project_mongo_doc(mongo_doc))

        for source_file, layer_hint in sorted(source_refs):
            layer_candidates = self._candidate_layers(layer_hint=layer_hint, ordered_layers=ordered_layers)
            mongo_doc = None
            for layer in layer_candidates:
                mongo_doc = self.kb_service.get_document_by_source_file(layer=layer, source_file=source_file)
                if mongo_doc is not None:
                    break
            if mongo_doc is None:
                continue
            key = f"{_to_str(mongo_doc.get('layer'))}:{_to_str(mongo_doc.get('doc_id'))}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            output.append(self._project_mongo_doc(mongo_doc))

        output.sort(key=lambda row: (_to_str(row.get("layer")), _to_str(row.get("doc_id"))))
        return output

    def _candidate_layers(self, layer_hint: str, ordered_layers: List[str]) -> List[str]:
        """给定层级提示，返回优先层级候选序列。"""
        layer = _to_str(layer_hint).lower()
        if layer in ordered_layers:
            return [layer] + [item for item in ordered_layers if item != layer]
        return list(ordered_layers)

    def _project_mongo_doc(self, mongo_doc: Dict[str, Any]) -> Dict[str, Any]:
        """将 Mongo 原文档投影为检索链路需要的关键字段。"""
        layer = _to_str(mongo_doc.get("layer")).lower()
        data = mongo_doc.get("data") or {}
        return {
            "doc_id": _to_str(mongo_doc.get("doc_id")),
            "layer": layer,
            "source_file": _to_str(mongo_doc.get("source_file")),
            "title": _to_str(mongo_doc.get("title")),
            "subject": _to_str(mongo_doc.get("subject")),
            "structured_summary": self._summarize_data(layer=layer, data=data),
            "trace": {
                "collection": _to_str(mongo_doc.get("collection")),
                "updated_at": _to_str(mongo_doc.get("updated_at")),
            },
        }

    def _summarize_data(self, layer: str, data: Dict[str, Any]) -> str:
        """按层抽取结构化摘要，避免直接暴露大段原始字段。"""
        if layer == "syllabus":
            modules = [
                _to_str(item.get("module_name") or item.get("module_title"))
                for item in (data.get("course_modules") or [])
                if isinstance(item, dict)
            ]
            points = self._normalize_str_list(data.get("knowledge_points") or [])
            key_points = self._normalize_str_list(data.get("teaching_key_points") or [])
            return self._join_summary_parts(
                [
                    self._label_list("modules", modules, 3),
                    self._label_list("knowledge_points", points, 4),
                    self._label_list("key_points", key_points, 4),
                ]
            )
        if layer == "textbook":
            chapters = [
                _to_str(item.get("chapter_title") or item.get("chapter_index"))
                for item in (data.get("chapters") or [])
                if isinstance(item, dict)
            ]
            sections = [
                _to_str(item.get("section_title") or item.get("section_index"))
                for item in (data.get("sections") or [])
                if isinstance(item, dict)
            ]
            points = [
                _to_str(item.get("name"))
                for item in (data.get("knowledge_points") or [])
                if isinstance(item, dict)
            ]
            return self._join_summary_parts(
                [
                    self._label_list("chapters", chapters, 3),
                    self._label_list("sections", sections, 3),
                    self._label_list("knowledge_points", points, 4),
                ]
            )
        if layer == "resource":
            pages = []
            for page in data.get("pages") or []:
                if not isinstance(page, dict):
                    continue
                title = _to_str(page.get("page_title"))
                summary = _to_str(page.get("page_summary"))
                if title and summary:
                    pages.append(f"{title}: {summary}")
                elif title:
                    pages.append(title)
            return self._join_summary_parts([self._label_list("pages", pages, 3)])
        if layer == "hotspot":
            items = []
            for row in data.get("hotspot_item") or []:
                if not isinstance(row, dict):
                    continue
                title = _to_str(row.get("title"))
                summary = _to_str(row.get("summary"))
                if title and summary:
                    items.append(f"{title}: {summary}")
                elif title:
                    items.append(title)
            publish_date = _to_str((data.get("hotspot_info") or {}).get("publish_date"))
            return self._join_summary_parts(
                [
                    f"publish_date={publish_date}" if publish_date else "",
                    self._label_list("items", items, 3),
                ]
            )
        return ""

    def _assemble_context(
        self,
        vector_hits: List[Dict[str, Any]],
        graph_payload: Dict[str, Any],
        mongo_docs: List[Dict[str, Any]],
        top_k: int,
    ) -> Dict[str, Any]:
        """组装可读上下文：向量片段 + 图谱链路 + Mongo 溯源摘要。"""
        vector_lines: List[str] = []
        seen_vector_text = set()
        for hit in vector_hits[: max(top_k, 6)]:
            snippet = self._preview_text(hit.get("chunk_text"), max_len=180)
            if not snippet:
                continue
            norm = self._normalize_for_dedup(snippet)
            if norm in seen_vector_text:
                continue
            seen_vector_text.add(norm)
            layer_label = self.LAYER_LABELS.get(_to_str(hit.get("layer")), _to_str(hit.get("layer")) or "Unknown")
            vector_lines.append(
                f"- [{layer_label}] {_to_str(hit.get('title')) or _to_str(hit.get('doc_id'))}: {snippet}"
            )

        graph_lines: List[str] = []
        seen_graph_line = set()
        for edge in graph_payload.get("edges") or []:
            source = edge.get("source_node") or {}
            target = edge.get("target_node") or {}
            source_layer = self.LAYER_LABELS.get(_to_str(source.get("layer")), _to_str(source.get("layer")) or "Unknown")
            target_layer = self.LAYER_LABELS.get(_to_str(target.get("layer")), _to_str(target.get("layer")) or "Unknown")
            relation = _to_str(edge.get("relation")) or "related_to"
            source_label = _to_str(source.get("label")) or _to_str(source.get("node_id"))
            target_label = _to_str(target.get("label")) or _to_str(target.get("node_id"))
            marker = " [cross-layer]" if edge.get("cross_layer") else ""
            evidence = self._preview_text(edge.get("evidence_text"), max_len=80)
            line = f"- [{source_layer}: {source_label}] -> {relation} -> [{target_layer}: {target_label}]{marker}"
            if evidence:
                line += f" | evidence: {evidence}"
            key = self._normalize_for_dedup(line)
            if key in seen_graph_line:
                continue
            seen_graph_line.add(key)
            graph_lines.append(line)
            if len(graph_lines) >= max(top_k * 2, 12):
                break

        mongo_lines: List[str] = []
        seen_mongo = set()
        for doc in mongo_docs[: max(top_k * 2, 12)]:
            layer_label = self.LAYER_LABELS.get(_to_str(doc.get("layer")), _to_str(doc.get("layer")) or "Unknown")
            summary = self._preview_text(doc.get("structured_summary"), max_len=180)
            line = f"- [{layer_label}] {_to_str(doc.get('title')) or _to_str(doc.get('doc_id'))} (doc_id={_to_str(doc.get('doc_id'))})"
            if summary:
                line += f": {summary}"
            key = self._normalize_for_dedup(line)
            if key in seen_mongo:
                continue
            seen_mongo.add(key)
            mongo_lines.append(line)

        text_blocks: List[str] = ["[Vector Semantic Recall]"]
        if vector_lines:
            text_blocks.extend(vector_lines)
        else:
            text_blocks.append("- (no vector hits)")

        text_blocks.append("")
        text_blocks.append("[Graph Logic And Cross-layer Links]")
        if graph_lines:
            text_blocks.extend(graph_lines)
        else:
            text_blocks.append("- (no graph edges)")

        text_blocks.append("")
        text_blocks.append("[Mongo Trace-back Documents]")
        if mongo_lines:
            text_blocks.extend(mongo_lines)
        else:
            text_blocks.append("- (no mongo docs)")

        return {
            "vector_lines": vector_lines,
            "graph_lines": graph_lines,
            "mongo_lines": mongo_lines,
            "text": "\n".join(text_blocks).strip(),
        }

    def _get_embedding_service(self):
        """懒加载 embedding 服务。"""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def _label_list(self, label: str, values: Sequence[str], limit: int) -> str:
        """将列表字段压缩为 label=... 形式摘要。"""
        cleaned = [self._preview_text(item, max_len=80) for item in values if _to_str(item)]
        cleaned = [item for item in cleaned if item]
        if not cleaned:
            return ""
        return f"{label}={'; '.join(cleaned[: max(1, int(limit))])}"

    def _join_summary_parts(self, parts: Sequence[str]) -> str:
        """拼接摘要片段并过滤空值。"""
        clean = [_to_str(item) for item in parts if _to_str(item)]
        return " | ".join(clean)

    def _normalize_str_list(self, values: Sequence[Any]) -> List[str]:
        """字符串列表去重并保序。"""
        result: List[str] = []
        seen = set()
        for item in values:
            text = _to_str(item)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _preview_text(self, value: Any, max_len: int = 200) -> str:
        """生成短文本预览，避免上下文过长。"""
        text = _to_str(value)
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_len:
            return text
        return text[: max(8, int(max_len))].rstrip(" ,.;") + "..."

    def _normalize_for_dedup(self, value: Any) -> str:
        """文本归一化，用于近似去重。"""
        text = _to_str(value).lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text)
        return text[:320]

    def _dedupe_preserve_order(self, values: Sequence[Any]) -> List[str]:
        """保序去重。"""
        output: List[str] = []
        seen = set()
        for value in values:
            text = _to_str(value)
            if not text:
                continue
            marker = text.lower()
            if marker in seen:
                continue
            seen.add(marker)
            output.append(text)
        return output


_SERVICE_LOCK = threading.Lock()
_SERVICE_INSTANCE: Optional[HybridSearchService] = None


def get_hybrid_search_service() -> HybridSearchService:
    """获取混合检索单例。"""
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is None:
        with _SERVICE_LOCK:
            if _SERVICE_INSTANCE is None:
                _SERVICE_INSTANCE = HybridSearchService()
    return _SERVICE_INSTANCE


def orchestrate_search(query: str, top_k: int = 5, layers: Optional[List[str]] = None) -> Dict[str, Any]:
    """便捷函数：直接调用混合检索单例。"""
    return get_hybrid_search_service().orchestrate_search(query=query, top_k=top_k, layers=layers)
