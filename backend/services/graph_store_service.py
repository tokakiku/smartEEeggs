import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

"""图存储服务（NetworkX MultiDiGraph）。

负责三元组入图、跨层边构建、实体邻居查询与图产物导出。
"""

from schema.graph_schema import KnowledgeTriple
from utils.text_cleaner import clean_text


class GraphStoreService:
    """基于 NetworkX 的轻量图存储实现。"""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._node_index: Dict[Tuple[str, str], str] = {}
        self.cross_layer_triples: List[KnowledgeTriple] = []
        self.cross_layer_edge_count = 0

    def add_triples(self, triples: Iterable[KnowledgeTriple]) -> None:
        """批量入图。"""
        for triple in triples:
            self.add_triple(triple)

    def add_triple(self, triple: KnowledgeTriple) -> None:
        """添加单条三元组并维护节点聚合属性。"""
        head_id = self._get_or_create_node(
            layer=triple.layer,
            label=triple.head,
            entity_type=triple.head_type,
            doc_id=triple.doc_id,
            source_file=triple.source_file,
        )
        tail_id = self._get_or_create_node(
            layer=triple.layer,
            label=triple.tail,
            entity_type=triple.tail_type,
            doc_id=triple.doc_id,
            source_file=triple.source_file,
        )

        edge_key = self._allocate_edge_key(
            source=head_id,
            target=tail_id,
            base_key=f"{triple.relation}|{triple.doc_id}|{triple.source_field or ''}|{triple.chunk_id or ''}",
        )
        self.graph.add_edge(
            head_id,
            tail_id,
            key=edge_key,
            relation=triple.relation,
            layer=triple.layer,
            doc_id=triple.doc_id,
            doc_ids=[triple.doc_id],
            source_file=triple.source_file,
            source_field=triple.source_field,
            chunk_id=triple.chunk_id,
            confidence=triple.confidence,
            evidence_text=triple.evidence_text,
            head_type=triple.head_type,
            tail_type=triple.tail_type,
            cross_layer=False,
            metadata=triple.metadata or {},
        )

    def build_cross_layer_edges(self, min_score: float = 0.9, max_matches_per_source: int = 3) -> int:
        """构建跨层关联边，补齐 GraphRAG 关键连接。"""
        cross_pairs = [
            (
                "textbook",
                "syllabus",
                {"knowledge_point", "concept", "method", "chapter", "section"},
                {"knowledge_point", "module", "course"},
                "supports",
            ),
            (
                "resource",
                "syllabus",
                {"knowledge_point", "concept", "resource_topic", "resource_page", "resource_unit"},
                {"knowledge_point", "module", "course"},
                "explains",
            ),
            (
                "hotspot",
                "syllabus",
                {"knowledge_point", "technology_topic", "application_scenario", "event"},
                {"knowledge_point", "module", "course"},
                "applies",
            ),
            (
                "resource",
                "textbook",
                {"knowledge_point", "concept", "resource_topic", "resource_page", "resource_unit"},
                {"knowledge_point", "concept", "method", "chapter", "section"},
                "explains",
            ),
            (
                "hotspot",
                "textbook",
                {"knowledge_point", "technology_topic", "application_scenario", "event"},
                {"knowledge_point", "concept", "method", "chapter", "section"},
                "applies",
            ),
        ]

        added = 0
        for source_layer, target_layer, source_types, target_types, default_relation in cross_pairs:
            added += self._connect_layer_pair(
                source_layer=source_layer,
                target_layer=target_layer,
                source_types=source_types,
                target_types=target_types,
                default_relation=default_relation,
                min_score=min_score,
                max_matches_per_source=max_matches_per_source,
            )
        self.cross_layer_edge_count += added
        return added

    def query_entity(self, entity: str, max_neighbors: int = 20) -> Dict[str, Any]:
        """按实体查询匹配节点与邻居关系。"""
        normalized = self._normalize_text(entity)
        if not normalized:
            return {"query": entity, "matches": [], "neighbors": []}

        matched_nodes = self._find_nodes_by_entity(entity=entity, normalized=normalized)
        node_payload = [self._serialize_node(node_id) for node_id in matched_nodes]

        neighbors: List[Dict[str, Any]] = []
        for node_id in matched_nodes:
            neighbors.extend(self._collect_neighbors(node_id=node_id, max_neighbors=max_neighbors))

        neighbors.sort(key=lambda item: (float(item.get("confidence") or 0.0), item.get("relation") or ""), reverse=True)
        if len(neighbors) > max_neighbors:
            neighbors = neighbors[:max_neighbors]

        return {"query": entity, "matches": node_payload, "neighbors": neighbors}

    def export_graph_json(self, output_path: Path) -> Path:
        """导出图 JSON 产物。"""
        payload = {
            "nodes": [self._serialize_node(node_id) for node_id in self.graph.nodes],
            "edges": [self._serialize_edge(u, v, key, data) for u, v, key, data in self.graph.edges(keys=True, data=True)],
            "stats": self.get_stats(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def export_graphml(self, output_path: Path) -> Path:
        """导出 GraphML 产物。"""
        graphml_graph = nx.MultiDiGraph()
        for node_id, attrs in self.graph.nodes(data=True):
            graphml_graph.add_node(node_id, **self._to_graphml_attrs(attrs))
        for source, target, key, attrs in self.graph.edges(keys=True, data=True):
            graphml_graph.add_edge(source, target, key=key, **self._to_graphml_attrs(attrs))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graphml_graph, output_path)
        return output_path

    def get_stats(self) -> Dict[str, Any]:
        """返回图统计信息。"""
        return {
            "nodes": int(self.graph.number_of_nodes()),
            "edges": int(self.graph.number_of_edges()),
            "cross_layer_edges": int(self.cross_layer_edge_count),
            "cross_layer_triples": int(len(self.cross_layer_triples)),
        }

    def _connect_layer_pair(
        self,
        source_layer: str,
        target_layer: str,
        source_types: Set[str],
        target_types: Set[str],
        default_relation: str,
        min_score: float,
        max_matches_per_source: int,
    ) -> int:
        source_nodes = self._get_layer_nodes(layer=source_layer, allowed_types=source_types)
        target_nodes = self._get_layer_nodes(layer=target_layer, allowed_types=target_types)
        if not source_nodes or not target_nodes:
            return 0

        target_by_norm: Dict[str, List[str]] = {}
        target_by_first_char: Dict[str, List[str]] = {}
        for node_id in target_nodes:
            data = self.graph.nodes[node_id]
            norm = str(data.get("normalized") or "")
            if not norm:
                continue
            target_by_norm.setdefault(norm, []).append(node_id)
            first_char = norm[0]
            target_by_first_char.setdefault(first_char, []).append(node_id)

        added = 0
        for source_node in source_nodes:
            source_data = self.graph.nodes[source_node]
            source_norm = str(source_data.get("normalized") or "")
            if not source_norm:
                continue

            candidate_ids: Set[str] = set(target_by_norm.get(source_norm, []))
            if not candidate_ids and source_norm:
                for target_id in target_by_first_char.get(source_norm[0], []):
                    target_norm = str(self.graph.nodes[target_id].get("normalized") or "")
                    if not target_norm:
                        continue
                    if source_norm in target_norm or target_norm in source_norm:
                        candidate_ids.add(target_id)

            scored_candidates: List[Tuple[float, str]] = []
            for target_node in candidate_ids:
                target_data = self.graph.nodes[target_node]
                score = self._alignment_score(source_norm, str(target_data.get("normalized") or ""))
                if score < min_score:
                    continue
                scored_candidates.append((score, target_node))

            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            for score, target_node in scored_candidates[: max(1, max_matches_per_source)]:
                source_types_now = set(source_data.get("entity_types") or [])
                target_types_now = set(self.graph.nodes[target_node].get("entity_types") or [])
                relation = self._choose_cross_layer_relation(
                    default_relation=default_relation,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    source_types=source_types_now,
                    target_types=target_types_now,
                )
                if self._add_cross_layer_edge(
                    source_node=source_node,
                    target_node=target_node,
                    relation=relation,
                    score=score,
                    source_layer=source_layer,
                    target_layer=target_layer,
                ):
                    added += 1
        return added

    def _add_cross_layer_edge(
        self,
        source_node: str,
        target_node: str,
        relation: str,
        score: float,
        source_layer: str,
        target_layer: str,
    ) -> bool:
        existing = self.graph.get_edge_data(source_node, target_node, default={})
        for _, attrs in existing.items():
            if attrs.get("relation") == relation and bool(attrs.get("cross_layer")):
                return False

        source_data = self.graph.nodes[source_node]
        target_data = self.graph.nodes[target_node]
        source_doc_ids = list(source_data.get("doc_ids") or [])
        target_doc_ids = list(target_data.get("doc_ids") or [])
        merged_doc_ids = self._merge_unique(source_doc_ids, target_doc_ids)
        source_files = list(source_data.get("source_files") or [])

        edge_key = self._allocate_edge_key(
            source=source_node,
            target=target_node,
            base_key=f"cross|{relation}|{source_layer}->{target_layer}",
        )
        self.graph.add_edge(
            source_node,
            target_node,
            key=edge_key,
            relation=relation,
            layer=source_layer,
            doc_id=source_doc_ids[0] if source_doc_ids else "",
            doc_ids=merged_doc_ids,
            source_file=source_files[0] if source_files else "",
            source_field="cross_layer_alignment",
            chunk_id=None,
            confidence=float(score),
            evidence_text=None,
            head_type=self._primary_type(source_data),
            tail_type=self._primary_type(target_data),
            cross_layer=True,
            metadata={
                "source_layer": source_layer,
                "target_layer": target_layer,
                "alignment_score": float(score),
                "source_doc_ids": source_doc_ids,
                "target_doc_ids": target_doc_ids,
            },
        )

        cross_triple = KnowledgeTriple(
            head=str(source_data.get("label") or ""),
            relation=relation,
            tail=str(target_data.get("label") or ""),
            doc_id=source_doc_ids[0] if source_doc_ids else "",
            layer=source_layer,
            source_file=source_files[0] if source_files else "",
            head_type=self._primary_type(source_data),
            tail_type=self._primary_type(target_data),
            confidence=float(score),
            source_field="cross_layer_alignment",
            metadata={
                "target_layer": target_layer,
                "target_doc_ids": target_doc_ids,
            },
        )
        self.cross_layer_triples.append(cross_triple)
        return True

    def _choose_cross_layer_relation(
        self,
        default_relation: str,
        source_layer: str,
        target_layer: str,
        source_types: Set[str],
        target_types: Set[str],
    ) -> str:
        if "method" in source_types and ("knowledge_point" in target_types or "concept" in target_types):
            return "implements"
        if source_layer == "resource":
            return "explains"
        if source_layer == "textbook" and target_layer == "syllabus":
            return "supports"
        if source_layer == "hotspot":
            return "applies"
        return default_relation or "related_to"

    def _get_or_create_node(
        self,
        layer: str,
        label: str,
        entity_type: Optional[str],
        doc_id: str,
        source_file: str,
    ) -> str:
        normalized = self._normalize_text(label)
        if not normalized:
            normalized = self._normalize_text(f"{label}_{doc_id}")
        key = (layer, normalized)

        if key not in self._node_index:
            node_id = f"{layer}|{normalized}"
            self._node_index[key] = node_id
            self.graph.add_node(
                node_id,
                label=clean_text(label),
                normalized=normalized,
                layer=layer,
                entity_types=[entity_type or "entity"],
                doc_ids=[doc_id] if doc_id else [],
                source_files=[source_file] if source_file else [],
                aliases=[clean_text(label)] if label else [],
            )
        node_id = self._node_index[key]
        node_data = self.graph.nodes[node_id]
        self._append_unique(node_data, "entity_types", entity_type or "entity")
        self._append_unique(node_data, "doc_ids", doc_id)
        self._append_unique(node_data, "source_files", source_file)
        self._append_unique(node_data, "aliases", clean_text(label))
        if not node_data.get("label"):
            node_data["label"] = clean_text(label)
        return node_id

    def _get_layer_nodes(self, layer: str, allowed_types: Optional[Set[str]] = None) -> List[str]:
        nodes: List[str] = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("layer") != layer:
                continue
            if allowed_types and not set(attrs.get("entity_types") or []).intersection(allowed_types):
                continue
            nodes.append(node_id)
        return nodes

    def _find_nodes_by_entity(self, entity: str, normalized: str) -> List[str]:
        matches: List[str] = []
        for node_id, attrs in self.graph.nodes(data=True):
            node_norm = str(attrs.get("normalized") or "")
            if not node_norm:
                continue
            if node_norm == normalized:
                matches.append(node_id)
        if matches:
            return matches

        for node_id, attrs in self.graph.nodes(data=True):
            node_norm = str(attrs.get("normalized") or "")
            if not node_norm:
                continue
            if normalized in node_norm or node_norm in normalized:
                matches.append(node_id)
        return matches

    def _collect_neighbors(self, node_id: str, max_neighbors: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, target, _, attrs in self.graph.out_edges(node_id, keys=True, data=True):
            target_data = self.graph.nodes[target]
            rows.append(
                {
                    "direction": "out",
                    "node": self._serialize_node(node_id),
                    "neighbor": self._serialize_node(target),
                    "relation": attrs.get("relation"),
                    "layer": attrs.get("layer"),
                    "doc_id": attrs.get("doc_id"),
                    "doc_ids": attrs.get("doc_ids") or [],
                    "source_file": attrs.get("source_file"),
                    "confidence": attrs.get("confidence"),
                    "cross_layer": bool(attrs.get("cross_layer")),
                }
            )
        for source, _, _, attrs in self.graph.in_edges(node_id, keys=True, data=True):
            source_data = self.graph.nodes[source]
            rows.append(
                {
                    "direction": "in",
                    "node": self._serialize_node(node_id),
                    "neighbor": self._serialize_node(source),
                    "relation": attrs.get("relation"),
                    "layer": attrs.get("layer"),
                    "doc_id": attrs.get("doc_id"),
                    "doc_ids": attrs.get("doc_ids") or [],
                    "source_file": attrs.get("source_file"),
                    "confidence": attrs.get("confidence"),
                    "cross_layer": bool(attrs.get("cross_layer")),
                }
            )
        rows.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        if len(rows) > max_neighbors:
            return rows[:max_neighbors]
        return rows

    def _serialize_node(self, node_id: str) -> Dict[str, Any]:
        attrs = self.graph.nodes[node_id]
        return {
            "node_id": node_id,
            "label": attrs.get("label"),
            "layer": attrs.get("layer"),
            "normalized": attrs.get("normalized"),
            "entity_types": list(attrs.get("entity_types") or []),
            "doc_ids": list(attrs.get("doc_ids") or []),
            "source_files": list(attrs.get("source_files") or []),
            "aliases": list(attrs.get("aliases") or []),
        }

    def _serialize_edge(self, source: str, target: str, key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "edge_key": key,
            "source": source,
            "source_label": self.graph.nodes[source].get("label"),
            "source_layer": self.graph.nodes[source].get("layer"),
            "target": target,
            "target_label": self.graph.nodes[target].get("label"),
            "target_layer": self.graph.nodes[target].get("layer"),
            "relation": data.get("relation"),
            "layer": data.get("layer"),
            "doc_id": data.get("doc_id"),
            "doc_ids": list(data.get("doc_ids") or []),
            "source_file": data.get("source_file"),
            "source_field": data.get("source_field"),
            "chunk_id": data.get("chunk_id"),
            "confidence": data.get("confidence"),
            "evidence_text": data.get("evidence_text"),
            "cross_layer": bool(data.get("cross_layer")),
            "metadata": data.get("metadata") or {},
        }

    def _to_graphml_attrs(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, (list, dict, set, tuple)):
                result[key] = json.dumps(value, ensure_ascii=False)
            else:
                result[key] = value
        return result

    def _alignment_score(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        shorter = left if len(left) <= len(right) else right
        longer = right if shorter == left else left
        if len(shorter) >= 4 and shorter in longer:
            ratio = len(shorter) / max(1, len(longer))
            return min(0.99, 0.9 + 0.09 * ratio)
        ratio = SequenceMatcher(None, left, right).ratio()
        return float(ratio)

    def _normalize_text(self, value: Any) -> str:
        text = clean_text(str(value or "")).lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
        return text

    def _append_unique(self, attrs: Dict[str, Any], key: str, value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        values = list(attrs.get(key) or [])
        if text not in values:
            values.append(text)
        attrs[key] = values

    def _merge_unique(self, left: Sequence[str], right: Sequence[str]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in list(left) + list(right):
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
        return merged

    def _allocate_edge_key(self, source: str, target: str, base_key: str) -> str:
        candidate = base_key
        index = 1
        while self.graph.has_edge(source, target, key=candidate):
            candidate = f"{base_key}|{index}"
            index += 1
        return candidate

    def _primary_type(self, node_data: Dict[str, Any]) -> Optional[str]:
        entity_types = list(node_data.get("entity_types") or [])
        if not entity_types:
            return None
        return entity_types[0]
