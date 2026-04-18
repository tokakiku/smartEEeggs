import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

"""图谱三元组抽取编排服务。

负责从 Mongo 按层读取结构化文档，调用规则抽取器生成三元组，
并输出可用于建图的统计与产物文件。
"""

from extractors.graph_triple_extractor import GraphTripleExtractor
from schema.graph_schema import GraphBuildSummary, KnowledgeTriple, LayerTripleStats
from services.mongo_kb_service import MongoKBService, MongoKBUnavailableError


class GraphExtractionService:
    """编排 Mongo 读取与规则抽取，生成三元组记录。"""

    def __init__(
        self,
        kb_service: Optional[MongoKBService] = None,
        triple_extractor: Optional[GraphTripleExtractor] = None,
    ) -> None:
        self.kb_service = kb_service or MongoKBService.from_env()
        self.triple_extractor = triple_extractor or GraphTripleExtractor()

    def extract_from_mongo(
        self,
        layers: Optional[Sequence[str]] = None,
        per_layer_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """从 Mongo 抽取指定层级三元组并返回统计。"""
        self._ensure_mongo_available()
        target_layers = self._normalize_layers(layers)

        docs_by_layer: Dict[str, int] = {}
        triples: List[KnowledgeTriple] = []
        per_layer_entity_keys: Dict[str, set] = defaultdict(set)
        per_layer_relation_types: Dict[str, set] = defaultdict(set)
        per_layer_triple_count: Dict[str, int] = defaultdict(int)

        for layer in target_layers:
            docs = self._load_layer_documents(layer=layer, limit=per_layer_limit)
            docs_by_layer[layer] = len(docs)

            for doc in docs:
                extracted = self.triple_extractor.extract_document(doc)
                triples.extend(extracted)
                for triple in extracted:
                    per_layer_triple_count[triple.layer] += 1
                    per_layer_relation_types[triple.layer].add(triple.relation)
                    per_layer_entity_keys[triple.layer].add((triple.head.lower(), triple.head_type or "entity"))
                    per_layer_entity_keys[triple.layer].add((triple.tail.lower(), triple.tail_type or "entity"))

        per_layer_stats: List[LayerTripleStats] = []
        for layer in target_layers:
            relation_types = sorted(per_layer_relation_types.get(layer, set()))
            per_layer_stats.append(
                LayerTripleStats(
                    layer=layer,
                    doc_count=int(docs_by_layer.get(layer, 0)),
                    triple_count=int(per_layer_triple_count.get(layer, 0)),
                    relation_count=len(relation_types),
                    entity_count=len(per_layer_entity_keys.get(layer, set())),
                    relation_types=relation_types,
                )
            )

        summary = GraphBuildSummary(
            total_docs=sum(docs_by_layer.values()),
            triples_total=len(triples),
            per_layer=per_layer_stats,
        )
        return {
            "triples": triples,
            "summary": summary,
            "docs_by_layer": docs_by_layer,
        }

    def save_triples_json(self, triples: Iterable[KnowledgeTriple], output_path: Path) -> Path:
        """保存三元组 JSON 产物。"""
        payload = [item.model_dump(mode="json") for item in triples]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def _ensure_mongo_available(self) -> None:
        """确保 Mongo 可用。"""
        if not self.kb_service.is_available:
            reason = self.kb_service.unavailable_reason or "mongodb not connected"
            raise MongoKBUnavailableError(reason)

    def _normalize_layers(self, layers: Optional[Sequence[str]]) -> List[str]:
        """标准化层级参数。"""
        valid_layers = list(self.kb_service.COLLECTION_MAP.keys())
        if not layers:
            return valid_layers

        result: List[str] = []
        seen = set()
        for value in layers:
            layer = str(value or "").strip().lower()
            if layer not in valid_layers:
                continue
            if layer in seen:
                continue
            seen.add(layer)
            result.append(layer)
        return result or valid_layers

    def _load_layer_documents(self, layer: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """按层加载 active 文档。"""
        collection_name = self.kb_service.COLLECTION_MAP[layer]
        collection = self.kb_service.db[collection_name]
        mongo_query = {
            "layer": layer,
            "$or": [{"status": "active"}, {"status": {"$exists": False}}],
        }
        cursor = collection.find(mongo_query).sort("updated_at", -1)
        if isinstance(limit, int) and limit > 0:
            cursor = cursor.limit(limit)
        return [self.kb_service._serialize_doc(doc, collection_name) for doc in cursor]
