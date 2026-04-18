from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KnowledgeTriple(BaseModel):
    # 标准三元组记录，保留完整溯源信息。
    head: str
    relation: str
    tail: str
    doc_id: str
    layer: str
    source_file: str

    head_type: Optional[str] = None
    tail_type: Optional[str] = None
    evidence_text: Optional[str] = None
    source_field: Optional[str] = None
    chunk_id: Optional[str] = None
    confidence: float = 1.0

    # 扩展元数据，例如对齐分数或目标层信息。
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LayerTripleStats(BaseModel):
    layer: str
    doc_count: int = 0
    triple_count: int = 0
    relation_count: int = 0
    entity_count: int = 0
    relation_types: List[str] = Field(default_factory=list)


class GraphBuildSummary(BaseModel):
    # 抽取与建图后的统计摘要。
    total_docs: int = 0
    triples_total: int = 0
    nodes_total: int = 0
    edges_total: int = 0
    cross_layer_edges: int = 0
    per_layer: List[LayerTripleStats] = Field(default_factory=list)

