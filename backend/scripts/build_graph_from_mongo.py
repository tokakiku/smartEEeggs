import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence
import sys

"""从 Mongo 构建图谱产物的脚本。"""

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from extractors.graph_triple_extractor import GraphTripleExtractor
from schema.graph_schema import KnowledgeTriple
from services.graph_extraction_service import GraphExtractionService
from services.graph_store_service import GraphStoreService
from services.mongo_kb_service import MongoKBService, MongoKBUnavailableError


def _pick_default_query(triples: Sequence[KnowledgeTriple]) -> str:
    """自动选择一个常见概念，作为邻居查询验证词。"""
    preferred = ["机器学习", "machine learning", "梯度下降", "假设空间", "神经网络", "模式识别"]
    lowered_preferred = [item.lower() for item in preferred]

    for triple in triples:
        candidates = [triple.head, triple.tail]
        for candidate in candidates:
            lowered = str(candidate or "").strip().lower()
            if lowered in lowered_preferred:
                return candidate

    for triple in triples:
        if triple.layer == "syllabus" and triple.tail_type == "knowledge_point":
            return triple.tail

    for triple in triples:
        if triple.tail_type in {"knowledge_point", "concept", "method"}:
            return triple.tail

    return "机器学习"


def run_build(
    output_dir: Path,
    query: Optional[str] = None,
    layers: Optional[Sequence[str]] = None,
    per_layer_limit: Optional[int] = None,
) -> dict:
    """执行建图流程并输出报告。"""
    kb_service = MongoKBService.from_env()
    if not kb_service.is_available:
        reason = kb_service.unavailable_reason or "mongodb not connected"
        raise MongoKBUnavailableError(reason)

    extractor = GraphTripleExtractor(enable_semantic=False)
    extraction_service = GraphExtractionService(kb_service=kb_service, triple_extractor=extractor)
    extraction_result = extraction_service.extract_from_mongo(layers=layers, per_layer_limit=per_layer_limit)

    layer_triples: List[KnowledgeTriple] = extraction_result["triples"]
    summary = extraction_result["summary"]

    graph_store = GraphStoreService()
    graph_store.add_triples(layer_triples)
    cross_layer_edges = graph_store.build_cross_layer_edges(min_score=0.9, max_matches_per_source=3)
    all_triples = list(layer_triples) + list(graph_store.cross_layer_triples)

    output_dir.mkdir(parents=True, exist_ok=True)
    triples_json_path = extraction_service.save_triples_json(all_triples, output_dir / "triples.json")
    graph_json_path = graph_store.export_graph_json(output_dir / "graph.json")
    graphml_path = graph_store.export_graphml(output_dir / "graph.graphml")

    query_term = query or _pick_default_query(all_triples)
    query_result = graph_store.query_entity(entity=query_term, max_neighbors=30)
    traced_doc_ids = set()
    for row in query_result.get("neighbors", []):
        for doc_id in row.get("doc_ids") or []:
            if doc_id:
                traced_doc_ids.add(doc_id)

    per_layer_stats = []
    for layer_stat in summary.per_layer:
        per_layer_stats.append(
            {
                "layer": layer_stat.layer,
                "doc_count": layer_stat.doc_count,
                "entity_count": layer_stat.entity_count,
                "triple_count": layer_stat.triple_count,
                "relation_type_count": layer_stat.relation_count,
                "relation_types": layer_stat.relation_types,
            }
        )

    graph_stats = graph_store.get_stats()
    report = {
        "mongodb_doc_counts": extraction_result["docs_by_layer"],
        "per_layer_extraction": per_layer_stats,
        "triples_total": len(all_triples),
        "base_layer_triples": len(layer_triples),
        "cross_layer_triples": len(graph_store.cross_layer_triples),
        "graph": {
            "nodes_total": graph_stats["nodes"],
            "edges_total": graph_stats["edges"],
            "cross_layer_edges": graph_stats["cross_layer_edges"],
            "cross_layer_edges_newly_built": cross_layer_edges,
        },
        "query_validation": {
            "query": query_term,
            "matched_nodes": len(query_result.get("matches") or []),
            "neighbor_count": len(query_result.get("neighbors") or []),
            "sample_neighbors": (query_result.get("neighbors") or [])[:10],
            "traceable_doc_ids": sorted(traced_doc_ids),
            "is_traceable": bool(traced_doc_ids),
        },
        "llm_extraction": {
            "enabled": False,
            "reason": "rule-first pipeline for deterministic offline minimum closure",
        },
        "artifacts": {
            "triples_json": str(triples_json_path),
            "graph_json": str(graph_json_path),
            "graph_graphml": str(graphml_path),
        },
    }

    report_path = output_dir / "graph_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["artifacts"]["report_json"] = str(report_path)
    return report


def main() -> None:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="Build minimal graph pipeline from Mongo structured KB.")
    parser.add_argument("--output-dir", default="data/graph", help="Output directory under backend.")
    parser.add_argument("--query", default=None, help="Entity used for neighbor query validation.")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma separated layers, e.g. syllabus,textbook,resource,hotspot",
    )
    parser.add_argument("--per-layer-limit", type=int, default=None, help="Optional max docs per layer.")
    args = parser.parse_args()

    layers = None
    if args.layers:
        layers = [item.strip() for item in str(args.layers).split(",") if item.strip()]

    report = run_build(
        output_dir=Path(args.output_dir),
        query=args.query,
        layers=layers,
        per_layer_limit=args.per_layer_limit,
    )
    print(str(report["artifacts"]["report_json"]))
    print(
        json.dumps(
            {
                "docs": report["mongodb_doc_counts"],
                "triples_total": report["triples_total"],
                "nodes": report["graph"]["nodes_total"],
                "edges": report["graph"]["edges_total"],
                "cross_layer_edges": report["graph"]["cross_layer_edges"],
                "query": report["query_validation"]["query"],
                "query_neighbors": report["query_validation"]["neighbor_count"],
                "traceable_doc_ids": len(report["query_validation"]["traceable_doc_ids"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
