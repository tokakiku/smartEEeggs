from __future__ import annotations

"""混合检索调试脚本（Mongo + Vector + Graph）。"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.hybrid_search_service import HybridSearchService


def _to_str(value: Any) -> str:
    """安全转字符串。"""
    return str(value or "").strip()


def _parse_layers(value: Optional[str]) -> Optional[List[str]]:
    """解析逗号分隔层级参数。"""
    if not value:
        return None
    layers: List[str] = []
    for item in str(value).split(","):
        one = _to_str(item).lower()
        if not one:
            continue
        layers.append(one)
    return layers or None


def _parse_queries(value: Optional[str]) -> List[str]:
    """解析逗号分隔查询词。"""
    if not value:
        return ["梯度下降法", "大模型应用"]
    output = [_to_str(item) for item in str(value).split(",")]
    output = [item for item in output if item]
    return output or ["梯度下降法", "大模型应用"]


def _compact_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """提取紧凑调试结果，便于终端查看。"""
    vector_hits = payload.get("vector_hits") or []
    graph_hits = (payload.get("graph_hits") or {}).get("edges") or []
    mongo_docs = payload.get("mongo_docs") or []
    debug = payload.get("debug") or {}

    return {
        "query": payload.get("query"),
        "normalized_query": payload.get("normalized_query"),
        "query_entities": payload.get("query_entities"),
        "query_keywords": payload.get("query_keywords"),
        "vector_hit_count": len(vector_hits),
        "graph_hit_count": len(graph_hits),
        "merged_doc_count": len(payload.get("merged_doc_ids") or []),
        "mongo_doc_count": len(mongo_docs),
        "debug": debug,
        "vector_hits_sample": vector_hits[:3],
        "graph_edges_sample": graph_hits[:5],
        "mongo_docs_sample": mongo_docs[:5],
        "assembled_context_preview": _to_str((payload.get("assembled_context") or {}).get("text"))[:600],
    }


def run(queries: List[str], top_k: int, layers: Optional[List[str]], graph_hops: int) -> Dict[str, Any]:
    """执行多 query 调试并汇总报告。"""
    service = HybridSearchService()
    results: List[Dict[str, Any]] = []
    for query in queries:
        payload = service.orchestrate_search(query=query, top_k=top_k, layers=layers, graph_hops=graph_hops)
        results.append(_compact_result(payload))

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "queries": queries,
        "top_k": int(top_k),
        "layers": layers or HybridSearchService.DEFAULT_LAYERS,
        "graph_hops": int(graph_hops),
        "results": results,
    }


def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="Debug hybrid orchestrated search (Mongo + Vector + Graph)")
    parser.add_argument("--queries", default="梯度下降法,大模型应用", help="comma-separated queries")
    parser.add_argument("--top-k", type=int, default=5, help="top-k per retrieval path")
    parser.add_argument("--layers", default=None, help="comma-separated layer filter")
    parser.add_argument("--graph-hops", type=int, default=1, help="graph hop count, supports 1 or 2")
    parser.add_argument("--output", default=None, help="optional output json path")
    args = parser.parse_args()

    report = run(
        queries=_parse_queries(args.queries),
        top_k=max(1, int(args.top_k)),
        layers=_parse_layers(args.layers),
        graph_hops=max(1, min(int(args.graph_hops), 2)),
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
