from __future__ import annotations

"""单资源增量更新脚本。

仅处理指定 resource 文件，并联动 Mongo / Milvus / 图谱与检索验证。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from extractors.graph_triple_extractor import GraphTripleExtractor
from services.chunk_builder_service import ChunkBuilderService
from services.embedding_service import get_embedding_service
from services.graph_store_service import GraphStoreService
from services.hybrid_search_service import HybridSearchService
from services.ingest_service import DocumentIngestService
from services.milvus_service import MilvusService
from services.mongo_kb_service import MongoKBService


def _safe_model_dump(model: Any) -> Dict[str, Any]:
    """兼容 pydantic v1/v2 的 model dump。"""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _trim(text: Any, n: int = 120) -> str:
    """生成短文本预览。"""
    s = str(text or "").strip()
    if len(s) <= n:
        return s
    return s[:n] + "..."


def _escape_filter_value(value: str) -> str:
    """转义 Milvus filter 字符串。"""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def run(resource_path: Path, query: str, top_k: int, graph_hops: int) -> Dict[str, Any]:
    """执行单文件增量更新并返回验证报告。"""
    layer = "resource"
    source_file = resource_path.name

    if not resource_path.exists():
        raise FileNotFoundError(f"resource file not found: {resource_path}")

    kb = MongoKBService.from_env()
    kb._ensure_available()

    before_counts = kb.get_collection_counts()
    before_doc = kb.get_document_by_source_file(layer=layer, source_file=source_file)
    before_doc_id = (before_doc or {}).get("doc_id")

    milvus = MilvusService()
    milvus_before = {
        "available": False,
        "collection_exists": False,
        "old_chunk_count": None,
        "reason": None,
    }
    milvus_after = {
        "updated": False,
        "delete_count": None,
        "upserted": 0,
        "new_chunk_count": None,
        "replaced_old_chunks": False,
        "reason": None,
    }

    client = None
    filter_expr = (
        f"{milvus.FIELD_LAYER} == '{_escape_filter_value(layer)}' and "
        f"{milvus.FIELD_SOURCE_FILE} == '{_escape_filter_value(source_file)}'"
    )
    try:
        client = milvus._get_client()
        milvus_before["available"] = True
        milvus_before["collection_exists"] = bool(client.has_collection(milvus.collection_name))
        if milvus_before["collection_exists"]:
            old_rows = client.query(
                collection_name=milvus.collection_name,
                filter=filter_expr,
                output_fields=[milvus.FIELD_PK, milvus.FIELD_DOC_ID, milvus.FIELD_CHUNK_ID],
                limit=16384,
            )
            milvus_before["old_chunk_count"] = len(old_rows)
    except Exception as exc:
        milvus_before["reason"] = str(exc)

    ingest_service = DocumentIngestService(kb_service=kb)
    ingest_resp = ingest_service.ingest_document(
        file_bytes=resource_path.read_bytes(),
        file_name=source_file,
        layer=layer,
        source_type="local_file",
        source_name=str(resource_path),
    )
    ingest_payload = _safe_model_dump(ingest_resp)

    after_doc = kb.get_document_by_source_file(layer=layer, source_file=source_file)
    after_counts = kb.get_collection_counts()
    if not after_doc:
        raise RuntimeError("resource doc missing after ingest")

    after_doc_id = str(after_doc.get("doc_id") or "")
    operation = "insert" if before_doc is None else "overwrite_update"

    chunk_builder = ChunkBuilderService()
    chunks = chunk_builder.build_chunks_from_document(layer=layer, mongo_doc=after_doc)

    if milvus_before["available"] and milvus_before["collection_exists"]:
        try:
            delete_result = client.delete(collection_name=milvus.collection_name, filter=filter_expr)
            delete_count = int((delete_result or {}).get("delete_count") or 0)
            milvus_after["delete_count"] = delete_count

            embedder = get_embedding_service()
            texts = [str(item.get("chunk_text") or "").strip() or " " for item in chunks]
            vectors = embedder.embed(texts)
            upsert_result = milvus.upsert_chunks(chunks, vectors, batch_size=64)

            new_rows = client.query(
                collection_name=milvus.collection_name,
                filter=filter_expr,
                output_fields=[milvus.FIELD_PK, milvus.FIELD_DOC_ID, milvus.FIELD_CHUNK_ID],
                limit=16384,
            )
            milvus_after.update(
                {
                    "updated": True,
                    "upserted": int((upsert_result or {}).get("upserted") or 0),
                    "new_chunk_count": len(new_rows),
                    "replaced_old_chunks": bool(delete_count > 0 or (milvus_before.get("old_chunk_count") or 0) > 0),
                }
            )
        except Exception as exc:
            milvus_after["reason"] = str(exc)
    else:
        milvus_after["reason"] = milvus_before.get("reason") or "milvus_unavailable_or_collection_missing"

    mongo_data = after_doc.get("data") or {}
    pages = mongo_data.get("pages") or []
    core_summary = {
        "title": after_doc.get("title"),
        "subject": after_doc.get("subject"),
        "source_file": after_doc.get("source_file"),
        "resource_role": ((mongo_data.get("resource_info") or {}).get("resource_role")),
        "pages_count": len(pages),
        "sample_pages": [
            {
                "page_no": p.get("page_no"),
                "page_title": _trim(p.get("page_title"), 40),
                "page_role": p.get("page_role"),
                "page_summary": _trim(p.get("page_summary"), 80),
            }
            for p in pages[:3]
        ],
    }

    triple_extractor = GraphTripleExtractor(enable_semantic=False)
    resource_triples = triple_extractor.extract_document(after_doc)
    resource_graph = GraphStoreService()
    resource_graph.add_triples(resource_triples)
    resource_cross = resource_graph.build_cross_layer_edges(min_score=0.9, max_matches_per_source=3)
    resource_graph_stats = resource_graph.get_stats()

    hybrid = HybridSearchService(kb_service=kb)
    payload = hybrid.orchestrate_search(
        query=query,
        top_k=max(1, int(top_k)),
        layers=["resource", "textbook", "syllabus", "hotspot"],
        graph_hops=max(1, min(int(graph_hops), 2)),
    )
    vector_hits = payload.get("vector_hits") or []
    graph_edges = ((payload.get("graph_hits") or {}).get("edges") or [])

    resource_vector_hits = [h for h in vector_hits if str(h.get("source_file") or "") == source_file]

    resource_graph_edges: List[Dict[str, Any]] = []
    for edge in graph_edges:
        if str(edge.get("source_file") or "") == source_file:
            resource_graph_edges.append(edge)
            continue
        if str(edge.get("doc_id") or "") == after_doc_id:
            resource_graph_edges.append(edge)
            continue
        md = edge.get("metadata") or {}
        if isinstance(md, dict):
            source_ids = [str(x) for x in (md.get("source_doc_ids") or [])]
            target_ids = [str(x) for x in (md.get("target_doc_ids") or [])]
            if after_doc_id in source_ids or after_doc_id in target_ids:
                resource_graph_edges.append(edge)

    return {
        "status": "ok",
        "target_file": str(resource_path),
        "layer": layer,
        "mongo": {
            "collection_counts_before": before_counts,
            "collection_counts_after": after_counts,
            "doc_id_before": before_doc_id,
            "doc_id_after": after_doc_id,
            "operation": operation,
            "doc_id_replaced": bool(before_doc_id and before_doc_id != after_doc_id),
            "core_summary": core_summary,
            "ingest_preview": ingest_payload.get("preview") or {},
        },
        "milvus": {
            "before": milvus_before,
            "after": milvus_after,
            "chunks_rebuilt_for_doc": len(chunks),
        },
        "graph": {
            "resource_doc_triples": len(resource_triples),
            "resource_doc_graph_nodes": resource_graph_stats.get("nodes"),
            "resource_doc_graph_edges": resource_graph_stats.get("edges"),
            "resource_doc_cross_layer_edges": resource_graph_stats.get("cross_layer_edges"),
            "resource_doc_cross_layer_newly_built": resource_cross,
        },
        "retrieval_validation": {
            "query": query,
            "vector_hit_count_total": len(vector_hits),
            "vector_hits_for_target_file": len(resource_vector_hits),
            "graph_edge_count_total": len(graph_edges),
            "graph_edges_for_target_doc": len(resource_graph_edges),
            "vector_sample": [
                {
                    "score": h.get("score"),
                    "doc_id": h.get("doc_id"),
                    "chunk_id": h.get("chunk_id"),
                    "source_file": h.get("source_file"),
                    "chunk_preview": _trim(h.get("chunk_text"), 100),
                }
                for h in resource_vector_hits[:3]
            ],
            "graph_sample": [
                {
                    "relation": e.get("relation"),
                    "doc_id": e.get("doc_id"),
                    "source_file": e.get("source_file"),
                    "cross_layer": e.get("cross_layer"),
                    "source_node": (e.get("source_node") or {}).get("label"),
                    "target_node": (e.get("target_node") or {}).get("label"),
                }
                for e in resource_graph_edges[:5]
            ],
            "mongo_trace_doc_id": after_doc_id,
        },
    }


def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="Incremental update for one resource file only.")
    parser.add_argument("--file", required=True, help="Absolute path of target resource file")
    parser.add_argument("--query", default="梯度下降算法", help="Validation query")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k for retrieval validation")
    parser.add_argument("--graph-hops", type=int, default=2, help="Graph hops for retrieval validation")
    parser.add_argument("--output", default=None, help="Optional output json path")
    args = parser.parse_args()

    report = run(
        resource_path=Path(args.file),
        query=str(args.query),
        top_k=int(args.top_k),
        graph_hops=int(args.graph_hops),
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
