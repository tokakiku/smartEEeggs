from __future__ import annotations

"""从 Mongo 构建 Milvus 向量索引的主脚本。"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.chunk_builder_service import ChunkBuilderService
from services.embedding_service import EmbeddingServiceError, get_embedding_service
from services.milvus_service import MilvusService, MilvusServiceError, MilvusUnavailableError
from services.mongo_kb_service import MongoKBService, MongoKBUnavailableError


def _parse_layers(value: Optional[str]) -> List[str]:
    """解析并校验层级参数。"""
    if not value:
        return ["syllabus", "textbook", "resource", "hotspot"]
    output = []
    for item in str(value).split(","):
        one = item.strip().lower()
        if not one:
            continue
        if one not in {"syllabus", "textbook", "resource", "hotspot"}:
            raise ValueError(f"unsupported layer: {one}")
        output.append(one)
    return output or ["syllabus", "textbook", "resource", "hotspot"]


def _is_true(value: Any) -> bool:
    """解析布尔字符串。"""
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_preview(text: Any, max_len: int = 220) -> str:
    """生成短文本预览。"""
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max(0, max_len)] + "..."


def _iter_batches(items: List[Dict[str, Any]], batch_size: int):
    """按 batch_size 迭代批次。"""
    step = max(1, int(batch_size))
    for i in range(0, len(items), step):
        yield items[i : i + step]


def _embedding_alignment_check(
    chunks: List[Dict[str, Any]],
    embedding_service: Any,
    batch_size: int,
) -> Dict[str, Any]:
    """检查 chunk 与 embedding 是否一一对应。"""
    total_chunks = len(chunks)
    embedded_rows = 0
    dim = int(getattr(embedding_service, "dimension", 0) or 0)
    batch_count = 0
    for batch in _iter_batches(chunks, batch_size):
        texts = [str(item.get("chunk_text") or "").strip() or " " for item in batch]
        vectors = embedding_service.embed(texts)
        batch_count += 1
        embedded_rows += int(getattr(vectors, "shape", [0])[0] if hasattr(vectors, "shape") else len(vectors))
        if dim <= 0 and hasattr(vectors, "shape") and len(vectors.shape) >= 2:
            dim = int(vectors.shape[1])
    return {
        "chunk_count": total_chunks,
        "embedded_count": embedded_rows,
        "aligned_one_to_one": total_chunks == embedded_rows,
        "batch_count": batch_count,
        "dimension": dim,
    }


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="Build Milvus vector index from MongoKB")
    parser.add_argument("--layers", default="syllabus,textbook,resource,hotspot", help="comma-separated layers")
    parser.add_argument("--batch-size", type=int, default=64, help="embedding/upsert batch size")
    parser.add_argument("--max-chunks-per-doc", type=int, default=2400, help="max chunks generated per document")
    parser.add_argument("--query", default="机器学习 梯度下降 的核心思想", help="verification query")
    parser.add_argument("--top-k", type=int, default=5, help="top-k for query validation")

    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))
    parser.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN", ""))
    parser.add_argument("--milvus-db-name", default=os.getenv("MILVUS_DB_NAME", "default"))
    parser.add_argument("--milvus-collection", default=os.getenv("MILVUS_COLLECTION", "ruijie_kb_chunks"))
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="keep existing collection schema (default behavior is drop + rebuild)",
    )
    parser.add_argument("--skip-milvus", action="store_true", help="only build chunks without Milvus write")
    parser.add_argument("--require-milvus", action="store_true", help="exit non-zero if Milvus unavailable")
    parser.add_argument("--dry-run", action="store_true", help="read mongo + build chunks only")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL_NAME", ""))

    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", MongoKBService.URI_DEFAULT))
    parser.add_argument("--mongo-db-name", default=os.getenv("MONGODB_DB_NAME", MongoKBService.DB_NAME_DEFAULT))
    parser.add_argument("--mongo-required", default=os.getenv("MONGODB_REQUIRED", "1"))
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """执行构建流程并返回结构化报告。"""
    layers = _parse_layers(args.layers)
    started_at = datetime.now().isoformat(timespec="seconds")

    kb_service = MongoKBService(
        uri=args.mongo_uri,
        db_name=args.mongo_db_name,
        enabled=True,
        required=_is_true(args.mongo_required),
        use_mock=False,
    )
    kb_service._ensure_available()

    docs: List[Dict[str, Any]] = list(kb_service.iter_active_documents(layers=layers))
    doc_counts_by_layer: Dict[str, int] = defaultdict(int)
    for item in docs:
        doc_counts_by_layer[str(item.get("layer") or "unknown")] += 1

    chunk_builder = ChunkBuilderService(max_chunks_per_doc=int(args.max_chunks_per_doc))
    chunks: List[Dict[str, Any]] = []
    chunk_counts_by_layer: Dict[str, int] = defaultdict(int)
    chunk_counts_by_doc: Dict[str, int] = defaultdict(int)

    for doc in docs:
        layer = str(doc.get("layer") or "")
        generated = chunk_builder.build_chunks_from_document(layer=layer, mongo_doc=doc)
        chunks.extend(generated)
        chunk_counts_by_layer[layer] += len(generated)
        chunk_counts_by_doc[f"{layer}:{doc.get('doc_id') or doc.get('source_file')}"] += len(generated)

    required_check = chunk_builder.validate_required_metadata(chunks)
    identity_check = {
        "doc_id_layer_chunk_id_all_present": all(
            bool(str(item.get("doc_id") or "").strip())
            and bool(str(item.get("layer") or "").strip())
            and bool(str(item.get("chunk_id") or "").strip())
            for item in chunks
        )
    }

    report: Dict[str, Any] = {
        "timestamp": started_at,
        "mongo": {
            "uri": kb_service.uri,
            "db_name": kb_service.db_name,
            "backend": kb_service.backend,
            "collection_counts": kb_service.get_collection_counts(),
            "active_collection_counts": kb_service.get_active_collection_counts(),
            "docs_read_total": len(docs),
            "docs_read_by_layer": dict(doc_counts_by_layer),
        },
        "chunking": {
            "chunks_total": len(chunks),
            "chunks_by_layer": dict(chunk_counts_by_layer),
            "required_metadata_check": required_check,
            "identity_check": identity_check,
            "sample_chunks": [
                {
                    "doc_id": item.get("doc_id"),
                    "layer": item.get("layer"),
                    "chunk_id": item.get("chunk_id"),
                    "source_file": item.get("source_file"),
                    "title": item.get("title"),
                    "subject": item.get("subject"),
                    "chunk_preview": _safe_preview(item.get("chunk_text")),
                    "chapter": item.get("chapter"),
                    "section": item.get("section"),
                    "page_no": item.get("page_no"),
                }
                for item in chunks[:8]
            ],
            "chunks_per_doc_sample": [
                {"doc_key": key, "chunk_count": count}
                for key, count in list(chunk_counts_by_doc.items())[:12]
            ],
        },
    }

    if args.dry_run:
        report["pipeline_status"] = "dry_run_only"
        return report

    embedding_service = get_embedding_service(model_name=args.embedding_model or None)
    report["embedding"] = {
        "model_name": getattr(embedding_service, "model_name", None),
        "dimension": int(embedding_service.dimension),
    }

    if args.skip_milvus:
        report["embedding"]["alignment_check"] = _embedding_alignment_check(
            chunks=chunks,
            embedding_service=embedding_service,
            batch_size=int(args.batch_size),
        )
        report["pipeline_status"] = "skip_milvus"
        return report

    milvus = MilvusService(
        uri=args.milvus_uri,
        token=args.milvus_token,
        db_name=args.milvus_db_name,
        collection_name=args.milvus_collection,
    )
    drop_existing = not bool(args.keep_existing)

    try:
        ensure_result = milvus.ensure_collection(
            vector_dim=int(embedding_service.dimension),
            drop_existing=drop_existing,
        )
        report["milvus"] = {
            "ensure": ensure_result,
            "collection_rebuild": {
                "drop_existing": drop_existing,
                "collection_name": args.milvus_collection,
            },
        }
    except (MilvusUnavailableError, MilvusServiceError) as exc:
        report["milvus"] = {
            "status": "unavailable",
            "error": str(exc),
            "uri": args.milvus_uri,
            "collection": args.milvus_collection,
        }
        report["embedding"]["alignment_check"] = _embedding_alignment_check(
            chunks=chunks,
            embedding_service=embedding_service,
            batch_size=int(args.batch_size),
        )
        report["pipeline_status"] = "milvus_unavailable"
        if args.require_milvus:
            raise
        return report

    upserted_total = 0
    embedded_total = 0
    batch_errors: List[str] = []
    for batch in _iter_batches(chunks, args.batch_size):
        texts = [str(item.get("chunk_text") or "").strip() or " " for item in batch]
        vectors = embedding_service.embed(texts)
        embedded_total += len(batch)
        try:
            upsert_result = milvus.upsert_chunks(batch, vectors, batch_size=args.batch_size)
            upserted_total += int(upsert_result.get("upserted") or 0)
        except Exception as exc:
            batch_errors.append(str(exc))
            if len(batch_errors) >= 5:
                break

    report["embedding"]["embedded_chunks"] = embedded_total
    report["embedding"]["alignment_check"] = {
        "chunk_count": len(chunks),
        "embedded_count": embedded_total,
        "aligned_one_to_one": len(chunks) == embedded_total,
        "batch_count": max(1, (len(chunks) + max(1, int(args.batch_size)) - 1) // max(1, int(args.batch_size))),
        "dimension": int(embedding_service.dimension),
    }
    report["milvus"]["upserted_chunks"] = upserted_total
    report["milvus"]["batch_errors"] = batch_errors
    report["milvus"]["stats_after_upsert"] = milvus.get_stats()
    report["milvus"]["truncation"] = milvus.truncation_stats()

    query_text = str(args.query or "").strip()
    query_validation = {
        "query": query_text,
        "top_k": int(args.top_k),
        "results": [],
        "milvus_result_count": 0,
        "mongo_traceable_count": 0,
    }
    if query_text and not batch_errors:
        q_vec = embedding_service.embed_query(query_text)
        q_hits = milvus.search(query_vector=q_vec[0], top_k=int(args.top_k))
        query_validation["milvus_result_count"] = len(q_hits)

        traceable = 0
        for item in q_hits:
            layer = str(item.get("layer") or "")
            doc_id = str(item.get("doc_id") or "")
            source_file = str(item.get("source_file") or "")
            mongo_doc = None
            if layer and doc_id:
                mongo_doc = kb_service.get_document(layer=layer, doc_id=doc_id)
            if mongo_doc is None and layer and source_file:
                mongo_doc = kb_service.get_document_by_source_file(layer=layer, source_file=source_file)
            if mongo_doc is not None:
                traceable += 1

            query_validation["results"].append(
                {
                    "score": item.get("score"),
                    "layer": layer,
                    "doc_id": doc_id,
                    "chunk_id": item.get("chunk_id"),
                    "source_file": source_file,
                    "title": item.get("title"),
                    "subject": item.get("subject"),
                    "chunk_preview": _safe_preview(item.get("chunk_text")),
                    "mongo_traceable": mongo_doc is not None,
                    "mongo_trace_doc_id": mongo_doc.get("doc_id") if mongo_doc else None,
                    "mongo_trace_title": mongo_doc.get("title") if mongo_doc else None,
                }
            )
        query_validation["mongo_traceable_count"] = traceable
        query_validation["all_results_traceable"] = (
            len(q_hits) > 0 and traceable == len(q_hits)
        )
    report["query_validation"] = query_validation
    report["pipeline_status"] = "ok" if not batch_errors else "partial_batch_failure"
    return report


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()
    report_dir = Path(__file__).resolve().parents[1] / "data" / "vector_index"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"milvus_build_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        report = run(args)
    except (MongoKBUnavailableError, EmbeddingServiceError, MilvusUnavailableError, MilvusServiceError) as exc:
        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pipeline_status": "failed",
            "error": str(exc),
            "args": vars(args),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "report_path": str(report_path),
                },
                ensure_ascii=False,
            )
        )
        return 2

    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report.get("pipeline_status"),
                "docs_read_total": ((report.get("mongo") or {}).get("docs_read_total")),
                "chunks_total": ((report.get("chunking") or {}).get("chunks_total")),
                "milvus_upserted": ((report.get("milvus") or {}).get("upserted_chunks")),
                "query_result_count": ((report.get("query_validation") or {}).get("milvus_result_count")),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
