import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

"""从 Mongo 重建本地向量索引（FAISS）并做一致性校验。"""

from services.mongo_kb_service import MongoKBService
from services.vector_index_service import get_vector_index_service


def _is_true(value: str) -> bool:
    """解析布尔字符串。"""
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_identifier(value: Any) -> str:
    """规范化标识字段，过滤空值占位符。"""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() in {"none", "null", "nan", "na"}:
        return ""
    return text


def _collect_active_mongo_keys(kb_service: MongoKBService) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, int]]:
    """收集 Mongo active 文档主键集合。"""
    kb_service._ensure_available()
    keys: Dict[str, Dict[str, Set[str]]] = {}
    counts: Dict[str, int] = {}

    for layer, collection_name in kb_service.COLLECTION_MAP.items():
        collection = kb_service.db[collection_name]
        query = {"layer": layer, "$or": [{"status": "active"}, {"status": {"$exists": False}}]}
        docs = list(collection.find(query, {"doc_id": 1, "source_file": 1}))
        doc_ids = set()
        source_files = set()
        for doc in docs:
            doc_id = _normalize_identifier(doc.get("doc_id"))
            source_file = _normalize_identifier(doc.get("source_file"))
            if doc_id:
                doc_ids.add(doc_id)
            if source_file:
                source_files.add(source_file)
        keys[layer] = {"doc_ids": doc_ids, "source_files": source_files}
        counts[layer] = len(docs)

    return keys, counts


def _validate_vector_entries(
    entries: List[Dict[str, Any]],
    mongo_keys: Dict[str, Dict[str, Set[str]]],
    sample_limit: int = 12,
) -> Dict[str, Any]:
    """校验向量索引条目是否可追溯到 Mongo。"""
    invalid: List[Dict[str, Any]] = []
    invalid_by_reason: Dict[str, int] = defaultdict(int)
    invalid_by_layer: Dict[str, int] = defaultdict(int)

    for item in entries:
        layer = _normalize_identifier(item.get("layer"))
        doc_id = _normalize_identifier(item.get("doc_id"))
        source_file = _normalize_identifier(item.get("source_file"))
        chunk_id = _normalize_identifier(item.get("chunk_id"))

        reason = ""
        if not layer:
            reason = "missing_layer"
        elif layer not in mongo_keys:
            reason = "unknown_layer"
        elif not (doc_id or source_file):
            reason = "missing_doc_identity"
        elif not chunk_id:
            reason = "missing_chunk_id"
        else:
            layer_keys = mongo_keys.get(layer, {"doc_ids": set(), "source_files": set()})
            exists = (doc_id and doc_id in layer_keys["doc_ids"]) or (
                source_file and source_file in layer_keys["source_files"]
            )
            if not exists:
                reason = "missing_in_mongo"

        if reason:
            invalid_by_reason[reason] += 1
            invalid_by_layer[layer or "unknown"] += 1
            if len(invalid) < sample_limit:
                invalid.append(
                    {
                        "reason": reason,
                        "layer": layer,
                        "doc_id": doc_id or None,
                        "source_file": source_file or None,
                        "chunk_id": chunk_id or None,
                    }
                )

    return {
        "total_entries": len(entries),
        "invalid_entries": sum(invalid_by_reason.values()),
        "valid_entries": len(entries) - sum(invalid_by_reason.values()),
        "invalid_by_reason": dict(invalid_by_reason),
        "invalid_by_layer": dict(invalid_by_layer),
        "invalid_samples": invalid,
    }


def run() -> Dict[str, Any]:
    """执行重建并输出报告。"""
    kb = MongoKBService.from_env()
    kb._ensure_available()
    if kb.use_mock:
        raise RuntimeError("refuse to rebuild vector index from mongomock; set MONGODB_USE_MOCK=0")

    vector_service = get_vector_index_service()
    mongo_keys, mongo_active_counts = _collect_active_mongo_keys(kb)

    before_stats = vector_service.stats()
    before_entries = vector_service.list_active_metadata()
    before_validation = _validate_vector_entries(before_entries, mongo_keys=mongo_keys)

    rebuild_result = vector_service.rebuild_from_mongo(
        kb_service=kb,
        batch_size=max(8, int(os.getenv("VECTOR_REBUILD_BATCH_SIZE", "64"))),
        max_chunks_per_doc=max(20, int(os.getenv("VECTOR_REBUILD_MAX_CHUNKS_PER_DOC", "80"))),
    )

    after_stats = vector_service.stats()
    after_entries = vector_service.list_active_metadata()
    after_validation = _validate_vector_entries(after_entries, mongo_keys=mongo_keys)

    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mongo": {
            "uri": kb.uri,
            "db_name": kb.db_name,
            "backend": kb.backend,
            "use_mock": kb.use_mock,
            "collection_counts": kb.get_collection_counts(),
            "active_doc_counts": mongo_active_counts,
        },
        "vector_index_before": before_stats,
        "vector_entries_before": before_validation,
        "rebuild": rebuild_result,
        "vector_index_after": after_stats,
        "vector_entries_after": after_validation,
        "consistency_check": {
            "before_ok": before_validation["invalid_entries"] == 0,
            "after_ok": after_validation["invalid_entries"] == 0,
        },
    }

    report_dir = Path(vector_service.index_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"rebuild_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    os.environ.setdefault("MONGODB_URI", "mongodb://127.0.0.1:27017")
    os.environ.setdefault("MONGODB_DB_NAME", "ruijie_kb")
    os.environ.setdefault("MONGODB_ENABLED", "1")
    os.environ.setdefault("MONGODB_REQUIRED", "1")
    os.environ.setdefault("MONGODB_USE_MOCK", "0")

    if _is_true(os.getenv("MONGODB_USE_MOCK", "0")):
        raise RuntimeError("MONGODB_USE_MOCK is enabled; abort rebuild")

    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
