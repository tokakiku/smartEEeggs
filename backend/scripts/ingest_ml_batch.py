from __future__ import annotations

"""机器学习课程四层批量入库脚本。"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.ingest_service import DocumentIngestService
from services.mongo_kb_service import MongoKBService


VALID_LAYERS = ("syllabus", "textbook", "resource", "hotspot")
DEFAULT_ROOT = BACKEND_DIR / "data" / "raw" / "curriculum" / "machine_learning"

SYLLABUS_SUFFIX = {".pdf"}
TEXTBOOK_SUFFIX = {".pdf"}
RESOURCE_SUFFIX = {".pptx", ".pdf", ".docx", ".ppt"}
HOTSPOT_SUFFIX = {".html", ".pdf", ".docx"}


def _safe_model_dump(model: Any) -> Dict[str, Any]:
    """兼容 pydantic v1/v2 的 model dump。"""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _parse_layers(value: Optional[str]) -> List[str]:
    """解析并校验层级参数。"""
    if not value:
        return list(VALID_LAYERS)
    output: List[str] = []
    seen = set()
    for item in str(value).split(","):
        layer = str(item or "").strip().lower()
        if not layer:
            continue
        if layer not in VALID_LAYERS:
            raise ValueError(f"unsupported layer: {layer}")
        if layer in seen:
            continue
        seen.add(layer)
        output.append(layer)
    return output or list(VALID_LAYERS)


def _suffix_whitelist(layer: str) -> set[str]:
    """返回各层允许的文件后缀。"""
    if layer == "syllabus":
        return set(SYLLABUS_SUFFIX)
    if layer == "textbook":
        return set(TEXTBOOK_SUFFIX)
    if layer == "resource":
        return set(RESOURCE_SUFFIX)
    return set(HOTSPOT_SUFFIX)


def _is_hotspot_asset_file(path: Path) -> bool:
    """过滤网页保存时生成的 `<name>_files` 资源目录内文件。"""
    lowered_parts = [part.lower() for part in path.parts]
    return any(part.endswith("_files") for part in lowered_parts)


def _collect_layer_files(layer_root: Path, layer: str) -> List[Path]:
    """收集指定层级可入库文件。"""
    if not layer_root.exists():
        return []
    whitelist = _suffix_whitelist(layer)
    files: List[Path] = []
    for file_path in sorted(layer_root.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in whitelist:
            continue
        if layer == "hotspot" and _is_hotspot_asset_file(file_path):
            continue
        files.append(file_path)
    return files


def _guess_hotspot_source_name(file_name: str, fallback: str) -> str:
    """根据文件名推断热点来源 URL。"""
    lowered = file_name.lower()
    # 常见保存命名：aibase_25122.html -> https://news.aibase.com/news/25122
    aibase_match = re.match(r"^aibase[_-]?(\d+)\.html$", lowered)
    if aibase_match:
        return f"https://news.aibase.com/news/{aibase_match.group(1)}"
    return fallback


def _build_record_base(layer: str, file_path: Path) -> Dict[str, Any]:
    """构造单文件执行记录的基础字段。"""
    return {
        "layer": layer,
        "file_path": str(file_path),
        "file_name": file_path.name,
        "size_bytes": int(file_path.stat().st_size),
    }


def _enable_force_fallback_parser() -> None:
    """强制启用 fallback parser（用于批量稳定执行）。"""
    from parsers.unstructured_parser import UnstructuredParser

    def _forced_failure(self, file_bytes: bytes, file_name: str):
        raise RuntimeError("forced fallback parser for batch ingest")

    UnstructuredParser._partition_with_unstructured = _forced_failure


def run_batch(
    root_dir: Path,
    layers: List[str],
    mongo_uri: str,
    mongo_db_name: str,
    reset_collections: bool,
    force_fallback_parser: bool,
) -> Dict[str, Any]:
    """执行四层批量入库并输出报告。"""
    started = time.time()

    kb_service = MongoKBService(
        uri=mongo_uri,
        db_name=mongo_db_name,
        enabled=True,
        required=True,
        use_mock=False,
    )
    kb_service._ensure_available()
    if force_fallback_parser:
        _enable_force_fallback_parser()
    ingest_service = DocumentIngestService(kb_service=kb_service)

    mongo_before = kb_service.get_collection_counts()
    reset_result: Dict[str, Any] = {"enabled": bool(reset_collections), "deleted_by_collection": {}}
    if reset_collections:
        for _, collection_name in kb_service.COLLECTION_MAP.items():
            deleted = kb_service.db[collection_name].delete_many({})
            reset_result["deleted_by_collection"][collection_name] = int(deleted.deleted_count)

    discovered: Dict[str, Any] = {}
    layer_files: Dict[str, List[Path]] = {}
    for layer in layers:
        files = _collect_layer_files(layer_root=root_dir / layer, layer=layer)
        layer_files[layer] = files
        discovered[layer] = {
            "count": len(files),
            "samples": [str(path) for path in files[:6]],
        }

    records: List[Dict[str, Any]] = []
    success_by_layer: Dict[str, int] = defaultdict(int)
    failure_by_layer: Dict[str, int] = defaultdict(int)

    for layer in layers:
        for file_path in layer_files[layer]:
            row = _build_record_base(layer=layer, file_path=file_path)
            start_one = time.time()
            print(f"[ingest:start] layer={layer} file={file_path.name}", flush=True)
            try:
                source_name = str(file_path)
                if layer == "hotspot":
                    source_name = _guess_hotspot_source_name(file_name=file_path.name, fallback=str(file_path))

                response = ingest_service.ingest_document(
                    file_bytes=file_path.read_bytes(),
                    file_name=file_path.name,
                    layer=layer,
                    source_type="local_file",
                    source_name=source_name,
                )
                payload = _safe_model_dump(response)

                row.update(
                    {
                        "status": "success",
                        "duration_sec": round(time.time() - start_one, 3),
                        "doc_id": payload.get("doc_id"),
                        "mongo_status": (payload.get("preview") or {}).get("mongo_status"),
                        "mongo_collection": (payload.get("preview") or {}).get("mongo_collection"),
                        "vector_status": (payload.get("preview") or {}).get("vector_status"),
                        "chunk_count": (payload.get("preview") or {}).get("chunk_count"),
                        "relation_count": (payload.get("preview") or {}).get("relation_count"),
                        "parser_name": (payload.get("preview") or {}).get("parser_name"),
                    }
                )
                success_by_layer[layer] += 1
                print(
                    f"[ingest:ok] layer={layer} file={file_path.name} "
                    f"sec={row['duration_sec']} mongo={row.get('mongo_status')} "
                    f"chunks={row.get('chunk_count')}",
                    flush=True,
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "duration_sec": round(time.time() - start_one, 3),
                        "error": str(exc),
                    }
                )
                failure_by_layer[layer] += 1
                print(
                    f"[ingest:fail] layer={layer} file={file_path.name} sec={row['duration_sec']} error={exc}",
                    flush=True,
                )
            records.append(row)

    mongo_after = kb_service.get_collection_counts()
    mongo_active = kb_service.get_active_collection_counts()

    success_total = sum(success_by_layer.values())
    failure_total = sum(failure_by_layer.values())
    status = "ok" if failure_total == 0 else "partial_success"

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "duration_sec": round(time.time() - started, 3),
        "root_dir": str(root_dir),
        "layers": layers,
        "parser_mode": "forced_fallback" if force_fallback_parser else "default_unstructured",
        "mongo": {
            "uri": mongo_uri,
            "db_name": mongo_db_name,
            "before_counts": mongo_before,
            "after_counts": mongo_after,
            "after_active_counts": mongo_active,
            "reset": reset_result,
        },
        "discovery": discovered,
        "ingest_summary": {
            "success_total": success_total,
            "failure_total": failure_total,
            "success_by_layer": dict(success_by_layer),
            "failure_by_layer": dict(failure_by_layer),
        },
        "records": records,
    }


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="Batch ingest machine_learning raw materials into Mongo KB")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT), help="Root directory of 4-layer raw files")
    parser.add_argument("--layers", default="syllabus,textbook,resource,hotspot", help="Comma separated layers")
    parser.add_argument("--mongo-uri", default="mongodb://127.0.0.1:27017", help="Mongo URI")
    parser.add_argument("--mongo-db-name", default="ruijie_kb", help="Mongo DB name")
    parser.add_argument(
        "--reset-collections",
        action="store_true",
        help="Delete all four layer collections before ingest",
    )
    parser.add_argument(
        "--force-fallback-parser",
        action="store_true",
        help="Force unstructured parser fallback mode (faster, rule-based text extraction)",
    )
    parser.add_argument("--report-path", default=None, help="Optional report output json path")
    return parser


def main() -> int:
    """脚本入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    report = run_batch(
        root_dir=Path(args.root_dir),
        layers=_parse_layers(args.layers),
        mongo_uri=str(args.mongo_uri),
        mongo_db_name=str(args.mongo_db_name),
        reset_collections=bool(args.reset_collections),
        force_fallback_parser=bool(args.force_fallback_parser),
    )

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        report_path = BACKEND_DIR / "data" / "ingest_reports" / f"ingest_ml_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": report.get("status"),
                "success_total": (report.get("ingest_summary") or {}).get("success_total"),
                "failure_total": (report.get("ingest_summary") or {}).get("failure_total"),
                "mongo_after": (report.get("mongo") or {}).get("after_counts"),
                "report_path": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
