from __future__ import annotations

"""端到端生成验证脚本。

流程：检索 -> 证据整理 -> outline 生成。
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
for one in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if one not in sys.path:
        sys.path.insert(0, one)

from services.context_consolidator import ContextConsolidator
from services.hybrid_search_service import HybridSearchService


def _to_str(value: Any) -> str:
    """安全转字符串。"""
    return str(value or "").strip()


def _load_outline_generator():
    """加载队友 outline 生成函数。"""
    try:
        from app.services.llm_client import generate_outline_from_text  # type: ignore

        return generate_outline_from_text
    except Exception as exc:
        raise RuntimeError(f"outline generator import failed: {exc}") from exc


def run(query: str, top_k: int, graph_hops: int) -> Dict[str, Any]:
    """执行端到端链路并返回结构化结果。"""
    hybrid = HybridSearchService()
    payload = hybrid.orchestrate_search(query=query, top_k=top_k, graph_hops=graph_hops)

    consolidator = ContextConsolidator()
    consolidated = consolidator.consolidate(retrieval_payload=payload, course_topic=query, use_llm=True)
    clean_lesson_brief = _to_str(consolidated.get("clean_lesson_brief"))

    outline_generator = _load_outline_generator()
    outline_data = outline_generator(course_topic=query, extracted_text=clean_lesson_brief)

    schema_ok = (
        isinstance(outline_data, dict)
        and isinstance(outline_data.get("course_metadata"), dict)
        and isinstance(outline_data.get("syllabus_content"), list)
        and isinstance(outline_data.get("resource_pool"), list)
    )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "status": "success",
        "outline_data": outline_data,
        "validation": {
            "retrieval_ok": len(payload.get("vector_hits") or []) > 0 or len((payload.get("graph_hits") or {}).get("edges") or []) > 0,
            "clean_lesson_brief_ok": bool(clean_lesson_brief),
            "outline_schema_ok": bool(schema_ok),
        },
        "stats": {
            "vector_hit_count": len(payload.get("vector_hits") or []),
            "graph_hit_count": len((payload.get("graph_hits") or {}).get("edges") or []),
            "mongo_doc_count": len(payload.get("mongo_docs") or []),
            "clean_lesson_brief_length": len(clean_lesson_brief),
        },
        "clean_lesson_brief": clean_lesson_brief,
        "consolidator_debug": consolidated.get("debug") or {},
    }


def main() -> int:
    """脚本入口。"""
    parser = argparse.ArgumentParser(description="End-to-end generation test: retrieval -> consolidation -> outline")
    parser.add_argument("--query", default="梯度下降法")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--output", default=None, help="optional output json file path")
    args = parser.parse_args()

    result = run(
        query=_to_str(args.query) or "梯度下降法",
        top_k=max(1, int(args.top_k)),
        graph_hops=max(1, min(int(args.graph_hops), 2)),
    )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
