from __future__ import annotations

"""生成适配层（薄桥接）。

职责：
- 将 query 串联为「检索 -> 整理 -> 大纲生成 -> PPT 生成」；
- 复用现有 app 侧生成能力，不重写生成逻辑；
- 在依赖缺失时提供清晰错误与可控降级。
"""

import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from services.context_consolidator import ContextConsolidator
from services.hybrid_search_service import HybridSearchService


def _to_str(value: Any) -> str:
    return str(value or "").strip()


class GeneratorAdapterError(RuntimeError):
    """生成适配层通用异常。"""


class GeneratorAdapterInputError(GeneratorAdapterError):
    """输入参数不合法时抛出。"""


class GeneratorAdapterDependencyError(GeneratorAdapterError):
    """下游生成依赖不可用时抛出。"""


class GeneratorAdapterService:
    """薄适配服务。

    主流程：
    query -> HybridSearch -> ContextConsolidator -> 队友 outline 生成。
    """

    def __init__(
        self,
        hybrid_search_service: Optional[HybridSearchService] = None,
        consolidator: Optional[ContextConsolidator] = None,
        outline_generator: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        ppt_generator: Optional[Callable[[Dict[str, Any], int], str]] = None,
    ) -> None:
        self.hybrid_search_service = hybrid_search_service or HybridSearchService()
        self.consolidator = consolidator or ContextConsolidator()
        self._outline_generator = outline_generator
        self._ppt_generator = ppt_generator

    def generate_from_hybrid_context(
        self,
        query: str,
        top_k: int = 6,
        graph_hops: int = 2,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """基于混合检索上下文生成 outline_data。"""
        topic = _to_str(query)
        if not topic:
            raise GeneratorAdapterInputError("query is required")

        try:
            retrieval = self.hybrid_search_service.orchestrate_search(
                query=topic,
                top_k=max(1, int(top_k)),
                layers=layers,
                graph_hops=max(1, min(int(graph_hops), 2)),
            )
        except Exception as exc:
            raise GeneratorAdapterError(f"hybrid search failed: {exc}") from exc

        try:
            consolidated = self.consolidator.consolidate(
                retrieval_payload=retrieval,
                course_topic=topic,
                use_llm=True,
            )
        except Exception as exc:
            raise GeneratorAdapterError(f"context consolidation failed: {exc}") from exc

        clean_lesson_brief = _to_str(consolidated.get("clean_lesson_brief"))
        if not clean_lesson_brief:
            raise GeneratorAdapterError("context consolidation returned empty clean_lesson_brief")

        outline_data = self._generate_outline(topic=topic, clean_lesson_brief=clean_lesson_brief)
        if not isinstance(outline_data, dict):
            raise GeneratorAdapterError("outline generator returned non-dict result")

        retrieval_stats = {
            "vector_hit_count": len(retrieval.get("vector_hits") or []),
            "graph_hit_count": len((retrieval.get("graph_hits") or {}).get("edges") or []),
            "mongo_doc_count": len(retrieval.get("mongo_docs") or []),
        }

        return {
            "status": "success",
            "query": topic,
            "clean_lesson_brief": clean_lesson_brief,
            "outline_data": outline_data,
            "retrieval_stats": retrieval_stats,
            "debug": {
                "consolidator_debug": consolidated.get("debug") or {},
                "hybrid_debug": retrieval.get("debug") or {},
            },
        }

    def generate_ppt_from_hybrid_context(
        self,
        query: str,
        project_id: int,
        top_k: int = 6,
        graph_hops: int = 2,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """基于混合检索上下文生成 PPT。"""
        outline_payload = self.generate_from_hybrid_context(
            query=query,
            top_k=top_k,
            graph_hops=graph_hops,
            layers=layers,
        )
        outline_data = outline_payload.get("outline_data") or {}
        if not isinstance(outline_data, dict) or not outline_data:
            raise GeneratorAdapterError("outline generation returned empty outline_data")

        download_url = ""
        fallback_reason = ""
        ppt_generator = self._load_ppt_generator()
        if not callable(ppt_generator):
            raise GeneratorAdapterDependencyError("ppt generator unavailable")

        try:
            download_url = _to_str(ppt_generator(outline_data, int(project_id)))
        except Exception as exc:
            fallback_reason = _to_str(exc)
            download_url = self._try_generate_ppt_without_minio(
                outline_data=outline_data,
                project_id=int(project_id),
            )
            if not download_url:
                raise GeneratorAdapterError(f"ppt generation failed: {exc}") from exc

        if not download_url:
            raise GeneratorAdapterError("ppt generator returned empty download_url")
        download_url = self._normalize_download_url(download_url)

        return {
            "status": "success",
            "message": "ppt generated",
            "project_id": int(project_id),
            "download_url": download_url,
            "outline_data": outline_data,
            "clean_lesson_brief": outline_payload.get("clean_lesson_brief"),
            "retrieval_stats": outline_payload.get("retrieval_stats") or {},
            "debug": {
                **(outline_payload.get("debug") or {}),
                "ppt_fallback_used": bool(fallback_reason),
                "ppt_fallback_reason": fallback_reason,
            },
        }

    def _generate_outline(self, topic: str, clean_lesson_brief: str) -> Dict[str, Any]:
        """调用队友 outline 生成入口，并统一返回结构校验。"""
        generator = self._load_outline_generator()
        if not callable(generator):
            raise GeneratorAdapterDependencyError("outline generator unavailable")

        try:
            payload = generator(topic, clean_lesson_brief)
        except Exception as exc:
            raise GeneratorAdapterError(f"outline generation failed: {exc}") from exc

        if isinstance(payload, dict):
            return payload
        raise GeneratorAdapterError("outline generator returned invalid payload")

    def _load_outline_generator(self) -> Optional[Callable[[str, str], Dict[str, Any]]]:
        """懒加载队友 outline 生成函数。"""
        if callable(self._outline_generator):
            return self._outline_generator

        self._ensure_project_root_on_path()
        try:
            from app.services.llm_client import generate_outline_from_text  # type: ignore
        except Exception:
            return None

        self._outline_generator = generate_outline_from_text
        return self._outline_generator

    def _load_ppt_generator(self) -> Optional[Callable[[Dict[str, Any], int], str]]:
        """懒加载队友 PPT 生成函数。"""
        if callable(self._ppt_generator):
            return self._ppt_generator

        self._ensure_project_root_on_path()
        try:
            from app.services.ppt_generator import generate_ppt_from_json  # type: ignore
        except Exception:
            return None

        self._ppt_generator = generate_ppt_from_json
        return self._ppt_generator

    def _ensure_project_root_on_path(self) -> None:
        """确保可导入 `app.*` 模块。"""
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)

    def _try_generate_ppt_without_minio(self, outline_data: Dict[str, Any], project_id: int) -> str:
        """MinIO 不可用时的兜底：保留渲染逻辑，仅跳过上传步骤。"""
        self._ensure_project_root_on_path()
        try:
            import app.services.ppt_generator as ppt_module  # type: ignore
        except Exception:
            return ""

        if not hasattr(ppt_module, "generate_ppt_from_json"):
            return ""

        original_uploader = getattr(ppt_module, "upload_ppt_to_minio", None)
        try:
            # 保持队友渲染逻辑不变，仅在兜底时把上传步骤改为 no-op。
            setattr(ppt_module, "upload_ppt_to_minio", lambda *_args, **_kwargs: "")
            value = ppt_module.generate_ppt_from_json(outline_data, int(project_id))
            return _to_str(value)
        except Exception:
            return ""
        finally:
            with suppress(Exception):
                if original_uploader is not None:
                    setattr(ppt_module, "upload_ppt_to_minio", original_uploader)

    def _normalize_download_url(self, value: str) -> str:
        """统一本地下载地址格式，避免多目录风格混用。"""
        normalized = _to_str(value)
        if not normalized:
            return ""

        lowered = normalized.lower()
        if lowered.startswith("http://") or lowered.startswith("https://"):
            return normalized

        path_value = normalized.replace("\\", "/")
        if path_value.startswith("/downloads/"):
            return path_value
        if path_value.startswith("downloads/"):
            return f"/{path_value}"

        file_name = Path(path_value).name
        if file_name.lower().endswith(".pptx"):
            return f"/downloads/{file_name}"
        return path_value
