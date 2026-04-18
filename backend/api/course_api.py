from __future__ import annotations

"""后端统一 API 入口。

说明：
- 保持历史接口路径不变，避免前端联调断裂；
- API 层只做参数接收、错误映射与返回组装；
- 具体业务编排由 services 层承担。
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from schema.course_schema import CourseInstruction
from schema.document_ingest_schema import DocumentIngestResponse
from schema.syllabus_schema import SyllabusExtractTextRequest, SyllabusExtractionResult
from services.cross_layer_retrieval_service import CrossLayerRetrievalService
from services.generator_adapter_service import (
    GeneratorAdapterDependencyError,
    GeneratorAdapterError,
    GeneratorAdapterInputError,
    GeneratorAdapterService,
)
from services.hybrid_retrieval_service import HybridRetrievalService
from services.ingest_service import DocumentIngestService, IngestValidationError
from services.kb_rag_service import KBRAGService
from services.mongo_kb_service import MongoKBService, MongoKBUnavailableError
from services.syllabus_extractor import SyllabusExtractor
from utils.pdf_reader import read_pdf_text

router = APIRouter()

# ---------------------------------------------------------------------------
# 共享服务实例
# ---------------------------------------------------------------------------
_generator_adapter_service = GeneratorAdapterService()
ingest_service = DocumentIngestService()
extractor = SyllabusExtractor()
kb_service = MongoKBService.from_env()

_rag_engine = None
_rag_error: Optional[str] = None

_FALLBACK_DOCS = [
    "TCP three-way handshake is used to establish a connection.",
    "HTTP is an application layer protocol for web data transfer.",
    "IP protocol is responsible for network layer delivery.",
    "DNS resolves domain names to IP addresses.",
]


# ---------------------------------------------------------------------------
# 课程生成接口
# ---------------------------------------------------------------------------
class _RAGGenerationBaseRequest(BaseModel):
    query: Optional[str] = None
    course_topic: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=20)
    graph_hops: int = Field(default=2, ge=1, le=2)
    layers: Optional[List[str]] = None


class GenerateWithRAGRequest(_RAGGenerationBaseRequest):
    include_clean_lesson_brief: bool = False
    include_debug: bool = False


class GeneratePPTWithRAGRequest(_RAGGenerationBaseRequest):
    project_id: Optional[int] = Field(default=None, ge=1)
    include_outline_data: bool = True
    include_clean_lesson_brief: bool = False
    include_debug: bool = False


def _resolve_query(query: Optional[str], course_topic: Optional[str]) -> str:
    """统一解析 query，兼容 `query` 与 `course_topic` 两种入参。"""
    value = (query or course_topic or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="query is required")
    return value


def _call_adapter(action_name: str, fn: Callable[..., Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
    """统一将适配层异常映射为 HTTP 错误，保持接口返回一致。"""
    try:
        return fn(**kwargs)
    except GeneratorAdapterInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GeneratorAdapterDependencyError as exc:
        raise HTTPException(status_code=500, detail=f"generator dependency unavailable: {exc}") from exc
    except GeneratorAdapterError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{action_name} failed: {exc}") from exc


@router.post("/generate_instruction")
def generate_instruction(data: CourseInstruction):
    """基础演示接口（保留兼容，不属于 RAG 主链）。"""
    return {"message": "Instruction received", "data": data}


@router.post("/api/course/generate_with_rag")
def generate_with_rag(request: GenerateWithRAGRequest):
    """RAG 前置大纲生成入口。"""
    query = _resolve_query(request.query, request.course_topic)
    payload = _call_adapter(
        action_name="generate_with_rag",
        fn=_generator_adapter_service.generate_from_hybrid_context,
        query=query,
        top_k=request.top_k,
        graph_hops=request.graph_hops,
        layers=request.layers,
    )

    response = {
        "status": payload.get("status") or "success",
        "query": payload.get("query") or query,
        "outline_data": payload.get("outline_data") or {},
        "retrieval_stats": payload.get("retrieval_stats") or {},
    }
    if request.include_clean_lesson_brief:
        response["clean_lesson_brief"] = payload.get("clean_lesson_brief") or ""
    if request.include_debug:
        response["debug"] = payload.get("debug") or {}

    return response


@router.post("/api/course/generate_ppt_with_rag")
def generate_ppt_with_rag(request: GeneratePPTWithRAGRequest):
    """RAG 前置 PPT 生成入口。"""
    query = _resolve_query(request.query, request.course_topic)

    project_id = int(request.project_id or int(datetime.now().timestamp()))
    if project_id <= 0:
        raise HTTPException(status_code=400, detail="project_id must be positive")

    payload = _call_adapter(
        action_name="generate_ppt_with_rag",
        fn=_generator_adapter_service.generate_ppt_from_hybrid_context,
        query=query,
        project_id=project_id,
        top_k=request.top_k,
        graph_hops=request.graph_hops,
        layers=request.layers,
    )

    download_url = str(payload.get("download_url") or "").strip()
    if not download_url:
        raise HTTPException(status_code=500, detail="ppt generation returned empty download_url")

    response = {
        "status": payload.get("status") or "success",
        "query": payload.get("query") or query,
        "project_id": int(payload.get("project_id") or project_id),
        "download_url": download_url,
        "ppt_result": {
            "status": payload.get("status") or "success",
            "message": payload.get("message") or "ppt generated",
            "project_id": int(payload.get("project_id") or project_id),
            "download_url": download_url,
        },
        "retrieval_stats": payload.get("retrieval_stats") or {},
    }

    if request.include_outline_data:
        response["outline_data"] = payload.get("outline_data") or {}
    if request.include_clean_lesson_brief:
        response["clean_lesson_brief"] = payload.get("clean_lesson_brief") or ""
    if request.include_debug:
        response["debug"] = payload.get("debug") or {}

    return response


# ---------------------------------------------------------------------------
# 文档入库接口
# ---------------------------------------------------------------------------
@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    layer: str = Form(...),
    file_type: Optional[str] = Form(None),
    source_type: Optional[str] = Form(None),
    source_name: Optional[str] = Form(None),
):
    """统一文档入库入口：解析 -> 抽取 -> 存储/入库。"""
    try:
        file_bytes = await file.read()
        return ingest_service.ingest_document(
            file_bytes=file_bytes,
            file_name=file.filename or "unknown",
            layer=layer,
            source_type=source_type,
            source_name=source_name,
            file_type_hint=file_type,
        )
    except IngestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ingest failed: {exc}") from exc


@router.post("/documents/parse_only")
async def parse_only_document(
    file: UploadFile = File(...),
    layer: str = Form(...),
    file_type: Optional[str] = Form(None),
    source_type: Optional[str] = Form(None),
    source_name: Optional[str] = Form(None),
):
    """仅解析调试入口，不执行后续抽取和存储。"""
    try:
        file_bytes = await file.read()
        parsed_doc = ingest_service.parse_only(
            file_bytes=file_bytes,
            file_name=file.filename or "unknown",
            layer=layer,
            source_type=source_type,
            source_name=source_name,
            file_type_hint=file_type,
        )
        return {
            "doc_id": parsed_doc.doc_id,
            "file_name": parsed_doc.file_name,
            "file_type": parsed_doc.file_type,
            "layer": parsed_doc.layer,
            "title": parsed_doc.title,
            "element_count": len(parsed_doc.elements),
            "raw_text_preview": parsed_doc.raw_text[:600],
            "elements_preview": [item.model_dump() for item in parsed_doc.elements[:10]],
            "metadata": parsed_doc.metadata,
        }
    except IngestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"parse failed: {exc}") from exc


@router.get("/documents/supported_types")
def get_supported_types():
    """返回当前支持的文档类型。"""
    return {"supported_types": ingest_service.get_supported_file_types()}


@router.get("/documents/supported_layers")
def get_supported_layers():
    """返回当前支持的知识层级。"""
    return {"supported_layers": ingest_service.get_supported_layers()}


# ---------------------------------------------------------------------------
# 教学大纲辅助接口
# ---------------------------------------------------------------------------
@router.post("/syllabus/extract_text", response_model=SyllabusExtractionResult)
def extract_syllabus_text(data: SyllabusExtractTextRequest):
    """直接对文本执行 syllabus 抽取。"""
    if not data.text or not data.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    return extractor.extract_from_text(data.text)


@router.post("/syllabus/extract_pdf", response_model=SyllabusExtractionResult)
async def extract_syllabus_pdf(file: UploadFile = File(...)):
    """对 PDF 文件执行 syllabus 抽取。"""
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    if not filename.endswith(".pdf") and "pdf" not in content_type:
        raise HTTPException(status_code=400, detail="only PDF file is supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="empty file")

    text = read_pdf_text(file_bytes)
    if not text.strip():
        raise HTTPException(status_code=400, detail="no readable text in PDF")

    return extractor.extract_from_text(text)


# ---------------------------------------------------------------------------
# 知识库检索与维护接口
# ---------------------------------------------------------------------------
@router.get("/kb/search")
def search_kb(
    layer: Optional[str] = Query(default=None),
    source_file: Optional[str] = Query(default=None),
    title: Optional[str] = Query(default=None),
    subject: Optional[str] = Query(default=None),
    knowledge_point: Optional[str] = Query(default=None),
    page_role: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    """按元字段过滤检索 Mongo 知识库。"""
    try:
        items = kb_service.search_documents(
            layer=layer,
            source_file=source_file,
            title=title,
            subject=subject,
            knowledge_point=knowledge_point,
            page_role=page_role,
            event_type=event_type,
            limit=limit,
        )
        return {"total": len(items), "items": items}
    except MongoKBUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"kb unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"kb search failed: {exc}") from exc


@router.get("/kb/retrieve")
def retrieve_kb_across_layers(
    query: str = Query(..., min_length=1),
    subject: Optional[str] = Query(default=None),
    top_k: int = Query(default=5, ge=1, le=20),
    layers: Optional[str] = Query(default=None),
):
    """执行跨层检索，返回分层结果与统计信息。"""
    try:
        layer_list = None
        if layers and layers.strip():
            layer_list = [item.strip() for item in layers.split(",") if item.strip()]

        retrieval_service = HybridRetrievalService(kb_service=kb_service)
        retrieval_bundle = retrieval_service.retrieve_hybrid(
            query=query,
            subject=subject,
            layers=layer_list,
            top_k=top_k,
        )
        return {
            "query": query,
            "subject": subject,
            "top_k": top_k,
            "layers": layer_list or CrossLayerRetrievalService.DEFAULT_LAYERS,
            "results": retrieval_bundle["results"],
            "counts": retrieval_bundle["counts"],
            "debug": retrieval_bundle.get("debug", {}),
        }
    except MongoKBUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"kb unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"kb retrieve failed: {exc}") from exc


@router.get("/kb/doc/{layer}/{doc_id}")
def get_kb_doc(layer: str, doc_id: str):
    """按 layer + doc_id 获取单条结构化文档。"""
    try:
        doc = kb_service.get_document(layer=layer, doc_id=doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="document not found")
        return doc
    except MongoKBUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"kb unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"kb get failed: {exc}") from exc


@router.post("/kb/syllabus/{doc_id}/primary")
def set_primary_syllabus(
    doc_id: str,
    course_name: Optional[str] = Query(default=None),
    course_code: Optional[str] = Query(default=None),
):
    """将指定 syllabus 设为主大纲（primary）。"""
    try:
        return kb_service.set_primary_syllabus(
            doc_id=doc_id,
            course_name=course_name,
            course_code=course_code,
        )
    except MongoKBUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"kb unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"set primary failed: {exc}") from exc


@router.post("/kb/textbook/{doc_id}/main")
def set_main_textbook(
    doc_id: str,
    subject: Optional[str] = Query(default=None),
    title: Optional[str] = Query(default=None),
):
    """将指定 textbook 设为主教材（main）。"""
    try:
        return kb_service.set_primary_textbook(
            doc_id=doc_id,
            subject=subject,
            title=title,
        )
    except MongoKBUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"kb unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"set main textbook failed: {exc}") from exc


@router.get("/kb/rag_search")
def rag_search_from_kb(
    query: str = Query(..., min_length=1),
    subject: Optional[str] = Query(default=None),
    top_k: int = Query(default=5, ge=1, le=20),
    layers: Optional[str] = Query(default=None),
    max_contexts: int = Query(default=12, ge=1, le=50),
):
    """KB 驱动 RAG 检索入口（主链）。"""
    try:
        layer_list = None
        if layers and layers.strip():
            layer_list = [item.strip() for item in layers.split(",") if item.strip()]

        rag_service = KBRAGService(kb_service=kb_service)
        return rag_service.rag_search(
            query=query,
            subject=subject,
            top_k=top_k,
            layers=layer_list,
            max_contexts=max_contexts,
        )
    except MongoKBUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"kb unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"kb rag search failed: {exc}") from exc


# ---------------------------------------------------------------------------
# 旧版 RAG 兼容入口
# ---------------------------------------------------------------------------
def _fallback_search(query: str, top_k: int = 2) -> List[str]:
    """主链不可用时，使用内置示例文本提供兜底检索。"""
    q = (query or "").strip().lower()
    if not q:
        return _FALLBACK_DOCS[:top_k]

    scored = []
    for doc in _FALLBACK_DOCS:
        score = 2 if q in doc.lower() else 0
        score += sum(1 for token in q.split() if token and token in doc.lower())
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def _search_with_engine(query: str, top_k: int = 2) -> Optional[List[str]]:
    """尝试调用旧版 `rag_engine`，失败时返回 None。"""
    global _rag_engine, _rag_error

    if _rag_error:
        return None

    if _rag_engine is None:
        try:
            from rag.rag_engine import RAGEngine

            _rag_engine = RAGEngine()
        except Exception as exc:
            _rag_error = str(exc)
            return None

    try:
        return _rag_engine.search(query, top_k=top_k)
    except Exception as exc:
        _rag_error = str(exc)
        return None


def _parse_layers(layers: Optional[str]) -> Optional[List[str]]:
    """解析逗号分隔 layers 参数。"""
    if not layers:
        return None
    items = [item.strip() for item in str(layers).split(",") if item.strip()]
    return items or None


def _format_kb_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """将 KB-RAG 结果整理为历史兼容返回格式。"""
    contexts = payload.get("contexts") or []
    context_count = int(payload.get("context_count") or len(contexts))
    debug = dict(payload.get("debug") or {})
    debug.setdefault("entrypoint", "/rag_search")
    debug.setdefault("path", "kb_mainline")
    debug.setdefault("engine", "kb_rag")
    retrieval_mode = payload.get("retrieval_mode") or debug.get("retrieval_mode")

    return {
        "query": payload.get("query"),
        "subject": payload.get("subject"),
        "top_k": payload.get("top_k"),
        "layers": payload.get("layers"),
        "counts": payload.get("counts"),
        "answer": payload.get("answer"),
        "contexts": contexts,
        "context_count": context_count,
        "contexts_used": payload.get("contexts_used") or [],
        "retrieval": payload.get("retrieval") or {},
        "results": [str(item.get("text") or "").strip() for item in contexts if str(item.get("text") or "").strip()],
        "engine": "kb_rag",
        "mode": "kb_mainline",
        "retrieval_mode": retrieval_mode,
        "debug": debug,
    }


def _build_fallback_response(
    query: str,
    top_k: int,
    kb_warning: Optional[str],
    subject: Optional[str] = None,
    layers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """构建 `/rag_search` 的兜底响应。"""
    results = _search_with_engine(query=query, top_k=top_k)
    legacy_engine = "rag_engine"
    warnings: List[str] = []

    if kb_warning:
        warnings.append(kb_warning)

    if results is None:
        results = _fallback_search(query=query, top_k=top_k)
        legacy_engine = "fallback_static_docs"
        if _rag_error:
            warnings.append(f"legacy_rag_engine_unavailable: {_rag_error}")

    contexts = [
        {
            "layer": "fallback_demo",
            "doc_id": None,
            "source_file": "legacy_demo_docs",
            "title": f"fallback_doc_{idx + 1}",
            "retrieval_mode": "fallback_demo",
            "vector_score": None,
            "text": text,
        }
        for idx, text in enumerate(results)
    ]
    answer = (
        "当前返回的是兼容兜底结果（非知识库主线）。\n"
        + "\n".join([f"{idx + 1}. {text}" for idx, text in enumerate(results)])
    )

    debug = {
        "entrypoint": "/rag_search",
        "path": "fallback",
        "kb_mainline_available": False,
        "legacy_engine_used": legacy_engine,
        "retrieval_mode": "fallback_demo",
        "kb_error": kb_warning,
        "legacy_engine_error": _rag_error,
    }
    response: Dict[str, Any] = {
        "query": query,
        "subject": subject,
        "top_k": top_k,
        "layers": layers,
        "counts": {},
        "answer": answer,
        "contexts": contexts,
        "context_count": len(contexts),
        "contexts_used": [
            {
                "layer": item["layer"],
                "doc_id": item["doc_id"],
                "source_file": item["source_file"],
                "title": item["title"],
                "retrieval_mode": item["retrieval_mode"],
                "vector_score": item["vector_score"],
            }
            for item in contexts
        ],
        "retrieval": {"fallback": contexts},
        "results": results,
        "engine": "fallback_rag_demo",
        "mode": "fallback",
        "retrieval_mode": "fallback_demo",
        "debug": debug,
    }
    if warnings:
        response["warning"] = " | ".join([item for item in warnings if item])
    return response


@router.get("/rag_search")
def rag_search(
    query: str = Query(..., min_length=1),
    subject: Optional[str] = Query(default=None),
    top_k: int = Query(default=5, ge=1, le=20),
    layers: Optional[str] = Query(default=None),
    max_contexts: int = Query(default=12, ge=1, le=50),
):
    """历史兼容接口：优先走 KB 主链，失败时回落到兜底逻辑。"""
    layer_list = _parse_layers(layers)
    kb_warning: Optional[str] = None

    try:
        rag_service = KBRAGService(kb_service=kb_service)
        payload = rag_service.rag_search(
            query=query,
            subject=subject,
            top_k=top_k,
            layers=layer_list,
            max_contexts=max_contexts,
        )
        return _format_kb_response(payload)
    except Exception as exc:
        kb_warning = f"kb_mainline_failed: {exc}"

    return _build_fallback_response(
        query=query,
        top_k=top_k,
        kb_warning=kb_warning,
        subject=subject,
        layers=layer_list,
    )
