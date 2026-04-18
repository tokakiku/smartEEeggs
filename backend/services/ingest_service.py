import os
from typing import Any, Dict, Optional

"""统一入库编排服务。

负责文档解析、分层抽取、分块、关系构建、落盘与 Mongo 写入，
并在不影响主流程的前提下衔接向量化步骤。
"""

from extractors.hotspot_extractor import HotspotExtractor
from extractors.resource_extractor import ResourceExtractor
from extractors.syllabus_extractor import SyllabusExtractor
from extractors.textbook_extractor import TextbookExtractor
from parsers.file_type_utils import SUPPORTED_FILE_TYPES, UnsupportedFileTypeError, detect_file_type
from parsers.parser_factory import get_parser
from schema.document_ingest_schema import DocumentIngestResponse
from schema.parsed_document_schema import ParsedDocument
from services.chunk_service import ChunkService
from services.metadata_service import MetadataService
from services.mongo_kb_service import MongoKBService, MongoKBUnavailableError
from services.relation_service import RelationService
from services.storage_service import StorageService
from services.tag_service import TagService


SUPPORTED_LAYERS = {"syllabus", "textbook", "resource", "hotspot"}
HOTSPOT_STATIC_ALLOWED_TYPES = {"html", "pdf", "docx"}


class IngestValidationError(ValueError):
    """统一入库流程中的可预期校验错误。"""
    # 统一摄取过程中的可预期校验错误
    pass


class DocumentIngestService:
    """统一文档入库编排服务。"""
    # 统一文档摄取编排服务

    def __init__(
        self,
        storage_service: Optional[StorageService] = None,
        kb_service: Optional[MongoKBService] = None,
    ) -> None:
        self.storage_service = storage_service or StorageService()
        self.chunk_service = ChunkService()
        self.tag_service = TagService()
        self.relation_service = RelationService()
        self.metadata_service = MetadataService()
        self.kb_service = kb_service or MongoKBService.from_env()

        # 热点层当前主流程仅支持静态新闻源；动态网页留待后续专用 parser。
        self.hotspot_static_only = self._is_true(os.getenv("HOTSPOT_STATIC_ONLY"), True)
        self.hotspot_allowed_domains = self._split_csv_env(
            os.getenv("HOTSPOT_STATIC_ALLOWED_DOMAINS"),
            default=["news.aibase.com", "www.reuters.com", "reuters.com"],
        )
        self.hotspot_blocked_domains = self._split_csv_env(
            os.getenv("HOTSPOT_DYNAMIC_BLOCKED_DOMAINS"),
            default=["caip.org.cn", "www.caip.org.cn"],
        )

    def ingest_document(
        self,
        file_bytes: bytes,
        file_name: str,
        layer: str,
        source_type: Optional[str] = None,
        source_name: Optional[str] = None,
        file_type_hint: Optional[str] = None,
    ) -> DocumentIngestResponse:
        """完整入库流程：解析 -> 抽取 -> 后处理 -> 落盘/入库。"""
        # 完整摄取流程：解析 -> 抽取 -> 后处理 -> 落盘
        if not file_bytes:
            raise IngestValidationError("empty file")

        normalized_layer = self._normalize_layer(layer)
        file_type = self._detect_file_type(file_name, file_type_hint)
        self._validate_hotspot_source_policy(
            layer=normalized_layer,
            file_type=file_type,
            source_type=source_type,
            source_name=source_name,
            file_name=file_name,
        )
        parsed_doc = self.parse_only(
            file_bytes=file_bytes,
            file_name=file_name,
            layer=normalized_layer,
            source_type=source_type,
            source_name=source_name,
            file_type_hint=file_type,
        )

        if not parsed_doc.raw_text.strip() and not parsed_doc.elements:
            raise IngestValidationError("no readable text")

        extractor = self._get_extractor(normalized_layer)
        structured_output = extractor.extract(parsed_doc)
        relations = self.relation_service.build_relations(normalized_layer, structured_output)
        chunks = self.chunk_service.build_chunks(parsed_doc, normalized_layer, structured_output)
        chunks = self.tag_service.enrich_chunks(normalized_layer, parsed_doc.file_type, chunks)

        # ── 向量化步骤：对 chunks 生成 embedding 并写入本地 FAISS 索引 ──────────
        # 独立于主流程，失败时只记录 warning，不中断 ingest。
        added_vectors = 0
        vector_status = "disabled"
        try:
            from services.vector_index_service import get_vector_index_service
            vec_service = get_vector_index_service()
            # 给每个 chunk 补充 title 字段（用于元数据展示）
            chunks_for_index = [
                {**chunk, "title": parsed_doc.title or parsed_doc.file_name}
                for chunk in chunks
            ]
            added_vectors = vec_service.add_chunks(chunks_for_index)
            vector_status = f"ok: {added_vectors} vectors added"
        except Exception as vec_exc:
            vector_status = f"warning: {vec_exc}"
        # ─────────────────────────────────────────────────────────────────────────

        parser_name = str(parsed_doc.metadata.get("parser_name", "unstructured"))
        structured_output = self.metadata_service.append_common_metadata(parsed_doc, structured_output, parser_name)

        raw_file_path = self.storage_service.save_raw_file(file_bytes, file_name, parsed_doc.doc_id)
        structured_output_path = self.storage_service.save_structured_output(
            doc_id=parsed_doc.doc_id,
            layer=normalized_layer,
            payload=structured_output,
        )
        chunks_output_path = self.storage_service.save_chunks_output(
            doc_id=parsed_doc.doc_id,
            layer=normalized_layer,
            chunks=chunks,
        )
        self.storage_service.save_relations_output(
            doc_id=parsed_doc.doc_id,
            layer=normalized_layer,
            relations=relations,
        )

        mongo_status = "disabled"
        mongo_collection: Optional[str] = None
        mongo_backend: Optional[str] = None
        try:
            if self.kb_service and self.kb_service.enabled:
                save_result = self.kb_service.save_extraction_result(
                    layer=normalized_layer,
                    payload=structured_output,
                    metadata={
                        "doc_id": parsed_doc.doc_id,
                        "source_file": parsed_doc.file_name,
                        "source_type": parsed_doc.source_type,
                        "parser_name": parser_name,
                    },
                )
                mongo_status = save_result.get("status", "success")
                mongo_collection = save_result.get("collection")
                mongo_backend = save_result.get("backend")
        except MongoKBUnavailableError as exc:
            if self.kb_service and self.kb_service.required:
                raise IngestValidationError(f"mongodb unavailable: {exc}") from exc
            mongo_status = f"unavailable: {exc}"
        except Exception as exc:
            if self.kb_service and self.kb_service.required:
                raise IngestValidationError(f"mongodb write failed: {exc}") from exc
            mongo_status = f"error: {exc}"

        return DocumentIngestResponse(
            doc_id=parsed_doc.doc_id,
            file_name=file_name,
            file_type=parsed_doc.file_type,
            layer=normalized_layer,
            parse_status="success",
            extract_status="success",
            storage_status="success",
            structured_output_path=structured_output_path,
            chunks_output_path=chunks_output_path,
            raw_file_path=raw_file_path,
            preview={
                "title": parsed_doc.title,
                "parser_name": parser_name,
                "element_count": len(parsed_doc.elements),
                "chunk_count": len(chunks),
                "relation_count": len(relations),
                "structured_keys": list(structured_output.keys()),
                "mongo_status": mongo_status,
                "mongo_collection": mongo_collection,
                "mongo_backend": mongo_backend,
                "vector_status": vector_status,
                "added_vectors": added_vectors,
            },
        )

    def parse_only(
        self,
        file_bytes: bytes,
        file_name: str,
        layer: str,
        source_type: Optional[str] = None,
        source_name: Optional[str] = None,
        file_type_hint: Optional[str] = None,
    ) -> ParsedDocument:
        """仅解析，不执行后续抽取与存储。"""
        # 仅解析，不进行后续抽取与存储
        if not file_bytes:
            raise IngestValidationError("empty file")

        normalized_layer = self._normalize_layer(layer)
        file_type = self._detect_file_type(file_name, file_type_hint)
        self._validate_hotspot_source_policy(
            layer=normalized_layer,
            file_type=file_type,
            source_type=source_type,
            source_name=source_name,
            file_name=file_name,
        )
        parser = get_parser(file_type)
        parsed_doc = parser.parse(
            file_bytes=file_bytes,
            file_name=file_name,
            layer=normalized_layer,
            source_type=source_type,
            source_name=source_name,
        )
        if not parsed_doc.raw_text.strip() and not parsed_doc.elements:
            raise IngestValidationError("no readable text")
        return parsed_doc

    def _normalize_layer(self, layer: str) -> str:
        """统一处理 layer 大小写并校验合法性。"""
        # 统一处理 layer 大小写并检查合法性
        normalized = (layer or "").strip().lower()
        if normalized not in SUPPORTED_LAYERS:
            raise IngestValidationError(f"unsupported layer: {layer}")
        return normalized

    def _detect_file_type(self, file_name: str, file_type_hint: Optional[str]) -> str:
        """统一封装文件类型识别，并转换为业务异常。"""
        # 统一封装文件类型识别，转为业务异常
        try:
            return detect_file_type(file_name=file_name, file_type_hint=file_type_hint)
        except UnsupportedFileTypeError as exc:
            raise IngestValidationError(str(exc)) from exc

    def _get_extractor(self, layer: str):
        """层级到抽取器的分发映射。"""
        # 层级到抽取器的分发映射
        mapping = {
            "syllabus": SyllabusExtractor(),
            "textbook": TextbookExtractor(),
            "resource": ResourceExtractor(),
            "hotspot": HotspotExtractor(),
        }
        extractor = mapping.get(layer)
        if extractor is None:
            raise IngestValidationError(f"unsupported layer: {layer}")
        return extractor

    @staticmethod
    def get_supported_layers() -> list[str]:
        """返回系统当前支持的层级。"""
        # 返回系统当前支持的层级
        return ["syllabus", "textbook", "resource", "hotspot"]

    @staticmethod
    def get_supported_file_types() -> list[str]:
        """返回系统当前支持的文件类型。"""
        # 返回系统当前支持的文件类型
        return ["pdf", "pptx", "docx", "html"]

    def _validate_hotspot_source_policy(
        self,
        layer: str,
        file_type: str,
        source_type: Optional[str],
        source_name: Optional[str],
        file_name: str,
    ) -> None:
        """热点层静态源校验：动态网页默认不进入主流程。"""
        # 热点层静态源限制：当前 demo 主流程明确不纳入动态网页。
        if layer != "hotspot":
            return

        if file_type not in HOTSPOT_STATIC_ALLOWED_TYPES:
            raise IngestValidationError(
                "hotspot layer currently supports static sources only: html/pdf/docx"
            )
        if not self.hotspot_static_only:
            return

        normalized_source_type = (source_type or "").strip().lower()
        if normalized_source_type in {"dynamic_web", "dynamic_page", "dynamic_url"}:
            raise IngestValidationError(
                "hotspot layer currently excludes dynamic webpages from main flow; "
                "please use static article html/pdf/markdown snapshot"
            )

        domain = MongoKBService.extract_domain(source_name)
        if domain and domain in self.hotspot_blocked_domains:
            raise IngestValidationError(
                "hotspot layer currently excludes dynamic webpage sources from main flow; "
                "please use static article html/pdf/markdown snapshot"
            )

        if domain and normalized_source_type in {"web", "url", "web_url"}:
            if self.hotspot_allowed_domains and domain not in self.hotspot_allowed_domains:
                raise IngestValidationError(
                    f"hotspot static source domain not allowed in current V1: {domain}; "
                    "please provide local static article snapshot or configured static domain"
                )

        lowered_name = (file_name or "").lower()
        if "caip" in lowered_name and normalized_source_type in {"web", "url", "web_url"}:
            raise IngestValidationError(
                "hotspot layer currently excludes dynamic webpage sources from main flow; "
                "please use static article html/pdf/markdown snapshot"
            )

    @staticmethod
    def _split_csv_env(value: Optional[str], default: list[str]) -> list[str]:
        if not value:
            return default
        return [item.strip().lower() for item in value.split(",") if item.strip()]

    @staticmethod
    def _is_true(value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
