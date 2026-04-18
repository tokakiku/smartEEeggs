from __future__ import annotations

"""Milvus 服务封装。

提供集合初始化、向量写入、检索与追溯查询能力。
"""

import hashlib
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _to_str(value: Any) -> str:
    return str(value or "").strip()


class MilvusServiceError(RuntimeError):
    pass


class MilvusUnavailableError(MilvusServiceError):
    pass


class MilvusService:
    """最小 Milvus 封装。

    - 创建 collection / schema / index
    - 按 chunk upsert 向量
    - query/search 支持 layer/doc_id 过滤
    """

    FIELD_PK = "pk"
    FIELD_EMBEDDING = "embedding"
    FIELD_DOC_ID = "doc_id"
    FIELD_CHUNK_ID = "chunk_id"
    FIELD_LAYER = "layer"
    FIELD_SOURCE_FILE = "source_file"
    FIELD_TITLE = "title"
    FIELD_SUBJECT = "subject"
    FIELD_CHUNK_TEXT = "chunk_text"
    FIELD_COURSE_NAME = "course_name"
    FIELD_COURSE_CODE = "course_code"
    FIELD_TEXTBOOK_ROLE = "textbook_role"
    FIELD_IS_PRIMARY = "is_primary"
    FIELD_PAGE_NO = "page_no"
    FIELD_CHAPTER = "chapter"
    FIELD_SECTION = "section"
    FIELD_PAGE_ROLE = "page_role"
    FIELD_PUBLISH_DATE = "publish_date"
    FIELD_EVENT_TYPE = "event_type"
    FIELD_METADATA = "metadata"

    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        metric_type: Optional[str] = None,
        index_type: Optional[str] = None,
    ) -> None:
        self.uri = uri or os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
        self.token = token if token is not None else os.getenv("MILVUS_TOKEN", "")
        self.db_name = db_name if db_name is not None else os.getenv("MILVUS_DB_NAME", "default")
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "ruijie_kb_chunks")
        self.metric_type = metric_type or os.getenv("MILVUS_METRIC_TYPE", "COSINE")
        self.index_type = index_type or os.getenv("MILVUS_INDEX_TYPE", "AUTOINDEX")

        self.max_len_doc_id = int(os.getenv("MILVUS_MAX_LEN_DOC_ID", "256"))
        self.max_len_chunk_id = int(os.getenv("MILVUS_MAX_LEN_CHUNK_ID", "320"))
        self.max_len_layer = int(os.getenv("MILVUS_MAX_LEN_LAYER", "64"))
        self.max_len_source = int(os.getenv("MILVUS_MAX_LEN_SOURCE_FILE", "512"))
        self.max_len_title = int(os.getenv("MILVUS_MAX_LEN_TITLE", "512"))
        self.max_len_subject = int(os.getenv("MILVUS_MAX_LEN_SUBJECT", "256"))
        self.max_len_chunk_text = int(os.getenv("MILVUS_MAX_LEN_CHUNK_TEXT", "30000"))
        self.max_len_meta = int(os.getenv("MILVUS_MAX_LEN_META_TEXT", "1024"))
        self.max_len_pk = int(os.getenv("MILVUS_MAX_LEN_PK", "512"))
        if self.max_len_chunk_text >= 65535:
            default_safe_chunk_len = 65000
        elif self.max_len_chunk_text >= 30000:
            default_safe_chunk_len = 29500
        else:
            default_safe_chunk_len = max(1, self.max_len_chunk_text - 100)
        self.safe_chunk_text_len = int(
            os.getenv("MILVUS_SAFE_CHUNK_TEXT_LEN", str(default_safe_chunk_len))
        )
        self.safe_chunk_text_len = min(self.safe_chunk_text_len, self.max_len_chunk_text)

        self._client = None
        self._truncated_chunk_count = 0

    def _get_client(self):
        """懒加载 MilvusClient，并在首次调用时做连通性探测。"""
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient
        except Exception as exc:
            raise MilvusUnavailableError("pymilvus is not installed") from exc

        kwargs: Dict[str, Any] = {}
        if self.token:
            kwargs["token"] = self.token
        if self.db_name:
            kwargs["db_name"] = self.db_name
        try:
            self._client = MilvusClient(uri=self.uri, **kwargs)
            # 触发一次轻量调用，尽早暴露连接错误。
            self._client.has_collection(self.collection_name)
        except Exception as exc:
            raise MilvusUnavailableError(f"cannot connect milvus: {exc}") from exc
        return self._client

    def has_collection(self) -> bool:
        """检查目标 collection 是否存在。"""
        client = self._get_client()
        return bool(client.has_collection(self.collection_name))

    def ensure_collection(self, vector_dim: int, drop_existing: bool = False) -> Dict[str, Any]:
        """确保 collection 可用；必要时按新 schema 重建。"""
        client = self._get_client()
        if drop_existing and client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)

        if not client.has_collection(self.collection_name):
            self._create_collection(vector_dim=vector_dim)
            created = True
        else:
            created = False

        try:
            client.load_collection(self.collection_name)
        except Exception:
            # 某些环境首轮 load 可能由 search 触发，这里允许忽略。
            pass
        stats = self.get_stats()
        stats["created"] = created
        return stats

    def _create_collection(self, vector_dim: int) -> None:
        """创建 collection 与索引定义。"""
        client = self._get_client()
        try:
            from pymilvus import DataType, MilvusClient
        except Exception as exc:
            raise MilvusUnavailableError("pymilvus import failed while creating collection") from exc

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(self.FIELD_PK, datatype=DataType.VARCHAR, is_primary=True, max_length=self.max_len_pk)
        schema.add_field(self.FIELD_DOC_ID, datatype=DataType.VARCHAR, max_length=self.max_len_doc_id)
        schema.add_field(self.FIELD_CHUNK_ID, datatype=DataType.VARCHAR, max_length=self.max_len_chunk_id)
        schema.add_field(self.FIELD_LAYER, datatype=DataType.VARCHAR, max_length=self.max_len_layer)
        schema.add_field(self.FIELD_SOURCE_FILE, datatype=DataType.VARCHAR, max_length=self.max_len_source)
        schema.add_field(self.FIELD_TITLE, datatype=DataType.VARCHAR, max_length=self.max_len_title)
        schema.add_field(self.FIELD_SUBJECT, datatype=DataType.VARCHAR, max_length=self.max_len_subject)
        schema.add_field(self.FIELD_CHUNK_TEXT, datatype=DataType.VARCHAR, max_length=self.max_len_chunk_text)
        schema.add_field(self.FIELD_COURSE_NAME, datatype=DataType.VARCHAR, max_length=self.max_len_subject)
        schema.add_field(self.FIELD_COURSE_CODE, datatype=DataType.VARCHAR, max_length=self.max_len_subject)
        schema.add_field(self.FIELD_TEXTBOOK_ROLE, datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(self.FIELD_IS_PRIMARY, datatype=DataType.BOOL)
        schema.add_field(self.FIELD_PAGE_NO, datatype=DataType.INT64)
        schema.add_field(self.FIELD_CHAPTER, datatype=DataType.VARCHAR, max_length=self.max_len_title)
        schema.add_field(self.FIELD_SECTION, datatype=DataType.VARCHAR, max_length=self.max_len_title)
        schema.add_field(self.FIELD_PAGE_ROLE, datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(self.FIELD_PUBLISH_DATE, datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(self.FIELD_EVENT_TYPE, datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(self.FIELD_METADATA, datatype=DataType.JSON)
        schema.add_field(self.FIELD_EMBEDDING, datatype=DataType.FLOAT_VECTOR, dim=int(vector_dim))

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=self.FIELD_EMBEDDING,
            index_type=self.index_type,
            metric_type=self.metric_type,
        )
        # 标量索引：提升按 layer/doc_id 的过滤查询效率。
        index_params.add_index(field_name=self.FIELD_LAYER, index_type="INVERTED")
        index_params.add_index(field_name=self.FIELD_DOC_ID, index_type="INVERTED")

        client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert_chunks(
        self,
        chunks: Sequence[Dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 200,
    ) -> Dict[str, Any]:
        """批量写入或更新 chunk 向量。"""
        if len(chunks) != len(embeddings):
            raise MilvusServiceError("chunks and embeddings length mismatch")
        if not chunks:
            return {"upserted": 0}

        client = self._get_client()
        total = 0
        for start in range(0, len(chunks), max(1, int(batch_size))):
            end = start + max(1, int(batch_size))
            rows = [
                self._to_milvus_row(chunk=chunks[idx], vector=embeddings[idx])
                for idx in range(start, min(end, len(chunks)))
            ]
            result = client.upsert(collection_name=self.collection_name, data=rows)
            total += int(result.get("upsert_count") or result.get("insert_count") or len(rows))

        client.flush(self.collection_name)
        return {"upserted": total}

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        layer: Optional[str] = None,
        doc_id: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """向量检索接口，可按 layer/doc_id 过滤。"""
        client = self._get_client()
        client.load_collection(self.collection_name)
        expr = self._build_filter(layer=layer, doc_id=doc_id)
        fields = output_fields or self.default_output_fields()
        results = client.search(
            collection_name=self.collection_name,
            data=[list(query_vector)],
            filter=expr,
            limit=max(1, int(top_k)),
            output_fields=fields,
        )
        hits = results[0] if results else []
        return [self._normalize_hit(item) for item in hits]

    def query_by_doc_id(self, doc_id: str, layer: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """按 doc_id 回查 Milvus 记录。"""
        client = self._get_client()
        client.load_collection(self.collection_name)
        expr = self._build_filter(layer=layer, doc_id=doc_id)
        rows = client.query(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=self.default_output_fields(),
            limit=max(1, int(limit)),
        )
        return [dict(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """获取集合与索引统计信息。"""
        client = self._get_client()
        exists = bool(client.has_collection(self.collection_name))
        if not exists:
            return {
                "collection_name": self.collection_name,
                "exists": False,
                "uri": self.uri,
                "db_name": self.db_name,
            }

        stats = client.get_collection_stats(self.collection_name)
        desc = client.describe_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "exists": True,
            "uri": self.uri,
            "db_name": self.db_name,
            "stats": stats,
            "description": desc,
        }

    def default_output_fields(self) -> List[str]:
        """返回检索默认输出字段。"""
        return [
            self.FIELD_DOC_ID,
            self.FIELD_CHUNK_ID,
            self.FIELD_LAYER,
            self.FIELD_SOURCE_FILE,
            self.FIELD_TITLE,
            self.FIELD_SUBJECT,
            self.FIELD_CHUNK_TEXT,
            self.FIELD_COURSE_NAME,
            self.FIELD_COURSE_CODE,
            self.FIELD_TEXTBOOK_ROLE,
            self.FIELD_IS_PRIMARY,
            self.FIELD_PAGE_NO,
            self.FIELD_CHAPTER,
            self.FIELD_SECTION,
            self.FIELD_PAGE_ROLE,
            self.FIELD_PUBLISH_DATE,
            self.FIELD_EVENT_TYPE,
            self.FIELD_METADATA,
        ]

    def _to_milvus_row(self, chunk: Dict[str, Any], vector: Sequence[float]) -> Dict[str, Any]:
        doc_id = self._truncate(_to_str(chunk.get(self.FIELD_DOC_ID) or chunk.get("doc_id")), self.max_len_doc_id)
        chunk_id = self._truncate(_to_str(chunk.get(self.FIELD_CHUNK_ID) or chunk.get("chunk_id")), self.max_len_chunk_id)
        layer = self._truncate(_to_str(chunk.get(self.FIELD_LAYER) or chunk.get("layer")), self.max_len_layer)
        source_file = self._truncate(
            _to_str(chunk.get(self.FIELD_SOURCE_FILE) or chunk.get("source_file")),
            self.max_len_source,
        )
        title = self._truncate(_to_str(chunk.get(self.FIELD_TITLE) or chunk.get("title")), self.max_len_title)
        subject = self._truncate(_to_str(chunk.get(self.FIELD_SUBJECT) or chunk.get("subject")), self.max_len_subject)
        chunk_text = self._normalize_chunk_text(
            value=_to_str(chunk.get(self.FIELD_CHUNK_TEXT) or chunk.get("chunk_text") or chunk.get("text")),
            layer=layer,
            doc_id=doc_id,
            chunk_id=chunk_id,
        )

        metadata = {
            "knowledge_points": chunk.get("knowledge_points") or [],
            "tags": chunk.get("tags") or [],
            "metadata": chunk.get("metadata") or {},
        }

        return {
            self.FIELD_PK: self.build_primary_key(layer=layer, doc_id=doc_id, chunk_id=chunk_id),
            self.FIELD_DOC_ID: doc_id,
            self.FIELD_CHUNK_ID: chunk_id,
            self.FIELD_LAYER: layer,
            self.FIELD_SOURCE_FILE: source_file,
            self.FIELD_TITLE: title,
            self.FIELD_SUBJECT: subject,
            self.FIELD_CHUNK_TEXT: chunk_text,
            self.FIELD_COURSE_NAME: self._truncate(_to_str(chunk.get(self.FIELD_COURSE_NAME)), self.max_len_subject),
            self.FIELD_COURSE_CODE: self._truncate(_to_str(chunk.get(self.FIELD_COURSE_CODE)), self.max_len_subject),
            self.FIELD_TEXTBOOK_ROLE: self._truncate(_to_str(chunk.get(self.FIELD_TEXTBOOK_ROLE)), 64),
            self.FIELD_IS_PRIMARY: bool(chunk.get(self.FIELD_IS_PRIMARY)) if chunk.get(self.FIELD_IS_PRIMARY) is not None else False,
            self.FIELD_PAGE_NO: int(chunk.get(self.FIELD_PAGE_NO)) if chunk.get(self.FIELD_PAGE_NO) is not None else -1,
            self.FIELD_CHAPTER: self._truncate(_to_str(chunk.get(self.FIELD_CHAPTER)), self.max_len_title),
            self.FIELD_SECTION: self._truncate(_to_str(chunk.get(self.FIELD_SECTION)), self.max_len_title),
            self.FIELD_PAGE_ROLE: self._truncate(_to_str(chunk.get(self.FIELD_PAGE_ROLE)), 64),
            self.FIELD_PUBLISH_DATE: self._truncate(_to_str(chunk.get(self.FIELD_PUBLISH_DATE)), 64),
            self.FIELD_EVENT_TYPE: self._truncate(_to_str(chunk.get(self.FIELD_EVENT_TYPE)), 64),
            self.FIELD_METADATA: metadata,
            self.FIELD_EMBEDDING: list(vector),
        }

    def _normalize_chunk_text(self, value: str, layer: str, doc_id: str, chunk_id: str) -> str:
        """入库前文本兜底截断，避免超长字段写入失败。"""
        chunk_text = str(value or "")
        if len(chunk_text) > self.safe_chunk_text_len:
            self._truncated_chunk_count += 1
            print(
                f"[WARN] milvus chunk_text truncated "
                f"(layer={layer}, doc_id={doc_id}, chunk_id={chunk_id}, "
                f"original_len={len(chunk_text)}, safe_len={self.safe_chunk_text_len}, "
                f"schema_len={self.max_len_chunk_text})"
            )
            chunk_text = chunk_text[: self.safe_chunk_text_len]
        if len(chunk_text) > self.max_len_chunk_text:
            chunk_text = chunk_text[: self.max_len_chunk_text]
        return chunk_text

    def truncation_stats(self) -> Dict[str, int]:
        """返回截断统计信息。"""
        return {
            "truncated_chunks": int(self._truncated_chunk_count),
            "safe_chunk_text_len": int(self.safe_chunk_text_len),
            "schema_chunk_text_len": int(self.max_len_chunk_text),
        }

    @classmethod
    def build_primary_key(cls, layer: str, doc_id: str, chunk_id: str) -> str:
        """构建稳定且可读的主键。"""
        raw = f"{layer}:{doc_id}:{chunk_id}"
        compact = re.sub(r"[^0-9A-Za-z_\-:.]+", "-", raw).strip("-")
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        if len(compact) > 480:
            compact = compact[:480]
        return f"{compact}:{digest}" if compact else digest

    def _build_filter(self, layer: Optional[str], doc_id: Optional[str]) -> str:
        """构建 Milvus 过滤表达式。"""
        clauses: List[str] = []
        if _to_str(layer):
            clauses.append(f"{self.FIELD_LAYER} == '{self._escape_filter_value(_to_str(layer))}'")
        if _to_str(doc_id):
            clauses.append(f"{self.FIELD_DOC_ID} == '{self._escape_filter_value(_to_str(doc_id))}'")
        return " and ".join(clauses)

    @staticmethod
    def _escape_filter_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @staticmethod
    def _truncate(value: str, max_len: int) -> str:
        if len(value) <= max_len:
            return value
        return value[: max(0, int(max_len))]

    def _normalize_hit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """统一 Milvus 命中结构（entity/score 字段兼容）。"""
        entity = item.get("entity") if isinstance(item.get("entity"), dict) else {}
        payload = dict(entity) if entity else {k: v for k, v in item.items() if k not in {"id", "distance", "score"}}
        payload["score"] = float(item.get("distance") if item.get("distance") is not None else item.get("score") or 0.0)
        payload[self.FIELD_PK] = item.get("id", payload.get(self.FIELD_PK))
        return payload

