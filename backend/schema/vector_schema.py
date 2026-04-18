from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VectorChunkRecord(BaseModel):
    # 统一向量化 chunk 结构，确保可追溯回 Mongo 原文档。
    doc_id: str
    layer: str
    chunk_id: str
    source_file: str
    title: str
    subject: str
    chunk_text: str

    course_name: Optional[str] = None
    course_code: Optional[str] = None
    textbook_role: Optional[str] = None
    is_primary: Optional[bool] = None
    page_no: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    page_role: Optional[str] = None
    publish_date: Optional[str] = None
    event_type: Optional[str] = None

    knowledge_points: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchHit(BaseModel):
    # Milvus 检索返回的标准结构。
    score: float
    doc_id: str
    layer: str
    chunk_id: str
    source_file: str
    title: str
    subject: str
    chunk_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
