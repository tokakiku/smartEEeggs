from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TextbookInfo(BaseModel):
    # 教材基础信息，供后续检索和追溯来源
    book_title: Optional[str] = None
    source_file: Optional[str] = None
    source_type: Optional[str] = None
    subject: Optional[str] = None
    textbook_role: Optional[str] = None
    priority_score: Optional[float] = None
    edition: Optional[str] = None
    academic_year: Optional[str] = None
    authors: List[str] = Field(default_factory=list)


class TextbookSection(BaseModel):
    # 小节结构，挂在章节下并同步输出平铺 sections
    section_id: str
    chapter_id: str
    section_index: str
    section_title: str
    raw_text: str = ""
    knowledge_points: List[str] = Field(default_factory=list)


class TextbookChapter(BaseModel):
    # 章节结构，作为教材层主干目录
    chapter_id: str
    chapter_index: str
    chapter_title: str
    raw_text: str = ""
    sections: List[TextbookSection] = Field(default_factory=list)


class TextbookKnowledgePoint(BaseModel):
    # 知识点结构，保留所属章节和小节上下文
    kp_id: str
    name: str
    chapter_id: Optional[str] = None
    section_id: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    source_text: str = ""


class TextbookChunk(BaseModel):
    # 教材层 chunk，供向量化与检索使用
    chunk_id: str
    doc_id: str
    chapter_id: str
    section_id: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    text: str
    knowledge_points: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextbookRelation(BaseModel):
    # 轻量关系边，后续可直接迁移到图数据库
    source: str
    target: str
    relation: str
    confidence: float = 0.8


class TextbookExtractionResult(BaseModel):
    # 教材层统一结构化输出
    textbook_info: TextbookInfo
    chapters: List[TextbookChapter] = Field(default_factory=list)
    sections: List[TextbookSection] = Field(default_factory=list)
    knowledge_points: List[TextbookKnowledgePoint] = Field(default_factory=list)
    chunks: List[TextbookChunk] = Field(default_factory=list)
    relations: List[TextbookRelation] = Field(default_factory=list)
