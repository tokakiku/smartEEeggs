from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PageRole(str, Enum):
    # 资源层页面角色枚举：用于教学表达单元分类
    title_page = "title_page"
    outline_page = "outline_page"
    definition_page = "definition_page"
    principle_page = "principle_page"
    formula_page = "formula_page"
    case_page = "case_page"
    application_page = "application_page"
    exercise_page = "exercise_page"
    summary_page = "summary_page"
    history_page = "history_page"
    comparison_page = "comparison_page"
    visual_page = "visual_page"
    other_page = "other_page"


class ResourceInfo(BaseModel):
    # 资源文档基础信息
    resource_id: str
    title: str
    subtitle: Optional[str] = None
    resource_type: str
    source_file: str
    source_type: Optional[str] = None
    subject: Optional[str] = None
    course_topic: Optional[str] = None
    resource_role: Optional[str] = None


class ResourcePage(BaseModel):
    # 页面级资源对象：资源层核心结构
    page_id: str
    resource_id: str
    page_no: int
    page_title: str
    page_text: str
    page_summary: str
    page_role: PageRole = PageRole.other_page
    knowledge_points: List[str] = Field(default_factory=list)
    has_image: bool = False
    has_formula: bool = False
    has_table: bool = False
    has_example: bool = False
    tags: List[str] = Field(default_factory=list)


class ReusableUnit(BaseModel):
    # 可复用教学单元：面向 PPT/教案生成复用
    unit_id: str
    page_id: str
    unit_title: str
    unit_summary: str
    unit_role: PageRole = PageRole.other_page
    knowledge_points: List[str] = Field(default_factory=list)
    reusability: float = 0.5
    recommended_use: List[str] = Field(default_factory=list)


class ResourceChunk(BaseModel):
    # 检索入库前 chunk：保留页面上下文
    chunk_id: str
    resource_id: str
    page_id: str
    page_no: int
    page_title: str
    page_role: PageRole = PageRole.other_page
    text: str
    knowledge_points: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    section: Optional[str] = None


class ResourceRelation(BaseModel):
    # 资源层基础关系
    source: str
    target: str
    relation: str
    confidence: float = 0.8


class ResourceIndex(BaseModel):
    # 资源索引：便于快速按角色和知识点反查页面
    page_roles: Dict[str, List[int]] = Field(default_factory=dict)
    knowledge_point_to_pages: Dict[str, List[int]] = Field(default_factory=dict)


class ResourceExtractionResult(BaseModel):
    # 资源层统一结构化输出
    resource_info: ResourceInfo
    pages: List[ResourcePage] = Field(default_factory=list)
    reusable_units: List[ReusableUnit] = Field(default_factory=list)
    chunks: List[ResourceChunk] = Field(default_factory=list)
    relations: List[ResourceRelation] = Field(default_factory=list)
    resource_index: Optional[ResourceIndex] = None
