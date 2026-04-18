from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HotspotEventType(str, Enum):
    # 热点事件类型枚举
    industry_application = "industry_application"
    research_breakthrough = "research_breakthrough"
    product_release = "product_release"
    policy_event = "policy_event"
    social_case = "social_case"
    historical_case = "historical_case"
    other_event = "other_event"


class TeachingUsage(str, Enum):
    # 热点教学用途枚举
    course_intro = "course_intro"
    case_analysis = "case_analysis"
    discussion = "discussion"
    extended_reading = "extended_reading"
    motivation = "motivation"


class HotspotInfo(BaseModel):
    # 热点元信息
    hotspot_id: str
    title: str
    source: Optional[str] = None
    publish_date: Optional[str] = None
    source_type: Optional[str] = None
    url: Optional[str] = None
    author: Optional[str] = None


class HotspotItem(BaseModel):
    # 教学可用的热点案例条目
    hotspot_id: str
    title: str
    summary: str
    event_type: HotspotEventType = HotspotEventType.other_event
    news_role: Optional[str] = None
    related_knowledge_points: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    teaching_usage: List[TeachingUsage] = Field(default_factory=list)
    evidence_snippets: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class HotspotChunk(BaseModel):
    # 热点层 chunk 结构
    chunk_id: str
    hotspot_id: str
    title: str
    text: str
    event_type: HotspotEventType = HotspotEventType.other_event
    related_knowledge_points: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    section: Optional[str] = None


class HotspotRelation(BaseModel):
    # 热点关系结构
    source: str
    target: str
    relation: str
    confidence: float = 0.85


class HotspotIndex(BaseModel):
    # 热点索引结构
    event_type: Dict[str, List[str]] = Field(default_factory=dict)
    knowledge_point_to_hotspots: Dict[str, List[str]] = Field(default_factory=dict)
    teaching_usage: Dict[str, List[str]] = Field(default_factory=dict)


class HotspotExtractionResult(BaseModel):
    # 热点层统一输出
    hotspot_info: HotspotInfo
    hotspot_item: List[HotspotItem] = Field(default_factory=list)
    chunks: List[HotspotChunk] = Field(default_factory=list)
    relations: List[HotspotRelation] = Field(default_factory=list)
    hotspot_index: Optional[HotspotIndex] = None
