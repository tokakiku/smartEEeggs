from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParsedElement(BaseModel):
    # 单个解析元素，来自统一 parser 输出
    element_id: str
    type: str
    text: str
    page_no: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    # 统一中间格式，供所有 extractor 消费
    doc_id: str
    file_name: str
    file_type: str
    layer: str
    source_type: Optional[str] = None
    source_name: Optional[str] = None
    title: Optional[str] = None
    raw_text: str = ""
    elements: List[ParsedElement] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
