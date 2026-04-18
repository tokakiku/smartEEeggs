from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DocumentIngestResponse(BaseModel):
    # 文档统一摄取返回结构
    doc_id: str
    file_name: str
    file_type: str
    layer: str
    parse_status: str
    extract_status: str
    storage_status: str
    structured_output_path: Optional[str] = None
    chunks_output_path: Optional[str] = None
    raw_file_path: Optional[str] = None
    preview: Dict[str, Any] = Field(default_factory=dict)
