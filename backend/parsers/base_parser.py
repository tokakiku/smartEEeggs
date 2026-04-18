from abc import ABC, abstractmethod
from typing import Optional

from schema.parsed_document_schema import ParsedDocument

"""解析器基类定义。"""


class BaseParser(ABC):
    """统一 parser 抽象接口。"""

    @abstractmethod
    def parse(
        self,
        file_bytes: bytes,
        file_name: str,
        layer: str,
        source_type: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> ParsedDocument:
        """将原始文件解析为 ParsedDocument。"""
        raise NotImplementedError
