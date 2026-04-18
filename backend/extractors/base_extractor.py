from abc import ABC, abstractmethod
from typing import Any, Dict

from schema.parsed_document_schema import ParsedDocument

"""抽取器基类定义。"""


class BaseExtractor(ABC):
    """四层 extractor 的统一接口。"""

    @abstractmethod
    def extract(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        """接收 ParsedDocument 并返回结构化 dict。"""
        raise NotImplementedError
