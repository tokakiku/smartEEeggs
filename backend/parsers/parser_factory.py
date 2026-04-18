"""解析器工厂。"""

from parsers.base_parser import BaseParser
from parsers.file_type_utils import SUPPORTED_FILE_TYPES, UnsupportedFileTypeError
from parsers.unstructured_parser import UnstructuredParser


def get_parser(file_type: str) -> BaseParser:
    """根据文件类型返回解析器实例。"""
    # 当前统一使用 unstructured parser，后续可按 file_type 扩展专用 parser。
    if file_type not in SUPPORTED_FILE_TYPES:
        raise UnsupportedFileTypeError(f"unsupported file_type: {file_type}")
    return UnstructuredParser()
