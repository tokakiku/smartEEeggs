"""文件类型识别工具。"""

from pathlib import Path
from typing import Optional


SUPPORTED_FILE_TYPES = {"pdf", "pptx", "docx", "html"}


class UnsupportedFileTypeError(ValueError):
    """文件类型不支持时抛出的异常。"""
    pass


def normalize_file_type(file_type: str) -> str:
    """将输入 file_type 统一为小写并处理别名。"""
    normalized = file_type.strip().lower().lstrip(".")
    if normalized == "htm":
        normalized = "html"
    return normalized


def detect_file_type(file_name: str, file_type_hint: Optional[str] = None) -> str:
    """优先使用显式传入 file_type，否则按扩展名识别。"""
    if file_type_hint:
        normalized_hint = normalize_file_type(file_type_hint)
        if normalized_hint not in SUPPORTED_FILE_TYPES:
            raise UnsupportedFileTypeError(f"unsupported file_type: {file_type_hint}")
        return normalized_hint

    suffix = Path(file_name or "").suffix.lower()
    mapping = {
        ".pdf": "pdf",
        ".pptx": "pptx",
        ".docx": "docx",
        ".html": "html",
        ".htm": "html",
    }
    file_type = mapping.get(suffix)
    if not file_type:
        raise UnsupportedFileTypeError(f"unsupported file extension: {suffix or 'unknown'}")
    return file_type
