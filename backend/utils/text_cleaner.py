import re
from typing import List


def clean_text(text: str) -> str:
    # 基础文本清洗：统一换行、压缩空白、去掉连续空行。
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_paragraphs(text: str) -> List[str]:
    # 将文本按段落切分为非空列表。
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return [line.strip() for line in cleaned.split("\n") if line.strip()]
