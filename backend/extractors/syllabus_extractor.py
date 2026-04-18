from typing import Any, Dict

"""教学大纲层抽取适配器（复用规则抽取器实现）。"""

from extractors.base_extractor import BaseExtractor
from schema.parsed_document_schema import ParsedDocument
from services.syllabus_extractor import SyllabusExtractor as RuleSyllabusExtractor


class SyllabusExtractor(BaseExtractor):
    """教学大纲层抽取器，复用既有规则抽取器。"""
    # 教学大纲层抽取器，复用已有规则提取器

    def __init__(self) -> None:
        self._rule_extractor = RuleSyllabusExtractor()

    def extract(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        return self._rule_extractor.extract(parsed_doc)
