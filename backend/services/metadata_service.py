from datetime import datetime, timezone
from typing import Any, Dict

from schema.parsed_document_schema import ParsedDocument


class MetadataService:
    # 通用元数据补全服务

    def append_common_metadata(
        self, parsed_doc: ParsedDocument, structured_output: Dict[str, Any], parser_name: str
    ) -> Dict[str, Any]:
        # 给结构化结果增加统一 metadata
        output = dict(structured_output)
        output.setdefault("metadata", {})
        output["metadata"].update(
            {
                "doc_id": parsed_doc.doc_id,
                "source_file": parsed_doc.file_name,
                "layer": parsed_doc.layer,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "parser_name": parser_name,
            }
        )
        return output
