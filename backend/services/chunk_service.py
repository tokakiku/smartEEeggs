import uuid
from typing import Any, Dict, List

from schema.parsed_document_schema import ParsedDocument


class ChunkService:
    # 分块标准化服务：统一输出向量化前的分块结构

    def build_chunks(self, parsed_doc: ParsedDocument, layer: str, structured_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        if layer == "syllabus":
            raw_chunks = self._from_syllabus(parsed_doc, structured_output)
        else:
            raw_chunks = structured_output.get("chunks", [])

        normalized: List[Dict[str, Any]] = []
        for index, chunk in enumerate(raw_chunks, start=1):
            normalized.append(
                {
                    "chunk_id": chunk.get("chunk_id") or f"{layer}-{parsed_doc.doc_id}-{index}",
                    "doc_id": parsed_doc.doc_id,
                    "layer": layer,
                    "source_file": parsed_doc.file_name,
                    "page_no": chunk.get("page_no"),
                    "section": chunk.get("section") or chunk.get("chapter"),
                    "text": chunk.get("text", ""),
                    "knowledge_points": chunk.get("knowledge_points", []),
                    "tags": chunk.get("tags", []),
                    "metadata": chunk.get("metadata", {}),
                }
            )
        return normalized

    def _from_syllabus(self, parsed_doc: ParsedDocument, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 教学大纲层由课程模块和教学目标生成分块
        chunks: List[Dict[str, Any]] = []
        for module in payload.get("course_modules", []):
            text_parts = [
                module.get("module_name", ""),
                module.get("description", ""),
                "；".join(module.get("learning_requirements", [])),
                "；".join(module.get("key_points", [])),
                "；".join(module.get("difficult_points", [])),
            ]
            chunks.append(
                {
                    "chunk_id": f"syl-{uuid.uuid4().hex[:12]}",
                    "section": module.get("module_name"),
                    "text": " ".join([item for item in text_parts if item]).strip(),
                    "knowledge_points": module.get("key_points", []),
                    "tags": ["syllabus", module.get("module_name", "")],
                    "metadata": {"hours": module.get("hours")},
                }
            )

        goals = payload.get("teaching_goals", [])
        if goals:
            chunks.append(
                {
                    "chunk_id": f"syl-{uuid.uuid4().hex[:12]}",
                    "section": "教学目标",
                    "text": "；".join(goals),
                    "knowledge_points": [],
                    "tags": ["syllabus", "教学目标"],
                    "metadata": {},
                }
            )

        if not chunks and parsed_doc.raw_text:
            chunks.append(
                {
                    "chunk_id": f"syl-{uuid.uuid4().hex[:12]}",
                    "section": "全文",
                    "text": parsed_doc.raw_text,
                    "knowledge_points": [],
                    "tags": ["syllabus"],
                    "metadata": {},
                }
            )
        return chunks
