from typing import Any, Dict, List


class RelationService:
    # 基础关系服务：构建轻量关系结果，便于后续接 GraphRAG

    def build_relations(self, layer: str, structured_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 优先使用 extractor 已提供关系
        if isinstance(structured_output.get("relations"), list):
            return structured_output["relations"]

        if layer == "syllabus":
            return self._build_syllabus_relations(structured_output)
        if layer == "textbook":
            return self._build_textbook_relations(structured_output)
        if layer == "resource":
            return self._build_resource_relations(structured_output)
        if layer == "hotspot":
            return self._build_hotspot_relations(structured_output)
        return []

    def _build_syllabus_relations(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 课程模块到知识点关系：module -> contains -> knowledge_point
        relations: List[Dict[str, Any]] = []
        for module in payload.get("course_modules", []):
            module_name = module.get("module_name", "")
            for point in module.get("key_points", []):
                relations.append(
                    {
                        "source": module_name,
                        "target": point,
                        "relation": "contains",
                        "confidence": 0.9,
                    }
                )
        return relations

    def _build_textbook_relations(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 章节到小节关系：chapter -> contains -> section
        relations: List[Dict[str, Any]] = []
        for chapter in payload.get("chapters", []):
            chapter_title = chapter.get("chapter_title", "")
            for section in chapter.get("sections", []):
                relations.append(
                    {
                        "source": chapter_title,
                        "target": section.get("section_title", ""),
                        "relation": "contains",
                        "confidence": 0.95,
                    }
                )
        return relations

    def _build_resource_relations(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 资源页到知识点关系：resource_page -> related_to -> knowledge_point
        relations: List[Dict[str, Any]] = []
        for unit in payload.get("reusable_units", []):
            unit_id = unit.get("unit_id", "")
            for point in unit.get("knowledge_points", []):
                relations.append(
                    {
                        "source": unit_id,
                        "target": point,
                        "relation": "related_to",
                        "confidence": 0.88,
                    }
                )
        return relations

    def _build_hotspot_relations(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 热点到知识点关系：hotspot -> related_to -> knowledge_point
        relations: List[Dict[str, Any]] = []
        hotspot_title = payload.get("hotspot_info", {}).get("title", "hotspot")
        for point in payload.get("related_knowledge_points", []):
            relations.append(
                {
                    "source": hotspot_title,
                    "target": point,
                    "relation": "related_to",
                    "confidence": 0.85,
                }
            )
        return relations
