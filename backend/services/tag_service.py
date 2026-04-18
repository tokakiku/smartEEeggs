from typing import Dict, List


class TagService:
    # 标签服务：对 chunk 进行轻量标签增强

    def enrich_chunks(self, layer: str, file_type: str, chunks: List[Dict]) -> List[Dict]:
        enriched: List[Dict] = []
        for chunk in chunks:
            tags = list(chunk.get("tags", []))
            tags.extend([layer, file_type])

            section = chunk.get("section")
            if section:
                tags.append(str(section))

            for point in chunk.get("knowledge_points", []):
                tags.append(str(point))

            metadata = chunk.get("metadata", {})
            page_role = metadata.get("page_role")
            if page_role:
                tags.append(str(page_role))

            for bool_key in ["has_image", "has_formula", "has_example"]:
                if bool(metadata.get(bool_key)):
                    tags.append(bool_key)

            text = str(chunk.get("text", ""))
            tags.append(self._infer_difficulty(text))

            chunk["tags"] = self._deduplicate(tags)
            enriched.append(chunk)
        return enriched

    def _infer_difficulty(self, text: str) -> str:
        # 简单难度标签推断
        if any(keyword in text for keyword in ["证明", "推导", "收敛", "复杂度"]):
            return "difficulty:high"
        if any(keyword in text for keyword in ["案例", "示例", "练习"]):
            return "difficulty:low"
        return "difficulty:medium"

    def _deduplicate(self, values: List[str]) -> List[str]:
        # 保序去重
        result: List[str] = []
        seen = set()
        for value in values:
            clean_value = (value or "").strip()
            if not clean_value or clean_value in seen:
                continue
            seen.add(clean_value)
            result.append(clean_value)
        return result
