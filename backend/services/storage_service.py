from pathlib import Path
from typing import Any, List, Optional

from utils.file_saver import save_binary_file
from utils.json_saver import save_json_file


class StorageService:
    # 本地存储服务：模拟未来文件存储、结构化库、向量库预存

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        backend_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir or (backend_dir / "data")
        self.raw_dir = self.base_dir / "raw"
        self.structured_dir = self.base_dir / "structured"
        self.chunks_dir = self.base_dir / "chunks"
        self.relations_dir = self.base_dir / "relations"

    def save_raw_file(self, file_bytes: bytes, file_name: str, doc_id: str) -> str:
        # 保存原始文件副本
        target_path = self.raw_dir / f"{doc_id}_{file_name}"
        return save_binary_file(target_path, file_bytes)

    def save_structured_output(self, doc_id: str, layer: str, payload: Any) -> str:
        # 保存结构化结果
        target_path = self.structured_dir / f"{layer}_{doc_id}.json"
        return save_json_file(target_path, payload)

    def save_chunks_output(self, doc_id: str, layer: str, chunks: List[dict]) -> str:
        # 保存 chunk 结果
        target_path = self.chunks_dir / f"{layer}_{doc_id}_chunks.json"
        return save_json_file(target_path, chunks)

    def save_relations_output(self, doc_id: str, layer: str, relations: List[dict]) -> str:
        # 保存关系结果
        target_path = self.relations_dir / f"{layer}_{doc_id}_relations.json"
        return save_json_file(target_path, relations)
