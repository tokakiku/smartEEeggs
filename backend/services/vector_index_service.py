from __future__ import annotations

"""鏈湴鍚戦噺绱㈠紩鏈嶅姟锛團AISS锛夈€?
鐢ㄤ簬鍘嗗彶鍏煎涓庣绾胯皟璇曞満鏅紝鏀寔锛?- chunk 鍚戦噺鍐欏叆涓庢绱紱
- 浠?Mongo 閲嶅缓绱㈠紩锛?- metadata 涓庡悜閲忎綅缃竴涓€瀵瑰簲绠＄悊銆?"""

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from services.mongo_kb_service import MongoKBService

_BACKEND_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_INDEX_DIR = str(_BACKEND_DIR / "data" / "vector_index")
INDEX_FILENAME = "index.faiss"
META_FILENAME = "metadata.json"

_LOCK = threading.Lock()
_INSTANCE: Optional["VectorIndexService"] = None


class VectorIndexService:
    """本地 FAISS 持久化索引。

    - 向量：IndexFlatIP + 归一化 embedding（近似余弦相似度）
    - 元数据：metadata.json，与向量位置一一对应
    - 支持软删除与重复 chunk upsert
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        embedding_service: Any = None,
        default_dim: int = 768,
    ) -> None:
        self.index_dir = Path(index_dir or os.getenv("VECTOR_INDEX_DIR", DEFAULT_INDEX_DIR)).resolve()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / INDEX_FILENAME
        self.meta_path = self.index_dir / META_FILENAME

        self.default_dim = int(os.getenv("VECTOR_DIM", str(default_dim)))
        self._embedding_service = embedding_service
        self._index = None
        self._meta: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._load_or_init()

    def _get_embedding_service(self):
        if self._embedding_service is None:
            from services.embedding_service import get_embedding_service

            self._embedding_service = get_embedding_service()
        return self._embedding_service

    def _load_or_init(self) -> None:
        import faiss

        if self.index_path.exists() and self.meta_path.exists():
            try:
                raw = self.index_path.read_bytes()
                array = np.frombuffer(raw, dtype=np.uint8)
                self._index = faiss.deserialize_index(array)
                with open(self.meta_path, "r", encoding="utf-8") as fh:
                    self._meta = json.load(fh)
                return
            except Exception:
                pass

        self._index = faiss.IndexFlatIP(self.default_dim)
        self._meta = []
        self._save()

    def _reset(self, dim: Optional[int] = None) -> None:
        import faiss

        target_dim = int(dim or self.default_dim)
        self._index = faiss.IndexFlatIP(target_dim)
        self._meta = []
        self._save()

    def _ensure_index_dim(self, dim: int) -> None:
        if self._index is None:
            self._reset(dim=dim)
            return
        if self._index.d == dim:
            return
        if int(self._index.ntotal) > 0:
            raise ValueError(f"vector dim mismatch: index={self._index.d}, embedding={dim}")
        self._reset(dim=dim)

    @staticmethod
    def _looks_like_toc_line(text: str) -> bool:
        value = str(text or "").strip().lower()
        if not value:
            return False
        if re.fullmatch(r"(鐩甛s*褰晐contents?)", value):
            return True
        if re.match(r"^\d+(\.\d+){0,3}\s+\S+", value) and re.search(r"\.{2,}\s*\d+$", value):
            return True
        if re.search(r"(绗琝s*\d+\s*椤祙\bpage\s*\d+\b)$", value):
            return True
        return False

    @staticmethod
    def _clean_text(text: Any, max_len: int = 900) -> str:
        value = str(text or "")
        value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", value)
        value = value.replace("\uFFFD", " ")
        value = re.sub(r"[路鈥⑩棌鈻犫枴鈼嗏柡鈻糕柂鈼+", " ", value)
        value = re.sub(r"(?:\s*[.銆俔\s*){3,}\d*", " ", value)
        value = re.sub(r"[=+\-/*]{8,}", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        if re.fullmatch(r"(鐩甛s*褰晐contents?)", value, flags=re.I):
            return ""
        if re.fullmatch(r"(绗琝s*\d+\s*椤祙\d+\s*/\s*\d+)", value, flags=re.I):
            return ""
        if len(value) > max_len:
            value = value[:max_len].rstrip("锛?銆?;锛?") + "..."
        return value

    @staticmethod
    def _is_noisy(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True
        if len(value) < 12:
            return True
        if VectorIndexService._looks_like_toc_line(value):
            return True
        if re.search(r"[=+\-/*]{8,}", value):
            return True
        if re.search(r"[.銆俔{10,}", value):
            return True
        if re.fullmatch(r"[\W_]+", value):
            return True
        useful = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", value)
        return len(useful) / max(1, len(value)) < 0.45

    @staticmethod
    def _normalize_identifier(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text.lower() in {"none", "null", "nan", "na"}:
            return ""
        return text

    @classmethod
    def _chunk_identity(cls, item: Dict[str, Any]) -> Tuple[str, str, str]:
        layer = cls._normalize_identifier(item.get("layer"))
        doc_key = cls._normalize_identifier(item.get("doc_id")) or cls._normalize_identifier(item.get("source_file"))
        chunk_id = cls._normalize_identifier(item.get("chunk_id"))
        return layer, doc_key, chunk_id

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        prepared: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            text = self._clean_text(chunk.get("text"))
            if self._is_noisy(text):
                continue
            layer = self._normalize_identifier(chunk.get("layer"))
            doc_id = self._normalize_identifier(chunk.get("doc_id"))
            source_file = self._normalize_identifier(chunk.get("source_file"))
            doc_key = doc_id or source_file or f"unknown-doc-{idx}"
            chunk_id = self._normalize_identifier(chunk.get("chunk_id"))
            if not chunk_id:
                chunk_id = f"{layer or 'unknown'}:{doc_key}:{idx}"
            prepared.append(
                {
                    "layer": layer,
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "title": str(chunk.get("title") or chunk.get("section") or ""),
                    "chunk_id": chunk_id,
                    "text": text,
                }
            )

        if not prepared:
            return 0

        es = self._get_embedding_service()
        embeddings = es.embed([item["text"] for item in prepared])
        if embeddings.size == 0:
            return 0

        with self._lock:
            self._ensure_index_dim(embeddings.shape[1])
            added = 0
            for idx, item in enumerate(prepared):
                chunk_id = item["chunk_id"]
                chunk_identity = self._chunk_identity(item)
                if chunk_id:
                    for meta_item in self._meta:
                        if meta_item.get("_deleted"):
                            continue
                        if self._chunk_identity(meta_item) == chunk_identity:
                            meta_item["_deleted"] = True

                self._meta.append(
                    {
                        "layer": item["layer"],
                        "doc_id": item["doc_id"],
                        "source_file": item["source_file"],
                        "title": item["title"],
                        "chunk_id": chunk_id,
                        "chunk_identity": f"{chunk_identity[0]}:{chunk_identity[1]}:{chunk_identity[2]}",
                        "text": item["text"],
                        "_deleted": False,
                    }
                )
                self._index.add(embeddings[idx].reshape(1, -1).astype(np.float32))
                added += 1

            self._save()
            return added

    def search(
        self,
        query: str,
        top_k: int = 10,
        layers: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if self._index is None or int(self._index.ntotal) == 0:
            return []

        q = str(query or "").strip()
        if not q:
            return []

        es = self._get_embedding_service()
        query_vec = es.embed_query(q).astype(np.float32)
        if query_vec.shape[1] != self._index.d:
            return []

        fetch_k = min(int(self._index.ntotal), max(top_k * 8, 40))
        scores, indices = self._index.search(query_vec, fetch_k)

        results: List[Dict[str, Any]] = []
        allowed_layers = set(layers or [])
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            item = self._meta[idx]
            if item.get("_deleted"):
                continue
            if allowed_layers and item.get("layer") not in allowed_layers:
                continue
            if float(score) < float(min_score):
                continue

            results.append(
                {
                    "layer": item.get("layer"),
                    "doc_id": item.get("doc_id"),
                    "source_file": item.get("source_file"),
                    "title": item.get("title"),
                    "chunk_id": item.get("chunk_id"),
                    "text": item.get("text"),
                    "vector_score": float(score),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def build_from_mongo(
        self,
        kb_service: MongoKBService,
        force: bool = False,
        batch_size: int = 64,
        max_chunks_per_doc: int = 80,
    ) -> Dict[str, Any]:
        kb_service._ensure_available()
        with self._lock:
            if force:
                self._reset()
            elif self.count_active() > 0:
                return {"added": 0, "status": "skip_non_empty"}

        all_chunks: List[Dict[str, Any]] = []
        docs_scanned = 0
        for layer, collection_name in kb_service.COLLECTION_MAP.items():
            collection = kb_service.db[collection_name]
            docs = list(
                collection.find(
                    {
                        "layer": layer,
                        "$or": [{"status": "active"}, {"status": {"$exists": False}}],
                    }
                )
            )
            docs_scanned += len(docs)
            for doc in docs:
                all_chunks.extend(
                    self._collect_doc_chunks(
                        layer=layer,
                        doc=doc,
                        max_chunks=max_chunks_per_doc,
                    )
                )

        added_total = 0
        for i in range(0, len(all_chunks), max(8, batch_size)):
            added_total += self.add_chunks(all_chunks[i : i + batch_size])

        return {
            "status": "ok",
            "added": added_total,
            "docs": docs_scanned,
            "chunks": len(all_chunks),
            "active_entries": self.count_active(),
        }

    def rebuild_from_mongo(
        self,
        kb_service: MongoKBService,
        batch_size: int = 64,
        max_chunks_per_doc: int = 80,
    ) -> Dict[str, Any]:
        return self.build_from_mongo(
            kb_service=kb_service,
            force=True,
            batch_size=batch_size,
            max_chunks_per_doc=max_chunks_per_doc,
        )

    def _collect_doc_chunks(self, layer: str, doc: Dict[str, Any], max_chunks: int = 80) -> List[Dict[str, Any]]:
        data = doc.get("data") or {}
        doc_id = self._normalize_identifier(doc.get("doc_id"))
        source_file = self._normalize_identifier(doc.get("source_file"))
        title = str(doc.get("title") or "")
        doc_key = doc_id or source_file or "unknown-doc"

        candidates: List[Dict[str, Any]] = []

        raw_chunks = data.get("chunks") or []
        for idx, chunk in enumerate(raw_chunks):
            text = chunk.get("text") if isinstance(chunk, dict) else chunk
            chunk_id = ""
            if isinstance(chunk, dict):
                chunk_id = self._normalize_identifier(chunk.get("chunk_id"))
            if not chunk_id:
                chunk_id = f"{layer}:{doc_key}:{idx}"
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "layer": layer,
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "title": title,
                    "text": text,
                }
            )

        if layer == "syllabus" and not candidates:
            for idx, module in enumerate(data.get("course_modules") or []):
                text = " ".join(
                    [
                        str(module.get("module_name") or ""),
                        ", ".join([str(x) for x in (module.get("key_points") or []) if str(x).strip()]),
                        ", ".join([str(x) for x in (module.get("difficult_points") or []) if str(x).strip()]),
                    ]
                ).strip()
                candidates.append(
                    {
                        "chunk_id": f"{layer}:{doc_key}:module:{idx}",
                        "layer": layer,
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "title": title,
                        "text": text,
                    }
                )

        if layer == "resource" and not candidates:
            for idx, page in enumerate(data.get("pages") or []):
                text = " ".join(
                    [
                        str(page.get("page_title") or ""),
                        str(page.get("page_summary") or ""),
                        ", ".join([str(x) for x in (page.get("knowledge_points") or []) if str(x).strip()]),
                    ]
                ).strip()
                candidates.append(
                    {
                        "chunk_id": f"{layer}:{doc_key}:page:{idx}",
                        "layer": layer,
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "title": title,
                        "text": text,
                    }
                )

        if layer == "hotspot" and not candidates:
            for idx, item in enumerate(data.get("hotspot_item") or []):
                text = " ".join(
                    [
                        str(item.get("summary") or ""),
                        ", ".join([str(x) for x in (item.get("related_knowledge_points") or []) if str(x).strip()]),
                        ", ".join([str(x) for x in (item.get("keywords") or []) if str(x).strip()]),
                    ]
                ).strip()
                candidates.append(
                    {
                        "chunk_id": f"{layer}:{doc_key}:hot:{idx}",
                        "layer": layer,
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "title": title,
                        "text": text,
                    }
                )

        dedup: List[Dict[str, Any]] = []
        seen = set()
        for item in candidates:
            text = self._clean_text(item.get("text"))
            if self._is_noisy(text):
                continue
            if self._looks_like_toc_line(text):
                continue
            key = f"{layer}:{doc_id or source_file}:{item.get('chunk_id')}::{text[:160]}"
            if key in seen:
                continue
            seen.add(key)
            dedup.append({**item, "text": text})
            if len(dedup) >= max_chunks:
                break
        return dedup

    def count_active(self) -> int:
        return sum(1 for item in self._meta if not item.get("_deleted"))

    def total_vectors(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def stats(self) -> Dict[str, Any]:
        layer_counts: Dict[str, int] = {}
        for item in self._meta:
            if item.get("_deleted"):
                continue
            layer = str(item.get("layer") or "unknown")
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        return {
            "active_entries": self.count_active(),
            "total_vectors": self.total_vectors(),
            "layer_counts": layer_counts,
            "index_dir": str(self.index_dir),
            "dim": int(self._index.d) if self._index is not None else None,
        }

    def list_active_metadata(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        active = [dict(item) for item in self._meta if not item.get("_deleted")]
        if limit is not None:
            return active[: max(0, int(limit))]
        return active

    def _save(self) -> None:
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        blob = faiss.serialize_index(self._index)
        self.index_path.write_bytes(blob.tobytes())
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, ensure_ascii=False, indent=2)


def get_vector_index_service(index_dir: Optional[str] = None) -> VectorIndexService:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = VectorIndexService(index_dir=index_dir)
    return _INSTANCE

