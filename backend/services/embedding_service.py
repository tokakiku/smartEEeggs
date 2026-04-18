from __future__ import annotations

"""向量嵌入服务单例封装。"""

import os
import threading
from typing import List, Optional

import numpy as np


def _is_true(value: Optional[str], default: bool) -> bool:
    """环境变量布尔解析。"""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# 默认使用中文检索效果更稳定的 m3e-base。
DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "moka-ai/m3e-base")
DEFAULT_LOCAL_ONLY = _is_true(os.getenv("EMBEDDING_LOCAL_ONLY"), True)
ALLOW_REMOTE_FALLBACK = _is_true(os.getenv("EMBEDDING_ALLOW_REMOTE"), False)

_LOCK = threading.Lock()
_INSTANCE: Optional["EmbeddingService"] = None


class EmbeddingServiceError(RuntimeError):
    """向量嵌入服务异常。"""
    pass


class EmbeddingService:
    """基于 sentence-transformers 的向量嵌入服务。

    加载策略：
    1) 优先 local_files_only=True（先走本地缓存）；
    2) 若允许远程回退，再尝试 local_files_only=False。
    """

    def __init__(self, model_name: Optional[str] = None, local_only: Optional[bool] = None) -> None:
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.local_only = DEFAULT_LOCAL_ONLY if local_only is None else local_only
        self._model = None
        self._dim: Optional[int] = None
        self._load_model()

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        errors: List[str] = []
        tried_modes: List[bool] = [True] if self.local_only else [False]
        if self.local_only and ALLOW_REMOTE_FALLBACK:
            tried_modes.append(False)

        for local_files_only in tried_modes:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=local_files_only,
                )
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dim = int(self._model.get_embedding_dimension())
                else:
                    self._dim = int(self._model.get_sentence_embedding_dimension())
                return
            except Exception as exc:
                mode = "local_only" if local_files_only else "remote_enabled"
                errors.append(f"{mode}: {exc}")

        raise EmbeddingServiceError(
            f"无法加载 embedding 模型 {self.model_name!r}. "
            f"尝试模式: {', '.join(errors)}"
        )

    @property
    def dimension(self) -> int:
        """返回当前模型向量维度。"""
        if self._dim is not None:
            return self._dim
        if self._model is not None:
            self._dim = int(self._model.get_sentence_embedding_dimension())
            return self._dim
        return 768

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量文本向量化。"""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        normalized = [str(text).strip() or " " for text in texts]
        embeddings = self._model.encode(
            normalized,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """单条查询向量化。"""
        return self.embed([str(query).strip() or " "])


def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """获取 Embedding 服务单例。"""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = EmbeddingService(model_name=model_name)
    return _INSTANCE
