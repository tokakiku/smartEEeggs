# 服务层分组说明（低风险收口版）

本文档用于说明 `backend/services/` 的职责分组，帮助交付与交接。  
当前策略是“**只补说明，不做高风险重构**”。

## 1) 核心主链

- `generator_adapter_service.py`：RAG 前置知识引擎到生成链路的薄适配。
- `hybrid_search_service.py`：Mongo + 向量 + 图谱统一检索编排。
- `context_consolidator.py`：检索证据过滤与教学素材整理。

## 2) 检索 / 兼容层

- `kb_rag_adapter.py`：旧 KB-RAG 上下文适配逻辑。
- `kb_rag_service.py`：旧 KB-RAG 服务入口（兼容保留）。
- `cross_layer_retrieval_service.py`：跨层检索与排序支持。
- `hybrid_retrieval_service.py`：混合检索兼容实现。

## 3) 数据底座

- `mongo_kb_service.py`：Mongo 事实源读写与查询。
- `milvus_service.py`：Milvus 向量集合管理与检索。
- `storage_service.py`：本地存储与产物读写。
- `metadata_service.py`：文档元信息构建。

## 4) 图谱相关

- `graph_extraction_service.py`：基于 Mongo 文档的图谱抽取编排。
- `graph_store_service.py`：图存储与查询能力。
- `relation_service.py`：关系构建与轻量关系补充。

## 5) 入库 / 切块 / 向量

- `ingest_service.py`：文档入库主编排（解析 -> 抽取 -> 落库）。
- `chunk_builder_service.py`：面向向量化的分块构建。
- `chunk_service.py`：分块标准化输出。
- `embedding_service.py`：向量模型加载与向量化。
- `vector_index_service.py`：向量索引构建与检索（兼容本地/服务端路径）。

## 6) 教学大纲 / 辅助能力

- `syllabus_extractor.py`：教学大纲规则抽取。
- `syllabus_normalizer.py`：教学大纲字段标准化。
- `topic_locator.py`：主题证据定位。
- `tag_service.py`：标签生成与标准化辅助。
- `syllabus_parser.py`：教学大纲解析辅助。

## 7) 维护约定

- 默认不移动 service 文件，不改主链 import 结构。
- 若后续需要重构，请先做“引用清单 + 最小 smoke check”再分阶段推进。
