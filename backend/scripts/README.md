# 脚本说明

本文件用于快速定位 `backend/scripts/` 下的正式入口脚本。

## 顶层保留原则（交付期）

- 顶层仅保留当前主流程会使用的入口脚本。
- 历史排查/一次性验证脚本统一归档到 `_archive_tmp/`。
- 对拿不准的脚本优先归档，不直接删除。

## A) 构建 / 入库（生产相关）

- `ingest_ml_batch.py`  
  从 `data/raw/curriculum/machine_learning/*` 批量入库到 Mongo。

- `build_milvus_index_from_mongo.py`  
  从 Mongo 构建/重建 Milvus 向量索引。

- `build_graph_from_mongo.py`  
  从 Mongo 构建图谱产物。

- `build_curriculum_layer.py`  
  基于教材/大纲产物构建课程层结构（用于课程层 JSON 产出与核对）。

- `rebuild_vector_index_from_mongo.py`  
  重建本地向量索引（FAISS 路径）。

- `incremental_update_single_resource.py`  
  对单个 `resource` 文件做增量更新（Mongo + Milvus + 检索验证）。

## B) 调试 / 检查

- `debug_hybrid_search.py`  
  运行混合检索诊断并输出摘要 JSON。

## C) 端到端 / 验证

- `test_end_to_end_generation.py`  
  检索 -> 整理 -> 大纲生成的端到端验证。

## D) 历史/一次性脚本

- `_archive_tmp/*`  
  历史临时脚本与一次性验证脚本。默认不参与当前主流程。  
  如需复用，请先确认依赖和输出结构是否仍兼容当前主链。

## 命名约定（后续新增建议）

- `build_*`：构建/重建索引与产物
- `ingest_*`：入库任务
- `incremental_*`：定向增量更新
- `debug_*` / `inspect_*`：诊断脚本
- `test_*` / `run_*_validation`：验证脚本
