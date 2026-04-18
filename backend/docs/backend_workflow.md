# 后端工作流（当前稳定路径）

本文档说明当前已打通的后端主流程（RAG + 生成融合版本）。

## 1) 主运行链路

### Query -> 大纲
1. 接口：`POST /api/course/generate_with_rag`
2. 服务链路：
   - `GeneratorAdapterService.generate_from_hybrid_context`
   - `HybridSearchService.orchestrate_search`
   - `ContextConsolidator.consolidate`
   - `app.services.llm_client.generate_outline_from_text`
3. 输出：`outline_data`（与队友 PPT 生成器兼容）

### Query -> PPT
1. 接口：`POST /api/course/generate_ppt_with_rag`
2. 服务链路：
   - 与上面的检索/整理/大纲链路一致
   - 然后调用 `app.services.ppt_generator.generate_ppt_from_json`
3. 输出：`project_id`、`download_url`，可选 `outline_data`、可选 `debug`

## 2) 核心服务边界

- `backend/api/*`：请求解析、响应组装、HTTP 异常映射
- `backend/services/hybrid_search_service.py`：向量 + 图谱 + Mongo 编排
- `backend/services/context_consolidator.py`：证据过滤与整理
- `backend/services/generator_adapter_service.py`：到队友生成链路的薄桥接
- `backend/services/mongo_kb_service.py`：Mongo 事实源访问
- `backend/services/milvus_service.py`：Milvus 向量存储与检索

## 3) 数据/索引构建路径

### 全量入库（四层）
- `python scripts/ingest_ml_batch.py --layers syllabus,textbook,resource,hotspot`

### 从 Mongo 构建向量索引
- `python scripts/build_milvus_index_from_mongo.py --layers syllabus,textbook,resource,hotspot`

### 从 Mongo 构建图谱
- `python scripts/build_graph_from_mongo.py --layers syllabus,textbook,resource,hotspot`

### 单资源增量更新（推荐用于单个 PPT 更新）
- `python scripts/incremental_update_single_resource.py --file <absolute_pptx_path>`

## 4) 前端主要接口

- `POST /api/course/generate_with_rag`
- `POST /api/course/generate_ppt_with_rag`

兼容性说明：
- `outline_data` 保持队友兼容字段：`course_metadata`、`syllabus_content`、`resource_pool`。

## 5) 目录职责

- `backend/data/raw/`：原始资源文件
- `backend/data/processed/`：解析/标准化中间产物
- `backend/data/debug/`：调试报告与验证输出
- `backend/data/graph/`：图谱产物
- `backend/data/vector_index/`：本地向量构建报告/产物
- `backend/downloads/`：生成产物目录（含本地 pptx 兜底），不属于核心源码模块
- `backend/scripts/`：可执行运维/调试脚本

## 6) 后端目录收口说明（主目录 / 兼容目录）

### 6.1 核心目录
- `backend/extractors/`：结构化抽取层，负责把解析后的文档内容转换为各层结构化结果（syllabus/textbook/resource/hotspot）以及图谱三元组。
- `backend/parsers/`：底层文件解析层，负责 PDF/PPTX/HTML 等输入文件的通用解析与元素标准化。
- `backend/rag/`：旧版 RAG 引擎目录。当前主链已切到 `services` 下的混合检索与适配服务；该目录主要用于兼容入口（见 `course_api` 的 legacy fallback）。
- `backend/schema/`：当前主 schema 目录，主链 API、extractor、service 统一依赖此目录下模型。

### 6.2 兼容目录
- `backend/schemas/`：历史兼容目录，当前仍有少量代码直接引用（例如 `services/syllabus_normalizer.py`）。
- 代码扫描结果（当前版本）：`schemas.syllabus_schema` 仍被 `backend/services/syllabus_normalizer.py` 直接依赖。
- 结论：本阶段不建议合并 `schema/` 与 `schemas/`，避免引发 import 连锁风险。优先保留并通过文档明确定位。

### 6.3 产物目录
- `backend/downloads/`：仅存放运行期生成产物（如 PPT 文件），不参与后端源码职责划分。
- 当前 `.gitignore` 已忽略 `downloads/` 与 `backend/downloads/`，可继续保持。
- 若 IDE 目录视觉干扰较大，建议在本地将 `backend/downloads/` 标记为 excluded（仅本地配置，不入库）。

## 7) schema / schemas 处理建议

- 默认策略：**不改 import，不合并目录，只补说明**。
- 触发合并的前提（当前不满足）：先清零 `schemas.*` 引用，再统一切到 `schema.*` 并完成主链 smoke check。
- 当前建议：在交付期维持双目录并存，待后续版本单独做“兼容目录下线”专项。

## 8) scripts 顶层与归档约定

- `backend/scripts/` 顶层只保留正式入口（`build_*` / `ingest_*` / `incremental_*` / 在用 `debug_*` / 在用 `test_*`）。
- 历史排查脚本统一放在 `backend/scripts/_archive_tmp/`，默认不参与主流程。
- 交付期建议：优先保持顶层清爽；历史脚本如无明确复用需求，不要再回迁到顶层。

## 9) services 分组与 tests 归类

- `backend/services/README.md`：服务层职责分组说明（仅说明，不改结构）。
- `backend/tests/README.md`：测试目录分组规则（api/services/pipeline/extractors/parsers）。
- 交付期原则：优先做目录清晰化与文档收口，避免高风险重构。
