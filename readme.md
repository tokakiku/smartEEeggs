#灵犀智课 - 后端核心服务

##目录结构与核心代码说明

```text
C:\Users\...\PycharmProjects\Rui_Jie_Project\
├── app/                        # 后端核心代码包
│   ├── core/                   # 核心配置模块
│   │   ├── config.py           # 全局配置中心 (读取 .env 变量)
│   │   └── database.py         # 数据库连接发牌员 (MySQL 会话管理)
│   │
│   ├── models/                 # 数据库图纸 (SQLAlchemy ORM 模型)
│   │   ├── project.py          # 课件项目表 (记录 teacher_id 和 outline_data)
│   │   └── user.py             # 教师用户表 (包含账号与加密密码)
│   │
│   ├── routers/                # 路由网关 (对外暴露的 API 接口)
│   │   ├── lessonprep.py       # 备课工作流路由 (串联大纲与 PPT 生成，包含 SSE 流式推送接口)
│   │   └── parser.py           # 多模态解析路由 (接收前端上传文件，并分发给对应物理引擎)
│   │
│   ├── services/               # 物理引擎与 AI 大脑 (核心业务逻辑)
│   │   ├── llm_client.py       # 大模型驱动引擎 (接入智谱 GLM-4，将文本转化为 BOPPPS 标准大纲)
│   │   ├── pdf_parser.py       # 静态文档解析引擎 (基于 PyMuPDF 提取 PDF 纯文本)
│   │   └── ppt_generator.py    # 高颜值 PPT 渲染引擎 (基于 python-pptx 精准排版，防文字溢出)
│   │
│   ├── utils/                  # 通用工具箱
│   │   └── minio_client.py     # MinIO 云端存储工具 (将本地课件上传并生成下载链接)
│   │
│   └── main.py                 # FastAPI 主入口 (挂载路由、配置跨域、数据库建表拦截)
│
├── backend/                        # 知识引擎与检索底座（当前推荐主链）
│   ├── api/
│   │   └── course_api.py           # 统一 API 入口（含 generate_with_rag / generate_ppt_with_rag）
│   │
│   ├── services/
│   │   ├── hybrid_search_service.py      # Mongo + Milvus + Graph 混合检索编排
│   │   ├── context_consolidator.py       # 检索证据整理（噪声过滤/去重/结构化）
│   │   ├── generator_adapter_service.py  # 薄适配层：RAG 输出接入 app 生成链
│   │   ├── ingest_service.py             # 离线入库编排（parser + extractor + Mongo）
│   │   ├── mongo_kb_service.py           # Mongo 知识库访问
│   │   ├── milvus_service.py             # Milvus 向量存取
│   │   ├── graph_store_service.py        # 图谱存储与查询
│   │   └── ...                           # 其余检索/抽取支撑服务
│   │
│   ├── scripts/
│   │   ├── ingest_ml_batch.py            # 四层批量入库
│   │   ├── build_milvus_index_from_mongo.py
│   │   ├── build_graph_from_mongo.py
│   │   ├── incremental_update_single_resource.py
│   │   └── ...
│   │
│   ├── data/                       # raw/processed/graph/vector/debug 等数据产物
│   ├── downloads/                  # 本地 PPT 生成产物目录（当前统一目录）
│   └── main.py                     # backend FastAPI 入口（挂载 /downloads 静态路径）
│
├── downloads/                  # 临时文件读写区 (PPT 生成目录、上传文件暂存区)
├── .env                        # 环境变量机密文件 (数据库密码等)
├── .gitignore                  # Git 忽略配置
├── docker-compose.yml          # 容器编排文件 (一键启动 MySQL、Redis、MinIO)
└── requirements.txt            # 项目依赖清单


## 核心工作流解析

本项目采用“生成引擎 + 知识引擎”双层协同架构，覆盖从资料入库、知识构建、检索增强到课件生成与导出的完整闭环（对应 ParserAgent、GeneratorAgent 与知识引擎层）。

### 1. 输入入口（运行时）
当前推荐对外入口为：
- `POST /api/course/generate_with_rag`（生成大纲）
- `POST /api/course/generate_ppt_with_rag`（从 query 直接到 PPT）

兼容/历史插件入口仍可用（非当前推荐主链）：
- `POST /api/plugins/parse_pdf`
- `POST /api/plugins/generate_outline`
- `POST /api/plugins/generate_ppt`

### 2. 知识库构建与资料预处理（离线阶段）
系统对教学资料的解析、抽取、结构化入库主要发生在“建库/增量更新”阶段，而非每次生成时重复执行。  
典型流程：
- `backend/services/ingest_service.py` 统一编排解析与抽取
- `backend/parsers/` 负责底层文件解析
- `backend/extractors/` 负责分层知识抽取
- 结果写入四层结构化知识库，并可同步更新 Mongo、向量索引与图谱

当前四层知识库：
- `syllabus`：课标 / 教学大纲
- `textbook`：教材知识
- `resource`：课件与教学资源
- `hotspot`：时事热点与案例

### 3. 知识库召回与检索增强（运行时）
当用户输入课程主题（如“神经网络”“梯度下降法”）后，`/api/course/generate_with_rag` 会触发 backend 检索增强链路：
- Mongo 结构化知识回查
- Milvus 向量检索
- 图谱关系扩展（含跨层边）
- Context Consolidator 证据整理与压缩

产出高质量 `clean_lesson_brief`，并将其作为生成引擎的输入，进一步生成结构化 `outline_data`。

### 4. 灵魂提炼与教学大纲生成
系统复用 `app/services/llm_client.py`，按 BOPPPS Schema 生成结构化教学大纲 `outline_data`。  
即：先做检索增强与上下文整理，再由生成引擎完成结构化教学设计。

### 5. 实体渲染与课件生成
前端可调用 `/api/course/generate_ppt_with_rag`，系统内部先生成 `outline_data`，再调用 `app/services/ppt_generator.py` 完成 PPT 渲染与排版，输出可交付课件文件。

### 6. 导出交付
生成结果通过 `download_url` 返回给前端：  
- MinIO 可用时返回云端链接  
- MinIO 不可用时走本地兜底路径（`/downloads/<file>.pptx`）  

当前本地产物统一目录为 `backend/downloads/`。

快速启动
启动基础设施：docker-compose up -d
安装依赖：pip install -r requirements.txt
启动服务：uvicorn app.main:app --reload
访问接口文档：http://127.0.0.1:8000/docs