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
├── downloads/                  # 临时文件读写区 (PPT 生成目录、上传文件暂存区)
├── .env                        # 环境变量机密文件 (数据库密码等)
├── .gitignore                  # Git 忽略配置
├── docker-compose.yml          # 容器编排文件 (一键启动 MySQL、Redis、MinIO)
└── requirements.txt            # 项目依赖清单


核心工作流解析
本后端完美实现了以下闭环（对应详细方案中的 ParserAgent 与 GeneratorAgent）：
多模态接收: 前端调用 /api/parser/upload 上传教学资料。
物理拆解: pdf_parser.py 等工具将资料拆解为纯文本。
灵魂提炼: llm_client.py 调用大模型，严格按照组长设计的 BOPPPS Schema 吐出教学大纲 JSON。
实体渲染: 前端将确认后的 JSON 发送至 /api/lessonprep/ppt/generate。
流式反馈: 后端通过 SSE (Server-Sent Events) 实时推送生成进度。
精美产出: ppt_generator.py 精准控制字号与排版生成 PPT，并通过 minio_client.py 存入云端返回链接。

快速启动
启动基础设施：docker-compose up -d
安装依赖：pip install -r requirements.txt
启动服务：uvicorn app.main:app --reload
访问接口文档：http://127.0.0.1:8000/docs