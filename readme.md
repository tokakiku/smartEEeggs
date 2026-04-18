二、完整 README.md 编写（直接复制使用）
markdown
# 灵犀智课 - 多模态AI互动式教学智能体
> 2026年大学生服务外包创新创业大赛 · 锐捷网络命题 · A04赛题

**一句话定位**：以多轮深度对话为核心、多智能体协作为引擎的教学课件共创系统，覆盖课前备课全流程。

## 🚀 快速启动（5分钟跑通）
### 1. 环境准备
- Python 3.10+
- Docker & Docker Compose
- 智谱 AI API Key（GLM-4-Flash + CogView-3-Plus）

### 2. 启动依赖服务
```bash
# 只启动 MySQL 和 MinIO（Redis 已注释）
docker-compose up -d mysql minio
3. 安装 Python 依赖
bash
运行
pip install -r requirements.txt
4. 配置环境变量
复制 .env.example 为 .env，填入你的智谱 API Key：
env
DATABASE_URL="mysql+pymysql://lingxi_user:lingxi_password@127.0.0.1:3306/lingxi_db"
ZHIPU_API_KEY="你的智谱APIKey"
5. 启动后端服务
bash
运行
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
6. 验证服务
后端健康检查：http://127.0.0.1:8000
API 文档：http://127.0.0.1:8000/docs
📁 项目核心结构
plaintext
backend/
├── app/
│   ├── core/                # 核心配置
│   │   ├── config.py        # 全局配置
│   │   └── database.py      # 数据库连接
│   ├── models/              # 数据库模型
│   │   ├── user.py          # 用户模型
│   │   └── project.py       # 课件项目模型
│   ├── routers/             # API 路由
│   │   ├── coze_plugins.py  # Coze 专属插件接口（核心）
│   │   └── word_plugins.py  # Word 教案生成接口
│   ├── services/            # 业务服务层
│   │   ├── llm_client.py    # 智谱大模型客户端
│   │   ├── document_parser.py # 多模态文档解析引擎
│   │   ├── ppt_generator.py # PPT 物理渲染引擎
│   │   └── word_generator.py # Word 教案渲染引擎
│   └── main.py              # FastAPI 应用入口
├── downloads/               # 生成文件存储目录（自动创建）
│   ├── uploads/             # 用户上传文件
│   └── exports/             # 生成的 PPT/Word 文件
├── template.pptx            # PPT 模板文件（必须存在）
├── .env                     # 环境变量
├── docker-compose.yml       # 依赖服务配置
└── requirements.txt         # Python 依赖
✨ 核心功能
1. 多模态文档解析
支持 PDF、Word、PPTX 三种格式
自动提取表格并转换为 HTML
保留文档结构和排版信息
2. BOPPPS 教学大纲生成
基于智谱 GLM-4-Flash 模型
严格遵循 BOPPPS 教学模型
输出标准化 JSON 格式大纲
3. 智能 PPT 生成
100% 可编辑的 PPTX 文件
根据 BOPPPS 阶段自动匹配布局
自动生成 AI 配图（CogView-3-Plus）
彻底解决模板索引崩溃问题
4. Word 教案生成
与 PPT 内容完全对齐
专业的教学详案格式
自动标注重点和难点
5. Coze 智能体无缝对接
专为 Coze 设计的插件接口
自动处理 JSON 字符串套娃问题
静态文件服务直接提供下载
🛠️ 技术栈
表格
领域	技术选型
Web 框架	FastAPI + Uvicorn
数据库	MySQL 8.0 + SQLAlchemy 2.0
大模型	智谱 GLM-4-Flash + CogView-3-Plus
文档解析	Unstructured + Tesseract + Poppler
文件生成	python-pptx + python-docx
部署	Docker + Docker Compose
🎯 演示流程（必须跑通）
调用 /api/plugins/parse_file 上传参考文档
调用 /api/plugins/generate_outline 生成 BOPPPS 大纲
调用 /api/plugins/generate_ppt 生成 PPT
调用 /api/word_plugins/generate_word 生成 Word 教案
通过返回的 download_url 下载生成的文件
⚠️ 常见问题
1. PPT 生成失败提示找不到 template.pptx
确保项目根目录下有 template.pptx 文件，且至少包含 3 种不同的布局。
2. 文档解析失败
Windows 系统需要安装 Tesseract 和 Poppler，并配置环境变量
确保文件没有损坏和密码保护
3. 大模型调用失败
检查 .env 中的 API Key 是否正确
确保账户有足够的余额
检查网络连接是否正常
4. 下载链接无法访问
确保 FastAPI 已经挂载了静态文件目录（main.py 中的 app.mount("/static", ...)）
📅 剩余 10 天开发计划
表格
时间	任务	优先级
第 1-2 天	跑通核心流程，确保 PPT 和 Word 生成正常	🔴 最高
第 3-4 天	对接 Coze 智能体，完成多轮对话功能	🔴 最高
第 5-6 天	制作演示视频，覆盖所有赛题要求场景	🔴 最高
第 7-8 天	编写答辩 PPT 和项目详细方案	🟡 高
第 9-10 天	全面测试，修复 Bug，准备提交材料	🟡 高
🤝 团队分工
组长（AI 架构师）：Coze 多智能体编排、Prompt 调优、演示视频制作
后端（AI 工程师）：PPT/Word 生成引擎优化、Bug 修复、接口联调
前端（全栈 / 交互）：一体化工作台开发、用户界面优化
知识库（教育专家）：本地知识库建设、学科资料整理
plaintext

# 三、紧急优化建议（10天冲刺必做）
1. **立即删除 `pdf_parser.py`**：避免代码冗余和混淆，所有解析都用 `document_parser.py`
2. **修复 Word 下载路径不一致问题**：
   - `word_generator.py` 中返回的下载URL是 `http://127.0.0.1:8000/downloads/...`
   - 但 `main.py` 中挂载的静态路径是 `/static`
   - **修改为**：`download_url = f"/static/word_project_{project_id}.docx"`
   - 与 PPT 的下载路径保持一致，避免 Coze 下载失败
3. **注释掉 MinIO 相关代码**：目前用不到，避免引入不必要的依赖和错误
4. **提前准备好 `template.pptx`**：至少包含封面、标题+正文、标题+正文+图片三种布局，这是 PPT 生成的基础
5. **测试所有接口**：用 Postman 或 FastAPI 文档逐个测试，确保每个接口都能正常返回

需要我帮你把**Word下载路径不一致的问题**直接改好，并提供一个可直接复制的修正后的 `word_generator.py` 文件吗？