from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 创建 FastAPI 实例
app = FastAPI(
    title="灵犀智课 API",
    description="多模态AI互动式教学智能体后端接口",
    version="1.0.0"
)

# === 新增：配置 CORS 跨域中间件 ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的跨域来源。开发阶段填 "*" 允许所有，上线时改成前端的真实域名
    allow_credentials=True, # 允许携带 Cookie 等凭证信息
    allow_methods=["*"],  # 允许所有的请求方法 (GET, POST, PUT, DELETE等)
    allow_headers=["*"],  # 允许所有的请求头
)
# ==================================

# 写第一个测试接口
@app.get("/")
def root():
    return {
        "message": "Hello, 灵犀智课后端已成功启动！",
        "status": "online"
    }