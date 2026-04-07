# ==========================================
# 1. 核心库与第三方包导入区
# ==========================================
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from passlib.context import CryptContext
import traceback
import jwt
from datetime import datetime, timedelta
from app.routers import lessonprep
# ==========================================
# 2. 数据库与数据模型 (图纸) 导入区
# ==========================================
from app.core.database import engine, SessionLocal, Base
from app.models.user import User

# 注意：如果你之前还写了 project.py (课件表)，请取消下面这行的注释
# from app.models.project import Project

# ==========================================
# 3. 【极其重要】开机自动建表动作
# ==========================================
# 这行代码会在服务器启动的第一秒，拿着上面导入的 User (和 Project) 图纸，
# 去 lingxi_db 数据库里检查。如果没有表，它会自动帮你把表建出来！
Base.metadata.create_all(bind=engine)

# ==========================================
# 4. FastAPI 实例与中间件配置
# ==========================================
app = FastAPI(
    title="灵犀智课 API",
    description="多模态AI互动式教学智能体后端接口",
    version="1.0.0"
)

# 配置 CORS 跨域中间件 (给前端 hqr 兄弟开绿灯)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有跨域请求，上线时应改为前端真实域名
    allow_credentials=True,  # 允许携带 Cookie
    allow_methods=["*"],  # 允许所有请求方法 (GET/POST等)
    allow_headers=["*"],  # 允许所有请求头
)

# ==========================================
# 5. 安全加密与数据校验配置
# ==========================================
# 初始化密码加密工具 (固定使用稳定的 bcrypt 算法)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# === JWT Token 配置参数 ===
SECRET_KEY = "lingxi_super_secret_key_2026"  # 这是用来给 Token 盖章的绝密印泥，千万不能泄露
ALGORITHM = "HS256"                          # 加密算法
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24        # 钥匙的有效期：24小时

# 定义前端注册时必须传的数据格式 (防格式报错 422 的功臣)
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str
# 数据库连接“发牌员”：每次请求来时发一个连接，请求结束时自动回收
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================
# 6. 核心业务接口路由区
# ==========================================

# 测试接口 1：系统状态根目录
@app.get("/", summary="系统状态检查")
def root():
    return {
        "message": "Hello, 灵犀智课后端已成功启动！",
        "status": "online"
    }


# 【主线任务完成】：真实的教师注册接口
@app.post("/api/users/register", summary="教师账号注册")
def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    try:
        # 第一关：去数据库查一下，这个用户名是不是已经被别人抢注了？
        existing_user = db.query(User).filter(User.username == user_data.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="该用户名已被注册，请换一个")

        # 第二关：给密码穿上“防弹衣”（哈希加密）
        hashed_pwd = pwd_context.hash(user_data.password)

        # 第三关：按照我们画好的图纸 (User 模型)，生成一条新数据
        new_user = User(
            username=user_data.username,
            hashed_password=hashed_pwd
        )

        # 第四关：正式存入 MySQL 仓库！
        db.add(new_user)
        db.commit()  # 确认提交
        db.refresh(new_user)  # 刷新获取自动生成的自增 ID

        # 返回成功信息给前端
        return {
            "message": "恭喜！教师账号注册成功",
            "user_id": new_user.id,
            "username": new_user.username
        }

    except HTTPException:
        # 如果是我们上面主动抛出的 400 错误，直接放行给前端
        raise
    except Exception as e:
        # 终极排雷大法：捕获其他所有未知崩溃，吐出真实原因
        error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"数据库操作崩溃啦！真实原因为: {error_detail}")


# 前端止渴接口：给 hqr 画界面的假数据
@app.get("/api/project/mock_outline", summary="获取假大纲数据")
def get_mock_course_outline():
    """返回符合队友 ljy 设计的 CourseInstruction 格式的假数据"""
    return {
        "course_topic": "计算机网络基础：TCP三次握手",
        "duration": "45分钟",
        "style": "BOPPPS互动式教学",
        "slides": [
            {
                "title": "什么是TCP连接？",
                "content": "TCP是一种面向连接的、可靠的、基于字节流的传输层通信协议。",
                "image_prompt": "一张展示两台电脑之间建立数据传输通道的科技感插图",
                "logic": "引入概念，吸引学生注意力"
            },
            {
                "title": "三次握手核心步骤",
                "content": "1. SYN (发起) \n2. SYN-ACK (响应) \n3. ACK (确认)",
                "image_prompt": "用生动的动画图解展示客户端和服务器抛球互动的过程",
                "logic": "拆解核心重难点"
            }
        ]
    }


# 基建探针接口：保留下来，以后如果 Docker 坏了可以随时自检
@app.post("/api/users/debug", summary="终极排雷探针")
def debug_system():
    try:
        test_pwd = pwd_context.hash("123456")
    except Exception as e:
        return {"案发现场": "密码加密库彻底崩溃", "详细死因": str(e)}

    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        return {"案发现场": "数据库连接彻底失败", "详细死因": str(e)}

    try:
        db = SessionLocal()
        db.execute(text("SELECT hashed_password FROM users LIMIT 1"))
        db.close()
    except Exception as e:
        return {"案发现场": "数据库 users 表结构不对", "详细死因": str(e)}

    return {"案发现场": "安全", "结果": "所有底层组件全部正常运转！"}


@app.post("/api/users/login", summary="教师账号登录")
def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    # 第一关：去数据库找找有没有这个人
    user = db.query(User).filter(User.username == login_data.username).first()

    # 第二关：如果人没找到，或者“用户输入的明文密码”和“数据库里的加密密码”对不上号！
    if not user or not pwd_context.verify(login_data.password, user.hashed_password):
        # 故意模糊提示，防黑客爆破
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 第三关：身份验证通过！开始给它制作“数字钥匙 (JWT Token)”
    # 1. 设定这把钥匙什么时候过期
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # 2. 把用户的核心信息（不含密码！）刻在钥匙上
    to_encode = {"sub": user.username, "user_id": user.id, "exp": expire}
    # 3. 盖上我们的绝密印章
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    # 第四关：把钥匙亲手交还给前端
    return {
        "message": "登录成功，欢迎回来！",
        "access_token": encoded_jwt,  # 这就是那把长长的数字钥匙
        "token_type": "bearer",  # 国际标准，告诉前端这是一把 bearer 类型的钥匙
        "user_id": user.id,
        "username": user.username
    }

app.include_router(lessonprep.router)