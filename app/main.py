# ==========================================
# 1. 核心库与第三方包导入区
# ==========================================
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta, timezone

# ==========================================
# 2. 数据库与数据模型导入区
# ==========================================
from app.core.database import engine, SessionLocal, Base
from app.models.user import User

# ==========================================
# 3. 开机自动建表动作
# ==========================================
Base.metadata.create_all(bind=engine)

# ==========================================
# 4. FastAPI 实例与中间件配置
# ==========================================
app = FastAPI(
    title="灵犀智课 API",
    description="多模态AI互动式教学智能体后端接口",
    version="1.0.0"
)

# 配置 CORS 跨域中间件 (全量开放，对接前端团队)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 5. 安全加密与数据校验配置
# ==========================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "lingxi_super_secret_key_2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================
# 6. 核心业务接口路由区
# ==========================================

@app.get("/", summary="系统状态检查")
def root():
    return {
        "message": "Hello, 灵犀智课后端已成功启动！",
        "status": "online"
    }


@app.post("/api/users/register", summary="教师账号注册")
def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    try:
        existing_user = db.query(User).filter(User.username == user_data.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="该用户名已被注册，请换一个")

        hashed_pwd = pwd_context.hash(user_data.password)
        new_user = User(username=user_data.username, hashed_password=hashed_pwd)

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return {
            "message": "恭喜！教师账号注册成功",
            "user_id": new_user.id,
            "username": new_user.username
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库操作崩溃: {str(e)}")


@app.post("/api/users/login", summary="教师账号登录")
def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == login_data.username).first()

    if not user or not pwd_context.verify(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": user.username, "user_id": user.id, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "message": "登录成功，欢迎回来！",
        "access_token": encoded_jwt,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }


# ==========================================
# 7. 接入 Coze 专属插件路由
# ==========================================
from app.routers import coze_plugins

app.include_router(coze_plugins.router)