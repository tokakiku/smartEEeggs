from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # 🌟 新增：静态文件服务
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta, timezone
import os # 🌟 新增

# 数据库与模型
from app.core.database import engine, SessionLocal, Base
from app.models.user import User

# 自动建表
Base.metadata.create_all(bind=engine)

app = FastAPI(title="灵犀智课 API", version="1.0.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🌟 核心新增：挂载本地目录到网络，供 Coze 下载
# ==========================================
os.makedirs("downloads/uploads", exist_ok=True)
os.makedirs("downloads/exports", exist_ok=True)
app.mount("/static", StaticFiles(directory="downloads"), name="static")


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "lingxi_super_secret_key_2026"
ALGORITHM = "HS256"


class UserAuth(BaseModel):
    username: str
    password: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"message": "灵犀智课后端服务运行中", "status": "online"}


@app.post("/api/users/register")
def register(user_data: UserAuth, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    hashed_pwd = pwd_context.hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    return {"message": "注册成功"}


@app.post("/api/users/login")
def login(login_data: UserAuth, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == login_data.username).first()
    if not user or not pwd_context.verify(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="账号或密码错误")

    expire = datetime.now(timezone.utc) + timedelta(minutes=1440)
    token = jwt.encode({"sub": user.username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "username": user.username}

# 1. 引入原有的 Coze 插件路由
from app.routers import coze_plugins
app.include_router(coze_plugins.router)

# 2. 引入新增的 Word 插件路由 (这一步很关键！)
from app.routers import word_plugins
app.include_router(word_plugins.router)