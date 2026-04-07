from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    __tablename__ = "users"  # 在 MySQL 中实际生成的表名

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="用户主键ID")
    username = Column(String(50), unique=True, index=True, nullable=False, comment="登录用户名")
    hashed_password = Column(String(255), nullable=False, comment="加密后的密码")

    # 针对业务的字段
    role = Column(String(20),default="teacher", comment="角色：teacher 或 admin")
    is_active = Column(Boolean, default=True, comment="账号是否启用")

    # 记录时间的通用字段
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), comment="最后更新时间")