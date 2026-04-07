from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.core.database import Base


class Project(Base):
    __tablename__ = "projects"  # 在 MySQL 中实际生成的表名

    id = Column(Integer, primary_key=True, index=True, autoincrement=True, comment="课件项目主键ID")
    title = Column(String(100), nullable=False, comment="课件/项目标题")

    # === 最关键的外键纽带 ===
    teacher_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属教师ID"
    )

    # === 针对多模态 AI 智能体的核心字段 ===
    # 存放 AI 生成的结构化大纲（比如按章节划分的树状结构数据）
    outline_data = Column(JSON, nullable=True, comment="教学大纲JSON数据")

    # 记录当前这个课件的生成进度
    status = Column(String(20), default="draft", comment="状态: draft(草稿), generating(AI生成中), completed(完成)")

    # 通用时间字段
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), comment="最后更新时间")