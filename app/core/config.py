from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 项目基础信息
    PROJECT_NAME: str = "灵犀智课 API"

    # 数据库配置
    DATABASE_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# 实例化配置对象，供全局调用
settings = Settings()