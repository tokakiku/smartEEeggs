# 文件路径: app/utils/minio_client.py
import os
from minio import Minio
from minio.error import S3Error

# ==========================================
# MinIO 配置信息 (请根据你的 docker-compose.yml 确认账号密码)
# 默认通常是 minioadmin / minioadmin
# ==========================================
MINIO_ENDPOINT = "127.0.0.1:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
BUCKET_NAME = "lingxi-coursewares" # 咱们专门存课件的桶

# 初始化 MinIO 客户端
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # 本地开发没配 HTTPS，所以写 False
)

def upload_ppt_to_minio(file_path: str, object_name: str) -> str:
    """
    将本地生成的 PPT 上传到 MinIO，并返回下载链接
    """
    try:
        # 1. 检查桶存不存在，不存在就建一个
        if not client.bucket_exists(BUCKET_NAME):
            client.make_bucket(BUCKET_NAME)
            print(f"📦 [MinIO] 创建了新的存储桶: {BUCKET_NAME}")

        # 2. 执行上传
        print(f"⏳ [MinIO] 正在将 {file_path} 上传至云端...")
        client.fput_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            file_path=file_path,
            content_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
        print("✅ [MinIO] 上传成功！")

        # 3. 拼接下载链接 (注意：本地测试时用 127.0.0.1，后续讲怎么给 Coze 用)
        download_url = f"http://{MINIO_ENDPOINT}/{BUCKET_NAME}/{object_name}"
        return download_url

    except S3Error as err:
        print(f"❌ [MinIO] 上传失败: {err}")
        return ""