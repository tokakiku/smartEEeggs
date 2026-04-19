import os
import json
from zhipuai import ZhipuAI
from pymilvus import MilvusClient
from dotenv import load_dotenv

# 加载 .env 里的环境变量
load_dotenv()

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# ==========================================
# 🌟 安全读取云端 Zilliz 配置
# ==========================================
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")

if not ZILLIZ_URI or not ZILLIZ_TOKEN:
    raise ValueError("🚨 严重错误：未在 .env 中找到 ZILLIZ 数据库的 URI 或 TOKEN！")

COLLECTION_NAME = "lingxi_knowledge_base"
DIMENSION = 1024

# 初始化云端客户端
milvus_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)


# 如果表已存在，重置它（演示阶段常用）
if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    milvus_client.drop_collection(collection_name=COLLECTION_NAME)

# 创建 Collection
milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=DIMENSION
)
print(f"✅ 云端向量表 {COLLECTION_NAME} 初始化成功！")


def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response.data[0].embedding


def build_database():
    export_dir = "kb_exports"
    if not os.path.exists(export_dir):
        print("⚠️ 找不到 kb_exports 文件夹！")
        return

    insert_data = []
    idx = 0

    for filename in os.listdir(export_dir):
        if not filename.endswith(".md"): continue
        filepath = os.path.join(export_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = [c.strip() for c in f.read().split("---") if len(c.strip()) > 10]

        for chunk in chunks:
            try:
                vector = get_embedding(chunk)
                insert_data.append({
                    "id": idx,
                    "vector": vector,
                    "text": chunk,
                    "source": filename
                })
                idx += 1
                print(f"   [Embedding] 进度: {idx}")
            except Exception as e:
                print(f"   ❌ 失败: {e}")

    if insert_data:
        # 批量上传到云端
        milvus_client.insert(collection_name=COLLECTION_NAME, data=insert_data)
        print(f"\n🚀 云端同步完成！共计 {len(insert_data)} 个知识块已存入 Zilliz Cloud！")


if __name__ == "__main__":
    build_database()