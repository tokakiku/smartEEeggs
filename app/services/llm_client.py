# ==========================================
# 灵犀智课 - 核心 LLM 大脑 (Zilliz Cloud 混合检索安全版)
# 特性：接入 GLM-4-Plus | 云端向量库安全调用 | 双路召回 RAG 融合
# ==========================================

import os
import re
import json
import logging
from zhipuai import ZhipuAI
from pymilvus import MilvusClient
from dotenv import load_dotenv

# 加载 .env 配置文件
load_dotenv()
logger = logging.getLogger("LLM_Client")

# 智谱 AI 配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
if not ZHIPU_API_KEY:
    logger.warning("🚨 警告：未找到 ZHIPU_API_KEY，请检查 .env 文件！")

client = ZhipuAI(api_key=ZHIPU_API_KEY)

# ==========================================
# 🌟 安全读取 Zilliz Cloud (云端 Milvus) 配置
# ==========================================
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "lingxi_knowledge_base"

# 安全初始化云端客户端
if ZILLIZ_URI and ZILLIZ_TOKEN:
    try:
        # 连接云端向量引擎，取代本地 .db 文件
        milvus_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        logger.info("✅ 已安全连接 Zilliz Cloud 云端向量库")
    except Exception as e:
        logger.error(f"❌ 连接 Zilliz 失败: {e}")
        milvus_client = None
else:
    logger.warning("⚠️ 未在 .env 中配置 ZILLIZ 信息，系统将降级为纯外部语料依赖。")
    milvus_client = None


def extract_json_from_text(text: str) -> dict:
    """
    从大模型返回的文本中提取并解析 JSON 对象
    """
    try:
        text = text.strip().strip("```json").strip("```").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("⚠️ 直接解析 JSON 失败，尝试正则提取...")
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"❌ 正则解析依然失败: {e}")
                raise ValueError("内容无法解析为有效 JSON")
        raise ValueError("未返回 JSON 格式数据")


def search_cloud_knowledge(query: str, top_k: int = 3) -> str:
    """
    从 Zilliz Cloud 云端召回最相关的知识片段
    """
    if not milvus_client:
        return ""

    try:
        # 1. 提问向量化
        query_vector = client.embeddings.create(
            model="embedding-2",
            input=query,
        ).data[0].embedding

        # 2. 云端相似度检索
        search_res = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "source"]
        )

        # 3. 拼接检索到的证据
        retrieved_texts = []
        for hits in search_res:
            for hit in hits:
                source = hit["entity"].get("source", "云端知识库")
                text_content = hit["entity"].get("text", "")
                retrieved_texts.append(f"【来源: {source}】\n{text_content}")

        if retrieved_texts:
            logger.info(f"✅ 成功从 Zilliz Cloud 召回 {len(retrieved_texts)} 条证据")
        return "\n\n".join(retrieved_texts)
    except Exception as e:
        logger.error(f"❌ 云端向量检索失败: {e}")
        return ""


def generate_outline_from_text(course_topic: str, extracted_text: str) -> dict:
    """
    结合云端检索证据与外部提取文本，生成标准 BOPPPS 课件大纲
    """
    logger.info(f"🧠 [Zhipu 引擎] 正在为主题《{course_topic}》生成命题标准施工图...")

    # 🚀 双路召回融合：云端检索 + 插件提取
    cloud_evidence = search_cloud_knowledge(course_topic, top_k=3)

    combined_evidence = ""
    if extracted_text:
        combined_evidence += f"【外部资料片段】\n{extracted_text}\n\n"
    if cloud_evidence:
        combined_evidence += f"【云端图谱增强知识】\n{cloud_evidence}\n\n"

    prompt = f"""你是一位顶级的教育架构师。请根据以下【知识库检索证据】，为主题《{course_topic}》生成一份符合 BOPPPS 教学模型，且严格满足命题格式要求的课件大纲 JSON。

    【知识库处理原则】
    1. 必须以检索到的“教材原文/大纲”作为核心知识点依据，绝不可胡编乱造。
    2. 将“时事热点”或“优秀资源”作为案例，巧妙融入到“导入(Bridge-in)”或“参与式学习(Participatory Learning)”环节中。

    【核心指令与规范】
    1. 必须输出名为 "outline_data" 的 JSON 对象。
    2. "course_metadata" 必须包含: "teaching_methods" 数组和 "homework" 字符串。
    3. "syllabus_content" 必须遵循 BOPPPS 的 6 个阶段。
    4. 互动游戏规则：在 P2 或 P3 环节插入字段 "interactive_game"。
       可选类型：memory_match, drag_sort, scenario_quiz。

    【知识库检索证据开始】
    {combined_evidence}
    【知识库检索证据结束】

    请直接输出 JSON 结果：
    """

    try:
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            top_p=0.9
        )
        raw_text = response.choices[0].message.content
        parsed_json = extract_json_from_text(raw_text)
        logger.info("✅ [Zhipu 引擎] 命题标准施工图生成完毕！")
        return parsed_json
    except Exception as e:
        logger.error(f"❌ [Zhipu 引擎] 大纲生成崩溃: {str(e)}")
        raise e