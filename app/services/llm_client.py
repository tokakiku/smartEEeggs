# ==========================================
# 灵犀智课 - 核心 LLM 大脑 (ZhipuAI 旗舰版 - 命题扣题版)
# 特性：接入 GLM-4-Plus | 强制输出教学方法、活动设计与作业
# ==========================================

import os
import re
import json
import logging
from zhipuai import ZhipuAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("LLM_Client")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
if not ZHIPU_API_KEY:
    logger.warning("🚨 警告：未找到 ZHIPU_API_KEY，请检查 .env 文件！")

client = ZhipuAI(api_key=ZHIPU_API_KEY)

def extract_json_from_text(text: str) -> dict:
    try:
        text = text.strip().strip("```json").strip("```").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("⚠️ 直接解析 JSON 失败，尝试正则暴力抠取...")
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as e:
                logger.error(f"❌ 正则解析依然失败: {e}")
                raise ValueError("大模型返回的内容无法解析为有效 JSON")
        raise ValueError("大模型未返回 JSON 格式数据")

def generate_outline_from_text(course_topic: str, extracted_text: str) -> dict:
    logger.info(f"🧠 [Zhipu 引擎] 正在为主题《{course_topic}》生成命题标准施工图...")

    # 🌟 终极优化 Prompt：严格扣紧“目标、过程、方法、活动设计、作业”五大命题要求，并加入互动游戏！
    prompt = f"""你是一位顶级的教育架构师。请根据以下[参考资料]，为主题《{course_topic}》生成一份符合 BOPPPS 教学模型，且严格满足命题格式要求的课件大纲 JSON。

    【核心指令与规范】
    1. 必须输出名为 "outline_data" 的 JSON 对象。
    2. "course_metadata" 中必须包含以下字段：
       - "teaching_methods": 数组格式，列出本课使用的主要教学方法（如讲授法、任务驱动法等）。
       - "homework": 字符串格式，为本节课设计的具体课后作业任务。
    3. "syllabus_content" 是代表教学过程的数组。`stage` 必须使用 "B (Bridge-in) - 导入" 等 6 个固定标识。
    4. 在每个环节的内部描述或知识点中，必须体现具体的“课堂活动设计”（如：小组讨论、案例分析、角色扮演等）。
    5. 互动游戏生成规则：为了增强课堂趣味性，请你根据教学内容，在合适的环节（如 P2 或 P3）中插入一个互动游戏。
    在该环节的 JSON 对象中新增一个 "interactive_game" 字段。游戏类型必须从以下三种中**随机选择最合适的一种**：

      - 类型 1：知识消消乐 (memory_match)。适用于概念匹配。
        格式要求：{{"game_type": "memory_match", "pairs": [{{"left": "概念1", "right": "解释1"}}, {{"left": "概念2", "right": "解释2"}}]}} (需提供 4 对)

      - 类型 2：概念分类 (drag_sort)。适用于归纳分类。
        格式要求：{{"game_type": "drag_sort", "categories": ["类别A", "类别B"], "items": [{{"word": "词条1", "target": "类别A"}}, {{"word": "词条2", "target": "类别B"}}]}} (需提供 6 个词条)

      - 类型 3：情景闯关 (scenario_quiz)。适用于逻辑推理或选择。
        格式要求：{{"game_type": "scenario_quiz", "questions": [{{"scenario": "情景描述和问题", "options": ["选项A", "选项B", "选项C"], "correct_index": 1}}]}} (需提供 3 个问题)

    【期望的 JSON 格式示例】
    {{
      "outline_data": {{
        "course_metadata": {{
          "title": "...",
          "target_audience": "...",
          "teaching_objectives": ["...", "..."],
          "difficulty_level": "...",
          "total_duration": 45,
          "teaching_methods": ["案例分析法", "启发式提问", "小组合作探究"],
          "homework": "请结合今天所学内容，完成一份关于...的实践报告，不少于500字。"
        }},
        "syllabus_content": [
          {{
            "stage": "B (Bridge-in) - 导入",
            "content_description": "【活动设计：视频赏析】播放一段...视频，提出核心问题...",
            "duration": 5,
            "interaction_type": "观看与问答"
          }}
        ]
      }}
    }}

    [参考资料开始]
    {extracted_text}
    [参考资料结束]

    请直接输出 JSON 结果：
    """

    try:
        response = client.chat.completions.create(
            model="glm-4-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.9
        )
        raw_text = response.choices[0].message.content
        parsed_json = extract_json_from_text(raw_text)
        logger.info("✅ [Zhipu 引擎] 命题标准施工图生成完毕！")
        return parsed_json
    except Exception as e:
        logger.error(f"❌ [Zhipu 引擎] 大纲生成崩溃: {str(e)}")
        raise e