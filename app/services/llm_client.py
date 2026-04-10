# ==========================================
# 灵犀智课 - LLM 大模型驱动引擎 (智谱 GLM-4)
# 文件路径: app/services/llm_client.py
# ==========================================

import json
from zhipuai import ZhipuAI

# 注入你的超级引擎钥匙
ZHIPU_API_KEY = "b2918f0171804e7cab964c4276bbd669.pqKTd3tMs7KszsEV"
client = ZhipuAI(api_key=ZHIPU_API_KEY)


def generate_outline_from_text(course_topic: str, extracted_text: str) -> dict:
    """
    根据组长 ljy 的最新 BOPPPS 标准 Schema，
    将枯燥的 PDF 纯文本转化为结构化的教学大纲 JSON。
    """
    print(f"🧠 [大模型引擎] 正在呼叫智谱 GLM-4 提炼《{course_topic}》的大纲...")

    # 核心改造：完全替换为 ljy 提供的完整版 BOPPPS 模板
    system_prompt = """
    你是一位资深的教育教学专家，精通 BOPPPS 教学模型。
    请根据我提供的【参考资料内容】，提取核心知识点，并自动生成一份结构化的教学大纲。

    【重要指令】
    必须严格按照以下 JSON Schema 输出，绝对不要包含任何额外的 Markdown 标记（如 ```json）或解释性文字，只能输出纯 JSON 字符串。
    你需要根据资料内容，智能填充以下所有字段（如果没有对应的素材资源，resource_pool 和 resource_refs 暂时置空或填空数组）：

    {
      "course_metadata": {
        "title": "课程标题",
        "target_audience": "目标学生群体（根据资料推测）",
        "teaching_objectives": ["目标1: 知识维度", "目标2: 能力维度", "目标3: 素养维度"],
        "difficulty_level": "难度系数（如：初级/中级/高级）",
        "total_duration": 45
      },
      "syllabus_content": [
        {
          "stage": "B (Bridge-in) - 导入",
          "content_description": "设计具体的导入情境或问题",
          "duration": 5,
          "interaction_type": "Q&A / 视频观察",
          "resource_refs": []
        },
        {
          "stage": "O (Objective) - 目标",
          "content_description": "本节课学习后学生应达到的具体指标",
          "duration": 2,
          "interaction_type": "展示",
          "resource_refs": []
        },
        {
          "stage": "P1 (Pre-assessment) - 前测",
          "content_description": "设计检测学生既有知识储备的问题",
          "duration": 5,
          "interaction_type": "匿名答疑墙 / 快速投票",
          "resource_refs": []
        },
        {
          "stage": "P2 (Participatory Learning) - 参与式学习",
          "core_knowledge_points": [
            {
              "point": "提取的核心知识点1",
              "is_key_point": true,
              "is_difficult_point": false,
              "explanation": "详细讲解逻辑",
              "game_hook": null
            }
          ],
          "duration": 25,
          "interaction_type": "讲授与互动",
          "resource_refs": []
        },
        {
          "stage": "P3 (Post-assessment) - 后测",
          "content_description": "设计课堂效果检测习题或任务",
          "duration": 5,
          "interaction_type": "在线测试",
          "resource_refs": []
        },
        {
          "stage": "S (Summary) - 总结",
          "content_description": "知识点梳理与思维导图生成",
          "duration": 3,
          "interaction_type": "思维导图展示",
          "resource_refs": []
        }
      ],
      "resource_pool": []
    }
    """

    user_prompt = f"【课程主题】{course_topic}\n\n【参考资料内容】\n{extracted_text}"

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )

        llm_output = response.choices[0].message.content.strip()

        # 脱掉 Markdown 外衣
        if llm_output.startswith("```json"):
            llm_output = llm_output.replace("```json", "", 1)
        if llm_output.endswith("```"):
            llm_output = llm_output.rsplit("```", 1)[0]

        llm_output = llm_output.strip()

        outline_dict = json.loads(llm_output)
        print("✅ [大模型引擎] 奇迹发生！已严格按照组长 ljy 的 BOPPPS 模板生成大纲！")
        return outline_dict

    except json.JSONDecodeError as e:
        print(f"❌ [大模型引擎] JSON解析失败: {str(e)}")
        print(f"模型的原始输出为:\n{llm_output}")
        return {"error": "JSON解析失败"}
    except Exception as e:
        print(f"❌ [大模型引擎] 调用失败: {str(e)}")
        return {"error": str(e)}