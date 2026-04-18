import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from zhipuai import ZhipuAI  # type: ignore
except Exception:  # pragma: no cover
    ZhipuAI = None

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


def _to_str(value: Any) -> str:
    return str(value or "").strip()


def _warn(message: str) -> None:
    print(f"[llm_client][warning] {message}")


def _strip_markdown_fence(text: str) -> str:
    value = _to_str(text)
    if value.startswith("```json"):
        value = value.replace("```json", "", 1).strip()
    elif value.startswith("```"):
        value = value.replace("```", "", 1).strip()
    if value.endswith("```"):
        value = value.rsplit("```", 1)[0].strip()
    return value


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidate = _strip_markdown_fence(text)
    try:
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        maybe = candidate[start : end + 1]
        try:
            loaded = json.loads(maybe)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return None
    return None


def _snippet_lines(text: str, limit: int = 6) -> List[str]:
    lines = []
    for row in re.split(r"[\n。；;]", _to_str(text)):
        line = re.sub(r"\s+", " ", row).strip()
        if len(line) < 8:
            continue
        if line in lines:
            continue
        lines.append(line)
        if len(lines) >= max(1, int(limit)):
            break
    return lines


def _fallback_outline(course_topic: str, extracted_text: str, reason: str = "") -> Dict[str, Any]:
    topic = _to_str(course_topic) or "未命名课程"
    snippets = _snippet_lines(extracted_text, limit=8)
    key_point = snippets[0] if snippets else f"{topic}的核心概念与应用"
    app_point = snippets[1] if len(snippets) > 1 else f"{topic}在教学与实践中的典型场景"
    intro_case = snippets[2] if len(snippets) > 2 else f"请从课程实际案例引入“{topic}”"
    summary_point = snippets[3] if len(snippets) > 3 else f"回顾“{topic}”的关键知识结构"

    if reason:
        _warn(f"outline generation fallback triggered: {reason}")

    return {
        "course_metadata": {
            "title": topic,
            "target_audience": "本科相关专业学生",
            "teaching_objectives": [
                f"理解{topic}的核心概念与基本原理",
                f"能够结合案例分析{topic}的应用逻辑",
                f"形成围绕{topic}的工程与实践意识",
            ],
            "difficulty_level": "中级",
            "total_duration": 45,
        },
        "syllabus_content": [
            {
                "stage": "B (Bridge-in) - 导入",
                "content_description": intro_case,
                "duration": 5,
                "interaction_type": "Q&A",
                "resource_refs": [],
            },
            {
                "stage": "O (Objective) - 目标",
                "content_description": f"明确本节课围绕“{topic}”的知识目标、能力目标与实践目标。",
                "duration": 2,
                "interaction_type": "展示",
                "resource_refs": [],
            },
            {
                "stage": "P1 (Pre-assessment) - 前测",
                "content_description": f"通过快速问答检测学生对“{topic}”先验认知。",
                "duration": 5,
                "interaction_type": "快速投票",
                "resource_refs": [],
            },
            {
                "stage": "P2 (Participatory Learning) - 参与式学习",
                "core_knowledge_points": [
                    {
                        "point": key_point,
                        "is_key_point": True,
                        "is_difficult_point": False,
                        "explanation": app_point,
                        "game_hook": None,
                    }
                ],
                "duration": 25,
                "interaction_type": "讲授与互动",
                "resource_refs": [],
            },
            {
                "stage": "P3 (Post-assessment) - 后测",
                "content_description": f"设置小任务，检验学生对“{topic}”的理解与应用能力。",
                "duration": 5,
                "interaction_type": "在线测试",
                "resource_refs": [],
            },
            {
                "stage": "S (Summary) - 总结",
                "content_description": summary_point,
                "duration": 3,
                "interaction_type": "思维导图展示",
                "resource_refs": [],
            },
        ],
        "resource_pool": [],
    }


def _fallback_brief(course_topic: str, evidence_text: str, reason: str = "") -> str:
    topic = _to_str(course_topic) or "未命名主题"
    lines = _snippet_lines(evidence_text, limit=10)
    if reason:
        _warn(f"brief consolidation fallback triggered: {reason}")

    core = lines[0] if lines else f"{topic}的概念定义与应用背景"
    key = lines[1] if len(lines) > 1 else f"{topic}在课程中的关键知识点"
    difficult = lines[2] if len(lines) > 2 else f"{topic}中的重难点与常见误区"
    cross = lines[3] if len(lines) > 3 else f"{topic}可从教材、资源与案例三层联动讲解"
    practice = lines[4] if len(lines) > 4 else f"结合前沿热点讨论{topic}的实践价值"

    return "\n".join(
        [
            f"【课程主题】{topic}",
            "【核心概念定义】",
            f"- {core}",
            "【教学重点】",
            f"- {key}",
            "【教学难点】",
            f"- {difficult}",
            "【跨层关联（教材↔资源↔热点）】",
            f"- {cross}",
            "【前沿/实践案例】",
            f"- {practice}",
            "【课堂引入与互动建议】",
            f"- 以“{topic}”真实场景引入，先问再讲，最后用小任务收束。",
        ]
    )


def _build_outline_prompts(course_topic: str, extracted_text: str) -> Tuple[str, str]:
    system_prompt = """
你是一位资深的教育教学专家，精通 BOPPPS 教学模型。
请根据参考资料生成结构化教学大纲。
必须严格输出 JSON 对象，不要输出 Markdown 代码块或解释性文字。

输出 JSON schema：
{
  "course_metadata": {
    "title": "课程标题",
    "target_audience": "目标学生群体",
    "teaching_objectives": ["目标1", "目标2", "目标3"],
    "difficulty_level": "初级/中级/高级",
    "total_duration": 45
  },
  "syllabus_content": [
    {
      "stage": "B (Bridge-in) - 导入",
      "content_description": "导入内容",
      "duration": 5,
      "interaction_type": "Q&A / 视频观察",
      "resource_refs": []
    },
    {
      "stage": "O (Objective) - 目标",
      "content_description": "学习目标",
      "duration": 2,
      "interaction_type": "展示",
      "resource_refs": []
    },
    {
      "stage": "P1 (Pre-assessment) - 前测",
      "content_description": "前测内容",
      "duration": 5,
      "interaction_type": "匿名答疑墙 / 快速投票",
      "resource_refs": []
    },
    {
      "stage": "P2 (Participatory Learning) - 参与式学习",
      "core_knowledge_points": [
        {
          "point": "核心知识点",
          "is_key_point": true,
          "is_difficult_point": false,
          "explanation": "讲解逻辑",
          "game_hook": null
        }
      ],
      "duration": 25,
      "interaction_type": "讲授与互动",
      "resource_refs": []
    },
    {
      "stage": "P3 (Post-assessment) - 后测",
      "content_description": "后测内容",
      "duration": 5,
      "interaction_type": "在线测试",
      "resource_refs": []
    },
    {
      "stage": "S (Summary) - 总结",
      "content_description": "总结内容",
      "duration": 3,
      "interaction_type": "思维导图展示",
      "resource_refs": []
    }
  ],
  "resource_pool": []
}
"""
    user_prompt = f"【课程主题】{course_topic}\n\n【参考资料内容】\n{_to_str(extracted_text)[:12000]}"
    return system_prompt.strip(), user_prompt


def _build_consolidation_prompts(course_topic: str, evidence_text: str) -> Tuple[str, str]:
    system_prompt = """
你是一位资深教学教研专家与课程设计专家。
你将收到混合检索后的证据材料（已做初步过滤），请再做精炼整理，输出“教学核心素材简报”。

要求：
1. 去除噪声、重复和无意义片段。
2. 提炼核心概念、教学重点、教学难点。
3. 保留跨层关联（教材↔资源↔热点）中真正有价值的链条。
4. 给出可用于课堂引入、互动、实践应用的素材。
5. 输出纯文本，严格使用以下标题：
【课程主题】
【核心概念定义】
【教学重点】
【教学难点】
【跨层关联（教材↔资源↔热点）】
【前沿/实践案例】
【课堂引入与互动建议】
"""
    user_prompt = f"【课程主题】{course_topic}\n\n【检索证据材料】\n{_to_str(evidence_text)[:12000]}"
    return system_prompt.strip(), user_prompt


def _call_zhipu(system_prompt: str, user_prompt: str, temperature: float) -> Tuple[Optional[str], Dict[str, Any]]:
    api_key = _to_str(os.getenv("ZHIPU_API_KEY"))
    model_name = _to_str(os.getenv("ZHIPU_MODEL")) or "glm-4-flash"
    if not api_key:
        return None, {"provider": "zhipu", "model": model_name, "reason": "ZHIPU_API_KEY not set"}
    if ZhipuAI is None:
        return None, {"provider": "zhipu", "model": model_name, "reason": "zhipuai package not installed"}

    try:
        client = ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = _to_str(getattr(response.choices[0].message, "content", ""))
        return content, {"provider": "zhipu", "model": model_name, "reason": ""}
    except Exception as exc:  # pragma: no cover
        return None, {"provider": "zhipu", "model": model_name, "reason": str(exc)}


def _call_openai(system_prompt: str, user_prompt: str, temperature: float) -> Tuple[Optional[str], Dict[str, Any]]:
    api_key = _to_str(os.getenv("OPENAI_API_KEY"))
    model_name = _to_str(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini"
    if not api_key:
        return None, {"provider": "openai", "model": model_name, "reason": "OPENAI_API_KEY not set"}
    if OpenAI is None:
        return None, {"provider": "openai", "model": model_name, "reason": "openai package not installed"}

    try:
        base_url = _to_str(os.getenv("OPENAI_BASE_URL")) or None
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = _to_str(response.choices[0].message.content)
        return content, {"provider": "openai", "model": model_name, "reason": ""}
    except Exception as exc:  # pragma: no cover
        return None, {"provider": "openai", "model": model_name, "reason": str(exc)}


def _call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Tuple[Optional[str], Dict[str, Any]]:
    preferred = _to_str(os.getenv("LLM_PROVIDER")).lower()
    call_chain = []

    if preferred == "zhipu":
        call_chain = [_call_zhipu, _call_openai]
    elif preferred == "openai":
        call_chain = [_call_openai, _call_zhipu]
    else:
        call_chain = [_call_zhipu, _call_openai]

    last_meta: Dict[str, Any] = {"provider": "none", "model": "", "reason": "no provider attempted"}
    for caller in call_chain:
        content, meta = caller(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature)
        if content:
            return content, meta
        last_meta = meta
    return None, last_meta


def generate_outline_from_text(course_topic: str, extracted_text: str) -> Dict[str, Any]:
    """
    复用已有大纲生成主链。若 API 不可用，则自动返回 schema 兼容的 fallback 大纲。
    """
    topic = _to_str(course_topic) or "未命名课程"
    system_prompt, user_prompt = _build_outline_prompts(topic, extracted_text)

    content, meta = _call_llm(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)
    if content:
        payload = _extract_json_object(content)
        if isinstance(payload, dict):
            return payload
        _warn("LLM returned non-JSON outline, using fallback schema.")
        return _fallback_outline(topic, extracted_text, reason="outline_json_parse_failed")

    reason = _to_str(meta.get("reason")) or "llm_unavailable"
    return _fallback_outline(topic, extracted_text, reason=reason)


def consolidate_lesson_brief(course_topic: str, retrieved_evidence: str) -> Dict[str, Any]:
    """
    将检索证据整理为 clean lesson brief，供后续生成链路直接使用。
    """
    topic = _to_str(course_topic) or "未命名主题"
    system_prompt, user_prompt = _build_consolidation_prompts(topic, retrieved_evidence)
    content, meta = _call_llm(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.2)

    if content:
        cleaned = _strip_markdown_fence(content)
        if cleaned:
            return {
                "text": cleaned,
                "provider": meta.get("provider"),
                "model": meta.get("model"),
                "is_fallback": False,
                "reason": "",
            }

    reason = _to_str(meta.get("reason")) or "llm_unavailable"
    fallback = _fallback_brief(topic, retrieved_evidence, reason=reason)
    return {
        "text": fallback,
        "provider": meta.get("provider"),
        "model": meta.get("model"),
        "is_fallback": True,
        "reason": reason,
    }
