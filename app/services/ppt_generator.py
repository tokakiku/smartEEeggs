# ==========================================
# 灵犀智课 - 高颜值 PPT 渲染引擎 (适配 BOPPPS 完整版)
# 文件路径: app/services/ppt_generator.py
# ==========================================

import os
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE
from app.utils.minio_client import upload_ppt_to_minio


def generate_ppt_from_json(json_data: dict, project_id: int) -> str:
    """
    核心 PPT 渲染引擎 (v2.1 防溢出升级版)
    精准控制字号与排版，防止大段文本撑爆页面
    """
    print(f"🚀 [渲染引擎] 收到任务！正在为课件 {project_id} 渲染实体 PPT...")

    template_path = "template.pptx"
    if os.path.exists(template_path):
        prs = Presentation(template_path)
    else:
        prs = Presentation()
        print("⚠️ [渲染引擎] 未找到 template.pptx，使用默认白板模式")

    metadata = json_data.get("course_metadata", {})
    syllabus = json_data.get("syllabus_content", [])

    # ==========================================
    # 2. 封面渲染
    # ==========================================
    title_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_layout)

    if slide.shapes.title:
        slide.shapes.title.text = metadata.get("title", "未命名课程")

    if len(slide.placeholders) > 1:
        target_audience = metadata.get("target_audience", "通用受众")
        difficulty = metadata.get("difficulty_level", "标准")
        duration = metadata.get("total_duration", 45)

        subtitle_text = f"🎯 授课对象：{target_audience}\n"
        subtitle_text += f"⏳ 课程时长：{duration} 分钟 | 📊 难度系数：{difficulty}\n"
        subtitle_text += "✨ 由灵犀智课 AI 备课引擎自动生成"

        tf = slide.placeholders[1].text_frame
        tf.text = subtitle_text
        # 控制副标题字号
        for paragraph in tf.paragraphs:
            paragraph.font.size = Pt(16)

    # ==========================================
    # 3. 教学目标页
    # ==========================================
    objectives = metadata.get("teaching_objectives", [])
    if objectives:
        content_layout = prs.slide_layouts[1]
        slide = prs.slides.add_slide(content_layout)

        if slide.shapes.title:
            slide.shapes.title.text = "教学目标 (Objectives)"

        if len(slide.placeholders) > 1:
            tf = slide.placeholders[1].text_frame
            tf.word_wrap = True
            tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE  # 开启自动缩放

            # 清空默认的第一个空段落
            tf.clear()

            for obj in objectives:
                p = tf.add_paragraph()
                p.text = f"✅ {obj}"
                p.level = 0
                p.font.size = Pt(20)  # 明确指定目标页字号

    # ==========================================
    # 4. 正文渲染：循环解析 syllabus_content
    # ==========================================
    content_layout = prs.slide_layouts[1]

    for item in syllabus:
        slide = prs.slides.add_slide(content_layout)

        if slide.shapes.title:
            slide.shapes.title.text = item.get("stage", "教学环节")

        if len(slide.placeholders) > 1:
            tf = slide.placeholders[1].text_frame
            tf.word_wrap = True
            tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE  # 开启自动缩放

            # 清空自带的空段落，防止顶部留白过多
            tf.clear()

            # 顶部标签 (灰色，小字号)
            meta_p = tf.add_paragraph()
            meta_p.text = f"⏱️ 建议时长: {item.get('duration', 0)}分钟 | 💡 互动形式: {item.get('interaction_type', '讲授')}"
            meta_p.level = 0
            meta_p.font.size = Pt(14)
            meta_p.font.color.rgb = RGBColor(128, 128, 128)  # 设置为高级灰

            # 加个空段落作为间距
            spacer = tf.add_paragraph()
            spacer.font.size = Pt(10)

            # 核心知识点列表
            if "core_knowledge_points" in item and item["core_knowledge_points"]:
                for kp in item["core_knowledge_points"]:
                    p = tf.add_paragraph()
                    prefix = "⭐ [重点] " if kp.get("is_key_point") else "🔸 "
                    prefix = "🔥 [重难点] " if kp.get("is_difficult_point") else prefix

                    p.text = f"{prefix}{kp.get('point', '')}"
                    p.level = 0
                    p.font.size = Pt(20)  # 主知识点字号控制
                    p.font.bold = True

                    if kp.get('explanation'):
                        sub_p = tf.add_paragraph()
                        sub_p.text = kp.get('explanation', '')
                        sub_p.level = 1
                        sub_p.font.size = Pt(16)  # 解释说明文字再小一号，拉开层次

            # 普通段落描述
            else:
                desc = item.get("content_description", "")
                if desc:
                    p = tf.add_paragraph()
                    p.text = desc
                    p.level = 0
                    p.font.size = Pt(20)

    # ==========================================
    # 5. 物理存储与云端上传
    # ==========================================
    output_dir = "downloads"
    os.makedirs(output_dir, exist_ok=True)
    local_file_path = f"{output_dir}/ppt_project_{project_id}.pptx"
    prs.save(local_file_path)
    print(f"💾 [渲染引擎] 本地文件已生成: {local_file_path}")

    object_name = f"ppt_{project_id}.pptx"
    cloud_url = upload_ppt_to_minio(local_file_path, object_name)

    if cloud_url:
        return cloud_url
    else:
        return local_file_path