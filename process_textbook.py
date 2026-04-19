import json
import os
import re

# 路径配置
KB_DATA_DIR = "kb_data"
EXPORT_DIR = "kb_exports"

os.makedirs(KB_DATA_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


def clean_ocr_text(raw_text: str) -> str:
    """清理破碎换行和多余空格的洗衣机"""
    if not raw_text: return ""
    # 去除中文和中文之间的换行与空格，实现无缝拼接
    cleaned = re.sub(r'([^\x00-\xff])\s+([^\x00-\xff])', r'\1\2', raw_text)
    # 将剩下的多余空格/换行替换为一个正常空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def recursive_extract_text(data) -> str:
    """兜底万能提取器：处理未知嵌套深度的字符串"""
    extracted = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ['id', '_id', 'created_at', 'layer', 'source_type', 'metadata', 'hotspot_index']:
                continue
            extracted.append(recursive_extract_text(value))
    elif isinstance(data, list):
        for item in data:
            extracted.append(recursive_extract_text(item))
    elif isinstance(data, str) and len(data.strip()) > 5:
        extracted.append(data.strip())
    return "\n".join(filter(None, extracted))


def process_document(doc: dict) -> str:
    """
    🌟 核心智能路由：根据不同知识层的真实数据结构精准扒取！
    """
    layer = doc.get("layer", "未知层级")
    data = doc.get("data", {})
    md_content = ""

    # 1. 解析【教材层 Textbook】
    if layer == "textbook":
        book_title = data.get("textbook_info", {}).get("book_title", "未命名教材")
        for chapter in data.get("chapters", []):
            chapter_title = chapter.get("chapter_title", "")
            for section in chapter.get("sections", []):
                section_title = section.get("section_title", "")
                raw_text = clean_ocr_text(section.get("raw_text", ""))
                if raw_text:
                    md_content += f"# 【教材】《{book_title}》- {chapter_title} - {section_title}\n\n{raw_text}\n\n---\n\n"

    # 2. 解析【资源层 Resource】 (PPT 课件解析)
    elif layer == "resource":
        title = data.get("resource_info", {}).get("title", "未命名课件")
        for page in data.get("pages", []):
            page_no = page.get("page_no", "")
            page_text = clean_ocr_text(page.get("page_text", ""))
            if page_text:
                md_content += f"# 【优质资源】{title} - 第 {page_no} 页\n\n{page_text}\n\n---\n\n"

    # 3. 解析【课标层 Syllabus】
    elif layer == "syllabus":
        course_name = data.get("course_info", {}).get("course_name", "未知课程大纲")
        # 提取直接的教学目标
        goals = data.get("teaching_goals", [])
        if goals:
            goals_text = clean_ocr_text(" ".join(goals))
            md_content += f"# 【课程标准】教学大纲核心目标\n\n{goals_text}\n\n---\n\n"

        # 提取散落的其他知识点
        other_text = clean_ocr_text(recursive_extract_text(data.get("knowledge_points", [])))
        if other_text:
            md_content += f"# 【课程标准】大纲知识点详述\n\n{other_text}\n\n---\n\n"

    # 4. 解析【热点层 Hotspot】
    elif layer == "hotspot":
        title = data.get("hotspot_info", {}).get("title", "未命名热点新闻")
        items = data.get("hotspot_item", [])
        item_text = clean_ocr_text(recursive_extract_text(items))
        if item_text:
            md_content += f"# 【时事热点】新闻素材：{title}\n\n{item_text}\n\n---\n\n"

    # 5. 兜底逻辑：处理其他未知结构
    else:
        raw_text = clean_ocr_text(recursive_extract_text(data))
        if raw_text:
            md_content += f"# 【{layer}】补充知识片段\n\n{raw_text}\n\n---\n\n"

    return md_content


def process_and_export_files(filename: str):
    filepath = os.path.join(KB_DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ 找不到文件: {filepath}")
        return

    print(f"⏳ 正在启动特种解析器处理: {filename}...")
    try:
        # 直接读取 JSON 格式的文件
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 {filename} 失败: {e}")
        return

    export_content = ""

    # 无论是列表还是单个字典，统统交给智能路由处理
    if isinstance(data, list):
        for doc in data:
            export_content += process_document(doc)
    elif isinstance(data, dict):
        export_content += process_document(data)

    # 导出为干净的 Markdown 文件
    export_filename = filename.replace(".json", "_clean.md")
    export_path = os.path.join(EXPORT_DIR, export_filename)

    with open(export_path, 'w', encoding='utf-8') as f:
        f.write(export_content)

    print(f"✅ 完成！结构化的高纯度语料已导出至: {export_path}")


if __name__ == "__main__":
    # 指定读取的都是 .json 文件
    files = [
        "ruijie_kb.textbook_docs.json",
        "ruijie_kb.syllabus_docs.json",
        "ruijie_kb.resource_docs.json",
        "ruijie_kb.hotspot_docs.json"
    ]

    for f in files:
        process_and_export_files(f)

    print("\n🚀 大满贯！四层知识库已经全部打平为带标题标记的极致 Markdown！请直接上传 Coze！")