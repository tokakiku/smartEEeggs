# ==========================================
# 灵犀智课 - PDF 静态文档解析引擎
# 文件路径: app/services/pdf_parser.py
# ==========================================

import fitz  # PyMuPDF 的导入名


def extract_content_from_pdf(file_path: str) -> dict:
    """
    深度解析 PDF 文件
    返回一个包含全文文本和基础元数据的字典
    """
    print(f"📄 [PDF引擎] 开始解析文件: {file_path}")

    result = {
        "total_pages": 0,
        "full_text": "",
        "error": None
    }

    try:
        # 打开 PDF 文件
        doc = fitz.open(file_path)
        result["total_pages"] = len(doc)

        extracted_text = ""
        # 逐页提取文字
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            # 加上页码标记，方便后续大模型理解上下文位置
            extracted_text += f"\n\n--- [第 {page_num + 1} 页] ---\n"
            extracted_text += text.strip()

        result["full_text"] = extracted_text
        print(f"✅ [PDF引擎] 解析完成，共提取 {result['total_pages']} 页文本。")

    except Exception as e:
        print(f"❌ [PDF引擎] 解析失败: {str(e)}")
        result["error"] = str(e)

    return result