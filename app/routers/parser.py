# ==========================================
# 灵犀智课 - 多模态解析入口 (ParserAgent)
# 文件路径: app/routers/parser.py
# ==========================================

import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

# 导入咱们的两大物理与灵魂引擎
from app.services.pdf_parser import extract_content_from_pdf
from app.services.llm_client import generate_outline_from_text

router = APIRouter(
    prefix="/api/parser",
    tags=["多模态解析 (ParserAgent)"]
)

UPLOAD_DIR = "downloads/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 升级响应模型，增加 generated_outline 字段接住大模型的产出
class UploadResponse(BaseModel):
    message: str
    filename: str
    saved_path: str
    engine_dispatched: str
    status: str
    generated_outline: Optional[dict] = None


@router.post("/upload", summary="节点：上传参考资料 -> 抽取文本 -> LLM生成大纲", response_model=UploadResponse)
async def upload_reference_material(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".docx", ".pptx", ".mp4", ".png", ".jpg", ".jpeg"}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}。"
        )

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件暂存失败: {str(e)}")

    engine_used = "none"
    extracted_info = None
    final_outline = None

    # 分支 A：静态长文本类 (PDF + GLM-4 完整链路)
    if file_ext == ".pdf":
        engine_used = "Static Document Parser (PyMuPDF) + GLM-4 LLM"

        # 1. 物理抽取文本
        extracted_info = extract_content_from_pdf(file_location)

        # 2. 灵魂提炼大纲
        if extracted_info and not extracted_info.get("error"):
            course_topic = os.path.splitext(file.filename)[0]  # 把文件名当作课程主题

            # 为了防止 PDF 太大撑爆 tokens，开发测试时我们截取前 8000 个字符
            text_to_process = extracted_info['full_text'][:8000]

            final_outline = generate_outline_from_text(
                course_topic=course_topic,
                extracted_text=text_to_process
            )

    elif file_ext in [".docx", ".pptx"]:
        engine_used = "Static Document Parser (待接入 Word/PPT 引擎)"
    elif file_ext == ".mp4":
        engine_used = "Dynamic Video Parser (待接入 Whisper)"
    elif file_ext in [".png", ".jpg", ".jpeg"]:
        engine_used = "Image Vision Parser (待接入 Qwen-VL)"

    return {
        "message": "文件处理完成！",
        "filename": file.filename,
        "saved_path": file_location,
        "engine_dispatched": engine_used,
        "status": "success" if final_outline and not final_outline.get("error") else "pending/error",
        "generated_outline": final_outline
    }