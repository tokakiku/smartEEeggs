# ==========================================
# 灵犀智课 - Coze 专属插件接口 (最终实战对齐版)
# 文件路径: app/routers/coze_plugins.py
# ==========================================

import os
import json
import asyncio
import shutil
from fastapi import APIRouter, HTTPException, Query, Body, UploadFile, File
from pydantic import BaseModel, Field
from typing import Any

# 导入底层引擎 (请确保路径正确)
from app.services.pdf_parser import extract_content_from_pdf
from app.services.llm_client import generate_outline_from_text
from app.services.ppt_generator import generate_ppt_from_json

router = APIRouter(prefix="/api/plugins", tags=["Coze 专属对接插件"])

UPLOAD_DIR = "downloads/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# 定义 Body 结构：根据 ngrok 截图，Body 里面其实只有 outline_data 这一项
class CozePluginBody(BaseModel):
    outline_data: Any = Field(..., description="BOPPPS 大纲数据")


# ==========================================
# 插件 1：PDF 文本解析器
# ==========================================
@router.post("/parse_pdf")
async def plugin_parse_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as f:
        shutil.copyfileobj(file.file, f)
    result = extract_content_from_pdf(file_location)
    return {"status": "success", "extracted_text": result.get("full_text", "")}


# ==========================================
# 插件 2：BOPPPS 大纲生成器
# ==========================================
class OutlineRequest(BaseModel):
    course_topic: str
    extracted_text: str


@router.post("/generate_outline")
def plugin_generate_outline(data: OutlineRequest):
    outline_json = generate_outline_from_text(data.course_topic, data.extracted_text)
    return {"status": "success", "outline_data": outline_json}


# ==========================================
# 插件 3：PPT 物理渲染器 (根据 ngrok 真相重构版)
# ==========================================
@router.post("/generate_ppt")
async def plugin_generate_ppt(
        # 1. 对应 ngrok 截图里的 Query Params，去网址后面捞 project_id
        project_id: int = Query(1024, description="项目索引ID"),
        # 2. 对应 ngrok 截图里的 JSON Body，只接 outline_data 这一项
        body: CozePluginBody = Body(...)
):
    """
    精准适配 Coze：Query 拿 ID，Body 拿大纲，手撕字符串套娃。
    """
    try:
        # 获取那坨可能带着引号的字符串
        actual_outline = body.outline_data

        print(f"\n[🚀 收到请求] Project ID: {project_id}")

        # --- 核心“防呆”逻辑：手撕字符串套娃 ---
        if isinstance(actual_outline, str):
            try:
                # 第一次解析：变回字典
                actual_outline = json.loads(actual_outline)
                print("✅ 成功将 outline_data 字符串解析为字典")
            except Exception as e:
                print(f"❌ 解析失败: {str(e)}")
                raise HTTPException(status_code=400, detail="大纲内容不是有效的 JSON 字符串")

        # 确保最后喂给渲染引擎的是个字典
        if not isinstance(actual_outline, dict):
            raise HTTPException(status_code=422, detail="数据格式错误，必须是 JSON 对象或字符串")

        # 物理渲染
        print(f"🎨 [渲染引擎] 开始为项目 {project_id} 生成 PPT...")
        download_url = await asyncio.to_thread(
            generate_ppt_from_json,
            actual_outline,
            project_id
        )

        return {
            "status": "success",
            "message": "PPT 渲染完毕",
            "download_url": download_url
        }

    except Exception as e:
        print(f"❌ 运行崩溃: {str(e)}")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))