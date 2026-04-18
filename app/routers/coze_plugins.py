# ==========================================
# 灵犀智课 - Coze 专属插件接口 (支持 outline_data 原生直插)
# ==========================================

import os
import json
import asyncio
import shutil
import logging
from fastapi import APIRouter, HTTPException, Query, Body, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Any

from app.services.document_parser import parse_document_to_text
from app.services.llm_client import generate_outline_from_text
from app.services.content_generator import generate_page_contents
from app.services.ppt_generator import generate_ppt_from_json
from app.services.export_service import convert_pptx_to_pdf, generate_preview_images, cleanup_temp_files

logger = logging.getLogger("Coze_Plugins")
router = APIRouter(prefix="/api/plugins", tags=["Coze 专属对接插件"])

UPLOAD_DIR = "downloads/uploads"
EXPORT_DIR = "downloads/exports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
import re


# ==========================================
# 🌟 核心提取器：把 outline_data 拍扁成引擎能懂的一维数组
# ==========================================
def flatten_outline_data(outline_data: dict) -> list:
    queue = []

    if "outline_data" in outline_data and isinstance(outline_data["outline_data"], dict):
        outline_data = outline_data["outline_data"]

    syllabus = outline_data.get("syllabus_content", [])

    for stage in syllabus:
        raw_stage = stage.get("stage", "未命名环节")
        stage_title = raw_stage.split("-")[-1].strip() if "-" in raw_stage else raw_stage

        # 🌟 1. 抓取游戏数据
        game_data = stage.get("interactive_game")

        desc = stage.get("content_description", "")
        # 🚨 如果大模型抗旨把 JSON 写在文本里了，用正则强行挖出来
        if not game_data and '"game_type"' in desc:
            match = re.search(r'\{[\s\S]*?"game_type"[\s\S]*?\}', desc)
            if match:
                try:
                    game_data = json.loads(match.group(0))
                    desc = desc.replace(match.group(0), "[🎮 互动游戏已载入]")
                except:
                    pass

        # 🌟 2. 组装当前页数据
        page_item = {"title": stage_title, "content": desc}

        if "core_knowledge_points" in stage:
            for kp in stage["core_knowledge_points"]:
                queue.append({
                    "title": kp.get("point", stage_title),
                    "content": kp.get("explanation", ""),
                    "interactive_game": game_data  # 把游戏数据挂在这一页上
                })
        else:
            page_item["interactive_game"] = game_data  # 把游戏数据挂在这一页上
            queue.append(page_item)

    return queue


# ==========================================
# 数据模型定义 (参数已改回 outline_data)
# ==========================================
class OutlineRequest(BaseModel):
    course_topic: str
    extracted_text: str


class GenerateContentRequest(BaseModel):
    outline_data: Any = Field(..., description="BOPPPS大纲施工图JSON")


class GeneratePPTRequest(BaseModel):
    # 插件4由于需要接收插件3加工后的图片路径，所以做成双重兼容
    outline_data: Any = Field(None, description="大纲施工图(直接测试用)")
    page_queue: Any = Field(None, description="带图片的成品队列(工作流用)")


# 插件 1：万能文档解析器 (现已支持图文及视频)
@router.post("/parse_file", summary="插件1: 万能文档解析")
async def plugin_parse_file(file: UploadFile = File(...)):
    filename = file.filename
    ext = filename.split(".")[-1].lower()

    # 🌟 更新白名单，允许图片与视频格式进入处理管线
    allowed_exts = [
        "pdf", "docx", "pptx", "xlsx", "xls",
        "png", "jpg", "jpeg", "bmp", "webp",
        "mp4", "avi", "mov", "mkv"
    ]
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {ext}")

    file_location = os.path.join(UPLOAD_DIR, filename)
    with open(file_location, "wb+") as f:
        shutil.copyfileobj(file.file, f)

    result = parse_document_to_text(file_location)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))

    return {
        "status": "success",
        "file_name": filename,
        "char_count": result.get("char_count"),
        "extracted_text": result.get("text", "")
    }


# 插件 2：通用大纲生成器
@router.post("/generate_outline", summary="插件2: 通用大纲生成")
def plugin_generate_outline(data: OutlineRequest):
    outline_json = generate_outline_from_text(data.course_topic, data.extracted_text)
    return {"status": "success", "outline_data": outline_json}


# ==========================================
# 🌟 插件 3：并发加工图文 (现已支持原生 outline_data)
# ==========================================
@router.post("/batch_generate_contents", summary="插件3: 并发加工图文")
async def plugin_batch_generate_contents(body: GenerateContentRequest):
    try:
        raw_data = body.outline_data
        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        # 🌟 调用智能提取器，把 BOPPPS 拍扁
        flat_queue = flatten_outline_data(raw_data)

        if not flat_queue:
            raise HTTPException(status_code=422, detail="未能从大纲中提取出有效页面")

        # 扔进加工厂并发处理
        finished_queue = await asyncio.to_thread(generate_page_contents, flat_queue)

        return {"status": "success", "page_queue": finished_queue}
    except Exception as e:
        logger.error(f"❌ 批量加工失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 🌟 插件 4：Banana PPT 全量物理出图
# ==========================================
@router.post("/generate_ppt", summary="插件4: Banana架构物理渲染与导出")
async def plugin_generate_ppt(
        background_tasks: BackgroundTasks,
        project_id: int = Query(1024),
        body: GeneratePPTRequest = Body(...)
):
    try:
        # 智能路由：优先使用带图片的 page_queue，如果没有，就自己把 outline_data 拍扁
        queue = []
        if body.page_queue:
            queue = body.page_queue
            if isinstance(queue, str): queue = json.loads(queue)
        elif body.outline_data:
            raw_data = body.outline_data
            if isinstance(raw_data, str): raw_data = json.loads(raw_data)
            queue = flatten_outline_data(raw_data)
        else:
            raise HTTPException(status_code=400, detail="必须提供 page_queue 或 outline_data")

        # 1. 通过 Banana 架构物理生成 PPTX
        local_pptx_path = await asyncio.to_thread(generate_ppt_from_json, queue, project_id)

        # 2. 生成 PDF
        local_pdf_path = await asyncio.to_thread(convert_pptx_to_pdf, local_pptx_path, EXPORT_DIR)

        # 3. 切片 PNG 预览图
        preview_urls = await asyncio.to_thread(generate_preview_images, local_pdf_path, EXPORT_DIR, project_id)

        # 4. 后台垃圾回收
        background_tasks.add_task(cleanup_temp_files, [UPLOAD_DIR, EXPORT_DIR], 24)

        return {
            "status": "success",
            "message": "Banana 全量渲染与导出完毕",
            "download_url_pptx": f"/static/exports/{os.path.basename(local_pptx_path)}",
            "download_url_pdf": f"/static/exports/{os.path.basename(local_pdf_path)}" if local_pdf_path else None,
            "preview_images": preview_urls
        }
    except Exception as e:
        logger.error(f"❌ 物理渲染失败: {str(e)}")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))