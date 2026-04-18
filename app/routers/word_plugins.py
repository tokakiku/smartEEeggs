# ==========================================
# 灵犀智课 - Word 文案生成插件模块
# 文件路径: app/routers/word_plugins.py
# ==========================================

import json
import asyncio
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import Any

# 导入你即将编写的底层 Word 渲染引擎
from app.services.word_generator import generate_word_from_json

router = APIRouter(
    prefix="/api/word_plugins",
    tags=["Word 详案插件"]
)


# 复用相同的 Body 结构，确保对接一致性
class WordPluginBody(BaseModel):
    outline_data: Any = Field(..., description="BOPPPS 大纲数据")


@router.post("/generate_word", summary="Coze插件4: 生成 Word 详细教案")
async def plugin_generate_word(
        project_id: int = Query(1024, description="项目索引ID"),
        body: WordPluginBody = Body(...)
):
    """
    接收 BOPPPS JSON，渲染成标准的 Word (.docx) 详案文档
    """
    try:
        actual_outline = body.outline_data

        # 延续今晚的“防呆”优良传统：如果是个长字符串，手动解析它
        if isinstance(actual_outline, str):
            try:
                actual_outline = json.loads(actual_outline)
                print("✅ Word 插件：成功解析大纲字符串")
            except Exception as e:
                print(f"❌ Word 插件：JSON 解析失败: {str(e)}")
                raise HTTPException(status_code=400, detail="大纲内容不是有效的 JSON 字符串")

        print(f"📄 [详案引擎] 正在为项目 {project_id} 撰写 Word 文案...")

        # 丢进线程池，防止 docx 处理过程中卡死主循环
        download_url = await asyncio.to_thread(
            generate_word_from_json,
            actual_outline,
            project_id
        )

        return {
            "status": "success",
            "message": "Word 详案生成完毕",
            "download_url": download_url
        }

    except Exception as e:
        print(f"❌ Word 接口崩溃: {str(e)}")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))