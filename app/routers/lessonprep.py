# ==========================================
# 灵犀智课 - 备课工作流核心路由 (集成 SSE 与 PPT 引擎)
# 文件路径: app/routers/lessonprep.py
# ==========================================

import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# 导入咱们优化过的物理渲染引擎
from app.services.ppt_generator import generate_ppt_from_json

router = APIRouter(
    prefix="/api/lessonprep",
    tags=["备课工作流"]
)


# ==========================================
# 模块 A：前端请求参数模型
# ==========================================
class OutlineEditRequest(BaseModel):
    project_id: int
    user_instruction: str = Field(..., example="把参与式学习的时间缩短5分钟")


class PPTGenerateRequest(BaseModel):
    project_id: int = Field(default=1024, description="课件项目ID")
    # 核心改造：直接接收前端传回来的、组长 ljy 定义的完整 BOPPPS JSON 数据
    outline_data: Dict[str, Any] = Field(..., description="前端确认后的 BOPPPS 大纲 JSON 数据")


# ==========================================
# 模块 B：状态机 API 节点实现
# ==========================================

@router.post("/outline/edit", summary="节点2：对话式增量修改大纲 (待接入LLM)")
def edit_outline(data: OutlineEditRequest):
    # 这里未来也是要调大模型的，根据 user_instruction 动态修改 JSON
    return {"message": "大纲已修改", "project_id": data.project_id}


@router.post("/plan/generate", summary="节点4：生成 Word 教案 (待开发)")
def generate_lesson_plan(project_id: int):
    return {"message": "Word教案生成成功 (开发中...)", "project_id": project_id}


# ==========================================
# 🔥 核心改造：节点5 - SSE 流式生成 PPT
# ==========================================
@router.post("/ppt/generate", summary="节点5：基于大纲生成 PPT (SSE流式响应)")
async def generate_ppt(data: PPTGenerateRequest):
    """
    接收前端确认后的 BOPPPS 大纲，使用 SSE 实时推送生成进度，
    并调用底层的 python-pptx 引擎渲染，最后返回云端下载链接。
    """

    # 定义发报机 (Generator) - SSE 规范要求必须是这种 yield 格式
    async def event_stream():
        # 推送进度 1：告诉前端开始处理了
        yield f"data: {json.dumps({'status': 'processing', 'progress': 20, 'message': '正在解析前端传回的 BOPPPS 教学结构...'})}\n\n"
        await asyncio.sleep(0.5)  # 稍微停顿，让前端进度条有平滑移动感

        # 推送进度 2：告诉前端正在渲染
        yield f"data: {json.dumps({'status': 'processing', 'progress': 60, 'message': '正在调动底层排版引擎注入多维数据与重难点标记...'})}\n\n"

        try:
            # 🔥 关键调用：把前端传来的 JSON 直接喂给你的渲染引擎
            # 使用 asyncio.to_thread 确保耗时的 PPT 读写操作不会卡死整个 FastAPI 服务器
            file_url_or_path = await asyncio.to_thread(
                generate_ppt_from_json,
                data.outline_data,
                data.project_id
            )

            # 推送进度 3：大功告成，发送最终下载链接
            yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'message': 'PPT渲染完毕！', 'download_url': file_url_or_path})}\n\n"

        except Exception as e:
            # 万一代码崩了，也要优雅地通知前端，不能白屏
            print(f"❌ [接口层异常] PPT生成失败: {str(e)}")
            yield f"data: {json.dumps({'status': 'error', 'message': f'服务器渲染失败: {str(e)}'})}\n\n"

    # 使用 StreamingResponse 返回流式数据，必须指定 media_type="text/event-stream"
    return StreamingResponse(event_stream(), media_type="text/event-stream")