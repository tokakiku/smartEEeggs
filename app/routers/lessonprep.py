# ==========================================
# 灵犀智课 - 备课工作流核心路由 (状态机 + BOPPPS契约)
# 文件路径: app/routers/lessonprep.py
# ==========================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter(
    prefix="/api/lessonprep",
    tags=["备课工作流"]
)

# ==========================================
# 模块 A：前端请求参数模型 (输入)
# ==========================================
class TopicInput(BaseModel):
    course_topic: str = Field(..., example="计算机网络：TCP三次握手")
    teaching_style: str = Field(default="BOPPPS")

class OutlineEditRequest(BaseModel):
    project_id: int
    user_instruction: str = Field(..., example="把参与式学习的时间缩短5分钟，加到后测里")

class ConfirmRequest(BaseModel):
    project_id: int

# ==========================================
# 模块 B：Agent 吐出的数据契约模型 (输出 - 由 ljy 设计)
# ==========================================
class ResourceItem(BaseModel):
    id: str
    type: str  # "video", "image", "pdf"
    source_path: str
    description: Optional[str] = None

class KnowledgePoint(BaseModel):
    point: str
    is_key_point: bool = False
    is_difficult_point: bool = False
    explanation: str
    game_hook: Optional[str] = None

class SyllabusStage(BaseModel):
    stage: str
    content_description: Optional[str] = None
    core_knowledge_points: Optional[List[KnowledgePoint]] = None
    duration: int
    interaction_type: str
    resource_refs: List[str] = []

class CourseMetadata(BaseModel):
    title: str
    target_audience: str
    teaching_objectives: List[str]
    difficulty_level: str
    total_duration: int

# 这是整个大纲的最终聚合体
class BopppsOutlineResponse(BaseModel):
    project_id: int
    message: str
    course_metadata: CourseMetadata
    syllabus_content: List[SyllabusStage]
    resource_pool: List[ResourceItem]


# ==========================================
# 模块 C：状态机 API 节点实现
# ==========================================

@router.post("/start", response_model=BopppsOutlineResponse, summary="节点1：输入主题，生成BOPPPS大纲")
def start_lesson_prep(data: TopicInput):
    """
    前端输入课题，后端调用大模型生成初始大纲。
    目前返回极度逼真的 Mock 数据供前端 hqr 对接。
    """
    return {
        "project_id": 1024,
        "message": "大纲生成成功",
        "course_metadata": {
            "title": data.course_topic,
            "target_audience": "计算机专业本科二年级",
            "teaching_objectives": ["知识：理解TCP面向连接特性", "能力：掌握三次握手时序图", "素养：建立严谨的网络协议思维"],
            "difficulty_level": "中等难度",
            "total_duration": 45
        },
        "syllabus_content": [
            {
                "stage": "B (Bridge-in) - 导入",
                "content_description": "通过日常生活中寄送快递需要确认收件人地址的例子，引入网络数据传输前也需要建立连接。",
                "duration": 5,
                "interaction_type": "生活化提问",
                "resource_refs": ["ref_video_01"]
            },
            {
                "stage": "O (Objective) - 目标",
                "content_description": "明确本节课需要大家掌握三次握手的具体过程和状态标志位。",
                "duration": 2,
                "interaction_type": "PPT直接展示",
                "resource_refs": []
            },
            {
                "stage": "P2 (Participatory Learning) - 参与式学习",
                "core_knowledge_points": [
                    {
                        "point": "第一步：客户端发送 SYN 包",
                        "is_key_point": True,
                        "is_difficult_point": False,
                        "explanation": "客户端主动打开连接，发送带有 SYN 标志的数据包。",
                        "game_hook": "h5_game_syn_send"
                    },
                    {
                        "point": "第二步：服务端回复 SYN-ACK 包",
                        "is_key_point": True,
                        "is_difficult_point": True,
                        "explanation": "服务端收到后，同时确认(ACK)并也发起同步(SYN)。",
                        "game_hook": None
                    }
                ],
                "duration": 20,
                "interaction_type": "动画演示 + H5互动游戏",
                "resource_refs": ["ref_img_01"]
            }
            # 注：为了代码简洁，这里略去了 P1, P3, S 阶段，前端渲染逻辑是一样的
        ],
        "resource_pool": [
            {
                "id": "ref_video_01",
                "type": "video",
                "source_path": "minio://lingxi-bucket/videos/kuaidi.mp4",
                "description": "顺丰快递寄件流程视频"
            },
            {
                "id": "ref_img_01",
                "type": "image",
                "source_path": "minio://lingxi-bucket/images/tcp_handshake.gif",
                "description": "TCP三次握手动图"
            }
        ]
    }


@router.post("/outline/edit", summary="节点2：对话式增量修改大纲")
def edit_outline(data: OutlineEditRequest):
    # TODO: 未来调用 ReviewAgent，传入 data.user_instruction 进行局部修改
    return {"message": "大纲已根据您的指令局部重写（开发中...）", "project_id": data.project_id}


@router.post("/outline/confirm", summary="节点3：确认大纲，锁定状态")
def confirm_outline(data: ConfirmRequest):
    # TODO: 将内存/Redis中的大纲正式持久化存入 MySQL 数据库的 projects 表
    return {"message": "大纲已锁定！随时可以开始生成物理课件", "project_id": data.project_id, "status": "locked"}


@router.post("/plan/generate", summary="节点4：生成 Word 详细教案")
def generate_lesson_plan(data: ConfirmRequest):
    # TODO: 读取数据库里的终版大纲，调用大模型扩写，用 python-docx 生成文件
    return {
        "message": "Word教案生成成功",
        "project_id": data.project_id,
        "word_download_url": "http://127.0.0.1:8000/downloads/plan_1024.docx"
    }


@router.post("/ppt/generate", summary="节点5：生成 PPT 课件")
def generate_ppt(data: ConfirmRequest):
    # TODO: 读取大纲和资源池，调用 python-pptx 模板引擎渲染
    return {
        "message": "PPT渲染完毕",
        "project_id": data.project_id,
        "ppt_download_url": "http://127.0.0.1:8000/downloads/ppt_1024.pptx"
    }