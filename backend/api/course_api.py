from fastapi import APIRouter
from schema.course_schema import CourseInstruction

router = APIRouter()

@router.post("/generate_instruction")
def generate_instruction(data: CourseInstruction):
    return {
        "message": "Instruction received",
        "data": data
    }