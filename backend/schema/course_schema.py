from pydantic import BaseModel
from typing import List


class Slide(BaseModel):
    title: str
    content: str
    image_prompt: str
    logic: str


class LessonPlan(BaseModel):
    objectives: List[str]
    steps: List[str]
    homework: List[str]


class CourseInstruction(BaseModel):
    course_topic: str
    duration: str
    style: str
    slides: List[Slide]
    lesson_plan: LessonPlan