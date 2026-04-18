from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SyllabusExtractTextRequest(BaseModel):
    text: str


class CourseInfo(BaseModel):
    course_name: Optional[str] = None
    course_code: Optional[str] = None
    credit_hours: Optional[str] = None
    course_type: Optional[str] = None
    applicable_major: List[str] = Field(default_factory=list)
    suggested_term: Optional[str] = None
    prerequisite_courses: List[str] = Field(default_factory=list)
    offering_institute: Optional[str] = None


class CourseModule(BaseModel):
    module_index: int
    module_name: str
    hours: Optional[str] = None
    description: Optional[str] = None
    learning_requirements: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    difficult_points: List[str] = Field(default_factory=list)
    assignments: List[str] = Field(default_factory=list)


class TeachingScheduleItem(BaseModel):
    order: int
    topic: str
    hours: Optional[str] = None
    teaching_method: Optional[str] = None


class TeachingMaterials(BaseModel):
    # 课标中的教材信息，区分主教材与参考教材
    main_textbooks: List[str] = Field(default_factory=list)
    reference_textbooks: List[str] = Field(default_factory=list)
    other_materials: List[str] = Field(default_factory=list)


class SyllabusExtractionResult(BaseModel):
    course_info: CourseInfo = Field(default_factory=CourseInfo)
    course_name: Optional[str] = None
    course_code: Optional[str] = None
    prerequisites: List[str] = Field(default_factory=list)
    teaching_goals: List[str] = Field(default_factory=list)
    course_modules: List[CourseModule] = Field(default_factory=list)
    teaching_key_points: List[str] = Field(default_factory=list)
    teaching_difficult_points: List[str] = Field(default_factory=list)
    knowledge_points: List[str] = Field(default_factory=list)
    teaching_schedule: List[TeachingScheduleItem] = Field(default_factory=list)
    textbooks: List[str] = Field(default_factory=list)
    teaching_materials: TeachingMaterials = Field(default_factory=TeachingMaterials)
    raw_sections: Dict[str, str] = Field(default_factory=dict)


class SyllabusPriorityMetadata(BaseModel):
    is_primary: bool = False
    course_name: Optional[str] = None
    course_code: Optional[str] = None
    school: Optional[str] = None
    department: Optional[str] = None
    major: Optional[str] = None
    academic_year: Optional[str] = None
    semester: Optional[str] = None
    version: Optional[str] = None
    teacher: Optional[str] = None
    effective_date: Optional[str] = None
    priority_score: float = 0.0
