from services.syllabus_extractor import SyllabusExtractor


SAMPLE_SYLLABUS = """
课程编号：CS301
课程名称：机器学习
学分/学时：3/48
课程性质：专业必修课
适用专业：计算机科学与技术、人工智能 / 数据科学
建议开设学期：第6学期
先修课程：高等数学、概率论与数理统计 / 线性代数
开课单位：计算机学院

一、课程的教学目标与任务
1. 使学生掌握机器学习基本概念和典型算法。
2. 培养学生使用机器学习方法分析问题与解决问题的能力。

二、课程具体内容及基本要求
（一）机器学习基本概念（2 学时）
介绍机器学习定义、任务分类与应用场景。
1.基本要求
理解监督学习、无监督学习与强化学习的区别。
2.重点、难点
重点：机器学习任务分类；模型评估方法。
难点：偏差与方差；过拟合与欠拟合。
3.作业及课外学习要求
阅读教材第一章，完成课后习题1-3。
""".strip()


SAMPLE_SYLLABUS_WITH_MATERIALS = """
课程编号：CS301
课程名称：机器学习

教材及参考书目
主课教材：机器学习（周志华）
指定教材：《模式识别与机器学习》
参考教材：机器学习实战、机器学习习题与解析
推荐阅读：深度学习（Goodfellow）
""".strip()


def test_extract_course_info():
    extractor = SyllabusExtractor()
    result = extractor.extract_from_text(SAMPLE_SYLLABUS)

    assert result.course_info.course_name == "机器学习"
    assert result.course_info.course_code == "CS301"
    assert result.course_info.course_type == "专业必修课"
    assert "人工智能" in result.course_info.applicable_major
    assert "线性代数" in result.course_info.prerequisite_courses


def test_extract_modules():
    extractor = SyllabusExtractor()
    result = extractor.extract_from_text(SAMPLE_SYLLABUS)

    assert len(result.course_modules) == 1
    assert result.course_modules[0].module_name == "机器学习基本概念"
    assert result.course_modules[0].hours == "2 学时"
    assert len(result.course_modules[0].learning_requirements) > 0


def test_extract_key_and_difficult_points():
    extractor = SyllabusExtractor()
    result = extractor.extract_from_text(SAMPLE_SYLLABUS)

    first_module = result.course_modules[0]
    assert any("任务分类" in item for item in first_module.key_points)
    assert any("过拟合" in item for item in first_module.difficult_points)


def test_extract_teaching_materials_main_and_reference():
    extractor = SyllabusExtractor()
    result = extractor.extract_from_text(SAMPLE_SYLLABUS_WITH_MATERIALS)

    assert any("机器学习（周志华）" in item for item in result.teaching_materials.main_textbooks)
    assert any("模式识别与机器学习" in item for item in result.teaching_materials.main_textbooks)
    assert any("机器学习实战" in item for item in result.teaching_materials.reference_textbooks)
    assert any("习题" in item for item in result.teaching_materials.reference_textbooks)
    assert len(result.textbooks) >= len(result.teaching_materials.main_textbooks)
