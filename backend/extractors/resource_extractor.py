import re
from typing import Any, Dict, List, Optional, Tuple

"""资源层抽取器：以页级教学表达单元为核心。"""

from extractors.base_extractor import BaseExtractor
from schema.parsed_document_schema import ParsedDocument, ParsedElement
from schema.resource_schema import (
    PageRole,
    ResourceChunk,
    ResourceExtractionResult,
    ResourceIndex,
    ResourceInfo,
    ResourcePage,
    ResourceRelation,
    ReusableUnit,
)
from utils.text_cleaner import clean_text, split_paragraphs


class ResourceExtractor(BaseExtractor):
    """资源层 extractor。"""
    # 资源层 extractor：面向“页面级教学表达单元”，而非教材章节结构

    KNOWLEDGE_TERMS = [
        "机器学习",
        "人工智能",
        "模式识别",
        "监督学习",
        "无监督学习",
        "强化学习",
        "假设空间",
        "归纳偏好",
        "NFL 定理",
        "No Free Lunch",
        "损失函数",
        "泛化能力",
        "过拟合",
        "欠拟合",
        "训练集",
        "验证集",
        "测试集",
        "自动驾驶",
        "推荐系统",
        "文献筛选",
        "古文献修复",
        "画作鉴别",
        "John McCarthy",
    ]
    KP_STOP_WORDS = {
        "绪论",
        "导论",
        "导读",
        "目录",
        "大纲",
        "本章内容",
        "总结",
        "小结",
        "思考题",
        "究竟",
        "例子",
        "例如",
    }

    def extract(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        # 主流程：按页切分 -> 页级结构抽取 -> 可复用单元 -> chunks -> relations -> 索引
        resource_id = f"resource-{parsed_doc.doc_id}"
        page_groups = self._group_elements_by_page(parsed_doc)
        pages = self._build_pages(parsed_doc, resource_id, page_groups)

        if not pages and parsed_doc.raw_text.strip():
            pages = self._build_fallback_page(parsed_doc, resource_id)
        if not pages:
            pages = [
                ResourcePage(
                    page_id=f"{resource_id}-p1",
                    resource_id=resource_id,
                    page_no=1,
                    page_title=parsed_doc.title or "未命名页面",
                    page_text="",
                    page_summary=parsed_doc.title or "资源页面内容为空",
                    page_role=PageRole.other_page,
                    knowledge_points=[],
                    has_image=False,
                    has_formula=False,
                    has_table=False,
                    has_example=False,
                    tags=["resource", "empty_page"],
                )
            ]

        resource_info = self._build_resource_info(parsed_doc, resource_id, pages)
        reusable_units = self._build_reusable_units(pages)
        chunks = self._build_chunks(resource_id, parsed_doc.file_name, pages)
        relations = self._build_relations(pages, reusable_units)
        resource_index = self._build_resource_index(pages)

        result_model = ResourceExtractionResult(
            resource_info=resource_info,
            pages=pages,
            reusable_units=reusable_units,
            chunks=chunks,
            relations=relations,
            resource_index=resource_index,
        )
        if hasattr(result_model, "model_dump"):
            return result_model.model_dump(mode="json")
        return result_model.dict()

    def _group_elements_by_page(self, parsed_doc: ParsedDocument) -> Dict[int, List[ParsedElement]]:
        # 优先使用 ParsedElement.page_no 分页；缺失时按顺序合并到第一页
        grouped: Dict[int, List[ParsedElement]] = {}
        for element in parsed_doc.elements:
            page_no = element.page_no if isinstance(element.page_no, int) and element.page_no > 0 else 1
            grouped.setdefault(page_no, []).append(element)
        return grouped

    def _build_pages(
        self,
        parsed_doc: ParsedDocument,
        resource_id: str,
        page_groups: Dict[int, List[ParsedElement]],
    ) -> List[ResourcePage]:
        # 构建页面对象：提取标题、摘要、角色、知识点及多模态标记
        pages: List[ResourcePage] = []
        page_numbers = sorted(page_groups.keys())
        total_pages = len(page_numbers)

        for page_no in page_numbers:
            elements = page_groups.get(page_no, [])
            text_lines = self._collect_page_lines(elements)
            page_text = clean_text("\n".join(text_lines))
            page_title = self._guess_page_title(elements, text_lines, page_no)

            has_image = self._detect_has_image(elements, page_text)
            has_table = self._detect_has_table(elements, page_text)
            has_formula = self._detect_has_formula(elements, page_text)
            has_example = self._detect_has_example(page_title, page_text)
            page_role = self._classify_page_role(
                page_title=page_title,
                page_text=page_text,
                page_no=page_no,
                total_pages=total_pages,
                has_image=has_image,
                has_table=has_table,
                has_formula=has_formula,
                has_example=has_example,
            )
            knowledge_points = self._extract_knowledge_points(page_title, page_text)
            page_summary = self._build_page_summary(page_title, page_text)
            tags = self._deduplicate(
                [
                    "resource",
                    parsed_doc.file_type,
                    page_role.value,
                    page_title,
                    *knowledge_points[:5],
                ]
            )

            pages.append(
                ResourcePage(
                    page_id=f"{resource_id}-p{page_no}",
                    resource_id=resource_id,
                    page_no=page_no,
                    page_title=page_title,
                    page_text=page_text,
                    page_summary=page_summary,
                    page_role=page_role,
                    knowledge_points=knowledge_points,
                    has_image=has_image,
                    has_formula=has_formula,
                    has_table=has_table,
                    has_example=has_example,
                    tags=tags,
                )
            )
        return pages

    def _build_fallback_page(self, parsed_doc: ParsedDocument, resource_id: str) -> List[ResourcePage]:
        # 没有可用元素时，使用 raw_text 兜底构建单页
        page_title = self._clean_heading(parsed_doc.title or parsed_doc.file_name)
        page_text = clean_text(parsed_doc.raw_text)
        knowledge_points = self._extract_knowledge_points(page_title, page_text)
        role = self._classify_page_role(
            page_title=page_title,
            page_text=page_text,
            page_no=1,
            total_pages=1,
            has_image=False,
            has_table=False,
            has_formula=self._detect_has_formula([], page_text),
            has_example=self._detect_has_example(page_title, page_text),
        )
        return [
            ResourcePage(
                page_id=f"{resource_id}-p1",
                resource_id=resource_id,
                page_no=1,
                page_title=page_title,
                page_text=page_text,
                page_summary=self._build_page_summary(page_title, page_text),
                page_role=role,
                knowledge_points=knowledge_points,
                has_image=False,
                has_formula=self._detect_has_formula([], page_text),
                has_table=False,
                has_example=self._detect_has_example(page_title, page_text),
                tags=self._deduplicate(["resource", role.value, page_title, *knowledge_points[:4]]),
            )
        ]

    def _collect_page_lines(self, elements: List[ParsedElement]) -> List[str]:
        # 清洗页面行文本，并按出现顺序去重
        lines: List[str] = []
        seen = set()
        for element in elements:
            text = clean_text(element.text or "")
            if not text:
                continue
            for line in split_paragraphs(text):
                normalized = clean_text(line)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                lines.append(normalized)
        return lines

    def _guess_page_title(self, elements: List[ParsedElement], lines: List[str], page_no: int) -> str:
        # 页面标题优先级：Title 元素 > 第一条短句 > 默认页名
        for element in elements:
            if "title" in (element.type or "").lower():
                candidate = self._clean_heading(element.text)
                if self._is_valid_page_title(candidate):
                    return candidate

        for line in lines:
            candidate = self._clean_heading(line)
            if self._is_valid_page_title(candidate):
                return candidate

        return f"第{page_no}页"

    def _clean_heading(self, text: str) -> str:
        # 清理标题中的序号、末尾页码与冗余空格
        value = clean_text(text or "")
        if not value:
            return ""
        value = re.sub(r"^\s*(第[一二三四五六七八九十百零〇\d]+[章节]|[一二三四五六七八九十]+、)\s*", "", value)
        value = re.sub(r"^\s*\d+(?:\.\d+){0,2}\s*", "", value)
        value = self._strip_trailing_page_no(value)
        value = re.sub(r"\s{2,}", " ", value)
        return value.strip()

    def _strip_trailing_page_no(self, text: str) -> str:
        # 去掉标题或知识点末尾可能混入的页码
        result = text
        result = re.sub(r"\.{2,}\s*\d+\s*$", "", result)
        result = re.sub(r"[·•—-]{2,}\s*\d+\s*$", "", result)
        result = re.sub(r"\s+\d{1,4}\s*$", "", result)
        return result.strip()

    def _is_valid_page_title(self, title: str) -> bool:
        # 页面标题有效性：避免“权/有/机”等单字或噪声标题
        if not title:
            return False
        if len(title) < 2 or len(title) > 50:
            return False
        if re.match(r"^[\W_]+$", title):
            return False
        if re.match(r"^[A-Za-z]$", title):
            return False
        if re.match(r"^[\u4e00-\u9fa5]$", title):
            return False
        return True

    def _detect_has_image(self, elements: List[ParsedElement], page_text: str) -> bool:
        # 图片判定：元素类型/元数据 + 文本线索联合判断
        for element in elements:
            element_type = (element.type or "").lower()
            if "image" in element_type or "figure" in element_type or "picture" in element_type:
                return True
            if bool((element.metadata or {}).get("has_image")):
                return True
        return bool(re.search(r"(图\s*\d+|如下图|示意图|流程图)", page_text))

    def _detect_has_table(self, elements: List[ParsedElement], page_text: str) -> bool:
        # 表格判定：元素类型/元数据 + 文本线索联合判断
        for element in elements:
            element_type = (element.type or "").lower()
            if "table" in element_type:
                return True
            if bool((element.metadata or {}).get("has_table")):
                return True
        return bool(re.search(r"(表\s*\d+|对照表|统计表)", page_text))

    def _detect_has_formula(self, elements: List[ParsedElement], page_text: str) -> bool:
        # 公式判定：元素元数据 + 数学符号密度 + 公式关键词
        for element in elements:
            if bool((element.metadata or {}).get("has_formula")):
                return True
        symbol_hits = re.findall(r"[=≈∑∏√∂≤≥∞λμσθ]", page_text)
        keyword_hit = re.search(r"(公式|定理|损失函数|目标函数|argmin|argmax|f\(x\)|P\()", page_text, flags=re.I)
        return len(symbol_hits) >= 2 or bool(keyword_hit)

    def _detect_has_example(self, page_title: str, page_text: str) -> bool:
        # 案例判定：标题和正文关键词
        content = f"{page_title}\n{page_text}"
        return bool(re.search(r"(案例|故事|实例|example|例如|比如)", content, flags=re.I))

    def _classify_page_role(
        self,
        page_title: str,
        page_text: str,
        page_no: int,
        total_pages: int,
        has_image: bool,
        has_table: bool,
        has_formula: bool,
        has_example: bool,
    ) -> PageRole:
        # 页面角色规则分类：保证资源层以“教学表达角色”组织
        title = page_title.lower()
        text = page_text.lower()
        content = f"{title}\n{text}"

        if page_no == 1 and len(page_text) <= 220 and not re.search(r"(目录|大纲|contents|agenda)", content):
            return PageRole.title_page
        if re.search(r"(目录|大纲|提纲|本章内容|本讲内容|contents|agenda)", content):
            return PageRole.outline_page
        if re.search(r"(基本术语)", content):
            return PageRole.definition_page
        if re.search(r"(总结|小结|回顾|本讲小结|takeaway)", content):
            return PageRole.summary_page
        if re.search(r"(练习|思考|作业|讨论题|quiz|习题)", content):
            return PageRole.exercise_page
        if re.search(r"(应用|能做什么|应用场景|落地|自动驾驶|推荐系统|古文献修复|画作鉴别|奥巴马)", content):
            return PageRole.application_page
        if re.search(r"(john mccarthy|图灵|达特茅斯)", content, flags=re.I):
            return PageRole.history_page
        if re.search(r"(人工智能.*阶段|发展阶段|历史|发展史|年代|提出|诞生)", content):
            return PageRole.history_page
        if re.search(r"(19\d{2}|20\d{2})", content) and re.search(r"(发展|历史|阶段|提出|诞生)", content):
            return PageRole.history_page
        if has_formula:
            return PageRole.formula_page
        if re.search(r"(什么是|是什么|定义|概念|是指|指的是|基本术语)", content):
            return PageRole.definition_page
        if re.search(r"(案例|故事|实例|实战|example|文献筛选)", content):
            return PageRole.case_page
        if has_example:
            return PageRole.case_page
        if re.search(r"(原理|机制|过程|流程|推导|证明|假设空间|归纳偏好|nfl|no free lunch)", content):
            return PageRole.principle_page
        if re.search(r"(对比|比较|区别|联系|优缺点|vs)", content):
            return PageRole.comparison_page
        if has_image and len(page_text) <= 120 and not has_table:
            return PageRole.visual_page
        return PageRole.other_page

    def _extract_knowledge_points(self, page_title: str, page_text: str) -> List[str]:
        # 规则法知识点抽取：标题优先 + 关键词 + 定义句，避免整句入库
        candidates: List[str] = []
        title_candidate = self._normalize_kp_name(page_title)
        if self._is_valid_kp(title_candidate):
            candidates.append(title_candidate)

        content = f"{page_title}\n{page_text}"
        for term in self.KNOWLEDGE_TERMS:
            if term.lower() in content.lower():
                candidates.append(term)

        for line in split_paragraphs(page_text):
            normalized_line = clean_text(line)
            if not normalized_line:
                continue

            short_line = self._normalize_kp_name(normalized_line)
            if len(short_line) <= 20 and self._looks_like_term(short_line):
                candidates.append(short_line)

            define_match = re.match(
                r"^([A-Za-z\u4e00-\u9fa5][A-Za-z0-9\u4e00-\u9fa5\-\s]{1,30}?)(?:是|指|定义为|属于)",
                normalized_line,
            )
            if define_match:
                candidates.append(self._normalize_kp_name(define_match.group(1)))

        deduped = self._deduplicate(candidates)
        return [item for item in deduped if self._is_valid_kp(item)][:10]

    def _normalize_kp_name(self, text: str) -> str:
        # 知识点名称清洗：保留概念名，去掉编号和末尾页码
        value = clean_text(text or "")
        if not value:
            return ""
        value = re.sub(r"^\s*(第[一二三四五六七八九十百零〇\d]+[章节]|[一二三四五六七八九十]+、)\s*", "", value)
        value = re.sub(r"^\s*\d+(?:\.\d+){0,2}\s*", "", value)
        value = re.sub(r"^[\-\*•●]\s*", "", value)
        value = self._strip_trailing_page_no(value)
        value = re.split(r"[，,。；;：:（）()]", value, maxsplit=1)[0].strip()
        value = re.sub(r"\s{2,}", " ", value)
        return value[:40].strip()

    def _looks_like_term(self, text: str) -> bool:
        # 启发式术语判定：模型/方法/定理等短语优先
        if not text:
            return False
        if any(keyword in text for keyword in ["模型", "算法", "方法", "空间", "偏好", "定理", "学习", "网络", "分类", "回归"]):
            return True
        if text in self.KNOWLEDGE_TERMS:
            return True
        return False

    def _is_valid_kp(self, candidate: str) -> bool:
        # 过滤噪声知识点：过短、过长、模板词、纯数字
        if not candidate:
            return False
        if candidate in self.KP_STOP_WORDS:
            return False
        if len(candidate) < 2 or len(candidate) > 30:
            return False
        if re.match(r"^\d+(?:\.\d+)*$", candidate):
            return False
        if re.search(r"(。|；|;|，|,)", candidate):
            return False
        return True

    def _build_page_summary(self, page_title: str, page_text: str) -> str:
        # 页面摘要：优先抽取前两条有效句，不直接返回整页全文
        sentences: List[str] = []
        for paragraph in split_paragraphs(page_text):
            for piece in re.split(r"[。；;!?！？]", paragraph):
                sentence = clean_text(piece)
                if len(sentence) < 6:
                    continue
                sentences.append(sentence)
                if len(sentences) >= 2:
                    break
            if len(sentences) >= 2:
                break

        if sentences:
            core = "；".join(sentences)[:180]
            if page_title and page_title not in core:
                return f"{page_title}：{core}"
            return core
        if page_title:
            return page_title
        return clean_text(page_text[:180]) or "本页内容较少，摘要为空"

    def _build_resource_info(
        self,
        parsed_doc: ParsedDocument,
        resource_id: str,
        pages: List[ResourcePage],
    ) -> ResourceInfo:
        # 汇总资源基本信息，供索引和落盘检索使用
        primary_title = self._clean_heading(parsed_doc.title or "")
        if self._is_invalid_title(primary_title):
            primary_title = pages[0].page_title if pages else parsed_doc.file_name
        if self._is_invalid_title(primary_title):
            primary_title = parsed_doc.file_name

        subtitle = self._extract_subtitle(pages[0] if pages else None, primary_title)
        subject = self._infer_subject(primary_title, parsed_doc.raw_text)
        course_topic = self._infer_course_topic(primary_title, pages)

        return ResourceInfo(
            resource_id=resource_id,
            title=primary_title,
            subtitle=subtitle,
            resource_type=self._infer_resource_type(parsed_doc.file_type),
            source_file=parsed_doc.file_name,
            source_type=parsed_doc.source_type,
            subject=subject,
            course_topic=course_topic,
            resource_role=self._infer_resource_role(title=primary_title, source_file=parsed_doc.file_name),
        )

    def _extract_subtitle(self, first_page: Optional[ResourcePage], title: str) -> Optional[str]:
        # 提取副标题：优先取首页第二条短文本
        if first_page is None:
            return None
        lines = split_paragraphs(first_page.page_text)
        for line in lines:
            candidate = self._clean_heading(line)
            if not candidate or candidate == title:
                continue
            if 3 <= len(candidate) <= 40:
                return candidate
        return None

    def _infer_subject(self, title: str, text: str) -> Optional[str]:
        # 学科猜测：仅做轻量关键词规则
        content = f"{title}\n{text}"
        if "机器学习" in content or "神经网络" in content or "监督学习" in content:
            return "机器学习"
        if "人工智能" in content:
            return "人工智能"
        if "数据挖掘" in content:
            return "数据挖掘"
        return None

    def _infer_course_topic(self, title: str, pages: List[ResourcePage]) -> Optional[str]:
        # 课程主题猜测：优先标题，再用首个有效知识点兜底
        cleaned_title = self._normalize_kp_name(title)
        if cleaned_title and cleaned_title not in self.KP_STOP_WORDS:
            return cleaned_title
        for page in pages:
            if page.knowledge_points:
                return page.knowledge_points[0]
        return None

    def _infer_resource_type(self, file_type: str) -> str:
        # 资源类型映射
        mapping = {
            "pptx": "presentation",
            "pdf": "courseware_pdf",
            "docx": "handout_doc",
            "html": "web_resource",
        }
        return mapping.get(file_type.lower(), file_type.lower())

    def _infer_resource_role(self, title: str, source_file: str) -> str:
        value = f"{title} {source_file}".lower()
        if any(token in value for token in ["习题", "练习", "作业", "quiz", "exercise"]):
            return "exercise_slides"
        if any(token in value for token in ["复习", "总结", "review", "summary"]):
            return "review_slides"
        if any(token in value for token in ["讲义", "lecture notes", "notes"]):
            return "lecture_notes"
        return "main_courseware"

    def _is_invalid_title(self, title: str) -> bool:
        # 标题质量判定：过滤空值、目录词、压缩包二进制残片
        if not title:
            return True
        lowered = title.lower().replace(" ", "")
        if lowered in {"目录", "contents", "untitled"}:
            return True
        if "pk\x03\x04" in lowered or "[content_types].xml" in lowered:
            return True
        return False

    def _build_reusable_units(self, pages: List[ResourcePage]) -> List[ReusableUnit]:
        # 每页至少产出一个可复用教学单元
        units: List[ReusableUnit] = []
        for page in pages:
            unit_id = f"{page.page_id}-u1"
            role = page.page_role
            units.append(
                ReusableUnit(
                    unit_id=unit_id,
                    page_id=page.page_id,
                    unit_title=page.page_title,
                    unit_summary=page.page_summary,
                    unit_role=role,
                    knowledge_points=page.knowledge_points,
                    reusability=self._score_reusability(role),
                    recommended_use=self._recommended_use(role),
                )
            )
        return units

    def _score_reusability(self, role: PageRole) -> float:
        # 可复用性评分：供后续排序筛选
        score_map = {
            PageRole.definition_page: 0.95,
            PageRole.principle_page: 0.95,
            PageRole.formula_page: 0.93,
            PageRole.case_page: 0.9,
            PageRole.application_page: 0.9,
            PageRole.summary_page: 0.88,
            PageRole.comparison_page: 0.88,
            PageRole.history_page: 0.85,
            PageRole.exercise_page: 0.84,
            PageRole.visual_page: 0.82,
            PageRole.outline_page: 0.78,
            PageRole.title_page: 0.75,
            PageRole.other_page: 0.72,
        }
        return score_map.get(role, 0.72)

    def _recommended_use(self, role: PageRole) -> List[str]:
        # 页面角色到教学用途映射
        mapping = {
            PageRole.title_page: ["课程导入"],
            PageRole.outline_page: ["课程导览", "教学节奏说明"],
            PageRole.definition_page: ["概念讲解", "课堂引入"],
            PageRole.principle_page: ["理论讲授", "板书推导"],
            PageRole.formula_page: ["公式推导", "习题讲解"],
            PageRole.case_page: ["案例分析", "课堂讨论"],
            PageRole.application_page: ["应用拓展", "课程导入"],
            PageRole.exercise_page: ["课堂练习", "课后作业"],
            PageRole.summary_page: ["课堂总结", "复习回顾"],
            PageRole.history_page: ["历史背景铺垫", "课堂导入"],
            PageRole.comparison_page: ["方法对比", "知识辨析"],
            PageRole.visual_page: ["图示讲解", "课堂互动"],
            PageRole.other_page: ["通用讲解"],
        }
        return mapping.get(role, ["通用讲解"])

    def _build_chunks(
        self,
        resource_id: str,
        source_file: str,
        pages: List[ResourcePage],
    ) -> List[ResourceChunk]:
        # 按页构建 chunk；超长页再按段落切分
        chunks: List[ResourceChunk] = []
        for page in pages:
            pieces = self._split_page_text(page.page_text, max_chars=360)
            if not pieces:
                pieces = [page.page_summary or page.page_title]

            for idx, piece in enumerate(pieces, start=1):
                tags = self._deduplicate(
                    [
                        "resource",
                        page.page_role.value,
                        page.page_title,
                        *page.knowledge_points[:5],
                    ]
                )
                chunks.append(
                    ResourceChunk(
                        chunk_id=f"{page.page_id}-c{idx}",
                        resource_id=resource_id,
                        page_id=page.page_id,
                        page_no=page.page_no,
                        page_title=page.page_title,
                        page_role=page.page_role,
                        text=clean_text(piece),
                        knowledge_points=page.knowledge_points,
                        tags=tags,
                        metadata={
                            "source_file": source_file,
                            "chunk_index_in_page": idx,
                            "has_image": page.has_image,
                            "has_formula": page.has_formula,
                            "has_table": page.has_table,
                            "has_example": page.has_example,
                        },
                        section=page.page_title,
                    )
                )
        return chunks

    def _split_page_text(self, page_text: str, max_chars: int) -> List[str]:
        # 页内分块：避免过长 chunk
        text = clean_text(page_text)
        if not text:
            return []
        if len(text) <= max_chars:
            return [text]

        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return [text[:max_chars]]

        pieces: List[str] = []
        buffer = ""
        for paragraph in paragraphs:
            paragraph = clean_text(paragraph)
            if not paragraph:
                continue
            if len(buffer) + len(paragraph) + 1 <= max_chars:
                buffer = f"{buffer}\n{paragraph}".strip()
                continue
            if buffer:
                pieces.append(buffer)
            if len(paragraph) <= max_chars:
                buffer = paragraph
            else:
                sentence_buffer = ""
                for sentence in re.split(r"[。；;]", paragraph):
                    sentence = clean_text(sentence)
                    if not sentence:
                        continue
                    if len(sentence_buffer) + len(sentence) + 1 <= max_chars:
                        sentence_buffer = f"{sentence_buffer}。{sentence}".strip("。")
                    else:
                        if sentence_buffer:
                            pieces.append(sentence_buffer)
                        sentence_buffer = sentence
                buffer = sentence_buffer
        if buffer:
            pieces.append(buffer)
        return pieces

    def _build_relations(self, pages: List[ResourcePage], units: List[ReusableUnit]) -> List[ResourceRelation]:
        # 资源关系：page->role、page->knowledge_point、page->next、unit->recommended_use
        relations: List[ResourceRelation] = []
        seen = set()

        def append(source: str, target: str, relation: str, confidence: float) -> None:
            key = (source, target, relation)
            if not source or not target or key in seen:
                return
            seen.add(key)
            relations.append(
                ResourceRelation(
                    source=source,
                    target=target,
                    relation=relation,
                    confidence=confidence,
                )
            )

        sorted_pages = sorted(pages, key=lambda item: item.page_no)
        for index, page in enumerate(sorted_pages):
            append(page.page_id, page.page_role.value, "has_role", 0.98)
            for point in page.knowledge_points:
                append(page.page_id, point, "related_to", 0.9)
            if index < len(sorted_pages) - 1:
                append(page.page_id, sorted_pages[index + 1].page_id, "next", 0.99)

        for unit in units:
            for use in unit.recommended_use:
                append(unit.unit_id, use, "recommended_for", 0.86)

        return relations

    def _build_resource_index(self, pages: List[ResourcePage]) -> ResourceIndex:
        # 建立角色索引和知识点索引，便于后续检索
        role_map: Dict[str, List[int]] = {}
        kp_map: Dict[str, List[int]] = {}

        for page in pages:
            role_map.setdefault(page.page_role.value, []).append(page.page_no)
            for point in page.knowledge_points:
                kp_map.setdefault(point, []).append(page.page_no)

        for role, page_nos in role_map.items():
            role_map[role] = sorted(set(page_nos))
        for point, page_nos in kp_map.items():
            kp_map[point] = sorted(set(page_nos))

        return ResourceIndex(
            page_roles=role_map,
            knowledge_point_to_pages=kp_map,
        )

    def _deduplicate(self, values: List[str]) -> List[str]:
        # 保序去重
        result: List[str] = []
        seen = set()
        for value in values:
            normalized = clean_text(value or "")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result
