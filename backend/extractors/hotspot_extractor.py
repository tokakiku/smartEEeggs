import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

"""热点层抽取器：将新闻/行业案例转为教学可复用结构。"""

from extractors.base_extractor import BaseExtractor
from schema.hotspot_schema import (
    HotspotChunk,
    HotspotEventType,
    HotspotExtractionResult,
    HotspotIndex,
    HotspotInfo,
    HotspotItem,
    HotspotRelation,
    TeachingUsage,
)
from schema.parsed_document_schema import ParsedDocument
from utils.text_cleaner import clean_text, split_paragraphs


class HotspotExtractor(BaseExtractor):
    """热点层 extractor。"""
    # 热点层 extractor：将新闻/行业案例转为可教学复用的现实案例条目

    KNOWLEDGE_TERMS = [
        "机器学习",
        "人工智能",
        "深度学习",
        "神经网络",
        "计算机视觉",
        "自然语言处理",
        "预测模型",
        "时间序列预测",
        "遥感",
        "卫星图像",
        "森林监测",
        "森林砍伐",
        "气候变化",
        "推荐系统",
        "强化学习",
        "特征工程",
        "模型泛化",
        "可解释性",
        "大模型应用",
        "AI 产品生态",
        "开源社区",
        "GitHub 社区传播",
        "模型命名与品牌传播",
    ]

    EVENT_PATTERNS = {
        HotspotEventType.product_release: [
            r"(发布|推出|上线|开源|release|launched|unveiled|announced|renamed|rebrand|更名|版本|update|plugin|集成)",
        ],
        HotspotEventType.research_breakthrough: [
            r"(研究|论文|突破|准确率|提升|实验|scientists|study|paper|breakthrough|improve)",
        ],
        HotspotEventType.policy_event: [
            r"(政策|监管|法案|审批|政府|条例|compliance|regulation|policy|\blaw\b)",
        ],
        HotspotEventType.industry_application: [
            r"(应用于|用于|部署于|落地|帮助|投入使用|deployment|applied|used to|prevent)",
        ],
        HotspotEventType.social_case: [
            r"(竞选|公共治理|社会|伦理|争议|风险|public|election|debate|controversy|community|fuss|热议|讨论)",
        ],
        HotspotEventType.historical_case: [
            r"(早在|历史上|回顾|年代|在.*年|historically|decades ago|history)",
        ],
    }

    NOISE_LINE_PATTERNS = [
        r"^(home|en|zh|中文|ai news|latest ai news|article content)$",
        r"^(published in|time\s*:|read\s*:)",
        r"^(related recommendations?)$",
        r"^(aibase)$",
        r"^(intelligent future)",
        r"^english.*",
        r"^漏\s*\d{4}",
        r"^\d+(\.\d+)?\s*[km]?$",
        r".*\.png$",
    ]

    STOP_SECTION_PATTERNS = [
        r"^(related recommendations?)$",
        r"^project address\s*:",
        r"^official website\s*:",
        r"^aibase$",
    ]

    def extract(self, parsed_doc: ParsedDocument) -> Dict[str, Any]:
        # 主流程：元信息 -> 摘要 -> 事件分类 -> 知识点 -> 教学用途 -> 证据片段 -> chunk/relations/index
        hotspot_id = f"hotspot-{parsed_doc.doc_id}"
        title = self._extract_title(parsed_doc)
        paragraphs = self._collect_effective_paragraphs(parsed_doc.raw_text)
        paragraphs = self._filter_noise_paragraphs(title, paragraphs)
        publish_date = self._extract_publish_date(parsed_doc.raw_text, parsed_doc.file_name)
        url = self._extract_url(parsed_doc.raw_text, parsed_doc.source_name, parsed_doc.file_name)
        source = self._extract_source(parsed_doc.raw_text, parsed_doc.source_name, url)
        author = self._extract_author(parsed_doc.raw_text)

        summary = self._build_summary(title, paragraphs)
        event_type = self._classify_event_type(title, summary, paragraphs)
        related_points = self._extract_related_knowledge_points(title, summary, paragraphs)
        keywords = self._extract_keywords(title, summary, paragraphs, related_points)
        evidence_snippets = self._extract_evidence_snippets(title, paragraphs, related_points)
        teaching_usage = self._infer_teaching_usage(event_type, summary, paragraphs)
        tags = self._build_tags(event_type, related_points, keywords, teaching_usage)

        hotspot_info = HotspotInfo(
            hotspot_id=hotspot_id,
            title=title,
            source=source,
            publish_date=publish_date,
            source_type=parsed_doc.source_type,
            url=url,
            author=author,
        )
        hotspot_item = HotspotItem(
            hotspot_id=hotspot_id,
            title=title,
            summary=summary,
            event_type=event_type,
            news_role=self._infer_news_role(event_type=event_type, title=title, summary=summary),
            related_knowledge_points=related_points,
            keywords=keywords,
            teaching_usage=teaching_usage,
            evidence_snippets=evidence_snippets,
            tags=tags,
        )
        chunks = self._build_chunks(hotspot_item, paragraphs, evidence_snippets)
        relations = self._build_relations(hotspot_id, event_type, related_points, teaching_usage)
        hotspot_index = self._build_hotspot_index(hotspot_item)

        result_model = HotspotExtractionResult(
            hotspot_info=hotspot_info,
            hotspot_item=[hotspot_item],
            chunks=chunks,
            relations=relations,
            hotspot_index=hotspot_index,
        )
        if hasattr(result_model, "model_dump"):
            return result_model.model_dump(mode="json")
        return result_model.dict()

    def _extract_title(self, parsed_doc: ParsedDocument) -> str:
        # 标题优先 parsed_doc.title，其次正文首条有效段
        title = clean_text(parsed_doc.title or "")
        if title:
            return title
        lines = self._collect_effective_paragraphs(parsed_doc.raw_text)
        if lines:
            return lines[0][:120]
        return parsed_doc.file_name

    def _collect_effective_paragraphs(self, text: str) -> List[str]:
        # 有效段落过滤：去掉过短、纯符号、无意义站点信息
        paragraphs = split_paragraphs(text)
        result: List[str] = []
        for paragraph in paragraphs:
            cleaned = clean_text(paragraph)
            if not cleaned:
                continue
            if len(cleaned) < 12:
                continue
            if re.match(r"^[\W_]+$", cleaned):
                continue
            if re.search(r"(copyright|all rights reserved)", cleaned, flags=re.I):
                continue
            result.append(cleaned)
        return result

    def _extract_publish_date(self, text: str, file_name: str) -> Optional[str]:
        # 日期提取：兼容中英文格式
        target = f"{file_name}\n{text}"
        patterns = [
            r"(\d{4}-\d{1,2}-\d{1,2})",
            r"(\d{4}/\d{1,2}/\d{1,2})",
            r"(\d{4}\.\d{1,2}\.\d{1,2})",
            r"(\d{4}年\d{1,2}月\d{1,2}日)",
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, target, flags=re.I)
            if match:
                return match.group(1)
        return None

    def _extract_source(self, text: str, source_name: Optional[str], url: Optional[str]) -> Optional[str]:
        # 来源提取：优先 source_name，再从正文匹配“来源/Source/Reuters”
        if source_name:
            if source_name.startswith("http://") or source_name.startswith("https://"):
                domain = urlparse(source_name).netloc.lower()
                if "aibase.com" in domain:
                    return "AIbase"
                if domain:
                    return domain
            return source_name
        if url:
            domain = urlparse(url).netloc.lower()
            if "aibase.com" in domain:
                return "AIbase"
        source_patterns = [
            r"(?:来源|來源|source)\s*[:：]\s*([^\n]+)",
            r"\b(reuters|ap|bbc|wsj|nyt|新华网|人民日报|澎湃新闻)\b",
        ]
        for pattern in source_patterns:
            match = re.search(pattern, text, flags=re.I)
            if not match:
                continue
            value = clean_text(match.group(1) if match.lastindex else match.group(0))
            if value:
                return value
        return None

    def _extract_url(self, text: str, source_name: Optional[str], file_name: str) -> Optional[str]:
        # 链接提取：优先显式来源链接，再抓站内链接，最后抓首个 http(s) 链接
        if source_name and (source_name.startswith("http://") or source_name.startswith("https://")):
            return source_name.strip()

        lower_name = (file_name or "").lower()
        match_id = re.match(r"^aibase_(\d+)\.html$", lower_name)
        if match_id:
            return f"https://news.aibase.com/news/{match_id.group(1)}"

        urls = re.findall(r"(https?://[^\s]+)", text)
        cleaned_urls = [item.strip(".,)") for item in urls if item]
        for link in cleaned_urls:
            if "news.aibase.com/news/" in link:
                return link
        for link in cleaned_urls:
            if link.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            return link
        return None

    def _extract_author(self, text: str) -> Optional[str]:
        # 作者提取：兼容 By xxx / 作者：xxx
        patterns = [
            r"(?m)^\s*(?:By)\s+([^\n]{2,60})\s*$",
            r"(?m)^\s*(?:作者|撰稿人)\s*[:：]\s*([^\n]{2,30})\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                return clean_text(match.group(1))
        return None

    def _filter_noise_paragraphs(self, title: str, paragraphs: List[str]) -> List[str]:
        # 过滤导航栏/推荐位/版权等噪声，并尽量截取正文主区间
        normalized_title = clean_text(title).lower()
        baseline: List[str] = []
        for paragraph in paragraphs:
            normalized = clean_text(paragraph)
            lowered = normalized.lower()
            if not normalized:
                continue
            if normalized_title and lowered == normalized_title:
                continue
            if self._is_noise_line(lowered):
                continue
            baseline.append(normalized)

        if not baseline:
            return paragraphs

        start_index = 0
        for idx, paragraph in enumerate(baseline):
            lowered = paragraph.lower()
            if re.match(r"^(published in|time\s*:|read\s*:)", lowered):
                start_index = idx + 1
                continue
            if len(paragraph) >= 30 and not self._looks_like_heading(paragraph):
                start_index = idx
                break

        filtered: List[str] = []
        for paragraph in baseline[start_index:]:
            lowered = paragraph.lower()
            if any(re.match(pattern, lowered) for pattern in self.STOP_SECTION_PATTERNS):
                break
            if self._is_noise_line(lowered):
                continue
            filtered.append(paragraph)

        if filtered:
            return filtered
        return paragraphs

    def _build_summary(self, title: str, paragraphs: List[str]) -> str:
        # 规则摘要：覆盖“更名 + 热度指标 + 产品更新”三类核心信息
        if not paragraphs:
            return title
        summary_parts: List[str] = []
        key_buckets = [
            r"(rename|renamed|rebrand|更名|命名|OpenClaw|Moltbot|Clawd)",
            r"(github|stars?|100,000|traffic|2 million)",
            r"(core features|latest updates|update|plugin|integration|supports|security|maintainer|multi-model|multi-platform|models such as|支持|集成|多模型|多平台)",
        ]
        for pattern in key_buckets:
            for paragraph in paragraphs[:12]:
                if self._looks_like_heading(paragraph):
                    continue
                if re.search(pattern, paragraph, flags=re.I):
                    summary_parts.append(paragraph)
                    break

        update_pattern = (
            r"(core features|latest updates|update|plugin|integration|supports|security|maintainer|"
            r"multi-model|multi-platform|models such as|支持|集成|多模型|多平台)"
        )
        if not any(re.search(update_pattern, item, flags=re.I) for item in summary_parts):
            for paragraph in paragraphs[:20]:
                if self._looks_like_heading(paragraph):
                    continue
                if re.search(update_pattern, paragraph, flags=re.I):
                    summary_parts.append(paragraph)
                    break

        if not summary_parts:
            for paragraph in paragraphs:
                if self._looks_like_heading(paragraph):
                    continue
                summary_parts.append(paragraph)
                if len(summary_parts) >= 2:
                    break

        summary_parts = self._deduplicate(summary_parts)[:3]
        compact_parts: List[str] = []
        for paragraph in summary_parts:
            compact = paragraph
            if len(compact) > 220:
                compact = compact[:220].rstrip("，,。.") + "..."
            compact_parts.append(compact)
        summary = " ".join(compact_parts)
        # 去掉混入正文中的“纯标题句”，避免摘要尾部出现章节名
        summary_sentences = re.split(r"(?<=[。！？.!?])\s+", summary)
        summary = " ".join([item for item in summary_sentences if not self._looks_like_heading(item)])
        summary = re.sub(r"\s{2,}", " ", summary).strip()
        if len(summary) > 560:
            summary = summary[:560].rstrip("，,。.") + "..."
        return summary

    def _classify_event_type(
        self,
        title: str,
        summary: str,
        paragraphs: List[str],
    ) -> HotspotEventType:
        # 事件类型规则分类：按命中数打分，平分时按教学优先级选择
        content = f"{title}\n{summary}\n" + "\n".join(paragraphs[:4])
        priority = [
            HotspotEventType.product_release,
            HotspotEventType.social_case,
            HotspotEventType.industry_application,
            HotspotEventType.research_breakthrough,
            HotspotEventType.policy_event,
            HotspotEventType.historical_case,
        ]
        scores: Dict[HotspotEventType, int] = {}
        for event_type in priority:
            score = 0
            for pattern in self.EVENT_PATTERNS.get(event_type, []):
                score += len(re.findall(pattern, content, flags=re.I))
            scores[event_type] = score

        if re.search(r"(rename|renamed|rebrand|更名|版本|star|github)", content, flags=re.I):
            scores[HotspotEventType.product_release] = scores.get(HotspotEventType.product_release, 0) + 2
        if re.search(r"(community|fuss|热议|讨论)", content, flags=re.I):
            scores[HotspotEventType.social_case] = scores.get(HotspotEventType.social_case, 0) + 1

        best_score = max(scores.values()) if scores else 0
        if best_score <= 0:
            return HotspotEventType.other_event

        best_types = [event_type for event_type in priority if scores.get(event_type, 0) == best_score]
        if best_types:
            return best_types[0]
        return HotspotEventType.other_event

    def _extract_related_knowledge_points(
        self,
        title: str,
        summary: str,
        paragraphs: List[str],
    ) -> List[str]:
        # 相关知识点映射：标题/摘要优先 + 正文补充
        content = f"{title}\n{summary}\n" + "\n".join(paragraphs[:5])
        points: List[str] = []
        for term in self.KNOWLEDGE_TERMS:
            if term.lower() in content.lower():
                points.append(term)

        pattern_points = [
            (r"(openclaw|clawd|moltbot|ai assistant|ai agent|个人ai助手)", "大模型应用"),
            (r"(open source|开源|community|社区|maintainer)", "开源社区"),
            (r"(github|stars?|repository|repo)", "GitHub 社区传播"),
            (r"(rename|rebrand|更名|命名|品牌)", "模型命名与品牌传播"),
            (r"(platform|生态|plugin|integration|provider|多模型|多平台)", "AI 产品生态"),
        ]
        for pattern, point in pattern_points:
            if re.search(pattern, content, flags=re.I):
                points.append(point)

        # 额外规则：森林砍伐预警类案例映射预测建模知识点
        if re.search(r"(deforestation|forest|rainforest|砍伐|森林)", content, flags=re.I):
            points.extend(["森林砍伐", "预测模型", "机器学习"])
        if re.search(r"(satellite|imagery|remote sensing|卫星|遥感)", content, flags=re.I):
            points.extend(["卫星图像", "遥感"])
        return self._deduplicate(points)[:8]

    def _extract_keywords(
        self,
        title: str,
        summary: str,
        paragraphs: List[str],
        related_points: List[str],
    ) -> List[str]:
        # 关键词抽取：知识点 + 标题短词 + 领域词
        candidates: List[str] = []
        candidates.extend(related_points)
        candidates.extend(self._extract_short_terms(title))
        candidates.extend(self._extract_short_terms(summary))
        for paragraph in paragraphs[:2]:
            candidates.extend(self._extract_short_terms(paragraph))
        return self._deduplicate(candidates)[:10]

    def _extract_short_terms(self, text: str) -> List[str]:
        # 轻量短词提取：英文术语与中文名词短语
        result: List[str] = []
        for token in re.findall(r"\b[A-Za-z][A-Za-z\-]{3,30}\b", text):
            lowered = token.lower()
            if lowered in {"with", "from", "that", "this", "have", "will", "were"}:
                continue
            result.append(token)

        # 中文短语提取（2~8个字）
        for token in re.findall(r"[\u4e00-\u9fa5]{2,8}", text):
            if token in {"通过", "可以", "相关", "以及", "进行", "对于"}:
                continue
            result.append(token)
        return result

    def _extract_evidence_snippets(
        self,
        title: str,
        paragraphs: List[str],
        related_points: List[str],
    ) -> List[str]:
        # 证据片段抽取：保留 1~3 条有代表性的支持句
        sentences: List[str] = []
        for paragraph in paragraphs[:8]:
            if self._looks_like_heading(paragraph):
                continue
            pieces = re.split(r"[。！？!?;；]", paragraph)
            for piece in pieces:
                sentence = clean_text(piece)
                if len(sentence) < 18:
                    continue
                if self._is_noise_line(sentence.lower()):
                    continue
                if self._looks_like_heading(sentence):
                    continue
                if len(sentence) > 180:
                    sentence = sentence[:180].rstrip("，,。.") + "..."
                sentences.append(sentence)

        scored: List[tuple[int, str]] = []
        for sentence in sentences:
            score = 0
            if any(point.lower() in sentence.lower() for point in related_points):
                score += 2
            if re.search(r"(help|improve|部署|应用|研究|发布|更新|更名|社区|star|github|风险|伦理)", sentence, flags=re.I):
                score += 1
            if any(word in sentence for word in title.split()):
                score += 1
            scored.append((score, sentence))

        scored.sort(key=lambda item: item[0], reverse=True)
        picked: List[str] = []
        for _, sentence in scored:
            if sentence in picked:
                continue
            picked.append(sentence)
            if len(picked) >= 3:
                break

        if not picked and paragraphs:
            picked.append(paragraphs[0][:160])
        return picked

    def _is_noise_line(self, lowered: str) -> bool:
        # 噪声识别：站点导航/模板文案/统计数字等
        for pattern in self.NOISE_LINE_PATTERNS:
            if re.match(pattern, lowered, flags=re.I):
                return True
        if lowered.startswith(("source:", "来源:", "by ", "作者:", "title:")):
            return True
        return False

    def _looks_like_heading(self, text: str) -> bool:
        # 标题样式判断：用于摘要和证据阶段过滤章节名
        cleaned = clean_text(text)
        if not cleaned:
            return False
        if (
            len(cleaned) <= 80
            and re.match(r"^[A-Z][A-Za-z0-9'\-\s:&]+$", cleaned)
            and "." not in cleaned
            and "," not in cleaned
        ):
            return True
        if len(cleaned) <= 28 and re.match(r"^[A-Z][A-Za-z0-9'\-\s:]+$", cleaned):
            return True
        if len(cleaned) <= 20 and re.match(r"^[\u4e00-\u9fa5A-Za-z0-9：:、，,]+$", cleaned):
            return True
        return False

    def _infer_teaching_usage(
        self,
        event_type: HotspotEventType,
        summary: str,
        paragraphs: List[str],
    ) -> List[TeachingUsage]:
        # 教学用途推断：课程导入/案例分析/讨论/拓展阅读/动机激发
        content = f"{summary}\n" + "\n".join(paragraphs[:3])
        usage: List[TeachingUsage] = [TeachingUsage.extended_reading]

        if event_type in {
            HotspotEventType.industry_application,
            HotspotEventType.research_breakthrough,
            HotspotEventType.product_release,
            HotspotEventType.historical_case,
        }:
            usage.append(TeachingUsage.course_intro)
            usage.append(TeachingUsage.motivation)

        if event_type in {HotspotEventType.industry_application, HotspotEventType.social_case}:
            usage.append(TeachingUsage.case_analysis)

        if re.search(
            r"(风险|伦理|争议|偏见|监管|治理|隐私|controversy|risk|ethic|fuss|community debate|community|热议|社区)",
            content,
            flags=re.I,
        ):
            usage.append(TeachingUsage.discussion)

        return self._deduplicate(usage)

    def _build_tags(
        self,
        event_type: HotspotEventType,
        related_points: List[str],
        keywords: List[str],
        teaching_usage: List[TeachingUsage],
    ) -> List[str]:
        # 标签构建：用于检索和过滤
        tags: List[str] = ["hotspot", event_type.value]
        tags.extend(related_points[:5])
        tags.extend(keywords[:5])
        tags.extend([usage.value for usage in teaching_usage])
        return self._deduplicate(tags)

    def _build_chunks(
        self,
        hotspot_item: HotspotItem,
        paragraphs: List[str],
        evidence_snippets: List[str],
    ) -> List[HotspotChunk]:
        # 分块生成：每条热点至少一个分块，保留可检索核心文本
        chunks: List[HotspotChunk] = []
        base_texts: List[str] = [hotspot_item.summary]
        base_texts.extend(evidence_snippets[:2])
        if len(base_texts) < 2 and paragraphs:
            base_texts.append(paragraphs[0])

        for index, text in enumerate(base_texts, start=1):
            cleaned = clean_text(text)
            if not cleaned:
                continue
            chunk = HotspotChunk(
                chunk_id=f"{hotspot_item.hotspot_id}-c{index}",
                hotspot_id=hotspot_item.hotspot_id,
                title=hotspot_item.title,
                text=cleaned,
                event_type=hotspot_item.event_type,
                related_knowledge_points=hotspot_item.related_knowledge_points,
                tags=self._deduplicate(
                    ["hotspot", hotspot_item.event_type.value] + hotspot_item.related_knowledge_points[:4]
                ),
                metadata={
                    "chunk_index": index,
                    "text_length": len(cleaned),
                    "teaching_usage": [usage.value for usage in hotspot_item.teaching_usage],
                },
                section=hotspot_item.title,
            )
            chunks.append(chunk)
        return chunks

    def _build_relations(
        self,
        hotspot_id: str,
        event_type: HotspotEventType,
        related_points: List[str],
        teaching_usage: List[TeachingUsage],
    ) -> List[HotspotRelation]:
        # 热点关系构建：类型、知识点、教学用途
        relations: List[HotspotRelation] = []
        seen = set()

        def add(source: str, target: str, relation: str, confidence: float) -> None:
            key = (source, target, relation)
            if not source or not target or key in seen:
                return
            seen.add(key)
            relations.append(
                HotspotRelation(
                    source=source,
                    target=target,
                    relation=relation,
                    confidence=confidence,
                )
            )

        add(hotspot_id, event_type.value, "has_type", 0.98)
        for point in related_points:
            add(hotspot_id, point, "related_to", 0.9)
        for usage in teaching_usage:
            add(hotspot_id, usage.value, "recommended_for", 0.88)
        return relations

    def _build_hotspot_index(self, item: HotspotItem) -> HotspotIndex:
        # 热点索引构建：按事件类型、知识点、教学用途反查
        event_index = {item.event_type.value: [item.hotspot_id]}
        kp_index = {point: [item.hotspot_id] for point in item.related_knowledge_points}
        usage_index = {usage.value: [item.hotspot_id] for usage in item.teaching_usage}
        return HotspotIndex(
            event_type=event_index,
            knowledge_point_to_hotspots=kp_index,
            teaching_usage=usage_index,
        )

    def _infer_news_role(self, event_type: HotspotEventType, title: str, summary: str) -> str:
        content = f"{title} {summary}".lower()
        if event_type in {HotspotEventType.product_release, HotspotEventType.industry_application}:
            return "case"
        if event_type in {HotspotEventType.research_breakthrough, HotspotEventType.historical_case}:
            return "industry_update"
        if event_type == HotspotEventType.policy_event:
            return "background"
        if event_type == HotspotEventType.social_case:
            return "case"
        if any(token in content for token in ["案例", "example", "实践", "应用"]):
            return "case"
        return "background"

    def _deduplicate(self, values: List[Any]) -> List[Any]:
        # 保序去重
        result: List[Any] = []
        seen = set()
        for value in values:
            key = str(value).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result
