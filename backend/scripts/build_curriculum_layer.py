from __future__ import annotations

"""构建 syllabus 层处理产物的脚本。"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.syllabus_extractor import CurriculumSyllabusExtractor  # noqa: E402
from services.syllabus_normalizer import SyllabusNormalizer  # noqa: E402
from services.syllabus_parser import parse_pdf_directory  # noqa: E402
from services.topic_locator import TopicLocator  # noqa: E402


RAW_DIR = BACKEND_DIR / "data" / "raw" / "curriculum" / "machine_learning" / "syllabus"
PROCESSED_ROOT = BACKEND_DIR / "data" / "processed" / "curriculum" / "machine_learning"
SYLLABUS_JSON_DIR = PROCESSED_ROOT / "syllabus_json"
NORMALIZED_DIR = PROCESSED_ROOT / "normalized"
TOPIC_DIR = PROCESSED_ROOT / "topic_gradient_descent"


def ensure_dirs() -> None:
    """确保输出目录存在。"""
    for path in [SYLLABUS_JSON_DIR, NORMALIZED_DIR, TOPIC_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def to_json_file(path: Path, payload: Dict[str, Any]) -> None:
    """写入 UTF-8 JSON 文件。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_single_filename(source_file: str) -> str:
    """统一输出文件名（去重 .pdf 后缀）。"""
    name = source_file
    lower = name.lower()
    if lower.endswith(".pdf.pdf"):
        return f"{name[:-8]}.json"
    if lower.endswith(".pdf"):
        return f"{name[:-4]}.json"
    return f"{name}.json"


def summarize_chapter_hits(record: Dict[str, Any]) -> List[str]:
    """提取命中章节位置并去重。"""
    hit_locations: List[str] = []
    for hit in record.get("target_topic_hits") or []:
        location = str(hit.get("location") or "").strip()
        if location.startswith("chapter::"):
            hit_locations.append(location.replace("chapter::", "", 1))
    # 去重保序
    deduped: List[str] = []
    seen = set()
    for item in hit_locations:
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def collect_key_difficult_points(record: Dict[str, Any]) -> Dict[str, List[str]]:
    """收集章节中的重点与难点。"""
    key_points: List[str] = []
    difficult_points: List[str] = []
    for chapter in record.get("chapters") or []:
        if not isinstance(chapter, dict):
            continue
        key_points.extend([str(item).strip() for item in chapter.get("key_points") or [] if str(item).strip()])
        difficult_points.extend(
            [str(item).strip() for item in chapter.get("difficulties") or [] if str(item).strip()]
        )
    return {"key_points": _dedup(key_points), "difficult_points": _dedup(difficult_points)}


def _dedup(values: List[str]) -> List[str]:
    """保序去重。"""
    result: List[str] = []
    seen = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def build_curriculum_layer() -> Dict[str, Any]:
    """执行 syllabus 解析、标准化与主题定位主流程。"""
    ensure_dirs()

    parsed_items = parse_pdf_directory(RAW_DIR, recursive=False)
    if not parsed_items:
        raise RuntimeError(f"未在目录中发现 PDF：{RAW_DIR}")

    extractor = CurriculumSyllabusExtractor()
    normalizer = SyllabusNormalizer()
    locator = TopicLocator()

    normalized_records: List[Dict[str, Any]] = []
    process_logs: List[Dict[str, Any]] = []

    for parsed in parsed_items:
        extracted = extractor.extract_from_text(
            text=parsed.text,
            source_file=parsed.source_file,
            source_type="pdf",
        )
        normalized = normalizer.normalize(extracted)
        topic_result = locator.locate(normalized)
        normalized.update(topic_result)

        out_file_name = build_single_filename(parsed.source_file)
        to_json_file(SYLLABUS_JSON_DIR / out_file_name, normalized)
        normalized_records.append(normalized)

        log_item = {
            "source_file": parsed.source_file,
            "page_count": parsed.page_count,
            "warnings": parsed.warnings,
            "chapter_count": len(normalized.get("chapters") or []),
            "topic_hit_count": len(normalized.get("target_topic_hits") or []),
            "prerequisites_count": len(normalized.get("prerequisites") or []),
        }
        process_logs.append(log_item)

        print(
            "[OK] {file} | pages={pages} | chapters={chapters} | topic_hits={hits} | warnings={warns}".format(
                file=parsed.source_file,
                pages=parsed.page_count,
                chapters=log_item["chapter_count"],
                hits=log_item["topic_hit_count"],
                warns=len(parsed.warnings),
            )
        )

    merged_payload = _build_merged_payload(normalized_records)
    merged_path = NORMALIZED_DIR / "all_syllabi_merged.json"
    to_json_file(merged_path, merged_payload)

    topic_payload = _build_topic_payload(normalized_records)
    topic_path = TOPIC_DIR / "gradient_descent_curriculum_view.json"
    to_json_file(topic_path, topic_payload)

    return {
        "processed_count": len(normalized_records),
        "syllabus_json_dir": str(SYLLABUS_JSON_DIR),
        "merged_path": str(merged_path),
        "topic_path": str(topic_path),
        "process_logs": process_logs,
    }


def _build_merged_payload(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构建多 syllabus 合并视图。"""
    base_info_summary: List[Dict[str, Any]] = []
    topic_position_comparison: List[Dict[str, Any]] = []
    objectives_comparison: List[Dict[str, Any]] = []
    key_difficulty_comparison: List[Dict[str, Any]] = []
    prerequisites_comparison: List[Dict[str, Any]] = []

    for record in records:
        source_file = str(record.get("source_file") or "")

        base_info_summary.append(
            {
                "source_file": source_file,
                "course_name": record.get("course_name"),
                "course_code": record.get("course_code"),
                "course_type": record.get("course_type"),
                "target_major": record.get("target_major") or [],
                "credits": record.get("credits"),
                "total_hours": record.get("total_hours"),
                "theory_hours": record.get("theory_hours"),
                "lab_hours": record.get("lab_hours"),
            }
        )

        topic_position_comparison.append(
            {
                "source_file": source_file,
                "topic": "梯度下降法",
                "hit_count": len(record.get("target_topic_hits") or []),
                "chapter_locations": summarize_chapter_hits(record),
            }
        )

        objectives_comparison.append(
            {
                "source_file": source_file,
                "course_objectives": record.get("course_objectives") or [],
            }
        )

        key_difficult = collect_key_difficult_points(record)
        key_difficulty_comparison.append(
            {
                "source_file": source_file,
                "key_points": key_difficult["key_points"],
                "difficult_points": key_difficult["difficult_points"],
            }
        )

        prerequisites_comparison.append(
            {
                "source_file": source_file,
                "prerequisites": record.get("prerequisites") or [],
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "topic": "梯度下降法",
        "syllabus_count": len(records),
        "all_syllabi": records,
        "base_info_summary": base_info_summary,
        "topic_position_comparison": topic_position_comparison,
        "teaching_objectives_comparison": objectives_comparison,
        "key_difficulty_comparison": key_difficulty_comparison,
        "prerequisites_comparison": prerequisites_comparison,
    }


def _build_topic_payload(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构建目标主题视图。"""
    topic_views: List[Dict[str, Any]] = []
    for record in records:
        topic_views.append(
            {
                "source_file": record.get("source_file"),
                "course_name": record.get("course_name"),
                "target_topic": "梯度下降法",
                "target_topic_hits": record.get("target_topic_hits") or [],
                "target_topic_summary": record.get("target_topic_summary") or "",
                "target_topic_prerequisites": record.get("target_topic_prerequisites") or [],
                "target_topic_followups": record.get("target_topic_followups") or [],
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "topic": "梯度下降法",
        "keywords": TopicLocator.KEYWORDS,
        "syllabus_topic_views": topic_views,
    }


if __name__ == "__main__":
    result = build_curriculum_layer()
    print(
        json.dumps(
            {
                "processed_count": result["processed_count"],
                "syllabus_json_dir": result["syllabus_json_dir"],
                "merged_path": result["merged_path"],
                "topic_path": result["topic_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
