import pytest
from pathlib import Path
import uuid

from services.ingest_service import DocumentIngestService, IngestValidationError
from services.storage_service import StorageService


def _build_service_with_local_storage() -> DocumentIngestService:
    # 在项目内创建测试目录，避免系统临时目录权限问题。
    backend_dir = Path(__file__).resolve().parents[2]
    base_dir = backend_dir / "data" / "test_outputs" / uuid.uuid4().hex
    base_dir.mkdir(parents=True, exist_ok=True)
    return DocumentIngestService(storage_service=StorageService(base_dir=base_dir))


def test_document_ingest_syllabus():
    # 简化教学大纲样例应能走完整 ingest 并生成落盘文件。
    service = _build_service_with_local_storage()

    content = (
        "课程编号：CS101\n"
        "课程名称：机器学习\n"
        "一、课程的教学目标与任务\n"
        "1. 掌握基本概念。\n"
        "二、课程具体内容及基本要求\n"
        "（一）绪论（2 学时）\n"
        "1.基本要求\n"
        "理解机器学习概念。\n"
        "2.重点、难点\n"
        "重点：基本概念。\n"
    ).encode("utf-8")

    response = service.ingest_document(
        file_bytes=content,
        file_name="syllabus.docx",
        layer="syllabus",
        source_type="upload",
        source_name="test",
    )

    assert response.parse_status == "success"
    assert response.extract_status == "success"
    assert response.storage_status == "success"
    assert response.structured_output_path and Path(response.structured_output_path).exists()
    assert response.chunks_output_path and Path(response.chunks_output_path).exists()
    assert response.raw_file_path and Path(response.raw_file_path).exists()


def test_invalid_layer():
    # 非法 layer 需要返回明确错误。
    service = _build_service_with_local_storage()

    with pytest.raises(IngestValidationError):
        service.ingest_document(
            file_bytes=b"demo",
            file_name="sample.docx",
            layer="invalid-layer",
        )


def test_unsupported_file_type():
    # 不支持的扩展名需要返回明确错误。
    service = _build_service_with_local_storage()

    with pytest.raises(IngestValidationError):
        service.ingest_document(
            file_bytes=b"demo",
            file_name="sample.txt",
            layer="syllabus",
        )
