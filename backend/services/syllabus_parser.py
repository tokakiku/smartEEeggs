from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader


@dataclass
class ParsedSyllabusText:
    source_file: str
    source_type: str
    text: str
    page_count: int
    warnings: List[str]


def parse_pdf_file(file_path: Path) -> ParsedSyllabusText:
    warnings: List[str] = []
    try:
        reader = PdfReader(str(file_path))
        pages: List[str] = []
        for index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"page_{index + 1}_extract_failed: {exc}")

        raw_text = "\n".join(pages).strip()
        return ParsedSyllabusText(
            source_file=file_path.name,
            source_type="pdf",
            text=raw_text,
            page_count=len(reader.pages),
            warnings=warnings,
        )
    except Exception as exc:  # noqa: BLE001
        return ParsedSyllabusText(
            source_file=file_path.name,
            source_type="pdf",
            text="",
            page_count=0,
            warnings=[f"parse_failed: {exc}"],
        )


def _is_pdf_like(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(".pdf") or ".pdf." in lower_name


def parse_pdf_directory(directory: Path, recursive: bool = False) -> List[ParsedSyllabusText]:
    if not directory.exists():
        raise FileNotFoundError(f"directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    files = sorted([path for path in directory.glob(pattern) if path.is_file() and _is_pdf_like(path)])
    outputs: List[ParsedSyllabusText] = []
    for file_path in files:
        outputs.append(parse_pdf_file(file_path))
    return outputs


def parse_single_pdf(file_path: str | Path) -> ParsedSyllabusText:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return parse_pdf_file(path)


def find_first_text(parsed_items: List[ParsedSyllabusText]) -> Optional[str]:
    for item in parsed_items:
        if item.text.strip():
            return item.text
    return None
