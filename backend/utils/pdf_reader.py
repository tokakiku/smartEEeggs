from io import BytesIO

from pypdf import PdfReader


def read_pdf_pages(file_bytes: bytes) -> list[str]:
    # 按页提取 PDF 文本，单页失败时跳过并保留顺序。
    if not file_bytes:
        return []

    try:
        reader = PdfReader(BytesIO(file_bytes))
    except Exception:
        return []

    pages: list[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            pages.append("")
            continue
        pages.append(page_text.strip())
    return pages


def read_pdf_text(file_bytes: bytes) -> str:
    # 读取 PDF 全部页面文本，单页失败时跳过但不中断整体流程。
    text_parts = [page for page in read_pdf_pages(file_bytes) if page.strip()]

    return "\n".join(text_parts).strip()
