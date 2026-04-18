# ==========================================
# 灵犀智课 - 多格式导出与预览引擎 (阶段5核心)
# 特性：PPT转PDF | PDF切片PNG预览 | 24小时垃圾文件回收
# ==========================================

import os
import time
import shutil
import logging
import subprocess
import fitz  # 需要安装: pip install PyMuPDF

logger = logging.getLogger("Export_Service")


def convert_pptx_to_pdf(pptx_path: str, output_dir: str) -> str:
    """调用 LibreOffice 将 PPTX 转换为 PDF (支持 Docker/Linux/Windows)"""
    if not os.path.exists(pptx_path):
        return ""

    try:
        # 兼容 Windows 和 Linux 的 LibreOffice 调用命令
        command = "soffice" if os.name != 'nt' else "soffice.exe"

        logger.info(f"📄 正在将 PPT 转换为 PDF: {pptx_path}")
        subprocess.run(
            [command, "--headless", "--convert-to", "pdf", "--outdir", output_dir, pptx_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
        )

        # 构建预期输出的 PDF 路径
        base_name = os.path.splitext(os.path.basename(pptx_path))[0]
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")

        if os.path.exists(pdf_path):
            return pdf_path
        else:
            logger.warning("⚠️ LibreOffice 执行完毕但未找到 PDF 文件。")
            return ""
    except FileNotFoundError:
        logger.warning("⚠️ 操作系统未安装 LibreOffice，自动降级：跳过 PDF 生成。")
        return ""
    except Exception as e:
        logger.error(f"❌ PDF 转换失败: {str(e)}")
        return ""


def generate_preview_images(pdf_path: str, output_dir: str, project_id: int) -> list:
    """利用 PyMuPDF 将 PDF 的每一页切成高清 PNG 预览图"""
    if not pdf_path or not os.path.exists(pdf_path):
        return []

    preview_urls = []
    try:
        logger.info(f"🖼️ 正在生成项目 {project_id} 的高清预览图...")
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # 缩放系数，2.0 代表生成 2 倍高清图
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))

            img_filename = f"preview_{project_id}_page_{page_num + 1}.png"
            img_path = os.path.join(output_dir, img_filename)
            pix.save(img_path)

            preview_urls.append(f"/static/exports/{img_filename}")

        doc.close()
        return preview_urls
    except Exception as e:
        logger.error(f"❌ 预览图切片失败: {str(e)}")
        return []


def cleanup_temp_files(directories: list, max_age_hours: int = 24):
    """垃圾回收：清理超过指定小时数的旧文件"""
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for filename in os.listdir(directory):
            # 跳过模板或占位文件
            if filename in ["template.pptx", ".gitkeep"]:
                continue

            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"⚠️ 清理文件失败 {filepath}: {e}")

    if deleted_count > 0:
        logger.info(f"🧹 垃圾回收完成，清理了 {deleted_count} 个过期文件。")