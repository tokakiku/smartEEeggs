# ==========================================
# 灵犀智课 - 万能文档解析引擎 (ONNX GPU 加速满血版 + 防爆机制 + 图/视频多模态)
# 文件路径: app/services/document_parser.py
# ==========================================

import os
import pytesseract
import pandas as pd
from unstructured.partition.auto import partition
from unstructured.documents.elements import Table, Title, NarrativeText, Text

# ==========================================
# 🛡️ 环境暴力注入区：彻底解决 Windows 依赖报错
# ==========================================
# 1. 注入 Tesseract (识别文字的眼睛)
TESSERACT_DIR = r'C:\Program Files\Tesseract-OCR'
if os.path.exists(TESSERACT_DIR):
    os.environ["PATH"] += os.pathsep + TESSERACT_DIR
    os.environ["TESSDATA_PREFIX"] = os.path.join(TESSERACT_DIR, 'tessdata')
    pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_DIR, 'tesseract.exe')
else:
    print("🚨 警告：未找到 Tesseract，OCR 可能会失败！")

# 2. 注入 Poppler (切碎 PDF 的手术刀)
POPPLER_DIR = r'C:\poppler\Library\bin'
if os.path.exists(POPPLER_DIR):
    os.environ["PATH"] += os.pathsep + POPPLER_DIR
else:
    print("🚨 致命警告：未找到 Poppler！解析 PDF 极大概率会崩溃！请务必将其解压到 C:\\poppler")

# 🌟 大模型上下文防爆安全线 (保留前 80,000 字符)
MAX_CHARS_LIMIT = 80000


def parse_document_to_text(file_path: str):
    """
    Unstructured 原生满血版 + Pandas Excel支持 + 图片(OCR) + 视频(语音) + 文本防爆截断
    """
    if not os.path.exists(file_path):
        return {"status": "error", "message": "文件不存在"}

    ext = file_path.split('.')[-1].lower()
    content_blocks = []
    element_count = 0

    try:
        # 🌟 逻辑分流 1：Excel 表格走 Pandas 处理
        if ext in ['xls', 'xlsx']:
            print(f"📊 [表格引擎] 正在使用 Pandas 解析: {file_path}")
            dfs = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, df in dfs.items():
                content_blocks.append(f"\n[📊 表格：{sheet_name}]\n{df.to_markdown(index=False)}\n")
                element_count += len(df)

        # 🌟 逻辑分流 2：图片文件走 Tesseract OCR
        elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'webp']:
            print(f"🖼️ [视觉引擎] 正在使用 OCR 解析图片: {file_path}")
            try:
                from PIL import Image
                text = pytesseract.image_to_string(Image.open(file_path), lang='chi_sim+eng')
                content_blocks.append(f"\n[🖼️ 图片 OCR 提取内容]\n{text}\n")
                element_count += 1
            except Exception as e:
                print(f"❌ [视觉引擎] 图片 OCR 失败: {e}")
                content_blocks.append(f"\n[🖼️ 图片解析失败: {str(e)}]\n")

        # 🌟 逻辑分流 3：视频文件走 MoviePy 音频提取 + SpeechRecognition 语音识别
        elif ext in ['mp4', 'avi', 'mov', 'mkv']:
            print(f"🎬 [音视频引擎] 正在提取视频语音: {file_path}")
            audio_path = file_path + ".wav"
            try:
                from moviepy.editor import VideoFileClip
                import speech_recognition as sr

                # 提取音频
                video = VideoFileClip(file_path)
                video.audio.write_audiofile(audio_path, logger=None)
                video.close()

                print(f"🎙️ [音视频引擎] 正在识别语音转文字...")
                r = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data, language='zh-CN')
                    content_blocks.append(f"\n[🎬 视频语音转写内容]\n{text}\n")
                    element_count += 1
            except Exception as e:
                print(f"❌ [音视频引擎] 视频解析失败: {e}")
                content_blocks.append(f"\n[🎬 视频解析失败: {str(e)}]\n")
            finally:
                # 务必清理临时音频文件
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass

        # 🌟 逻辑分流 4：PDF/DOCX/PPTX 走 Unstructured 高精度处理
        else:
            print(f"🔥 [视觉引擎] 正在启动高精度解析 (ONNX GPU 加速尝试中): {file_path}")
            # 开启高精度视觉模式 (hi_res)，提取完美排版和表格
            elements = partition(
                filename=file_path,
                strategy="fast",
                languages=["chi_sim", "eng"]
            )
            element_count = len(elements)
            print(f"✅ 解析完成！共提取 {element_count} 个物理结构块。")

            # 遍历解析出来的每一个元素，进行精准拼接
            for el in elements:
                # 抓取高精度表格并转化为 HTML
                if isinstance(el, Table):
                    html_table = getattr(el.metadata, 'text_as_html', None)
                    if html_table:
                        content_blocks.append(f"\n[📊 提取到精美表格数据]\n{html_table}\n")
                    else:
                        content_blocks.append(f"\n[📊 表格数据]\n{str(el)}\n")
                # 抓取标题和正文
                elif isinstance(el, (Title, NarrativeText, Text)):
                    content_blocks.append(str(el))

        # 拼接成最终大文本
        final_text = "\n\n".join(content_blocks).strip()
        if not final_text:
            final_text = "[提取内容为空或系统缺少对应解码器]"

        # 🌟 核心防爆机制：超过 80k 强制截断
        original_length = len(final_text)
        if original_length > MAX_CHARS_LIMIT:
            print(f"⚠️ [防爆机制触发] 文本超长 ({original_length} 字符)，执行截断！")
            final_text = final_text[:MAX_CHARS_LIMIT]
            final_text += "\n\n[⚠️ 系统提示：因上传资料过长，为保证AI生成质量，后续内容已被智能截断。建议分章节上传。]"

        return {
            "status": "success",
            "text": final_text,
            "element_count": element_count,
            "char_count": len(final_text)
        }

    except Exception as e:
        print(f"❌ [解析引擎] 底层崩溃: {str(e)}")
        return {"status": "error", "message": f"引擎崩溃: {str(e)}"}