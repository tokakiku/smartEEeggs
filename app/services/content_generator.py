# ==========================================
# 灵犀智课 - 多线程 AI 内容加工厂 (带互动游戏组装版)
# 特性：纯异步并发 | 基于大纲提炼正文 | 图文游三位一体
# ==========================================

import os
import requests
import logging
import concurrent.futures
from zhipuai import ZhipuAI
from dotenv import load_dotenv

from app.services.game_generator import build_html_game # 🌟 新增：引入游戏组装车间

load_dotenv()
logger = logging.getLogger("Content_Generator")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
IMAGE_CACHE_DIR = "downloads/images"
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)


class ContentGenerator:
    def __init__(self):
        if not ZHIPU_API_KEY:
            logger.warning("🚨 未配置 ZHIPU_API_KEY，大模型生成将失败！")
        self.client = ZhipuAI(api_key=ZHIPU_API_KEY)
        self.global_style = "3D极简学术插画，教育科技风格，质感高级，背景干净纯白，画面中绝对不能出现任何文字、数字或字母拼写。"

    def _generate_text(self, page_task: dict) -> dict:
        title = page_task.get("title", "未命名页面")
        original_content = page_task.get("content", "")

        logger.info(f"✍️ 正在提炼排版文字: {title}")

        prompt = f"""你是一个顶级的 PPT 文案提炼专家。
        当前页面标题：【{title}】
        原始大纲描述：【{original_content}】

        任务：
        请根据上述原始描述，为该页 PPT 提炼出适合直接展示的正文。
        要求：
        1. 提炼核心信息，绝不能偏离原始描述的意思。
        2. 必须分点列出（如 1. 2. 3.），逻辑清晰，每点一句话。
        3. 总字数严格控制在 150 字以内，字字珠玑。
        4. 绝对不要返回除了正文内容以外的任何客套话、Markdown 格式词或解释。"""

        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            page_task["content"] = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ 文字提炼失败 ({title}): {e}")
            page_task["content"] = original_content

        return page_task

    def _generate_image(self, page_task: dict) -> dict:
        title = page_task.get("title", "未命名页面")
        context_snippet = page_task.get("content", "")[:50]

        if page_task.get("need_image", True) is False:
            return page_task

        logger.info(f"🎨 正在生成配图: {title}")
        prompt = f"核心主题：{title}。内容参考：{context_snippet}。画风要求：{self.global_style}"

        try:
            response = self.client.images.generations(
                model="cogview-3-plus",
                prompt=prompt
            )
            img_url = response.data[0].url
            img_data = requests.get(img_url, timeout=15).content

            img_filename = f"img_{hash(title)}_{os.urandom(4).hex()}.png"
            img_path = os.path.join(IMAGE_CACHE_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_data)

            page_task["image_path"] = img_path
        except Exception as e:
            logger.error(f"⚠️ 图片生成降级 ({title}): {e}")
            page_task["image_path"] = ""

        return page_task

    def batch_generate(self, raw_queue: list) -> list:
        # 1. 并发生成文字 (线程数=5)
        logger.info("🚀 [加工厂] 启动文字提炼线程池...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            text_completed_queue = list(executor.map(self._generate_text, raw_queue))

        # 2. 并发生成图片 (线程数=3，防限流)
        logger.info("🚀 [加工厂] 启动图片绘制线程池...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            final_queue = list(executor.map(self._generate_image, text_completed_queue))

        # 🌟 3. 新增：组装互动游戏！
        logger.info("🚀 [加工厂] 检查并组装互动游戏...")
        for page in final_queue:
            if page.get("interactive_game"):
                try:
                    logger.info(f"🎮 发现互动游戏配置，正在送往车间组装...")
                    # 调用刚才写的车间，默认传入项目ID 1024 方便演示
                    game_link = build_html_game(page["interactive_game"], 1024)
                    page["game_url"] = game_link
                    logger.info(f"✅ 游戏装载成功：{game_link}")
                except Exception as e:
                    logger.error(f"❌ 游戏组装失败: {e}")
                    page["game_url"] = None

        logger.info("✅ [加工厂] 批量生成车间任务完成！")
        return final_queue


# 对外工厂函数
def generate_page_contents(raw_queue: list) -> list:
    engine = ContentGenerator()
    return engine.batch_generate(raw_queue)