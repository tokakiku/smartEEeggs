# ==========================================
# 灵犀智课 - 互动游戏组装车间
# 文件路径: app/services/game_generator.py
# ==========================================

import os
import json
import logging

logger = logging.getLogger("Game_Engine")


def build_html_game(game_json: dict, project_id: int) -> str:
    """
    核心逻辑：将 AI 生成的 JSON 数据，强行注入到本地的 HTML 模板中
    """
    game_type = game_json.get("game_type", "memory_match")

    logger.info(f"🕹️ [游戏车间] 正在为项目 {project_id} 组装 {game_type} 游戏...")

    # 1. 定位模板文件 (由于你放在了项目根目录，也就是和 app 文件夹平级，所以直接写文件名)
    template_path = f"template_{game_type}.html"

    if not os.path.exists(template_path):
        logger.error(f"❌ [游戏车间] 找不到游戏模板: {template_path}，请确保它和 README 放在同一级目录！")
        return ""

    # 2. 读取 HTML 空壳
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 3. 将大模型给的 Python 字典，变成前端 JS 能认的纯 JSON 字符串
    game_data_str = json.dumps(game_json, ensure_ascii=False)

    # 4. 灵魂附体：把占位符 /*INJECT_DATA*/ 替换成真正的游戏数据
    if "/*INJECT_DATA*/" not in html_content:
        logger.warning("⚠️ [游戏车间] 模板里没有找到 /*INJECT_DATA*/ 占位符，数据可能无法注入！")

    final_html = html_content.replace('/*INJECT_DATA*/', game_data_str)

    # 5. 保存为最终可以玩的网页文件 (输出到统一的 exports 目录)
    output_dir = "downloads/exports"
    os.makedirs(output_dir, exist_ok=True)

    # 文件名加上随机特征，防止被覆盖
    output_filename = f"interactive_game_{project_id}_{game_type}.html"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    logger.info(f"✅ [游戏车间] 互动游戏网页已生成: {output_path}")

    # 返回相对路径，方便前端下载或展示
    return f"/static/exports/{output_filename}"


# 测试桩 (你可以直接在本地右键运行这个文件测试)
if __name__ == "__main__":
    test_json = {
        "game_type": "memory_match",
        "pairs": [
            {"left": "BOPPPS", "right": "闭环教学法"},
            {"left": "MinerU", "right": "多模态文档提取工具"},
            {"left": "CogView", "right": "文生图模型"},
            {"left": "FastAPI", "right": "高性能Python后端框架"}
        ]
    }
    # 假设你在根目录运行
    os.chdir("../../")  # 回退到根目录找 html
    result = build_html_game(test_json, 9999)
    print(f"生成的游戏地址: {result}")