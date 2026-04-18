from pathlib import Path
import sys


# 将 backend 目录加入导入路径，保证测试可直接导入业务模块。
TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
