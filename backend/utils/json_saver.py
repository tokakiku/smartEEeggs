import json
from pathlib import Path
from typing import Any


def save_json_file(path: Path, payload: Any) -> str:
    # 保存 JSON 文件，使用 UTF-8 与中文可读格式。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return str(path.resolve())
