from pathlib import Path


def save_binary_file(path: Path, file_bytes: bytes) -> str:
    # 保存二进制文件到本地，自动创建目录。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(file_bytes)
    return str(path.resolve())
