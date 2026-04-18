"""
兼容性 shim：在无法下载官方 spaCy 模型包时，提供 en_core_web_sm.load()。

说明：
- unstructured 的部分文本流程会调用 `en_core_web_sm.load()`
- 当前环境网络受限，无法从 GitHub 下载官方模型 wheel
- 这里返回一个轻量 `spacy.blank("en")` 管线，确保解析流程可运行
"""

from __future__ import annotations

import spacy

__version__ = "0.0.0-local-shim"


def load(**kwargs):
    """返回可用的英文基础 NLP 管线。"""
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp
