# 测试目录说明

`backend/tests/` 按职责分组，交付期优先保持“结构清晰 + 行为不变”。

## 分组规则

- `api/`：路由与接口行为测试（FastAPI/TestClient）。
- `services/`：服务层单元测试（检索、Mongo、教学大纲辅助等）。
- `pipeline/`：入库/图谱/优先级等流程级测试。
- `extractors/`：各层抽取器测试。
- `parsers/`：底层解析器测试。

## 根目录约定

- `conftest.py` 保留在 `tests/` 根目录，统一注入 `backend` 路径，确保子目录测试都能复用同一套导入上下文。

## 新增测试放置建议

- 新增接口测试：放到 `tests/api/`
- 新增服务测试：放到 `tests/services/`
- 新增流程联测：放到 `tests/pipeline/`
- 新增抽取/解析测试：放到 `tests/extractors/` 或 `tests/parsers/`

## 维护原则

- 优先做低风险目录整理，不随意改断言与测试语义。
- 需要移动测试文件时，仅做必要的路径修正（例如基于 `__file__` 的目录计算）。
