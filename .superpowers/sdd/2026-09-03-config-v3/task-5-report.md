# Task 5 报告

## 状态

已完成 Config v3 文档修正和最终验证。更新了 `configs/README.md`、根目录 `README.md`，
并更新本报告；未修改代码或公开 GT 契约。

文档已覆盖：

- Config v3 完整示例和 `configs/local.machine.example.json` 示例；
- 真实 `configs/local.machine.json` 的忽略和不得入库约定；
- `--local-config` > `FUTSALMOT_LOCAL_CONFIG` > 缺少配置报错的固定优先级；
- 不自动搜索 task 同目录或其他目录的 local config；
- dataset/UE 路径及 `.uproject` 校验门禁；
- 失败不写半完整 resolved 输出，resolve 原子替换并保留旧文件；
- resolver 派生的旧运行时字段和完整 `task resolve` 摘要；
- v2 deprecated warning 与兼容行为；
- unchanged public GT contract，以及当前不提供 batch/split/assembler。
- Config v3 为 `configs/` 推荐工作流；`configs/example.json` 和 inline-path 说明明确标为 v2 legacy。

## Commit

前序提交：`86d4095`、`ba1b2c0`、`b19f20a`、`a6a2245`。
本次修正提交：待提交，提交信息为 `修正Config v3文档兼容说明`。

## 测试与检查

- `uv run pytest`：通过，`691 passed, 7 deselected`。
- `git diff --check`：通过；仅输出 Git 的 LF/CRLF 提示。
- `configs/local.machine.json`：不存在，未创建真实本机配置。
- tracked `configs/local.machine.json`：0；tracked runtime files：0；tracked secret-like files：0。
- v3 resolve：未运行。具体限制是 `configs/local.machine.json` 不存在，且不能用占位路径
  伪造真实本机路径；因此无法执行要求的 `task resolve ... --local-config ...` 命令。

## 关注事项

当前工作树在任务开始时已有未跟踪的 `.superpowers/sdd/` 资料和 `docs/superpowers/`
目录，本任务未修改或清理这些既有内容。

本次修正更新 `README.md`、`CLAUDE.md`、`configs/README.md`、
`docs/REPRODUCIBILITY_AND_MANIFEST.md` 和本报告；未修改代码或公开 GT 契约。

## 本轮验证原始输出

`uv run pytest`

```text
============================= test session starts =============================
platform win32 -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
collected 698 items / 7 deselected / 691 selected
======================= 691 passed, 7 deselected, 61 warnings in 11.91s =======================
```

`git diff --check`

```text
warning: in the working copy of '.superpowers/sdd/2026-09-03-config-v3/task-5-report.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'CLAUDE.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'configs/README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'docs/REPRODUCIBILITY_AND_MANIFEST.md', LF will be replaced by CRLF the next time Git touches it
```

`git diff --check` 无空白错误；以上仅为 Git 的换行符提示。
