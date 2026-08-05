#!/usr/bin/env python
"""【已废弃】兼容包装：请改用 `uv run grf-ue task audit <task>`。

本文件不含业务逻辑，仅转发到正式包 grf_ue_bridge.workflows.task_audit。
"""
import sys
import warnings

from grf_ue_bridge.workflows.task_audit import main

warnings.warn(
    "scripts/audit_soak_episode.py 已废弃，请改用 `uv run grf-ue task audit <task>`",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.exit(main())
