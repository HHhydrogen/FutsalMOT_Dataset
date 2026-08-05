#!/usr/bin/env python
"""【已废弃】兼容包装：请改用 `uv run grf-ue benchmark ...`。

本文件不含业务逻辑，仅转发到正式包 grf_ue_bridge.tools.benchmark_postprocess。
"""
import sys
import warnings

from grf_ue_bridge.tools.benchmark_postprocess import main

warnings.warn(
    "scripts/benchmark_postprocess.py 已废弃，请改用 `uv run grf-ue benchmark ...`",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.exit(main())
