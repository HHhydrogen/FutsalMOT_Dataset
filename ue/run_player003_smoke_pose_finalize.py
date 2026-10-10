"""通过 UE MCP 文件入口完成 Player_003 smoke Runtime Pose finalize。"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ["C5_RESOLVED_TASK"] = str(
    ROOT.parent / ".futsalmot" / "runtime" / "player003_smoke_10frames_1cam" / "resolved-task.json"
)
os.environ["C5_RUN_MODE"] = "pose-finalize"
try:
    runpy.run_path(str(ROOT / "run_task.py"), run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
