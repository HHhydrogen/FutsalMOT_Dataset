"""从 Player_003 smoke 已完成的 MRQ 原始帧恢复 canonical 输出。"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import render_episode  # noqa: E402

importlib.reload(render_episode)

resolved = json.loads(
    (ROOT.parent / ".futsalmot" / "runtime" / "player003_smoke_10frames_1cam" / "resolved-task.json")
    .read_text(encoding="utf-8")
)
ue_profile = resolved["ue_profile"]
annotation_cfg = dict(ue_profile.get("annotation_export") or {})
annotation_cfg["playback_fps"] = int(resolved["export_profile"]["playback_fps"])
status, details = render_episode.recover_render_to_img1(
    ue_profile.get("sequences") or [],
    annotation_cfg,
    Path(resolved["trajectory_output"]),
    Path(resolved["dataset_root"]),
)
print(json.dumps({"status": status, "details": details}, ensure_ascii=False, indent=2))
if status != "success":
    raise SystemExit(1)
