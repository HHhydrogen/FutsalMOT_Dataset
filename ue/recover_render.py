"""从已有 MRQ render/ 目录恢复 img1/（无需重新渲染）。

场景：上一次 MRQ 渲染已把 PNG 输出到各 camera 的 render/，但完成回调
（finished delegate / watchdog）未触发，导致 img1/ 为空。本脚本从现有
render/ 帧复制对齐帧到 img1/，并写 render_summary.json。

用法（P1，.venv python）：
    uv run python ue/recover_render.py

或在 UE 控制台：
    py "D:/path/to/code/ue/recover_render.py"

读取脚本上两级（仓库根）的 ue_import_config.json。纯 Python，不依赖 unreal，
无需编辑器即可完成恢复。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_episode import recover_render_to_img1  # noqa: E402

_CFG = Path(__file__).resolve().parent.parent / "ue_import_config.json"


def main() -> int:
    if not _CFG.exists():
        print(f"ERROR: 未找到配置文件: {_CFG}")
        return 1
    with open(_CFG, encoding="utf-8") as f:
        raw = json.load(f)
    cfg = {k: v for k, v in raw.items() if not k.startswith("comment_")}
    episode = Path(cfg["episode"])
    ann = cfg.get("annotation_export") or {}
    output_dir = Path(ann.get("output_dir") or (episode.parent / "dataset"))
    seqs = cfg.get("sequences") or []
    status, _per_cam = recover_render_to_img1(seqs, ann, episode, output_dir)
    print(f"恢复状态: {status}")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
