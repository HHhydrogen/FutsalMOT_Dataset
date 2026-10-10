"""在 Unreal Editor 中应用 resolved task 的角色视觉 Profile。"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import player_visuals as _player_visuals  # noqa: E402

_player_visuals = importlib.reload(_player_visuals)
apply_player_visuals = _player_visuals.apply_player_visuals
json_default = _player_visuals.json_default
normalize_player_visuals = _player_visuals.normalize_player_visuals


def serialize_result(payload: object) -> str:
    """将 UE 返回的 Text、Name 和枚举值稳定转换为 JSON。"""
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=json_default,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="应用 Player_N 视觉 Profile")
    parser.add_argument("--resolved-task", default=os.environ.get("C5_RESOLVED_TASK"))
    parser.add_argument("--rollback", required=False)
    args = parser.parse_args()
    if not args.resolved_task:
        print(serialize_result({"ok": False, "errors": ["缺少 resolved task"], "warnings": []}))
        return 2

    resolved_path = Path(args.resolved_task)
    if not resolved_path.is_file():
        print(serialize_result({"ok": False, "errors": [f"resolved task 不存在: {resolved_path}"], "warnings": []}))
        return 2
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    ue_profile = resolved.get("ue_profile") or {}
    normalized = normalize_player_visuals(ue_profile.get("player_visuals"))
    if normalized is None:
        print(serialize_result({"ok": True, "operation": "apply", "skipped": True, "errors": [], "warnings": []}))
        return 0

    mapping_path = Path(resolved["actor_mapping"])
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rollback = args.rollback or str(
        Path(resolved.get("repo_root", resolved_path.parent))
        / ".futsalmot"
        / "runtime"
        / resolved["task_id"]
        / "player-visuals-rollback.json"
    )
    result = apply_player_visuals(normalized, mapping, rollback_path=rollback)
    print(serialize_result(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
