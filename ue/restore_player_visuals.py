"""在 Unreal Editor 中按 rollback JSON 恢复角色 Mesh 和 AnimBP。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from player_visuals import restore_player_visuals  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="恢复 Player_N 视觉 Profile")
    parser.add_argument("--rollback", required=True)
    args = parser.parse_args()
    result = restore_player_visuals(args.rollback)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
