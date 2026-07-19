#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: run Unreal preflight and build/export annotations.

Run this file from the Unreal Editor Python console.
"""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from futsalmot.scripts import ue_preflight  # noqa: E402


def main() -> None:
    ue_preflight.main()

    build_path = CODE_DIR / "futsalmot" / "scripts" / "ue_build_sequences.py"
    source = build_path.read_text(encoding="utf-8")
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(build_path),
    }
    try:
        exec(source, globals_dict)
    except SystemExit:
        pass
    except Exception as exc:
        try:
            import unreal
            unreal.log_error("[02_run_unreal] ue_build_sequences 失败: {}: {}".format(type(exc).__name__, exc))
        except Exception:
            print("[02_run_unreal] ue_build_sequences 失败: {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
