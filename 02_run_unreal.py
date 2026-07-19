#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: run Unreal preflight and build/export annotations.

Run this file from the Unreal Editor Python console.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from futsalmot.scripts import ue_preflight  # noqa: E402


def main() -> None:
    ue_preflight.main()
    importlib.import_module("futsalmot.scripts.ue_build_sequences")


if __name__ == "__main__":
    main()
