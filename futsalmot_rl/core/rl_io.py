"""Atomic I/O utilities for FutsalMOT-RL.

Reuses patterns from futsalmot/core/io.py — adds npz support.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: str | Path) -> Any:
    """Read a JSON file (utf-8 with optional BOM)."""
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json_atomic(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """Atomically write a JSON file (tmp + replace)."""
    text = json.dumps(data, ensure_ascii=False, indent=indent, allow_nan=False) + "\n"
    write_text_atomic(path, text)


def write_text_atomic(path: str | Path, text: str) -> None:
    """Atomically write a text file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.{}".format(os.getpid()))
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(target))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def save_npz(path: str | Path, **kwargs: Any) -> None:
    """Save a .npz file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert scalars/lists to arrays for consistent saving
    arrays = {}
    for key, value in kwargs.items():
        if isinstance(value, (int, float, str)):
            arrays[key] = np.array(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
            # String arrays need special handling
            arrays[key] = np.array(value, dtype=object)
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(str(path), **arrays)


def load_npz(path: str | Path) -> dict[str, Any]:
    """Load a .npz file, returning a dict of arrays."""
    data = np.load(str(path), allow_pickle=True)
    return dict(data)


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
