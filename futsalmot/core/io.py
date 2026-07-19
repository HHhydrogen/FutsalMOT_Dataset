from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_text_atomic(path: str | Path, text: str) -> None:
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


def write_json_atomic(path: str | Path, data: Any, *, indent: int = 2) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=indent, allow_nan=False) + "\n"
    write_text_atomic(path, text)
