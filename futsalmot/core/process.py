from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class StepResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float


def format_command(cmd: Sequence[object]) -> str:
    parts = []
    for value in cmd:
        text = str(value)
        if any(ch.isspace() for ch in text) or '"' in text:
            text = '"{}"'.format(text.replace('"', '\\"'))
        parts.append(text)
    return " ".join(parts)


def tail_lines(text: str, count: int = 20) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()][-count:]


def run_logged_step(
    label: str,
    cmd: Sequence[object],
    *,
    timeout: int,
    log_dir: Path,
) -> StepResult:
    print("\n[RUN] {}".format(label))
    print("  {}".format(format_command(cmd)))
    start = time.time()
    try:
        completed = subprocess.run(
            [str(item) for item in cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        rc = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
    except OSError as exc:
        rc = 127
        stdout = ""
        stderr = str(exc)

    elapsed = time.time() - start
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    (log_dir / "{}_stdout.log".format(safe)).write_text(stdout, encoding="utf-8")
    (log_dir / "{}_stderr.log".format(safe)).write_text(stderr, encoding="utf-8")

    if rc == 0:
        print("  OK ({:.1f}s)".format(elapsed))
    else:
        print("  FAILED (rc={}, {:.1f}s)".format(rc, elapsed))
        if stdout.strip():
            print("  --- stdout tail ---")
            for line in tail_lines(stdout):
                print("    {}".format(line))
        if stderr.strip():
            print("  --- stderr tail ---", file=sys.stderr)
            for line in tail_lines(stderr):
                print("    {}".format(line), file=sys.stderr)

    return StepResult(rc, stdout, stderr, elapsed)
