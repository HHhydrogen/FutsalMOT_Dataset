#!/usr/bin/env python
"""运行一条命令并测量墙钟时间 + 进程树峰值 RSS（从 scripts/measure_run.py 迁移）。

CLI：`grf-ue measure -- <cmd...>`（推荐）；或 `python -m grf_ue_bridge.tools.process_measure <cmd...>`。
对目标命令的整棵进程树周期采样 RSS（root + recursive children），
报告 wall time / peak tree RSS / 退出码。用于正式后处理记录与性能基准。
"""

from __future__ import annotations

import subprocess
import sys
import time


def _tree_rss(root) -> int:
    try:
        rss = root.memory_info().rss
        children = root.children(recursive=True)
    except Exception:  # noqa: BLE001
        return 0
    total = rss
    for c in children:
        try:
            total += c.memory_info().rss
        except Exception:  # noqa: BLE001
            continue
    return total


def main(argv: object = None) -> int:
    cmd = argv if argv is not None else sys.argv[1:]
    if not cmd:
        print(__doc__)
        return 2
    import psutil

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd)
    peak = 0
    try:
        root = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak = max(peak, _tree_rss(root))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.2)
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        print("\nMEASURE: interrupted")
        return 130
    # 收尾采样（子进程可能在 poll 后短暂存活）
    for _ in range(3):
        try:
            peak = max(peak, _tree_rss(psutil.Process(proc.pid)))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.2)
    dt = time.monotonic() - t0
    print(
        f"\nMEASURE: wall={dt:.1f}s  peak_tree_rss={peak / 1024 / 1024:.0f}MB  rc={proc.returncode}",
        file=sys.stderr,
    )
    return proc.returncode if proc.returncode is not None else 1


if __name__ == "__main__":
    sys.exit(main())
