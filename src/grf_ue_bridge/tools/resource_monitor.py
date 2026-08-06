#!/usr/bin/env python
"""渲染期间的资源/目录增长监控。

每 interval 秒采样一次并追加 CSV：
  - 时间戳 / 运行秒数
  - 输出目录总大小、render/ PNG 数、render_mask/ EXR 数、img1/ PNG 数
  - 数据集盘剩余空间
  - UnrealEditor 进程树 RSS（如存在）
  - 系统总内存 / 可用内存
  - CPU 占用（psutil 聚合）
  - GPU 显存/占用（nvidia-smi，如可用）

CLI：`grf-ue monitor <task>`（推荐）；或 `python -m grf_ue_bridge.tools.resource_monitor --input ...`。
Ctrl-C 结束；CSV 头已写。
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _dir_size_gb(path: Path) -> float:
    try:
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return total / (1024 ** 3)
    except OSError:
        return float("nan")


def _count(path: Path, pattern: str) -> int:
    try:
        return len(list(path.rglob(pattern)))
    except OSError:
        return 0


def _ue_processes():
    """返回 UnrealEditor 相关进程列表（含子进程）。"""
    out = []
    if psutil is None:
        return out
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if "Unreal" in proc.info["name"] or "UnrealEditor" in proc.info["name"]:
                out.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def _ue_tree_rss_mb():
    if psutil is None:
        return float("nan")
    total = 0
    for proc in _ue_processes():
        try:
            total += sum(
                p.memory_info().rss for p in [proc, *proc.children(recursive=True)]
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 ** 2)


def _gpu_info() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else "n/a"
    except Exception:  # noqa: BLE001
        return "n/a"


def main(argv: object = None) -> int:
    ap = argparse.ArgumentParser(description="渲染资源监控")
    ap.add_argument("--input", required=True, help="数据集输出根目录")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--out", default="soak_resources.csv")
    args = ap.parse_args(argv)

    root = Path(args.input)
    out_path = Path(args.out)
    header = [
        "timestamp", "elapsed_s", "dir_size_gb", "render_png", "render_mask_exr",
        "img1_png", "disk_free_gb", "ue_tree_rss_mb",
        "sys_total_gb", "sys_avail_gb", "cpu_percent", "gpu",
    ]
    new = not out_path.exists()
    t0 = time.monotonic()
    print(f"监控开始: {root}  每 {args.interval}s 采样 → {out_path}（Ctrl-C 结束）")
    try:
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(header)
            while True:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row = [
                    now,
                    round(time.monotonic() - t0, 1),
                    round(_dir_size_gb(root), 3),
                    _count(root / "CineCam_01" / "render", "*.png")
                    if (root / "CineCam_01" / "render").exists() else 0,
                    _count(root, "*.exr"),
                    _count(root, "*.png"),
                ]
                try:
                    sh = psutil.disk_usage(str(root)) if psutil else None
                    row.append(round(sh.free / (1024 ** 3), 1) if sh else float("nan"))
                except OSError:
                    row.append(float("nan"))
                row.append(round(_ue_tree_rss_mb(), 1))
                if psutil:
                    vm = psutil.virtual_memory()
                    row.extend([
                        round(vm.total / (1024 ** 3), 1),
                        round(vm.available / (1024 ** 3), 1),
                        round(psutil.cpu_percent(interval=0.5), 1),
                    ])
                else:
                    row.extend([float("nan"), float("nan"), float("nan")])
                row.append(_gpu_info())
                writer.writerow(row)
                f.flush()
                print("  " + ", ".join(map(str, row[:10])) + ", ...", flush=True)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n监控结束，写入 {out_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
