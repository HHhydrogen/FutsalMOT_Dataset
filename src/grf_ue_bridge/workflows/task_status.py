"""task 工作流：只读状态显示（不修改任何文件）。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from grf_ue_bridge.config import models as m

RGB_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _count(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    try:
        return sum(1 for _ in path.rglob(pattern))
    except OSError:
        return 0


def _count_rgb(path: Path) -> int:
    if not path.is_dir():
        return 0
    try:
        return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in RGB_SUFFIXES)
    except OSError:
        return 0
def _lines(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def collect_status(resolved: m.ResolvedTask) -> Dict:
    """收集任务各产物的只读状态。"""
    from .artifact_cleanup import public_capabilities

    traj = Path(resolved.trajectory_output)
    ds = Path(resolved.dataset_episode_dir)
    capabilities = public_capabilities(resolved)

    cams = sorted(d.parent for d in ds.rglob("camera.json")) if ds.is_dir() else []

    st: Dict = {
        "task_id": resolved.task_id,
        "episode_name": resolved.episode_name,
        "trajectory_exists": (traj / "meta.json").is_file() and (traj / "frames.jsonl").is_file(),
        "camera_count": len(cams),
        "cameras": {},
        "render_summary": None,
        **capabilities,
        "cleanup_status": "pending",
    }
    manifest_path = ds / "dataset_manifest.json"
    if manifest_path.is_file():
        try:
            import json
            st["cleanup_status"] = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "cleanup_status", "pending"
            )
        except (OSError, ValueError):
            pass
    for cam in cams:
        st["cameras"][cam.name] = {
            "render_rgb": _count_rgb(cam / "render"),
            "object_id_exr": _count(cam / "render_mask", "*.exr"),
            "img1": _count_rgb(cam / "img1"),
            "mask": _count(cam / "mask", "*.png"),
            "annotations": _lines(cam / "annotations.jsonl"),
            "det": _count(cam / "labels" / "det", "*.txt"),
            "seg": _count(cam / "labels" / "seg", "*.txt"),
            "mot_lines": _lines(cam / "gt" / "gt.txt"),
        }
    summary_path = ds / "render_summary.json"
    if summary_path.is_file():
        try:
            import json
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            st["render_summary"] = {
                "status": summary.get("status"),
                "per_camera": {k: v.get("ok") for k, v in (summary.get("cameras") or {}).items()},
            }
        except Exception:  # noqa: BLE001
            st["render_summary"] = {"status": "unreadable"}
    return st


def print_status(resolved: m.ResolvedTask, st: Dict, print_fn: Callable[[str], None] = print) -> None:
    """打印人类可读状态。"""
    print_fn(f"Task: {resolved.task_id}  episode: {resolved.episode_name}")
    print_fn(f"  trajectory exists: {st['trajectory_exists']}  -> {resolved.trajectory_output}")
    print_fn(f"  dataset episode dir: {resolved.dataset_episode_dir}")
    print_fn(f"  cameras: {st['camera_count']}  render_summary: {st['render_summary']}")
    for cam, c in st["cameras"].items():
        print_fn(
            f"    {cam}: render={c['render_rgb']} exr={c['object_id_exr']} "
            f"img1={c['img1']} mask={c['mask']} ann={c['annotations']} "
            f"det={c['det']} seg={c['seg']} mot={c['mot_lines']}"
        )
