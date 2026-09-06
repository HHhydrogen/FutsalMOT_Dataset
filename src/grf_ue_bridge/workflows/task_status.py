"""task 工作流：只读状态显示（不修改任何文件）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

from grf_ue_bridge.config import models as m
from grf_ue_bridge.pipeline_state import FIXED_STEPS, load_state, state_path, create_state
from grf_ue_bridge.validation_result import validation_result_from_report


def _count(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    try:
        return sum(1 for _ in path.rglob(pattern))
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
    traj = Path(resolved.trajectory_output)
    ds = Path(resolved.dataset_episode_dir)

    cams = sorted(d.parent for d in ds.rglob("camera.json")) if ds.is_dir() else []

    st: Dict = {
        "task_id": resolved.task_id,
        "episode_name": resolved.episode_name,
        "trajectory_exists": (traj / "meta.json").is_file() and (traj / "frames.jsonl").is_file(),
        "camera_count": len(cams),
        "cameras": {},
        "render_summary": None,
    }
    pipeline_path = state_path(Path(resolved.repo_root), resolved.task_id)
    if pipeline_path.is_file():
        try:
            pipeline = load_state(pipeline_path).to_dict()
        except ValueError as exc:
            pipeline = {"state": "FAILED", "steps": {}, "error": str(exc)}
    else:
        pipeline = create_state(resolved.task_id).to_dict()
        pipeline["state_available"] = False
    st["pipeline"] = pipeline
    audit_path = ds / "audit" / "soak_audit_report.json"
    if audit_path.is_file():
        try:
            report = json.loads(audit_path.read_text(encoding="utf-8"))
            validation = validation_result_from_report(report)
            st["validation"] = {"passed": validation.passed, "exit_code": validation.exit_code}
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            st["validation"] = {"passed": False, "error": str(exc)}
    else:
        st["validation"] = None
    for cam in cams:
        st["cameras"][cam.name] = {
            "render_rgb": _count(cam / "render", "*.png"),
            "object_id_exr": _count(cam / "render_mask", "*.exr"),
            "img1": _count(cam / "img1", "*.png"),
            "mask": _count(cam / "mask", "*.png"),
            "annotations": _lines(cam / "annotations.jsonl"),
            "det": _count(cam / "labels" / "det", "*.txt"),
            "seg": _count(cam / "labels" / "seg", "*.txt"),
            "mot_lines": _lines(cam / "gt" / "gt.txt"),
        }
    summary_path = ds / "render_summary.json"
    if summary_path.is_file():
        try:
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
    pipeline = st.get("pipeline", {})
    print_fn(f"Pipeline State: {pipeline.get('state', 'UNKNOWN')}")
    print_fn(f"  Schema version: {pipeline.get('version', 'UNKNOWN')}")
    print_fn(f"  Last update: {pipeline.get('updated_at', 'UNKNOWN')}")
    current = next((step for step in FIXED_STEPS
                    if pipeline.get("steps", {}).get(step, {}).get("status") == "RUNNING"), None)
    failed = next((step for step in FIXED_STEPS
                   if pipeline.get("steps", {}).get(step, {}).get("status") == "FAILED"), None)
    print_fn(f"  Current step: {current or 'none'}")
    print_fn(f"  Failed step: {failed or 'none'}")
    if failed:
        print_fn("  Resume suggestion: retry " + failed)
    elif current:
        print_fn("  Resume suggestion: inspect " + current + " before retry")
    else:
        pending = next((step for step in FIXED_STEPS
                        if pipeline.get("steps", {}).get(step, {}).get("status") == "PENDING"), None)
        if pending:
            print_fn("  Resume suggestion: run " + pending)
    for step in FIXED_STEPS:
        detail = pipeline.get("steps", {}).get(step, {"status": "PENDING"})
        status = detail.get("status", "PENDING")
        marker = {"COMPLETED": "✓", "RUNNING": "▶", "FAILED": "✗", "SKIPPED": "-"}.get(status, "○")
        print_fn(f"  {marker} {step}: {status}")
        if status == "FAILED":
            print_fn(f"    error: {detail.get('error_type', '')}: {detail.get('error_message', '')}")
    validation = st.get("validation")
    if validation is not None:
        print_fn(f"ValidationResult: {'PASS' if validation.get('passed') else 'FAIL'}")
        if validation.get("passed") is False:
            print_fn("Task success: NO")
            print_fn("  action: inspect audit errors before treating the task as usable")
    run_manifest = st.get("run_manifest")
    if run_manifest:
        print_fn("Run Manifest: " + str(run_manifest.get("path", "available")))
        source = run_manifest.get("source") or {}
        print_fn(f"  code commit: {source.get('code_commit')}")
        print_fn(f"  ue commit: {source.get('ue_commit')}")
        manifest_validation = run_manifest.get("validation") or {}
        if manifest_validation.get("passed") is not None:
            print_fn(f"  manifest validation: {'PASS' if manifest_validation['passed'] else 'FAIL'}")
    for cam, c in st["cameras"].items():
        print_fn(
            f"    {cam}: render={c['render_rgb']} exr={c['object_id_exr']} "
            f"img1={c['img1']} mask={c['mask']} ann={c['annotations']} "
            f"det={c['det']} seg={c['seg']} mot={c['mot_lines']}"
        )
