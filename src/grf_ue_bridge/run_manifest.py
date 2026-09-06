"""一次 task pipeline run 的来源、结果和 artifact 轻量摘要。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from grf_ue_bridge.validation_result import validation_result_from_report


SCHEMA = "futsalmot_run_manifest"
CURRENT_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_manifest_path(repo_root: Path, task_id: str) -> Path:
    return Path(repo_root) / ".futsalmot" / "runtime" / task_id / "run_manifest.json"


def task_spec_hash(task_file: Path) -> str:
    digest = hashlib.sha256()
    with Path(task_file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _ue_commit(repo_root: Path) -> Optional[str]:
    outer_root = Path(repo_root).resolve()
    for parent in (outer_root, *outer_root.parents):
        if (parent / ".git").exists() and (parent / "Content" / "FutsalMOT" / "code").resolve() == outer_root:
            return git_commit(parent)
    return None


def create_run_manifest(
    task_id: str,
    task_file: Path,
    repo_root: Path,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": CURRENT_VERSION,
        "task_id": task_id,
        "source": {
            "task_hash": task_spec_hash(task_file),
            "code_commit": git_commit(repo_root),
            "ue_commit": _ue_commit(repo_root),
        },
        "runtime": {"started_at": started_at or _now(), "finished_at": None},
        "pipeline": {"state": None},
        "validation": {"passed": None, "audit_report": None},
        "artifacts": {},
    }


def _validate(manifest: Any) -> Dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("invalid run manifest schema")
    version = manifest.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("invalid run manifest version")
    if version > CURRENT_VERSION:
        raise ValueError(f"Unsupported run manifest version: {version}")
    if version != CURRENT_VERSION:
        raise ValueError(f"unsupported run manifest version: {version}")
    if not isinstance(manifest.get("task_id"), str) or not manifest["task_id"]:
        raise ValueError("invalid run manifest task_id")
    for key in ("source", "runtime", "pipeline", "validation", "artifacts"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"invalid run manifest {key}")
    return manifest


def load_run_manifest(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read run manifest: {path}: {exc}") from exc
    return _validate(data)


def save_run_manifest(manifest: Dict[str, Any], path: Path) -> None:
    _validate(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".run_manifest.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def update_validation(manifest: Dict[str, Any], audit_report_path: Path) -> None:
    try:
        report = json.loads(Path(audit_report_path).read_text(encoding="utf-8"))
        result = validation_result_from_report(report)
        manifest["validation"] = {
            "passed": result.passed,
            "audit_report": str(Path(audit_report_path)),
        }
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        manifest["validation"] = {
            "passed": None,
            "audit_report": str(Path(audit_report_path)),
            "error": str(exc),
        }


def update_artifacts(manifest: Dict[str, Any], dataset_episode_dir: Path) -> None:
    """尽力统计常用产物，统计失败只进入 artifacts.errors。"""
    artifacts: Dict[str, Any] = {}
    errors = []
    try:
        root = Path(dataset_episode_dir)
        if not root.is_dir():
            raise OSError(f"directory does not exist: {root}")
        cameras = sorted(path.parent for path in root.rglob("camera.json"))
        artifacts["cameras"] = {"count": len(cameras)}
        artifacts["images"] = {"count": sum(len(list((cam / "img1").glob("*.png"))) for cam in cameras)}
        artifacts["annotations"] = {"count": sum(1 for cam in cameras for _ in (cam / "annotations.jsonl").open(encoding="utf-8")) if cameras else 0}
        artifacts["mask"] = {"count": sum(len(list((cam / "mask").glob("*.png"))) for cam in cameras)}
        artifacts["mot"] = {"count": sum(1 for cam in cameras if (cam / "gt" / "gt.txt").is_file())}
        artifacts["pose"] = {"count": sum(1 for _ in (root / "coco17_3d.jsonl").open(encoding="utf-8")) if (root / "coco17_3d.jsonl").is_file() else 0}
    except (OSError, UnicodeError) as exc:
        errors.append(str(exc))
    if errors:
        artifacts["errors"] = errors
    manifest["artifacts"] = artifacts
