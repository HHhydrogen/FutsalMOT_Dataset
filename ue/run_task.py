"""UE 统一入口：读取 resolved task，调用既有导入/渲染逻辑。

C5.3 正式数据流（full 模式）：
    create_sequence → export_annotations → Runtime Pose prep → render_episode（RGB+BurnIn）
    （渲染异步完成后，再单独运行 pose_capture_export → build_coco17 → P1 postprocess/audit）

不再在 full 模式调用旧 pose_export.py（Editor 逐帧采样），避免新旧两套 Pose 混入正式
episode（旧 pose_export.py 保留为 Legacy，可经 --mode annotations 显式调用）。

用法（Unreal Editor Python Console）：
    py "D:/path/to/code/ue/run_task.py" --resolved-task "D:/.../resolved-task.json" --mode full
    py "D:/path/to/code/ue/run_task.py" --resolved-task "D:/.../resolved-task.json" --mode pose-finalize

也支持环境变量 C5_RESOLVED_TASK（MCP 调用友好）：
    export C5_RESOLVED_TASK=D:/.../resolved-task.json
    py ".../run_task.py" --mode full

纯 UE Python + 标准库，不导入 gfootball/.venv 任何模块。
"""

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESOLVED_TASK_SCHEMA = "futsalmot_resolved_task"

# UE Python 会话内，已 import 过的 ue/ 模块会缓存在 sys.modules 中；多次执行本脚本
# 时先强制重载，否则会运行到磁盘上已修改但会话里仍是旧版本的代码。
_UE_MODULE_NAMES = (
    "camera_projection", "annotation_utils", "dataset_export",
    "scene_apply", "annotation_exporter", "render_preset", "render_episode",
    "import_grf_episode",
    "pose_bones", "pose_export", "pose_render", "pose_capture_export",
    "build_coco17",
    "player_motion", "camera_distribution", "player_visuals",
    "episode_appearance",
    "episode_environment",
    "diagnose_episode_environment",
)
for _name in _UE_MODULE_NAMES:
    if _name in sys.modules:
        if (
            _name in {"render_episode", "episode_appearance"}
            and getattr(sys.modules.get("render_episode"), "_ACTIVE_RENDER", None) is not None
        ):
            continue
        importlib.reload(sys.modules[_name])


def _fail(msg: str, code: int = 2) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return code


def _fail_with_episode_transaction(transaction, msg: str, reason: str) -> int:
    """在已应用的 episode transaction 上记录并执行同步失败恢复。"""
    if transaction is not None:
        try:
            transaction.record_downstream_failure(RuntimeError(msg))
        except Exception as report_exc:
            print(
                "ERROR: 无法记录 episode downstream 失败: {}".format(report_exc),
                file=sys.stderr,
            )
        if (
            getattr(transaction, "restore_result", None) is None
            and not getattr(transaction, "callback_ownership_transferred", False)
        ):
            try:
                restored = transaction.restore(reason)
            except Exception as restore_exc:
                return _fail(
                    "{}；episode transaction restore 异常: {}".format(
                        msg, restore_exc
                    )
                )
            if not restored.get("ok"):
                return _fail(
                    "{}；episode transaction restore 失败: {}".format(
                        msg,
                        "; ".join(restored.get("errors") or ["未知错误"]),
                    )
                )
    return _fail(msg)


POSE_KEYPOINTS_SCHEMA = "grf_ue_pose_keypoints"


def _apply_resolved_anchor_cameras(camera_entries, annotation_cfg):
    """应用并回读 resolved 的静态相机状态。"""
    import math
    import unreal
    from camera_projection import focal_length_to_fov_deg
    from scene_apply import find_actor

    render_cfg = annotation_cfg.get("render_rgb") or {}
    resolution = (
        int(render_cfg.get("output_resolution_x")),
        int(render_cfg.get("output_resolution_y")),
    ) if render_cfg.get("output_resolution_x") is not None and render_cfg.get("output_resolution_y") is not None else None
    for entry in camera_entries:
        camera_id = entry["camera_id"]
        profile = entry["profile"]
        actor = find_actor(entry["camera_actor"])
        if actor is None:
            raise RuntimeError(f"Canonical ID {camera_id} 的 UE Camera Actor 不存在: {entry['camera_actor']}")
        components = actor.get_components_by_class(unreal.CineCameraComponent)
        if len(components) != 1:
            raise RuntimeError(
                f"Canonical ID {camera_id} 的 Camera Component 数量必须为 1，实际为 {len(components)}"
            )
        component = components[0]
        location = profile["position_m"]
        rotation = profile["rotation_deg"]
        actor.set_actor_location_and_rotation(
            unreal.Vector(float(location[0]) * 100.0, float(location[1]) * 100.0, float(location[2]) * 100.0),
            unreal.Rotator(
                pitch=float(rotation[0]),
                yaw=float(rotation[1]),
                roll=float(rotation[2]),
            ),
            False,
            False,
        )
        lens = profile.get("lens") or {}
        if lens.get("focal_length_mm") is not None:
            component.set_editor_property("current_focal_length", float(lens["focal_length_mm"]))
        actual_location = actor.get_actor_location()
        actual_rotation = actor.get_actor_rotation()
        actual_focal = float(component.get_editor_property("current_focal_length"))
        actual_fov = float(component.get_editor_property("current_horizontal_fov"))
        filmback = component.get_editor_property("filmback")
        sensor_width = float(filmback.sensor_width)
        expected_focal = lens.get("focal_length_mm")
        expected_fov = focal_length_to_fov_deg(actual_focal, sensor_width)
        if expected_focal is not None and not math.isclose(actual_focal, float(expected_focal), rel_tol=0.0, abs_tol=1e-4):
            raise RuntimeError(f"Canonical ID {camera_id} 的 focal_length_mm 应用失败")
        if not math.isclose(actual_fov, expected_fov, rel_tol=0.0, abs_tol=1e-3):
            raise RuntimeError(
                f"Canonical ID {camera_id} 的 derived horizontal FOV 与实际 filmback/focal 不一致: "
                f"expected={expected_fov} actual={actual_fov}"
            )
        if resolution is not None:
            profile_resolution = tuple(profile["resolution"])
            if resolution != profile_resolution:
                raise RuntimeError(
                    f"Canonical ID {camera_id} 的 profile resolution={profile_resolution} "
                    f"与 render resolution={resolution} 不一致"
                )
        for actual, expected, field in (
            (actual_location.x / 100.0, location[0], "position_m.x"),
            (actual_location.y / 100.0, location[1], "position_m.y"),
            (actual_location.z / 100.0, location[2], "position_m.z"),
            (actual_rotation.pitch, rotation[0], "rotation_deg.pitch"),
            (actual_rotation.yaw, rotation[1], "rotation_deg.yaw"),
            (actual_rotation.roll, rotation[2], "rotation_deg.roll"),
        ):
            if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-3):
                raise RuntimeError(f"Canonical ID {camera_id} 的 {field} 回读不匹配")
        print(
            f"  Camera {camera_id}: {entry['camera_actor']} verified "
            f"focal={actual_focal:.6f}mm fov={actual_fov:.6f}"
        )


def _resolve_runtime_camera_selection(
    resolved_task,
    *,
    legacy_sequences,
    legacy_cameras,
    camera_ids=None,
):
    """解析 episode camera policy，并保留显式 smoke/debug 覆盖。"""
    from camera_distribution import resolve_runtime_camera_selection

    return resolve_runtime_camera_selection(
        resolved_task,
        legacy_sequences=legacy_sequences,
        legacy_cameras=legacy_cameras,
        camera_ids=camera_ids,
    )


def _appearance_enabled_for_mode(ue_profile, mode):
    """判断当前模式是否消费 seeded appearance。"""
    return mode in ("sequence", "annotations", "full", "render") and ue_profile.get("appearance") is not None




def _render_environment_options(prepared):
    """返回渲染规划所需的 environment 可移植 metadata。"""
    if prepared is None:
        return {}
    manifest = prepared.get("manifest") or {}
    preflight = prepared.get("preflight") or {}
    return {
        "environment_active": True,
        "environment_policy": manifest.get("policy"),
        "environment_manifest_fingerprint": manifest.get("fingerprint"),
        "environment_preset": manifest.get("preset"),
        "environment_preflight_verified": preflight.get("ok") is True,
        "environment_transaction_state": "applied",
    }


def _validate_appearance_exclusivity(appearance, player_visuals):
    """拒绝同时配置静态外观和 episode seeded appearance。"""
    if appearance is not None and player_visuals is not None:
        raise ValueError("静态 player_visuals 与 seeded appearance 不能同时启用")


def _appearance_runtime_paths(repo_root, task_id):
    """返回本机外观事务证据路径，不写入 episode provenance。"""
    task_id = str(task_id or "task")
    if not task_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in task_id):
        raise ValueError("task_id 含有不允许用于 runtime 目录的字符")
    runtime_dir = Path(repo_root) / ".futsalmot" / "runtime" / task_id
    return (
        runtime_dir / "appearance-rollback.json",
        runtime_dir / "appearance-application.json",
    )


def _environment_enabled_for_mode(ue_profile, mode):
    """仅由当前 resolved environment profile 激活 v1.5。"""
    profile = ue_profile.get("environment_profile")
    return (
        mode in ("sequence", "annotations", "full", "render")
        and isinstance(profile, dict)
        and profile.get("policy") == "futsalmot_environment_v1_5"
    )


def _environment_runtime_paths(repo_root, task_id):
    """返回与 appearance 完全独立的本机 environment 恢复证据路径。"""
    task_id = str(task_id or "task")
    if not task_id or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in task_id
    ):
        raise ValueError("task_id 含有不允许用于 runtime 目录的字符")
    runtime_dir = Path(repo_root) / ".futsalmot" / "runtime" / task_id
    return (
        runtime_dir / "environment-rollback.json",
        runtime_dir / "environment-application.json",
    )


def _prepare_episode_environment(
    ue_profile, mode, episode_dir, repo_root, task_id
):
    """校验环境 manifest、轨迹和 Task 3 diagnostic，不修改 UE 状态。"""
    if not _environment_enabled_for_mode(ue_profile, mode):
        return None
    import episode_environment as adapter

    episode_dir = Path(episode_dir)
    meta_path = episode_dir / "meta.json"
    frames_path = episode_dir / "frames.jsonl"
    manifest_path = episode_dir / "provenance" / "environment.json"
    for required in (meta_path, frames_path, manifest_path):
        if not required.is_file():
            raise ValueError("environment 输入文件缺失: {}".format(required.name))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = dict(ue_profile["environment_profile"])
    profile.update(
        {
            "_episode_dir": episode_dir,
            "_frames_path": frames_path,
            "_meta": meta,
        }
    )
    preflight = adapter.preflight_episode_environment(manifest, profile)
    if not preflight.get("ok"):
        raise RuntimeError(
            "MANUAL_REQUIRED: episode environment preflight failed: {}".format(
                "; ".join(preflight.get("errors") or ["未知错误"])
            )
        )
    snapshot_path, report_path = _environment_runtime_paths(repo_root, task_id)
    public_report_path = episode_dir / "provenance" / "environment_application.json"
    if snapshot_path.exists() or report_path.exists():
        raise RuntimeError(
            "MANUAL_REQUIRED: environment recovery evidence 已存在，拒绝覆盖"
        )
    if public_report_path.exists():
        try:
            existing = json.loads(public_report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise RuntimeError(
                "MANUAL_REQUIRED: 既有 environment report 无法验证，拒绝覆盖"
            ) from exc
        if not isinstance(existing, dict) or existing.get("state") not in {
            "not_started", "preflight_failed", "restored"
        }:
            raise RuntimeError(
                "MANUAL_REQUIRED: 既有 environment report 未终结或状态未知"
            )
    return {
        "adapter": adapter,
        "manifest": manifest,
        "profile": profile,
        "preflight": preflight,
        "snapshot_path": snapshot_path,
        "report_path": report_path,
        "public_report_path": public_report_path,
    }


def _retire_runtime_evidence(rollback_path, report_path, label):
    """删除已验证恢复证据；任一删除失败时恢复已删文件。"""
    paths = [Path(rollback_path), Path(report_path)]
    backups = {}
    for path in paths:
        if path.is_file():
            backups[path] = path.read_bytes()
        elif path.exists():
            raise RuntimeError("{} 恢复证据不是普通文件: {}".format(label, path))
    removed = []
    try:
        for path in paths:
            if path.exists():
                path.unlink()
                removed.append(path)
        if any(path.exists() for path in paths):
            raise OSError("恢复证据删除后仍存在")
    except Exception as exc:
        restore_errors = []
        for path in removed:
            if path.exists():
                continue
            try:
                path.write_bytes(backups[path])
            except Exception as restore_exc:
                restore_errors.append("{}: {}".format(path, restore_exc))
        detail = "{} 恢复证据清理失败: {}".format(label, exc)
        if restore_errors:
            detail += "; 无法恢复已删除证据: {}".format("; ".join(restore_errors))
        raise RuntimeError(detail) from exc


def _retire_appearance_runtime_evidence(rollback_path, report_path):
    """删除已验证 appearance 恢复证据。"""
    return _retire_runtime_evidence(rollback_path, report_path, "appearance")


def _retire_environment_runtime_evidence(rollback_path, report_path):
    """删除已验证 environment 恢复证据。"""
    return _retire_runtime_evidence(rollback_path, report_path, "environment")


def _recovery_render_result(report):
    """把遗留的非终态 MRQ 记录封闭为恢复失败终态。"""
    jobs = report.get("mrq_jobs")
    if not isinstance(jobs, list):
        jobs = []
    terminal = {"success", "failed", "cancelled", "canceled", "skipped"}
    normalized_jobs = []
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("camera_id"), str):
            continue
        normalized_jobs.append({
            "camera_id": job["camera_id"],
            "status": job["status"] if job.get("status") in terminal else "failed",
        })
    return {
        "status": "failed",
        "jobs": normalized_jobs,
        "reason": "recovery finalized an interrupted appearance transaction",
    }


def _recover_local_appearance_evidence(
    adapter, rollback_path, report_path, public_report_path,
    snapshot, local_report, manifest, catalog, mapping,
):
    """验证并完成本机 appearance 恢复事务，成功后才退休证据。"""
    recovery_errors = adapter.validate_local_recovery_evidence(
        snapshot, local_report, manifest, catalog
    )
    if recovery_errors:
        raise RuntimeError(
            "appearance 恢复证据无效，拒绝覆盖并保留现有文件: {}".format(
                "; ".join(recovery_errors)
            )
        )

    if public_report_path.exists():
        try:
            public_report = json.loads(public_report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise RuntimeError(
                "既有公共 appearance report 无法验证，保留本机恢复证据"
            ) from exc
        if not isinstance(public_report, dict):
            raise RuntimeError("既有公共 appearance report 无效，保留本机恢复证据")
        if public_report.get("state") in {"applied", "rendering", "restore_failed"}:
            if public_report.get("transaction_id") != snapshot.get("transaction_id"):
                raise RuntimeError(
                    "公共 report 存在未解决事务且 transaction_id 不匹配，保留本机恢复证据"
                )
            for field, expected in (
                ("manifest_fingerprint", manifest.get("fingerprint")),
                ("catalog_fingerprint", manifest.get("catalog_fingerprint")),
                ("trajectory_sha256", manifest.get("trajectory_sha256")),
                ("episode_id", manifest.get("episode_id")),
                ("catalog_id", manifest.get("catalog_id")),
            ):
                if public_report.get(field) != expected:
                    raise RuntimeError(
                        "公共 report {} 与本机恢复事务不匹配，保留本机恢复证据".format(field)
                    )
            try:
                adapter._validate_report(public_report)
            except (TypeError, ValueError, KeyError, OverflowError, RecursionError) as exc:
                raise RuntimeError(
                    "公共 report 未解决事务结构无效，保留本机恢复证据"
                ) from exc
        elif public_report.get("state") not in {
            "not_started", "preflight_failed", "restored"
        }:
            raise RuntimeError("公共 report 状态未知，保留本机恢复证据")
        if public_report.get("state") == "restored" and (
            public_report.get("transaction_id") != snapshot.get("transaction_id")
        ):
            raise RuntimeError("公共 report restored transaction_id 与本机证据不匹配")

    recovery_preflight = adapter.preflight_episode_appearance_recovery(
        snapshot, mapping, local_report.get("state") == "restored"
    )
    if not recovery_preflight.get("ok"):
        raise RuntimeError(
            "appearance 恢复前 formal-map/ten-actor gate 失败，保留现有证据: {}".format(
                "; ".join(recovery_preflight.get("errors") or ["未知错误"])
            )
        )

    if local_report.get("state") in {"applied", "rendering", "restore_failed"}:
        import render_episode
        if getattr(render_episode, "_ACTIVE_RENDER", None) is not None:
            raise RuntimeError(
                "MRQ pipeline 仍在当前 Editor 活跃，拒绝并发恢复并保留现有证据"
            )
        render_result = _recovery_render_result(local_report)
        recovered = adapter.restore_episode_appearance(
            str(rollback_path), str(report_path), "failed", render_result=render_result
        )
        if not recovered.get("ok"):
            raise RuntimeError(
                "appearance 恢复失败，保留现有证据: {}".format(
                    "; ".join(recovered.get("errors") or ["未知错误"])
                )
            )

    restored_report = adapter.read_application_report(report_path)
    if (
        restored_report.get("state") != "restored"
        or restored_report.get("transaction_id") != snapshot.get("transaction_id")
    ):
        raise RuntimeError("appearance 恢复未验证为匹配的 restored 状态，保留现有证据")
    restored_preflight = adapter.preflight_episode_appearance_recovery(
        snapshot, mapping, True
    )
    if not restored_preflight.get("ok"):
        raise RuntimeError(
            "appearance 恢复后的十人 baseline readback 失败，保留现有证据: {}".format(
                "; ".join(restored_preflight.get("errors") or ["未知错误"])
            )
        )
    try:
        adapter.write_application_report(public_report_path, restored_report)
        published = adapter.read_application_report(public_report_path)
        if any((
            published.get("state") != "restored",
            published.get("transaction_id") != snapshot.get("transaction_id"),
            published.get("manifest_fingerprint") != manifest.get("fingerprint"),
            published.get("catalog_fingerprint") != manifest.get("catalog_fingerprint"),
            published.get("trajectory_sha256") != manifest.get("trajectory_sha256"),
        )):
            raise RuntimeError("公共 appearance report 未验证为 restored")
    except Exception as exc:
        raise RuntimeError(
            "appearance 恢复报告发布失败，保留现有证据: {}".format(exc)
        ) from exc
    _retire_appearance_runtime_evidence(rollback_path, report_path)


def _portable_render_diagnostic(value):
    """保留渲染诊断文本，同时移除其中的本机绝对路径。"""
    if not isinstance(value, str):
        return value
    value = re.sub(
        r"(['\"])(?:[A-Za-z]:[\\/]|\\\\|/(?!Game/))[^'\"]*\1",
        "'<local-path>'",
        value,
    )
    value = re.sub(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s,;\"']+", "<local-path>", value)
    value = re.sub(r"(?<![\w.])/(?!Game/)[^\s,;\"']+", "<local-path>", value)
    return re.sub(r"(?<![A-Za-z0-9_])\\\\[^\s,;\"']+", "<local-path>", value)


class _AppearanceTransaction:
    """把本机恢复证据和 episode 可移植报告关联起来。"""

    def __init__(self, adapter, rollback_path, report_path, public_report_path):
        self.adapter = adapter
        self.rollback_path = Path(rollback_path)
        self.report_path = Path(report_path)
        self.public_report_path = Path(public_report_path)
        self.restore_result = None
        self.callback_ownership_transferred = False

    def _publish_report(self):
        if not self.report_path.is_file():
            return
        report = self.adapter.read_application_report(self.report_path)
        self.adapter.write_application_report(self.public_report_path, report)

    def mark_rendering(self):
        report = self.adapter.read_application_report(self.report_path)
        if report.get("state") != "rendering":
            report = self.adapter.transition_application_report(
                report, "rendering", mrq_pending=True
            )
            self.adapter.write_application_report(self.report_path, report)
            self._publish_report()

    def record_downstream_failure(self, exc):
        report = self.adapter.read_application_report(self.report_path)
        report["errors"].append(
            "downstream work raised {}".format(type(exc).__name__)
        )
        self.adapter.write_application_report(self.report_path, report)
        self._publish_report()

    def restore(self, final_status, render_result=None):
        if self.restore_result is not None:
            if not self.restore_result.get("ok") and self.report_path.is_file():
                try:
                    self._publish_report()
                except Exception as report_exc:
                    errors = self.restore_result.setdefault("errors", [])
                    message = "appearance report 重试发布失败: {}: {}".format(
                        type(report_exc).__name__, report_exc
                    )
                    if message not in errors:
                        errors.append(message)
            return self.restore_result
        if render_result is not None and self.report_path.is_file():
            try:
                render_result = self._record_render_result(render_result)
            except Exception as report_exc:
                print(
                    "ERROR: 无法在 restore 前记录 terminal MRQ 结果: {}".format(report_exc),
                    file=sys.stderr,
                )
        if render_result is None:
            result = self.adapter.restore_episode_appearance(
                str(self.rollback_path), str(self.report_path), final_status
            )
        else:
            result = self.adapter.restore_episode_appearance(
                str(self.rollback_path), str(self.report_path), final_status,
                render_result=render_result,
            )
        if not result.get("ok") and self.report_path.is_file():
            try:
                report = self.adapter.read_application_report(self.report_path)
                if report.get("state") != "restore_failed":
                    report["state"] = "restore_failed"
                    report["mrq_pending"] = False
                    for actor in report.get("actors", []):
                        if isinstance(actor, dict):
                            actor["restore_status"] = "failed"
                    report.setdefault("errors", []).extend(
                        str(error) for error in result.get("errors", [])
                    )
                    self.adapter.write_application_report(self.report_path, report)
            except Exception as report_exc:
                print(
                    "ERROR: 无法将 appearance report 标记为 restore_failed: {}".format(
                        report_exc
                    ),
                    file=sys.stderr,
                )
        if self.report_path.is_file():
            try:
                self._publish_report()
            except Exception as report_exc:
                message = "appearance report 发布失败: {}: {}".format(
                    type(report_exc).__name__, report_exc
                )
                result = {
                    "ok": False,
                    "operation": "restore",
                    "errors": list(result.get("errors") or []) + [message],
                    "warnings": list(result.get("warnings") or []),
                    "state": "restore_failed",
                }
                try:
                    report = self.adapter.read_application_report(self.report_path)
                    if report.get("state") != "restore_failed":
                        report["state"] = "restore_failed"
                        report["mrq_pending"] = False
                        report.setdefault("errors", []).append(message)
                        self.adapter.write_application_report(self.report_path, report)
                    self._publish_report()
                except Exception as retry_exc:
                    result["errors"].append(
                        "restore_failed report 持久化/发布失败: {}: {}".format(
                            type(retry_exc).__name__, retry_exc
                        )
                    )
        self.restore_result = result
        return self.restore_result

    def _record_render_result(self, render_result):
        report = self.adapter.read_application_report(self.report_path)
        raw_status = render_result.get("status")
        if raw_status in {"skipped", "no_render"}:
            raw_status = "failed"
        portable_status = raw_status if raw_status in {
            "success", "failed", "partial", "cancelled", "canceled", "error"
        } else "error"
        portable_result = {"status": portable_status}
        jobs = render_result.get("jobs")
        if isinstance(jobs, list):
            terminal = {"success", "failed", "cancelled", "canceled", "skipped"}
            normalized_jobs = [
                {
                    "camera_id": job.get("camera_id"),
                    "status": job.get("status")
                    if job.get("status") in terminal
                    else "failed",
                }
                for job in jobs
                if isinstance(job, dict)
            ]
            report["mrq_jobs"] = normalized_jobs
            portable_result["jobs"] = normalized_jobs
        cameras = render_result.get("cameras")
        if isinstance(cameras, dict):
            portable_cameras = {}
            for camera_id, camera in cameras.items():
                if not isinstance(camera_id, str) or not isinstance(camera, dict):
                    continue
                portable_cameras[camera_id] = {
                    key: camera[key]
                    for key in (
                        "ok", "img1_frames", "mask_frames", "expected_frames",
                        "mask_source", "annotations_jsonl_frames", "annotation_img1_match",
                    )
                    if key in camera and isinstance(
                        camera[key], (str, bool, int, float, type(None))
                    )
                }
            portable_result["cameras"] = portable_cameras
        for key in ("reason", "summary_write_error"):
            value = render_result.get(key)
            if key in render_result and (value is None or isinstance(value, str)):
                portable_result[key] = _portable_render_diagnostic(value)
        mrq_errors = render_result.get("mrq_errors")
        if mrq_errors is None and "mrq_errors" in render_result:
            portable_result["mrq_errors"] = None
        elif isinstance(mrq_errors, list):
            portable_result["mrq_errors"] = [
                _portable_render_diagnostic(error)
                for error in mrq_errors
                if isinstance(error, str)
            ]
        report["render_result"] = portable_result
        self.adapter.write_application_report(self.report_path, report)
        return portable_result



def _portable_environment_render_result(render_result):
    """抽取 environment report 需要的 portable render terminal 记录。"""
    if not isinstance(render_result, Mapping):
        return None
    cameras = render_result.get("cameras")
    if isinstance(cameras, Mapping):
        records = [
            {
                "camera_id": camera_id,
                "status": _portable_camera_status(value),
            }
            for camera_id, value in cameras.items()
        ]
    elif isinstance(cameras, list):
        records = [
            {"camera_id": camera.get("camera_id"), "status": _portable_camera_status(camera)}
            for camera in cameras
            if isinstance(camera, Mapping)
        ]
    else:
        records = []
    return {
        "status": render_result.get("status"),
        "cameras": records,
        "environment_restore_ready": render_result.get("environment_restore_ready"),
    }


def _portable_camera_status(value):
    """把 camera 记录归一化为 terminal 状态字符串。"""
    if not isinstance(value, Mapping):
        return None
    status = value.get("status")
    if isinstance(status, str) and status:
        return status
    ok = value.get("ok")
    if ok is True:
        return "success"
    if ok is False:
        return "failed"
    return None


class _EnvironmentTransaction:
    """独立持有 environment rollback/report 生命周期。"""
    def __init__(self, adapter, snapshot_path, report_path, public_report_path):
        self.adapter = adapter
        self.snapshot_path = Path(snapshot_path)
        self.report_path = Path(report_path)
        self.public_report_path = Path(public_report_path)
        self.restore_result = None

    def mark_rendering(self):
        if not self.report_path.is_file():
            return
        report = self.adapter._read_json(self.report_path)
        report["state"] = "rendering"
        self.adapter._write_json(self.report_path, report)
        self.adapter._write_json(self.public_report_path, report)

    def record_downstream_failure(self, exc):
        if not self.report_path.is_file():
            return
        report = self.adapter._read_json(self.report_path)
        report.setdefault("errors", []).append(
            "downstream work raised {}".format(type(exc).__name__)
        )
        self.adapter._write_json(self.report_path, report)
        self.adapter._write_json(self.public_report_path, report)

    def restore(self, reason, render_result=None):
        if self.restore_result is not None:
            return self.restore_result
        self.restore_result = self.adapter.restore_episode_environment(
            self.snapshot_path, self.report_path,
            {"reason": reason, "render_result": render_result},
        )
        if self.report_path.is_file():
            try:
                report = self.adapter._read_json(self.report_path)
                report["state"] = (
                    "restored" if self.restore_result.get("ok") else "restore_failed"
                )
                report["restore_status"] = (
                    "restored" if self.restore_result.get("ok") else "failed"
                )
                if not self.restore_result.get("ok"):
                    report.setdefault("errors", []).extend(
                        self.restore_result.get("errors") or ["environment restore failed"]
                    )
                if render_result is not None:
                    report["render_result"] = _portable_environment_render_result(
                        render_result
                    )
                self.adapter._write_json(self.report_path, report)
                self.adapter._write_json(self.public_report_path, report)
            except Exception as report_exc:
                self.restore_result = {
                    "ok": False,
                    "state": "restore_failed",
                    "status": "MANUAL_REQUIRED",
                    "errors": list(self.restore_result.get("errors") or [])
                    + ["environment report publication failed: {}".format(report_exc)],
                }
        if self.restore_result.get("ok"):
            try:
                _retire_environment_runtime_evidence(
                    self.snapshot_path, self.report_path
                )
            except Exception as retire_exc:
                self.restore_result = {
                    "ok": False,
                    "state": "restore_failed",
                    "status": "MANUAL_REQUIRED",
                    "errors": ["environment 恢复证据清理失败: {}".format(retire_exc)],
                }
        return self.restore_result


def _apply_prepared_environment(prepared):
    """应用已通过只读诊断的 environment transaction。"""
    result = prepared["adapter"].apply_episode_environment(
        prepared["manifest"],
        prepared["profile"],
        prepared["snapshot_path"],
        prepared["report_path"],
    )
    if not result.get("ok"):
        if prepared["report_path"].is_file():
            prepared["adapter"]._write_json(
                prepared["public_report_path"],
                prepared["adapter"]._read_json(prepared["report_path"]),
            )
        raise RuntimeError(
            "episode environment apply 失败: {}".format(
                "; ".join(result.get("errors") or ["MANUAL_REQUIRED"])
            )
        )
    return _EnvironmentTransaction(
        prepared["adapter"],
        prepared["snapshot_path"],
        prepared["report_path"],
        prepared["public_report_path"],
    )


def _apply_prepared_environment_and_appearance(environment_prepared, appearance_prepared):
    """按顺序应用两笔独立事务，失败时回滚已成功应用的另一笔。"""
    environment_transaction = None
    appearance_transaction = None
    try:
        if environment_prepared is not None:
            environment_transaction = (
                environment_prepared.apply()
                if hasattr(environment_prepared, "apply")
                else _apply_prepared_environment(environment_prepared)
            )
        if appearance_prepared is not None:
            appearance_transaction = (
                appearance_prepared.apply()
                if hasattr(appearance_prepared, "apply")
                else _apply_prepared_appearance(appearance_prepared)
            )
    except BaseException:
        if appearance_transaction is not None:
            appearance_transaction.restore("apply_failed")
        if environment_transaction is not None:
            environment_transaction.restore("apply_failed")
        raise
    if environment_transaction is None:
        return appearance_transaction
    if appearance_transaction is None:
        return environment_transaction
    return _CombinedEnvironmentAppearanceTransaction(
        environment_transaction, appearance_transaction
    )


def _combine_environment_appearance_transactions(
    environment_transaction, appearance_transaction
):
    if environment_transaction is None:
        return appearance_transaction
    if appearance_transaction is None:
        return environment_transaction
    return _CombinedEnvironmentAppearanceTransaction(
        environment_transaction, appearance_transaction
    )


class _CombinedEnvironmentAppearanceTransaction:
    """共同持有 MRQ lifecycle，但分别写入和恢复两类 report。"""

    def __init__(self, environment_transaction, appearance_transaction):
        self.environment = environment_transaction
        self.appearance = appearance_transaction
        self.restore_result = None
        self.render_deferred = False
        self.callback_ownership_transferred = False

    def mark_rendering(self):
        self.environment.mark_rendering()
        self.appearance.mark_rendering()

    def record_downstream_failure(self, exc):
        if hasattr(self.appearance, "record_downstream_failure"):
            self.appearance.record_downstream_failure(exc)
        if hasattr(self.environment, "record_downstream_failure"):
            self.environment.record_downstream_failure(exc)

    def restore(self, final_status, render_result=None):
        if self.restore_result is not None:
            return self.restore_result
        try:
            appearance_result = self.appearance.restore(final_status, render_result)
        except BaseException as exc:
            appearance_result = {
                "ok": False,
                "state": "restore_failed",
                "errors": [
                    "appearance restore exception: {}: {}".format(
                        type(exc).__name__, exc
                    )
                ],
            }
        try:
            environment_result = self.environment.restore(final_status, render_result)
        except BaseException as exc:
            environment_result = {
                "ok": False,
                "state": "restore_failed",
                "errors": [
                    "environment restore exception: {}: {}".format(
                        type(exc).__name__, exc
                    )
                ],
            }
        self.render_deferred = bool(self.appearance.render_deferred)
        self.callback_ownership_transferred = bool(
            self.appearance.callback_ownership_transferred
        )
        errors = []
        if not appearance_result.get("ok"):
            errors.extend(appearance_result.get("errors") or ["appearance restore failed"])
        if not environment_result.get("ok"):
            errors.extend(environment_result.get("errors") or ["environment restore failed"])
        self.restore_result = {
            "ok": not errors,
            "state": "restored" if not errors else "restore_failed",
            "errors": errors,
        }
        return self.restore_result

    def _record_incomplete_render_result(self, render_result):
        result = self.appearance._record_incomplete_render_result(render_result)
        environment_result = self.environment.restore("failed", render_result)
        if not environment_result.get("ok"):
            errors = list(result.get("errors") or [])
            errors.extend(
                environment_result.get("errors") or ["environment restore failed"]
            )
            return {
                "ok": False,
                "operation": "restore_deferred",
                "state": "restore_failed",
                "errors": errors,
                "restore_deferred": True,
            }
        self.render_deferred = True
        return result


def _prepare_episode_appearance(
    ue_profile, mode, episode_dir, mapping, repo_root, task_id, player_visuals=None
):
    """校验 episode manifest、轨迹和十人 UE roster，暂不修改 Actor。"""
    appearance = ue_profile.get("appearance")
    if not _appearance_enabled_for_mode(ue_profile, mode):
        return None

    import episode_appearance as adapter

    episode_dir = Path(episode_dir)
    meta_path = episode_dir / "meta.json"
    frames_path = episode_dir / "frames.jsonl"
    manifest_path = episode_dir / "provenance" / "appearance.json"
    for required in (meta_path, frames_path, manifest_path):
        if not required.is_file():
            raise ValueError("appearance 输入文件缺失: {}".format(required.name))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = adapter.validate_appearance_inputs(
        manifest, appearance, meta, frames_path
    )
    if errors:
        raise ValueError("appearance 输入校验失败: {}".format("; ".join(errors)))
    rollback_path, report_path = _appearance_runtime_paths(repo_root, task_id)
    public_report_path = episode_dir / "provenance" / "appearance_application.json"
    if rollback_path.exists() != report_path.exists():
        raise RuntimeError("appearance 恢复证据不完整，拒绝覆盖并保留现有文件")
    if rollback_path.exists():
        try:
            snapshot = json.loads(rollback_path.read_text(encoding="utf-8"))
            local_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise RuntimeError("appearance 恢复证据无法读取，拒绝覆盖并保留现有文件") from exc
        _recover_local_appearance_evidence(
            adapter, rollback_path, report_path, public_report_path,
            snapshot, local_report, manifest, appearance, mapping,
        )
    if public_report_path.exists():
        try:
            existing_report = json.loads(public_report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise RuntimeError("既有 appearance application report 无法验证，拒绝覆盖") from exc
        if not isinstance(existing_report, dict) or existing_report.get("state") in {
            "applied", "rendering", "restore_failed"
        }:
            raise RuntimeError("既有 appearance application report 含未解决事务，拒绝覆盖")
        if existing_report.get("state") not in {
            "not_started", "preflight_failed", "restored"
        }:
            raise RuntimeError("既有 appearance application report 状态未知，拒绝覆盖")
    preflight = adapter.preflight_episode_appearance(manifest, appearance, mapping)
    if not preflight.get("ok"):
        raise RuntimeError(
            "appearance 十人 preflight 失败: {}".format(
                "; ".join(preflight.get("errors") or ["未知错误"])
            )
        )
    return {
        "adapter": adapter,
        "catalog": appearance,
        "manifest": manifest,
        "mapping": mapping,
        "preflight": preflight,
        "rollback_path": rollback_path,
        "report_path": report_path,
        "public_report_path": public_report_path,
    }



def _apply_prepared_appearance(prepared):
    """应用已完成纯输入和 Editor roster preflight 的外观事务。"""
    result = prepared["adapter"].apply_episode_appearance(
        prepared["manifest"],
        prepared["catalog"],
        prepared["mapping"],
        str(prepared["rollback_path"]),
        str(prepared["report_path"]),
    )
    if not result.get("ok"):
        if prepared["report_path"].is_file():
            prepared["adapter"].write_application_report(
                prepared["public_report_path"],
                prepared["adapter"].read_application_report(prepared["report_path"]),
            )
        raise RuntimeError(
            "episode appearance apply 失败: {}".format(
                "; ".join(result.get("errors") or ["未知错误"])
            )
        )
    transaction = _AppearanceTransaction(
        prepared["adapter"],
        prepared["rollback_path"],
        prepared["report_path"],
        prepared["public_report_path"],
    )
    try:
        report = prepared["adapter"].read_application_report(prepared["report_path"])
        actor_paths = {
            actor["entity_id"]: actor.get("actor_path")
            for actor in prepared["preflight"].get("actors", [])
            if isinstance(actor, dict)
        }
        for actor in report["actors"]:
            actor_path = actor_paths.get(actor["entity_id"])
            if actor_path:
                actor["actor_path"] = actor_path
        prepared["adapter"].write_application_report(prepared["report_path"], report)
        transaction._publish_report()
    except Exception as report_exc:
        restored = transaction.restore("failed")
        if not restored.get("ok"):
            raise RuntimeError(
                "application report 发布失败且 appearance restore 失败: {} / {}".format(
                    report_exc,
                    "; ".join(restored.get("errors") or ["未知错误"]),
                )
            ) from report_exc
        raise
    return transaction



def _run_with_optional_appearance(transaction, downstream):
    """同步执行模式工作并在成功或异常时恢复外观。"""
    if transaction is None:
        return downstream()
    downstream_error = None
    try:
        result = downstream()
    except BaseException as exc:
        downstream_error = exc
        try:
            transaction.record_downstream_failure(exc)
        except Exception as report_exc:
            print(
                "ERROR: 无法记录 episode appearance downstream 失败: {}".format(report_exc),
                file=sys.stderr,
            )
        raise
    finally:
        try:
            restore_result = transaction.restore(
                "failed" if downstream_error is not None else "success"
            )
            if not restore_result.get("ok"):
                raise RuntimeError(
                    "episode appearance restore 失败: {}".format(
                        "; ".join(restore_result.get("errors") or ["未知错误"])
                    )
                )
        except Exception as restore_exc:
            if downstream_error is None:
                raise
            print("ERROR: episode appearance restore 失败: {}".format(restore_exc), file=sys.stderr)
    return result


def _run_prepared_appearance_synchronously(prepared, downstream):
    """应用外观后执行同步模式工作，并在所有路径恢复。"""
    if prepared is None:
        return downstream()
    transaction = _apply_prepared_appearance(prepared)
    return _run_with_optional_appearance(transaction, downstream)


def _submit_render_with_appearance(render_submit, transaction):
    """提交 MRQ；同步终态由调用方恢复，异步提交才转移 callback 所有权。"""
    callback_called = False

    def on_complete(render_result):
        nonlocal callback_called
        if callback_called:
            return
        callback_called = True
        if not isinstance(render_result, dict):
            render_result = {"status": "failed", "jobs": []}
        final_status = render_result.get("status")
        if final_status == "skipped":
            final_status = "no_render"
        if final_status == "partial":
            final_status = "failed"
        if final_status not in {"success", "failed", "cancelled", "canceled", "no_render"}:
            final_status = "failed"
        if render_result.get("appearance_restore_ready") is False:
            transaction.callback_ownership_transferred = True
            result = transaction._record_incomplete_render_result(render_result)
        elif (
            render_result.get("environment_restore_ready") is False
            and not hasattr(transaction, "_record_incomplete_render_result")
        ):
            result = transaction.restore(final_status, render_result)
        else:
            result = transaction.restore(final_status, render_result)
        if not result.get("ok"):
            raise RuntimeError(
                "MRQ callback appearance restore 失败: {}".format(
                    "; ".join(result.get("errors") or ["未知错误"])
                )
            )

    try:
        transaction.mark_rendering()
        result = render_submit(on_complete=on_complete)
    except BaseException as render_exc:
        if not callback_called and not transaction.callback_ownership_transferred:
            restored = transaction.restore("failed")
            if not restored.get("ok"):
                raise RuntimeError(
                    "render 提交异常且 appearance restore 失败: {} / {}".format(
                        render_exc,
                        "; ".join(restored.get("errors") or ["未知错误"]),
                    )
                ) from render_exc
        raise
    if not isinstance(result, dict):
        result = {"callback_ownership_transferred": False}
    if callback_called:
        result = dict(result)
        result["callback_ownership_transferred"] = False
        if (
            transaction.restore_result is None
            or not transaction.restore_result.get("ok")
        ) and not transaction.render_deferred:
            raise RuntimeError("render 在 submit 返回前终止，但 appearance restore/report 未成功")
        return result
    if result.get("callback_ownership_transferred") is not True:
        if not callback_called:
            terminal_result = result.get("terminal_result")
            if not isinstance(terminal_result, dict):
                terminal_result = {"status": "failed", "jobs": []}
            final_status = terminal_result.get("status")
            if final_status == "skipped":
                final_status = "no_render"
            if final_status not in {"success", "failed", "cancelled", "canceled", "no_render"}:
                final_status = "failed"
            restored = transaction.restore(final_status, terminal_result)
            if not restored.get("ok"):
                raise RuntimeError(
                    "render 同步终态 appearance restore 失败: {}".format(
                        "; ".join(restored.get("errors") or ["未知错误"])
                    )
                )
        if (
            transaction.restore_result is None
            or not transaction.restore_result.get("ok")
        ) and not transaction.render_deferred:
            raise RuntimeError("render 已同步终止，但 appearance 未成功恢复")
        return result
    transaction.callback_ownership_transferred = True
    if callback_called and (
        transaction.restore_result is None
        or not transaction.restore_result.get("ok")
    ) and not transaction.render_deferred:
        raise RuntimeError("render callback 已执行但 appearance 未成功恢复")
    return result


def _runtime_pose_to_pose_keypoints(ep_dir, cameras, num_steps, ann_cfg):
    """把 Runtime Pose 12 点输出桥接为 P1 pose_annotator 契约的 pose_keypoints.jsonl。

    读 <ep>/coco17_3d.jsonl（每行 {actor_id, root, keypoints_3d_m: [[x,y,z]×12]}），
    写 <ep>/<cam>/pose_keypoints.jsonl（meta + frame 行，与 pose_export.py 格式一致，
    keypoints_world 单位为米，无 occluded——P1 用 mask 判定 visibility）。
    """
    from pose_bones import POSE_KEYPOINT_NAMES, NUM_POSE_KEYPOINTS
    from annotation_utils import entity_id_to_track_id

    # 解析 image_width/height（来自 render_rgb 或 annotation_export 配置）
    render_cfg = ann_cfg.get("render_rgb") or {}
    image_width = int(render_cfg.get("output_resolution_x", 1920))
    image_height = int(render_cfg.get("output_resolution_y", 1080))

    c3d = ep_dir / "coco17_3d.jsonl"
    if not c3d.is_file():
        print(f"  WARN: 缺 {c3d}，跳过 pose_keypoints 生成")
        return
    rows = [json.loads(l) for l in c3d.read_text(encoding="utf-8").splitlines() if l.strip()]
    # 按 root（帧）分组
    by_root = {}
    for r in rows:
        by_root.setdefault(r["root"], []).append(r)

    for cam in cameras:
        cam_dir = ep_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        meta_line = {
            "kind": "meta",
            "schema": POSE_KEYPOINTS_SCHEMA,
            "episode_id": ep_dir.name,
            "camera_id": cam,
            "image_width": image_width,
            "image_height": image_height,
            "keypoint_names": POSE_KEYPOINT_NAMES,
            "coordinate_convention": (
                "world keypoints in meters（Runtime Pose SaveGame cm / 100）；"
                "UE 左手系 X 前 Y 右 Z 上；按 12 点肢体关键点顺序"
            ),
            "occlusion_method": "none（P1 仅用 mask 判定 visibility）",
        }
        frame_lines = []
        for root in sorted(by_root.keys()):
            objects = []
            for r in by_root[root]:
                kps = r.get("keypoints_3d_m") or []
                if len(kps) != NUM_POSE_KEYPOINTS:
                    continue
                # None → 仍占位（pose_annotator 检查 len==17）
                kp_world = []
                for p in kps:
                    if p is None:
                        kp_world.append([None, None, None])
                    else:
                        kp_world.append([p[0], p[1], p[2]])
                eid = r["actor_id"]
                objects.append({
                    "entity_id": eid,
                    "track_id": entity_id_to_track_id(eid),
                    "keypoints_world": kp_world,
                })
            frame_lines.append({
                "kind": "frame",
                "frame_index": root + 1,  # GRF step+1 约定（img1 000001 对应 step 0）
                "source_step": root,
                "objects": objects,
            })
        out = [meta_line] + frame_lines
        pk_path = cam_dir / "pose_keypoints.jsonl"
        pk_path.write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
        print(f"  Wrote: {pk_path} ({len(frame_lines)} 帧，{len(rows)} actor×帧）")


def _load_resolved_task(args) -> dict:
    rt_path = args.resolved_task or os.environ.get("C5_RESOLVED_TASK")
    if not rt_path:
        raise SystemExit(_fail("缺少 --resolved-task（或环境变量 C5_RESOLVED_TASK）"))
    rt_path = Path(rt_path)
    if not rt_path.is_file():
        raise SystemExit(_fail(f"resolved task 不存在: {rt_path}"))
    with open(rt_path, encoding="utf-8") as f:
        rt = json.load(f)
    if rt.get("schema") != RESOLVED_TASK_SCHEMA:
        raise SystemExit(_fail(f"resolved task schema 非法: {rt.get('schema')!r}"))
    if rt.get("version") != 1:
        raise SystemExit(_fail(f"resolved task version 非法: {rt.get('version')!r}"))
    return rt


def main() -> int:
    ap = argparse.ArgumentParser(description='按 resolved task 运行 UE 导入/渲染')
    ap.add_argument('--resolved-task', required=False, help='resolved task JSON 路径（或环境变量 C5_RESOLVED_TASK）')
    ap.add_argument('--mode', default=None, choices=['sequence', 'annotations', 'full', 'render', 'pose-finalize'], help='覆盖执行模式（缺省 full）')
    args = ap.parse_args()
    mode = args.mode or os.environ.get('C5_RUN_MODE') or 'full'
    rt = _load_resolved_task(args)
    import unreal
    ue_profile = rt.get('ue_profile') or {}
    ann_cfg = dict(ue_profile.get('annotation_export') or {})
    export_profile = rt.get('export_profile') or {}
    if export_profile.get('playback_fps') is not None:
        ann_cfg["playback_fps"] = int(export_profile['playback_fps'])
    pose_cfg = (rt.get('postprocess') or {}).get('yolo_pose') or {}
    pose_enabled = bool(pose_cfg.get('enabled', False))
    episode_dir = Path(rt['trajectory_output'])
    dataset_root = Path(rt['dataset_root'])
    mapping_path = Path(rt['actor_mapping'])
    seq_list = ue_profile.get('sequences') or []
    seq_pkg = ue_profile.get('sequence_package_path') or '/Game/FutsalMOT/Sequences'
    replace_existing = bool(ue_profile.get('replace_existing', True))
    ball_rolling = ue_profile.get('ball_rolling') or None
    episode_name = rt.get('episode_name') or 'episode'
    requested_camera_ids = os.environ.get('C5_CAMERA_IDS')
    camera_ids = [camera_id.strip() for camera_id in requested_camera_ids.split(',') if camera_id.strip()] if requested_camera_ids else None
    camera_selection = _resolve_runtime_camera_selection(rt, legacy_sequences=seq_list, legacy_cameras=ann_cfg.get('cameras') or ['CineCam_01'], camera_ids=camera_ids)
    camera_entries = camera_selection['entries']
    appearance_requested = _appearance_enabled_for_mode(ue_profile, mode)
    environment_requested = _environment_enabled_for_mode(ue_profile, mode)
    if camera_selection['canonical'] and (not appearance_requested) and (not environment_requested):
        _apply_resolved_anchor_cameras(camera_entries, ann_cfg)
    seq_list = camera_selection['sequences']
    cameras = camera_selection['cameras'] or ['CineCam_01']
    if camera_selection['canonical']:
        ann_cfg['cameras'] = cameras
        resolution = camera_selection['resolution']
        render_cfg = dict(ann_cfg.get('render_rgb') or {})
        configured = (render_cfg.get('output_resolution_x'), render_cfg.get('output_resolution_y'))
        if configured != (None, None) and configured != tuple(resolution):
            raise RuntimeError(f'canonical camera resolution={tuple(resolution)} 与 render_rgb.output_resolution={configured} 不一致')
        (render_cfg['output_resolution_x'], render_cfg['output_resolution_y']) = resolution
        ann_cfg['render_rgb'] = render_cfg
    _export_ns = int(export_profile.get('num_steps') or 3)
    _export_fps = int(export_profile.get('target_fps') or 10)
    _factor = max(1, _export_fps // 10)
    meta_timing_num_steps = _export_ns * _factor
    if not episode_dir.is_dir():
        return _fail(f'trajectory 目录不存在（先运行 grf-ue task export）: {episode_dir}')
    if not mapping_path.is_file():
        return _fail(f'actor mapping 不存在: {mapping_path}')
    from player_visuals import apply_player_visuals, normalize_player_visuals, preflight_player_visuals, write_player_visuals_provenance
    from import_grf_episode import create_sequence, load_episode, load_mapping
    from annotation_exporter import export_annotations
    from render_episode import render_sequences
    mapping = load_mapping(mapping_path)
    normalized_player_visuals = normalize_player_visuals(ue_profile.get('player_visuals'))
    try:
        environment_prepared = _prepare_episode_environment(ue_profile, mode, episode_dir, rt.get('repo_root', episode_dir.parent), rt.get('task_id', 'task'))
        _validate_appearance_exclusivity(ue_profile.get('appearance'), normalized_player_visuals)
        appearance_prepared = _prepare_episode_appearance(ue_profile, mode, episode_dir, mapping, rt.get('repo_root', episode_dir.parent), rt.get('task_id', 'task'), normalized_player_visuals)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return _fail(str(exc))
    episode_transaction = None
    if environment_prepared is not None:
        try:
            episode_transaction = _apply_prepared_environment_and_appearance(environment_prepared, appearance_prepared)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return _fail(str(exc))
    if camera_selection['canonical'] and (appearance_prepared is not None or environment_prepared is not None):
        try:
            _apply_resolved_anchor_cameras(camera_entries, ann_cfg)
        except BaseException:
            if episode_transaction is not None:
                episode_transaction.restore('camera_apply_failed')
            raise
    if normalized_player_visuals is not None:
        rollback_path = Path(rt.get('repo_root', episode_dir.parent)) / '.futsalmot' / 'runtime' / str(rt.get('task_id', 'task')) / 'player-visuals-rollback.json'
        if mode in ('full', 'sequence'):
            applied = apply_player_visuals(normalized_player_visuals, mapping, rollback_path=str(rollback_path))
            if not applied.get('ok'):
                return _fail_with_episode_transaction(episode_transaction, f"player_visuals apply 失败: {applied.get('errors')}", 'player_visuals_apply_failed')
        preflight = preflight_player_visuals(normalized_player_visuals, mapping)
        if not preflight.get('ok'):
            return _fail_with_episode_transaction(episode_transaction, f"player_visuals preflight 失败: {preflight.get('errors')}", 'player_visuals_preflight_failed')
        provenance_path = write_player_visuals_provenance(episode_dir, normalized_player_visuals)
        print(f'  player_visuals provenance: {provenance_path}')
    print(f"run_task: task={rt.get('task_id')} episode={episode_name} mode={mode}")
    print(f'  trajectory: {episode_dir}')
    print(f'  dataset output root: {dataset_root}  (episode_id -> {dataset_root / episode_name})')
    print(f"  sequences: {[s.get('name') for s in seq_list]}")
    print(f'  cameras: {cameras}')
    print(f'  num_steps={meta_timing_num_steps}  yolo_pose enabled: {pose_enabled}')
    if mode == 'pose-finalize':
        os.environ['C5_POSE_TASK'] = str(args.resolved_task or os.environ.get('C5_RESOLVED_TASK'))
        os.environ['C5_POSE_EXPECT'] = str(meta_timing_num_steps)
        ep_out = dataset_root / episode_name
        for stale in ('pose_capture.jsonl', 'pose_session.json'):
            stale_path = ep_out / stale
            if stale_path.exists():
                stale_path.unlink()
                print(f'  [防残留] 删除旧 {stale}（避免误读 incomplete 数据）')
        print('\n--- Runtime Pose 导出（pose_capture_export）---')
        import pose_capture_export
        pose_capture_export.main()
        pc_path = ep_out / 'pose_capture.jsonl'
        ps_path = ep_out / 'pose_session.json'
        if not pc_path.is_file() or not ps_path.is_file():
            return _fail('pose_capture_export 未生成 pose_capture.jsonl（Runtime Pose 捕获 incomplete 或失败），禁止生成最终 Pose')
        import json as _json
        ps = _json.loads(ps_path.read_text(encoding='utf-8'))
        if not ps.get('capture_complete'):
            return _fail(f"pose_session.capture_complete=False (captured={ps.get('captured_frame_count')}/expected={ps.get('expected_frame_count')})，禁止生成最终 Pose")
        print('\n--- 12 点肢体 Pose 3D/2D 生成（build_coco17）---')
        os.environ['C5_EPISODE_DIR'] = str(dataset_root / episode_name)
        os.environ['C5_COCO17_CAMERAS'] = ','.join(cameras)
        import build_coco17
        rc = build_coco17.main()
        if rc:
            return _fail('build_coco17 失败（见上）')
        print('\n--- 生成 pose_keypoints.jsonl（Runtime Pose → P1 pose_annotator 契约）---')
        _runtime_pose_to_pose_keypoints(dataset_root / episode_name, cameras, meta_timing_num_steps, ann_cfg)
        print('\nDone（pose-finalize）。')
        return 0
    appearance_transaction = episode_transaction if mode in ('full', 'render') else None
    if environment_prepared is None and appearance_prepared is not None and (mode in ('full', 'render')):
        try:
            appearance_transaction = _apply_prepared_appearance(appearance_prepared)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return _fail(str(exc))
    if mode in ('full', 'render'):
        try:
            (meta, frames) = load_episode(episode_dir)
            if mode == 'full':
                print('\n--- 全流程：创建 Level Sequence ---')
                create_sequence(meta, frames, mapping, replace_existing, seq_pkg, seq_list, ball_rolling)
                export_cfg = dict(ann_cfg)
                export_cfg['resolved_appearance_policy'] = (ue_profile.get('appearance') or {}).get('policy')
                export_annotations(episode_dir, mapping_path, dataset_root, export_cfg)
            for s in seq_list:
                seq = unreal.load_asset(f"{seq_pkg}/{s['name']}")
                if seq is not None:
                    seq.set_playback_start(0)
                    print(f"  序列 {s['name']} playback 保持创建时范围")
            if pose_enabled and mode == 'full':
                print(f'\n--- Runtime Pose prep（primary={cameras[0]}）---')
                from pose_render import _prep_recorders
                _prep_recorders(episode_name, cameras[0])
                for i in range(5):
                    try:
                        unreal.GameplayStatics.delete_game_in_slot(f'PoseCapture_{episode_name}_{cameras[0]}_G{i}', 0)
                    except Exception:
                        pass
                print(f'  {cameras[0]}: slots 已清空')

            def _render(on_complete=None):
                kwargs = {'appearance_active': True} if appearance_transaction is not None else {}
                kwargs.update(_render_environment_options(environment_prepared))
                if on_complete is not None:
                    kwargs['on_complete'] = on_complete
                return render_sequences(seq_list, ann_cfg, seq_pkg, episode_dir, dataset_root, mapping_path, **kwargs)
            if appearance_transaction is None:
                _render()
            else:
                _submit_render_with_appearance(_render, appearance_transaction)
        except BaseException as downstream_exc:
            if appearance_transaction is not None and appearance_transaction.restore_result is None and (not appearance_transaction.callback_ownership_transferred):
                try:
                    appearance_transaction.record_downstream_failure(downstream_exc)
                except Exception as report_exc:
                    print('ERROR: 无法记录 episode appearance downstream 失败: {}'.format(report_exc), file=sys.stderr)
                restore_result = appearance_transaction.restore('failed')
                if not restore_result.get('ok'):
                    print('ERROR: full/render episode appearance restore 失败: {}'.format('; '.join(restore_result.get('errors') or ['未知错误'])), file=sys.stderr)
            raise
        print('\n已提交。MRQ 渲染为异步执行（不阻塞编辑器），完成后自动复制 RGB 到 img1/（及统计 Instance-ID Mask 对齐帧）并写 render_summary.json。')
        print(f'Runtime Pose 捕获：BurnIn 写入 {cameras[0]} 的 5×SaveGame（世界 3D，camera 无关）。')
        print('渲染完成后运行：py run_task.py --mode pose-finalize（导出 raw 13 骨 + 12 点肢体 Pose）。')
        return 0
    if mode == 'annotations':

        def _annotations():
            export_cfg = dict(ann_cfg)
            export_cfg['resolved_appearance_policy'] = (ue_profile.get('appearance') or {}).get('policy')
            export_annotations(episode_dir, mapping_path, dataset_root, export_cfg)
            if pose_enabled:
                print('\n--- 导出 Pose 关键点（Legacy pose_export，Editor 逐帧采样）---')
                from pose_export import export_pose_keypoints
                export_pose_keypoints(episode_dir, mapping_path, dataset_root, ann_cfg, pose_cfg)
            print('\nDone（annotations）。')
            return 0
        if environment_prepared is not None:
            return _run_with_optional_appearance(episode_transaction, _annotations)
        return _run_prepared_appearance_synchronously(appearance_prepared, _annotations)
    if mode == 'sequence':

        def _sequence():
            (meta, frames) = load_episode(episode_dir)
            create_sequence(meta, frames, mapping, replace_existing, seq_pkg, seq_list, ball_rolling)
            print('\nDone（sequence）。')
            return 0
        if environment_prepared is not None:
            return _run_with_optional_appearance(episode_transaction, _sequence)
        return _run_prepared_appearance_synchronously(appearance_prepared, _sequence)
    return _fail(f'未知模式: {mode!r}')


if __name__ == "__main__":
    sys.exit(main())
