"""Dataset Artifact Lifecycle Cleanup（C6-P1.6）。

按 artifact_policy.profile 清理 transient 产物：
  - 默认 research_minimal：清理 render/FinalImage 重复、BurnIn、debug overlay、
    mask EXR、mask PNG、raw pose_capture、render scratch、SaveGame slots。
  - 保留 canonical：RGB img1、camera calibration、frames/meta、annotations、
    MOT、12 点 Pose 2D/3D、YOLO det/seg/pose labels、provenance、audit、manifest。

安全规则：
  - 必须 validation-gated：若 episode 不完整（audit FAIL / pose incomplete /
    RGB 缺失）→ 拒绝 cleanup，保留用于诊断。
  - dry-run 默认，apply 才真正删除。
  - idempotent：重复运行不报错、不删除 canonical、不改 semantics。
  - 从不删除 canonical。

CLI：
  grf-ue task cleanup <task> --dry-run   # 默认
  grf-ue task cleanup <task> --apply
"""

import json
import shutil
from pathlib import Path

from grf_ue_bridge.task_requirements import resolve_task_requirements
from grf_ue_bridge.validation_result import validation_result_from_report
from grf_ue_bridge.environment import validate_environment_manifest

# 每 camera 下 transient 的相对路径（相对 cam_dir）
_TRANSIENT_CAM_RELS = {
    "render": None,          # render/* （FinalImage + BurnIn）
    "render_mask": None,     # render_mask/*.exr（Cryptomatte EXR）
    "debug": None,           # debug/* overlay
    "mask": "*.png",         # mask/*.png（research_minimal 可删）
}
# global transient 相对 episode 目录
_TRANSIENT_EP_RELS = {
    "pose_capture.jsonl": None,  # raw 13 骨 pose（12 点最终 Pose 已验证后可删）
}
# yolo_pose/images 是复制自 img1 的重复 RGB（YOLO 配置应引用原始 img1），属 transient
_YOLO_IMAGE_DIRS = ("yolo_pose/images", "yolo_det/images", "yolo_seg/images")


def _path_matches(p: Path, cam_rel: str):
    """判断 cam_rel 是否为 p 的父目录之一或 p 本身。"""
    parts = p.parts
    for i in range(len(parts) - 1, -1, -1):
        if Path(*parts[i:]).as_posix() == cam_rel:
            return True
    return False


def _cam_dir(dataset_episode_dir: Path, cam: str) -> Path:
    return dataset_episode_dir / cam




def _resolved_environment_profile(resolved):
    if isinstance(resolved, dict):
        ue_profile = resolved.get("ue_profile") or {}
    else:
        ue_profile = getattr(resolved, "ue_profile", {}) or {}
    profile = ue_profile.get("environment_profile") if isinstance(ue_profile, dict) else None
    if isinstance(profile, dict) and profile.get("policy") == "futsalmot_environment_v1_5":
        return profile
    return None


def collect_transient(dataset_episode_dir: Path, cams, profile='research_minimal', resolved=None):
    """收集将删除的文件（canonical 永不包含）。返回 {path: bytes}。"""
    to_delete = {}
    for cam in cams:
        cam_dir = _cam_dir(dataset_episode_dir, cam)
        if not cam_dir.is_dir():
            continue
        for (rel, pat) in _TRANSIENT_CAM_RELS.items():
            target = cam_dir / rel
            if target.is_dir():
                for f in target.rglob('*'):
                    if f.is_file() and (pat is None or f.name.endswith(pat.replace('*', ''))):
                        to_delete[str(f)] = f.stat().st_size
            elif target.is_file():
                to_delete[str(target)] = target.stat().st_size
    for (rel, pat) in _TRANSIENT_EP_RELS.items():
        target = dataset_episode_dir / rel
        if target.is_file():
            to_delete[str(target)] = target.stat().st_size
    for ydir in _YOLO_IMAGE_DIRS:
        t = dataset_episode_dir / ydir
        if t.is_dir():
            for f in t.rglob('*.png'):
                to_delete[str(f)] = f.stat().st_size
    return to_delete


def _validation_gate(dataset_episode_dir: Path, resolved=None) -> list:
    """返回阻止 cleanup 的原因列表（空 = 可通过）。"""
    problems = []
    requirements = resolve_task_requirements(resolved) if resolved is not None else None
    resolved_get = (resolved.get if isinstance(resolved, dict) else lambda key, default=None: getattr(resolved, key, default)) if resolved is not None else lambda key, default=None: default
    ue_profile = resolved_get('ue_profile', {}) or {}
    appearance_enabled = ue_profile.get('appearance') is not None
    environment_profile = _resolved_environment_profile(resolved)
    task_id = resolved_get('task_id')
    repo_root = resolved_get('repo_root')
    rollback_path = None
    local_report_path = None
    if isinstance(task_id, str) and task_id and all((char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for char in task_id)) and isinstance(repo_root, str) and repo_root:
        runtime_dir = Path(repo_root) / '.futsalmot' / 'runtime' / task_id
        rollback_path = runtime_dir / 'appearance-rollback.json'
        local_report_path = runtime_dir / 'appearance-application.json'
    elif appearance_enabled:
        problems.append('appearance: 缺少 task_id/repo_root，无法检查本机恢复证据')
    public_report_path = Path(dataset_episode_dir) / 'provenance' / 'appearance_application.json'
    evidence_paths = [path for path in (rollback_path, local_report_path, public_report_path) if path is not None and path.exists()]
    report_states = {}
    for (label, path) in (('本机', local_report_path), ('公共', public_report_path)):
        if path is None or not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            problems.append(f'appearance: {label} application report 无法读取: {exc}')
            continue
        state = report.get('state') if isinstance(report, dict) else None
        report_states[label] = state
        if state not in {'preflight_failed', 'restored'}:
            problems.append(f'appearance: {label} application report 状态未终结或无效 ({state!r})，拒绝 cleanup 并保留恢复证据')
    if rollback_path is not None and rollback_path.exists():
        problems.append('appearance: 存在本机 rollback snapshot；即使 report 标记 restored，P1 仍无法验证当前 UE Actor 完整 baseline，拒绝 cleanup 并保留恢复证据')
    elif evidence_paths and (not report_states):
        problems.append('appearance: 存在无法判定状态的事务恢复证据，拒绝 cleanup')
    if environment_profile is not None:
        if not isinstance(task_id, str) or not task_id or (not isinstance(repo_root, str)) or (not repo_root):
            problems.append('environment: 缺少 task_id/repo_root，无法检查本机恢复证据')
            environment_dir = None
        else:
            environment_dir = Path(repo_root) / '.futsalmot' / 'runtime' / task_id
        environment_snapshot_path = environment_dir / 'environment-rollback.json' if environment_dir else None
        environment_report_path = environment_dir / 'environment-application.json' if environment_dir else None
        if environment_snapshot_path is not None and environment_snapshot_path.exists():
            problems.append('environment: 存在本机 rollback snapshot，拒绝 cleanup')
        environment_public_path = Path(dataset_episode_dir) / 'provenance' / 'environment_application.json'
        for (label, path, required) in (('本机', environment_report_path, False), ('公共', environment_public_path, True)):
            if path is None or not path.exists():
                if required:
                    problems.append(f'environment: 缺少{label} application report，拒绝 cleanup')
                continue
            try:
                environment_report = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                problems.append(f'environment: {label} application report 无法读取: {exc}')
                continue
            if not isinstance(environment_report, dict) or environment_report.get('state') != 'restored':
                actual = environment_report.get('state') if isinstance(environment_report, dict) else None
                problems.append(f"environment: {label} application report state actual={actual!r}, expected='restored'")
        environment_manifest_path = Path(dataset_episode_dir) / 'provenance' / 'environment.json'
        try:
            manifest = json.loads(environment_manifest_path.read_text(encoding='utf-8'))
            meta_path = Path(dataset_episode_dir) / 'meta.json'
            frames_path = Path(dataset_episode_dir) / 'frames.jsonl'
            if not meta_path.is_file() or not frames_path.is_file():
                problems.append('environment: 缺少 meta.json/frames.jsonl，拒绝 cleanup')
            else:
                manifest_errors = validate_environment_manifest(manifest, meta=json.loads(meta_path.read_text(encoding='utf-8')), frames_path=frames_path, profile=environment_profile)
                problems.extend((f'environment: {message}' for message in manifest_errors))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
            problems.append(f'environment: environment.json 校验失败: {exc}')
    requires_render = requirements.requires_render if requirements else True
    requires_pose = requirements.requires_pose if requirements else True
    rs_path = dataset_episode_dir / 'render_summary.json'
    if not requires_render:
        pass
    elif rs_path.exists():
        try:
            rs = json.loads(rs_path.read_text(encoding='utf-8'))
            if not isinstance(rs, dict):
                problems.append('render_summary.json 顶层必须是 JSON 对象')
            elif rs.get('status') != 'success':
                problems.append(f"render_summary.status != success ({rs.get('status')})")
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as e:
            problems.append(f'render_summary.json 解析失败: {e}')
    else:
        problems.append('缺少 render_summary.json（渲染未成功/未完成）')
    ps_path = dataset_episode_dir / 'pose_session.json'
    if not requires_pose:
        pass
    elif ps_path.exists():
        try:
            ps = json.loads(ps_path.read_text(encoding='utf-8'))
            if not isinstance(ps, dict):
                problems.append('pose_session.json 顶层必须是 JSON 对象')
            elif not ps.get('capture_complete'):
                problems.append('pose_session.capture_complete != true')
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as e:
            problems.append(f'pose_session.json 解析失败: {e}')
    else:
        problems.append('缺少 pose_session.json（Runtime Pose 未导出）')
    audit_path = dataset_episode_dir / 'audit' / 'soak_audit_report.json'
    if not audit_path.exists():
        problems.append('缺少 canonical audit 报告，无法确认数据完整性')
    else:
        try:
            ar = json.loads(audit_path.read_text(encoding='utf-8'))
            result = validation_result_from_report(ar)
            if not result.passed:
                problems.extend((f'audit 未通过: {error}' for error in result.errors))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            problems.append(f'audit 报告读取失败: {exc}')
    return problems


def plan_cleanup(dataset_episode_dir, cams, profile="research_minimal", dry_run=True, resolved=None):
    """计算 cleanup 计划。返回 dict 报告。"""
    dataset_episode_dir = Path(dataset_episode_dir)
    gate_problems = _validation_gate(dataset_episode_dir, resolved=resolved)
    to_delete = collect_transient(dataset_episode_dir, cams, profile, resolved=resolved)

    delete_bytes = sum(to_delete.values())
    # keep = 全部文件 - 待删
    keep_files = 0
    keep_bytes = 0
    for f in dataset_episode_dir.rglob("*"):
        if f.is_file():
            if str(f) not in to_delete or gate_problems:
                keep_files += 1
                keep_bytes += f.stat().st_size

    report = {
        "profile": profile,
        "dry_run": dry_run,
        "gate_ok": not gate_problems,
        "gate_problems": gate_problems,
        "would_delete_files": len(to_delete),
        "would_delete_bytes": delete_bytes,
        "would_delete_gb": round(delete_bytes / 1e9, 3),
        "would_keep_files": keep_files,
        "would_keep_bytes": keep_bytes,
        "would_keep_gb": round(keep_bytes / 1e9, 3),
        "would_delete": sorted(to_delete.keys()),
        "blocked_delete": bool(gate_problems),
    }
    return report


def apply_cleanup(dataset_episode_dir, cams, profile="research_minimal", resolved=None):
    """执行 cleanup。返回结果 dict。先过 validation gate，失败则拒绝。"""
    dataset_episode_dir = Path(dataset_episode_dir)
    gate_problems = _validation_gate(dataset_episode_dir, resolved=resolved)
    if gate_problems:
        return {"ok": False, "reason": "validation_gate_failed",
                "gate_problems": gate_problems, "deleted_files": 0, "deleted_bytes": 0}
    to_delete = collect_transient(dataset_episode_dir, cams, profile, resolved=resolved)
    deleted = 0
    deleted_bytes = 0
    for path in to_delete:
        p = Path(path)
        if p.exists():
            try:
                deleted_bytes += p.stat().st_size
                p.unlink()
                deleted += 1
            except OSError:
                pass
    # 清理空 transient 目录
    for cam in cams:
        cam_dir = _cam_dir(dataset_episode_dir, cam)
        for rel in list(_TRANSIENT_CAM_RELS.keys()):
            t = cam_dir / rel
            if t.is_dir() and not any(t.iterdir()):
                try:
                    t.rmdir()
                except OSError:
                    pass
    return {"ok": True, "deleted_files": deleted, "deleted_bytes": deleted_bytes,
            "deleted_gb": round(deleted_bytes / 1e9, 3)}


def build_manifest(dataset_episode_dir, resolved, cams) -> dict:
    """生成 dataset_manifest.json 内容（不写盘，返回 dict）。"""
    dataset_episode_dir = Path(dataset_episode_dir)
    manifest = {
        "dataset_version": 1,
        "episode_id": resolved.get("episode_name"),
        "fps": int((resolved.get("export_profile") or {}).get("target_fps") or 30),
        "frames": int((resolved.get("export_profile") or {}).get("num_steps", 0)
                      * max(1, ((resolved.get("export_profile") or {}).get("target_fps") or 30) // 10)),
        "tempo": float((resolved.get("export_profile") or {}).get("trajectory_time_scale") or 1.0),
        "seed": int((resolved.get("export_profile") or {}).get("seed") or 0),
        "cameras": cams,
        "classes": ["player", "ball"],
        "global_track_id_policy": "L0=1..L4=5,R0=6..R4=10,BALL=100",
        "mot_ball_policy": "include_ball=true (BALL track_id=100)",
        "pose_schema": "limb_pose_3d/2d (12 keypoints, meters/pixels); raw capture keeps 13 bones",
        "artifact_profile": (resolved.get("artifact_policy") or {}).get("profile", "research_minimal"),
        "rgb_count_per_camera": len(list((dataset_episode_dir / cams[0] / "img1").glob("*.png")))
        if cams and (dataset_episode_dir / cams[0] / "img1").is_dir() else 0,
        "annotation_count": len(list((dataset_episode_dir / cams[0] / "annotations.jsonl").exists()
                                     and [0])) if cams else 0,
        "cleanup_status": "pending",
        "source_commit": "see provenance/task.json",
    }
    # 计算最终数据集大小（不含 pending cleanup）
    total = sum(f.stat().st_size for f in dataset_episode_dir.rglob("*") if f.is_file())
    manifest["final_dataset_bytes"] = total
    manifest["final_dataset_gb"] = round(total / 1e9, 3)
    return manifest
