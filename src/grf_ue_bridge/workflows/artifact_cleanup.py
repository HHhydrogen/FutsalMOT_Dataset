"""Dataset Artifact Lifecycle Cleanup（C6-P1.6）。

按 artifact_policy.profile 清理 transient 产物：
  - 默认 research_minimal：清理 render/FinalImage 重复、BurnIn、debug overlay、
    mask EXR、mask PNG、raw pose_capture、render scratch、SaveGame slots。
  - 保留 canonical：RGB img1、camera calibration、frames/meta、annotations、
    MOT、COCO17 2D/3D、YOLO det/seg/pose labels、provenance、audit、manifest。

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

RGB_SUFFIXES = {".png", ".jpg", ".jpeg"}

# 每 camera 下 transient 的相对路径（相对 cam_dir）
_TRANSIENT_CAM_RELS = {
    "render": None,          # render/* （FinalImage + BurnIn）
    "render_mask": None,     # render_mask/*.exr（Cryptomatte EXR）
    "debug": None,           # debug/* overlay
    "mask": "*.png",         # mask/*.png（research_minimal 可删）
}
# global transient 相对 episode 目录
_TRANSIENT_EP_RELS = {
    "pose_capture.jsonl": None,  # raw pose（coco17_3d 已验证后可删）
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


def collect_transient(dataset_episode_dir: Path, cams, profile="research_minimal"):
    """收集将删除的文件（canonical 永不包含）。返回 {path: bytes}。"""
    to_delete = {}
    for cam in cams:
        cam_dir = _cam_dir(dataset_episode_dir, cam)
        if not cam_dir.is_dir():
            continue
        for rel, pat in _TRANSIENT_CAM_RELS.items():
            target = cam_dir / rel
            if target.is_dir():
                for f in target.rglob("*"):
                    if f.is_file() and (pat is None or f.name.endswith(pat.replace("*", ""))):
                        to_delete[str(f)] = f.stat().st_size
            elif target.is_file():
                to_delete[str(target)] = target.stat().st_size
    # global transient（research_minimal：raw pose 在 coco17_3d 验证后可删）
    for rel, pat in _TRANSIENT_EP_RELS.items():
        target = dataset_episode_dir / rel
        if target.is_file():
            to_delete[str(target)] = target.stat().st_size
    # yolo_* 复制的 RGB images（YOLO 配置应引用原始 img1，不复制）
    for ydir in _YOLO_IMAGE_DIRS:
        t = dataset_episode_dir / ydir
        if t.is_dir():
            for f in t.rglob("*"):
                if f.is_file() and f.suffix.lower() in RGB_SUFFIXES:
                    to_delete[str(f)] = f.stat().st_size
    return to_delete


def _requested_contract(resolved=None):
    """读取 resolved Config v3 的公开 annotations/classes 请求。"""
    if resolved is None:
        return None
    if hasattr(resolved, "config_v3"):
        contract = resolved.config_v3 or {}
    elif isinstance(resolved, dict):
        contract = resolved.get("config_v3") or resolved
    else:
        return None
    if not isinstance(contract, dict) or not contract.get("annotations"):
        return None
    return set(contract["annotations"]), set(contract.get("classes") or ())


def _validation_gate(dataset_episode_dir: Path, resolved=None) -> list:
    """返回阻止 cleanup 的原因列表（空 = 可通过）。"""
    problems = []
    requested = _requested_contract(resolved)
    annotations = requested[0] if requested is not None else None
    public_output = (dataset_episode_dir / "episode_manifest.json").is_file()
    if public_output:
        try:
            from grf_ue_bridge.public_validator import validate_public_episode
            result = validate_public_episode(dataset_episode_dir, resolved_task=resolved)
            if not result.ok:
                problems.extend(f"public validation failed: {error}" for error in result.errors)
        except Exception as exc:
            problems.append(f"public validation exception: {exc}")
    if annotations is None or annotations:
        rs_path = dataset_episode_dir / "render_summary.json"
        if rs_path.exists():
            try:
                rs = json.loads(rs_path.read_text(encoding="utf-8"))
                if rs.get("status") != "success":
                    problems.append(f"render_summary.status != success ({rs.get('status')})")
            except Exception as e:
                problems.append(f"render_summary.json 解析失败: {e}")
        else:
            problems.append("缺少 render_summary.json（渲染未成功/未完成）")
    if annotations is None or "pose" in annotations:
        ps_path = dataset_episode_dir / "pose_session.json"
        if ps_path.exists():
            try:
                ps = json.loads(ps_path.read_text(encoding="utf-8"))
                if not ps.get("capture_complete"):
                    problems.append("pose_session.capture_complete != true")
            except Exception as e:
                problems.append(f"pose_session.json 解析失败: {e}")
        else:
            problems.append("缺少 pose_session.json（Runtime Pose 未导出）")
    # 若 audit 报告存在但 FAIL，阻止
    audit_path = dataset_episode_dir / "audit" / "soak_audit_report.json"
    if audit_path.exists():
        try:
            ar = json.loads(audit_path.read_text(encoding="utf-8"))
            if ar.get("ok") is False or ar.get("failed_checks"):
                problems.append("audit 未通过（soak_audit_report ok=false）")
        except Exception:
            if public_output:
                problems.append("audit 报告解析失败")
    return problems


def plan_cleanup(dataset_episode_dir, cams, profile="research_minimal", dry_run=True, resolved=None):
    """计算 cleanup 计划。返回 dict 报告。"""
    dataset_episode_dir = Path(dataset_episode_dir)
    gate_problems = _validation_gate(dataset_episode_dir, resolved)
    to_delete = collect_transient(dataset_episode_dir, cams, profile)

    delete_bytes = sum(to_delete.values())
    # keep = 全部文件 - 待删
    keep_files = 0
    keep_bytes = 0
    for f in dataset_episode_dir.rglob("*"):
        if f.is_file():
            if str(f) not in to_delete:
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
    }
    return report


def apply_cleanup(dataset_episode_dir, cams, profile="research_minimal", resolved=None):
    """执行 cleanup。返回结果 dict。先过 validation gate，失败则拒绝。"""
    dataset_episode_dir = Path(dataset_episode_dir)
    gate_problems = _validation_gate(dataset_episode_dir, resolved)
    if gate_problems:
        return {"ok": False, "reason": "validation_gate_failed",
                "gate_problems": gate_problems, "deleted_files": 0, "deleted_bytes": 0}
    to_delete = collect_transient(dataset_episode_dir, cams, profile)
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
        "pose_schema": "coco17_3d/2d (17 keypoints, meters/pixels)",
        "artifact_profile": (resolved.get("artifact_policy") or {}).get("profile", "research_minimal"),
        "rgb_count_per_camera": sum(
            1 for p in (dataset_episode_dir / cams[0] / "img1").iterdir()
            if p.is_file() and p.suffix.lower() in RGB_SUFFIXES
        )
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

