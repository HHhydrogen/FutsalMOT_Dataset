"""task 工作流：后处理（cryptomatte → annotate → 可选 validate）。

按 task 参数顺序调用现有命令，非自动门禁；支持跳过单阶段便于局部重跑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from grf_ue_bridge.config import models as m
from grf_ue_bridge.task_requirements import resolve_task_requirements


def _cryptomatte_camera(
    cam: Path,
    mapping_dict: dict,
    num_steps: int,
    step_sec: float,
    fps: int,
    pp: dict,
    print_fn: Callable[[str], None],
) -> bool:
    from grf_ue_bridge.cryptomatte import convert_render_mask_dir

    rmask = cam / "render_mask"
    mdir = cam / "mask"
    if not rmask.exists():
        print_fn(f"  SKIP {cam.name}: 无 render_mask/")
        return True
    status, per = convert_render_mask_dir(
        rmask, mapping_dict, mdir, num_steps, step_sec, fps,
        png_compress_level=pp.get("png_compress_level", 1),
        workers=pp.get("workers", 4),
        chunk_size=pp.get("chunk_size", 0),
    )
    print_fn(f"  [{status.upper()}] {cam.name}: {len(per)} 帧 mask 已生成")
    return status == "success"


def run_postprocess(
    resolved: m.ResolvedTask,
    *,
    skip_cryptomatte: bool = False,
    skip_annotate: bool = False,
    skip_validate: bool = False,
    skip_pose: bool = False,
    skip_debug: bool = False,
    print_fn: Callable[[str], None] = print,
) -> int:
    """按 resolved task 执行后处理，返回退出码（0=成功）。"""
    pp = resolved.postprocess
    requirements = resolve_task_requirements(resolved)
    dataset = Path(resolved.dataset_episode_dir)
    traj = Path(resolved.trajectory_output)
    mapping = Path(resolved.actor_mapping)

    yolo_pose = pp.get("yolo_pose") or {}
    pose_enabled = bool(yolo_pose.get("enabled", False))
    debug_cfg = pp.get("debug") or {}
    debug_enabled = bool(debug_cfg.get("enabled", False))

    print_fn(f"Postprocess task: {resolved.task_id}")
    print_fn(f"  dataset: {dataset}")
    print_fn(f"  yolo_pose enabled: {pose_enabled}")
    print_fn(f"  debug enabled: {debug_enabled}")
    if not requirements.requires_mot:
        print_fn(
            "  MOT requirement disabled: missing gt/gt.txt will be skipped; "
            "an existing MOT file will still be validated"
        )

    # 1) Cryptomatte EXR → mask
    if not skip_cryptomatte:
        # 先 import cryptomatte（会补 ue/ 到 sys.path），再导入 dataset_export
        from grf_ue_bridge.cryptomatte import convert_render_mask_dir  # noqa: F401
        from dataset_export import load_episode, load_mapping  # noqa: E402

        if not traj.is_dir() or not (traj / "meta.json").is_file():
            print_fn(f"ERROR: trajectory 输出不存在（先运行 task export）: {traj}", )
            return 1
        meta, _ = load_episode(traj)
        num_steps = int(meta["timing"]["num_steps"])
        step_sec = float(meta["timing"]["source_step_seconds"])
        fps = int(meta["timing"].get("playback_fps", 30))
        mapping_dict = load_mapping(mapping)

        cams = sorted(d.parent for d in dataset.rglob("camera.json")) if dataset.is_dir() else []
        if not cams:
            print_fn(f"ERROR: 数据集目录下没有 camera 子目录: {dataset}")
            return 1
        total_ok = 0
        for cam in cams:
            if _cryptomatte_camera(cam, mapping_dict, num_steps, step_sec, fps, pp, print_fn):
                total_ok += 1
        print_fn(f"cryptomatte-to-mask 完成（{total_ok}/{len(cams)} camera）")
        if total_ok == 0:
            return 1

    # 2) annotate-masks
    if not skip_annotate:
        from grf_ue_bridge.mask_annotator import annotate_masks_dir

        formats = ",".join(pp.get("formats") or ["all"]) or "all"
        rc = annotate_masks_dir(
            dataset,
            include_ball=pp.get("include_ball", True),
            workers=pp.get("workers", 4),
            chunk_size=pp.get("chunk_size", 50),
            formats=formats,
            clean_stale=pp.get("clean_stale", True),
        )
        print_fn(f"annotate-masks 完成（exit={rc}）")
        if rc != 0:
            return rc

    # 3) 可选 validate-annotations
    if not skip_validate:
        from grf_ue_bridge.annotation_validator import validate_annotation_dir

        level = pp.get("validation_level", "full")
        rc = validate_annotation_dir(
            dataset,
            workers=pp.get("workers", 4),
            validation_level=level,
            require_mot=requirements.requires_mot,
            require_mask=requirements.requires_instance_mask,
            require_yolo_det=requirements.requires_yolo_det,
            require_yolo_seg=requirements.requires_yolo_seg,
        )
        print_fn(f"validate-annotations({level}) 完成（exit={rc}）")
        if rc != 0:
            return rc

    # 4) 可选 YOLO Pose（postprocess.yolo_pose.enabled）
    if pose_enabled and not skip_pose:
        from grf_ue_bridge.pose_annotator import annotate_pose_dir

        print_fn("annotate-pose（YOLO Pose COCO 17 点）...")
        rc = annotate_pose_dir(
            dataset,
            pose_cfg=yolo_pose,
            workers=pp.get("workers", 4),
            write_yaml=bool(yolo_pose.get("write_dataset_yaml", True)),
        )
        print_fn(f"annotate-pose 完成（exit={rc}）")
        if rc != 0:
            return rc
        from grf_ue_bridge.pose_validator import validate_pose_dir

        rc = validate_pose_dir(
            dataset,
            workers=pp.get("workers", 4),
            validation_level=pp.get("validation_level", "full"),
            visibility_neighborhood_radius=int(yolo_pose.get("visibility_neighborhood_radius", 2)),
        )
        print_fn(f"validate-pose 完成（exit={rc}）")
        if rc != 0:
            return rc

    # 5) 可选 debug 全量可视化（postprocess.debug.enabled）：bbox / 彩色 mask / pose + 视频
    if debug_enabled and not skip_debug:
        from grf_ue_bridge.debug import debug_annotations_dir

        print_fn("debug 可视化（bbox / 彩色 mask / pose 关节点 + 三套视频）...")
        rc = debug_annotations_dir(dataset, cfg=debug_cfg, print_fn=print_fn)
        print_fn(f"debug 可视化完成（exit={rc}）")
        if rc != 0:
            return rc

    return 0
