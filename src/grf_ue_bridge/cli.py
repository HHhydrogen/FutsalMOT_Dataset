"""GRF-UE 桥接的 CLI。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .config import ExportConfig
from .exporter import export_episode
from .grf_runner import run_episode
from .seeds import derive_episode_seeds
from .validator import validate_episode

app = typer.Typer()


def _draw_frame_overlay(img, objects: list, include_ball: bool):
    """把一帧 objects 的 bbox + 标签画到 PIL Image 上，返回新 Image（annotate-overlay / make-video 共用）。"""
    from PIL import Image, ImageDraw

    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for obj in objects:
        if not obj.get("in_frame"):
            continue
        if obj.get("class") == "ball" and not include_ball:
            continue
        xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
        color = (0, 255, 0) if obj.get("class") == "player" else (255, 128, 0)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        label = f"{obj['entity_id']} #{obj['track_id']}"
        draw.text((xmin, max(0, ymin - 14)), label, fill=color)
    return img


def _draw_overlay(camera_dataset_dir: Path, include_ball: bool, mask_color: bool = False) -> int:
    """把标注 bbox 绘制到 img1/ 中对应的 RGB 帧上，输出到 debug/。

    mask_color=True 时额外把 mask/*.png 转成彩色可视化到 debug/{frame}_mask_color.png
    （仅查看，不改写 mask 数据契约）。

    需要 pillow（可选依赖，`uv sync --extra overlay` 安装）。无 RGB 帧时跳过。
    """
    try:
        from PIL import Image
    except ImportError:
        typer.echo(
            "需要 pillow：请运行 `uv sync --extra overlay` 或 `uv pip install pillow`",
            err=True,
        )
        return 1

    ann_path = camera_dataset_dir / "annotations.jsonl"
    img_dir = camera_dataset_dir / "img1"
    if not ann_path.exists() or not img_dir.exists():
        typer.echo(f"ERROR: 缺少 annotations.jsonl 或 img1/: {camera_dataset_dir}", err=True)
        return 1

    out_dir = camera_dataset_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    drawn = 0
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            frame_index = frame["frame_index"]
            img_path = img_dir / f"{frame_index:06d}.png"
            if not img_path.exists():
                continue
            img = _draw_frame_overlay(Image.open(img_path), frame.get("objects", []), include_ball)
            out_path = out_dir / f"{frame_index:06d}_bbox.png"
            img.save(out_path)
            drawn += 1

    # 彩色 mask 可视化（仅查看）：mask_id 1..11 → 固定鲜艳调色板，与背景区分
    if mask_color:
        try:
            import sys
            from pathlib import Path

            _ue_dir = Path(__file__).resolve().parent.parent.parent / "ue"
            if str(_ue_dir) not in sys.path:
                sys.path.insert(0, str(_ue_dir))
            from instance_mask import load_mask_array, mask_to_color_image
        except ImportError:
            typer.echo("WARNING: 无法 import instance_mask（缺 numpy），跳过 mask 彩色输出", err=True)
            mask_color = False
    mask_drawn = 0
    if mask_color:
        mask_dir = camera_dataset_dir / "mask"
        if mask_dir.exists():
            for p in sorted(mask_dir.glob("*.png")):
                arr = load_mask_array(p, "r")
                col = Image.fromarray(mask_to_color_image(arr))
                col.save(out_dir / f"{p.stem}_mask_color.png")
                mask_drawn += 1
        else:
            typer.echo(f"  (无 mask/ 目录，跳过彩色 mask 输出: {camera_dataset_dir})")

    typer.echo(f"Overlay 完成: {drawn} 帧 bbox -> {out_dir}"
               + (f"；{mask_drawn} 帧彩色 mask -> {out_dir}" if mask_drawn else ""))
    return 0


@app.command()
def export(
    config: Path = typer.Option(
        ...,
        "--config",
        help="导出配置文件 JSON 的路径",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="episode 的输出目录",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="覆盖配置文件的 root seed（优先级：CLI > 配置文件 > 默认值）",
    ),
):
    """运行一个 GRF episode 并导出为 GRF-UE JSONL 格式。"""
    # 加载配置
    if not config.exists():
        typer.echo(f"ERROR: Config file not found: {config}", err=True)
        raise typer.Exit(1)

    # 显式 UTF-8：配置文件可能含中文注释
    with open(config, encoding="utf-8") as f:
        config_data = json.load(f)
    export_cfg = ExportConfig(**config_data)

    # CLI --seed 覆盖配置文件的 seed（不修改磁盘上的配置文件）
    if seed is not None and seed != export_cfg.seed:
        export_cfg = export_cfg.model_copy(update={"seed": seed})

    typer.echo(f"Export config: {export_cfg.model_dump_json(indent=2)}")
    typer.echo(f"Running episode: scenario={export_cfg.scenario}, steps={export_cfg.num_steps}")

    # 派生并打印完整种子信息（root / GRF game-engine / policy）
    seeds = derive_episode_seeds(export_cfg.seed)
    typer.echo(
        f"Seed: root={seeds.root_seed}  "
        f"grf_game_engine={seeds.grf_game_engine_seed}  "
        f"policy={seeds.policy}"
    )

    # 运行 episode
    result = run_episode(
        scenario=export_cfg.scenario,
        seed=export_cfg.seed,
        num_steps=export_cfg.num_steps,
        render=export_cfg.render,
        game_duration=export_cfg.game_duration,
        left_team_difficulty=export_cfg.left_team_difficulty,
        right_team_difficulty=export_cfg.right_team_difficulty,
        number_of_left_players_agent_controls=export_cfg.number_of_left_players_agent_controls,
        number_of_right_players_agent_controls=export_cfg.number_of_right_players_agent_controls,
    )

    typer.echo(f"Episode complete: {len(result.snapshots)} snapshots")
    typer.echo(f"Final score: {result.score}")

    # 导出
    output_path = output if output.is_absolute() else Path.cwd() / output
    export_episode(export_cfg, result, output_path)

    typer.echo(f"Export complete: {output_path}")
    typer.echo(f"  meta.json: {(output_path / 'meta.json').stat().st_size} bytes")
    typer.echo(f"  frames.jsonl: {(output_path / 'frames.jsonl').stat().st_size} bytes")


@app.command()
def validate(
    episode_dir: Path = typer.Argument(
        ...,
        help="包含 meta.json 和 frames.jsonl 的 episode 目录",
    ),
):
    """验证一个导出的 episode 目录。"""
    exit_code = validate_episode(episode_dir.resolve())
    if exit_code != 0:
        typer.echo("VALIDATION FAILED", err=True)
        raise typer.Exit(exit_code)
    typer.echo("VALIDATION PASSED")


@app.command("build-manifest")
def build_manifest_cmd(
    dataset_root: Path = typer.Argument(
        ...,
        help="数据集根目录（manifest 写入 <root>/dataset_manifest.json，校验和写入 <root>/checksums/）",
    ),
    episode: List[str] = typer.Option(
        None,
        "--episode",
        help="要纳入的 episode id（可重复指定）。缺省只纳入满足合法 episode 结构（含 camera.json）的目录",
    ),
    dataset_id: str = typer.Option(
        "futsalmot_local_v001",
        "--dataset-id",
        help="数据集标识（不进 fingerprint）",
    ),
    checksum_profile: str = typer.Option(
        "final",
        "--checksum-profile",
        help="checksum profile：metadata / final（默认）/ all",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        help="并行 hash worker 数（磁盘 I/O 瓶颈，保守默认 4）",
    ),
    hash_chunk_size_mb: int = typer.Option(
        1,
        "--hash-chunk-size-mb",
        help="流式 SHA-256 的 chunk 大小（MB）",
    ),
    strict_duplicates: bool = typer.Option(
        False,
        "--strict-duplicates",
        help="检测到重复轨迹时以非零退出码结束",
    ),
):
    """构建数据集级 manifest（索引、校验和、去重检测、fingerprint）。"""
    from .dataset_manifest import build_manifest

    manifest = build_manifest(
        dataset_root.resolve(),
        episode_ids=episode,
        dataset_id=dataset_id,
        checksum_profile=checksum_profile,
        workers=workers,
        chunk_mb=hash_chunk_size_mb,
    )
    typer.echo(f"Manifest 构建完成: {dataset_root / 'dataset_manifest.json'}")
    typer.echo(f"  dataset_id: {manifest.dataset_id}")
    typer.echo(f"  episodes: {len(manifest.episodes)}")
    for e in manifest.episodes:
        typer.echo(
            f"    {e.episode_id}: {e.camera_count} 相机, "
            f"{e.frames_per_camera} 帧/相机, root_seed={e.root_seed}, "
            f"trajectory={e.content_hashes.get('trajectory_hash', 'N/A')[:12]}"
        )
    typer.echo(f"  totals: rgb={manifest.totals.rgb_final} mask={manifest.totals.instance_mask} "
               f"annotation_frames={manifest.totals.annotation_frames} "
               f"raw_rgb={manifest.totals.raw_rgb} raw_exr={manifest.totals.raw_object_id_exr}")
    typer.echo(f"  duplicate_seed_groups: {manifest.duplicate_seed_groups}")
    typer.echo(f"  duplicate_trajectory_groups: {manifest.duplicate_trajectory_groups}")
    typer.echo(f"  dataset_fingerprint: {manifest.dataset_fingerprint}")
    for w in manifest.warnings:
        typer.echo(f"  WARNING: {w}", err=True)
    if manifest.duplicate_seed_groups:
        typer.echo("  WARNING: possible duplicate seed/config combination", err=True)
    if manifest.duplicate_trajectory_groups:
        typer.echo("  WARNING: duplicate trajectory detected", err=True)
    if strict_duplicates and manifest.duplicate_trajectory_groups:
        typer.echo("  --strict-duplicates：存在重复轨迹，非零退出", err=True)
        raise typer.Exit(1)


@app.command("verify-manifest")
def verify_manifest_cmd(
    dataset_root: Path = typer.Argument(
        ...,
        help="数据集根目录",
    ),
    manifest: Optional[Path] = typer.Option(
        None,
        "--manifest",
        help="manifest 路径（默认 <root>/dataset_manifest.json）",
    ),
    workers: int = typer.Option(
        4,
        "--workers",
        help="并行校验 worker 数",
    ),
    hash_chunk_size_mb: int = typer.Option(
        1,
        "--hash-chunk-size-mb",
        help="流式 SHA-256 的 chunk 大小（MB）",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="存在未记录的额外文件时也以非零退出码结束",
    ),
):
    """校验 dataset manifest 与实际内容的一致性。"""
    from .dataset_manifest import verify_manifest

    result = verify_manifest(
        dataset_root.resolve(),
        manifest_path=manifest,
        workers=workers,
        chunk_mb=hash_chunk_size_mb,
        strict_extra=strict,
    )
    typer.echo(f"verify-manifest: {'PASS' if result.exit_code == 0 else 'FAIL'} "
               f"(checked={result.checked}, matched={result.matched}, "
               f"missing={len(result.missing)}, size_mismatch={len(result.size_mismatch)}, "
               f"hash_mismatch={len(result.hash_mismatch)}, extra={len(result.extra)})")
    for w in result.warnings:
        typer.echo(f"  WARNING: {w}", err=True)
    for e in result.errors[:20]:
        typer.echo(f"  ERROR: {e}", err=True)
    for m in result.missing[:20]:
        typer.echo(f"  MISSING: {m}", err=True)
    for m in result.size_mismatch[:20]:
        typer.echo(f"  SIZE-MISMATCH: {m}", err=True)
    for m in result.hash_mismatch[:20]:
        typer.echo(f"  HASH-MISMATCH: {m}", err=True)
    for m in result.extra[:20]:
        typer.echo(f"  EXTRA: {m}", err=True)
    if result.exit_code != 0:
        raise typer.Exit(result.exit_code)


@app.command()
def validate_annotations(
    annotation_dir: Path = typer.Argument(
        ...,
        help="标注输出目录（含多个 camera 子目录）",
    ),
    workers: int = typer.Option(
        0, "--workers",
        help="并行 worker 数：0=自动（min(相机数, cpu//2)），1=串行，>1=多进程",
    ),
    validation_level: str = typer.Option(
        "full", "--validation-level",
        help="验证级别：full（完整重新派生并比较，默认）/ quick（结构 + 抽样重算，快）",
    ),
):
    """验证导出的 CV 标注目录。"""
    from .annotation_validator import validate_annotation_dir

    exit_code = validate_annotation_dir(
        annotation_dir.resolve(),
        workers=workers,
        validation_level=validation_level,
    )
    if exit_code != 0:
        typer.echo("ANNOTATION VALIDATION FAILED", err=True)
        raise typer.Exit(exit_code)
    typer.echo("ANNOTATION VALIDATION PASSED")


@app.command()
def annotate_overlay(
    camera_dataset_dir: Path = typer.Argument(
        ...,
        help="单个 camera 的 dataset 目录（含 img1/ 与 annotations.jsonl）",
    ),
    include_ball: bool = typer.Option(
        False, "--include-ball", help="是否绘制球"
    ),
    mask_color: bool = typer.Option(
        False, "--mask-color",
        help="额外输出彩色 mask 可视化到 debug/{frame}_mask_color.png（仅查看，不改写 mask 数据）",
    ),
):
    """把标注 bbox 绘制到 img1/ 中对应的 RGB 帧上，输出到 debug/。需要 pillow。"""
    raise typer.Exit(_draw_overlay(camera_dataset_dir.resolve(), include_ball, mask_color))


@app.command()
def annotate_masks(
    annotation_dir: Path = typer.Argument(
        ...,
        help="标注输出目录（含多个 camera 子目录，每个含 mask/ 与 annotations.jsonl）",
    ),
    mask_channel: str = typer.Option(
        "r", "--mask-channel",
        help="mask PNG 中携带实例 ID 的通道（r/g/b/a/gray），需与 MRQ 输出一致",
    ),
    include_ball: bool = typer.Option(
        False, "--include-ball", help="MOT / YOLO 是否包含球（BALL）",
    ),
    polygon_tolerance_px: float = typer.Option(
        1.0, "--polygon-tolerance-px",
        help="YOLO segmentation 多边形 RDP 简化容差（像素）",
    ),
    max_polygon_points: int = typer.Option(
        64, "--max-polygon-points",
        help="每个实例多边形最大点数（超出均匀抽样）",
    ),
    id_scale: float = typer.Option(
        1.0, "--id-scale",
        help="mask 解码缩放：像素值量化 = round((v - id_offset) / id_scale)",
    ),
    id_offset: float = typer.Option(
        0.0, "--id-offset",
        help="mask 解码偏移：像素值量化 = round((v - id_offset) / id_scale)",
    ),
    workers: int = typer.Option(
        0, "--workers",
        help="并行 worker 数：0=自动（min(相机数, cpu//2)），1=串行，>1=多进程",
    ),
    chunk_size: int = typer.Option(
        0, "--chunk-size",
        help="单相机内帧分块大小（>0 时；相机数少于 worker 数时自动分块）",
    ),
    formats: str = typer.Option(
        "all", "--formats",
        help="导出的派生产物格式：all/mot/yolo-det/yolo-seg/json 或逗号组合",
    ),
    no_segmentation: bool = typer.Option(
        False, "--no-segmentation",
        help="跳过实例分割多边形（轮廓/polygon/桥接/质量检查），不生成 labels/seg/；"
             "bbox、像素数、MOT、YOLO Det 正常生成",
    ),
    clean_stale: bool = typer.Option(
        True, "--clean-stale/--no-clean-stale",
        help="清理与当前 --formats 不符的陈旧派生产物（gt/gt.txt、labels/det、labels/seg），"
             "保证目录反映本次运行的选择；默认开启。--no-clean-stale 保留旧文件",
    ),
):
    """从 Instance-ID Mask 计算 pixel-tight bbox / 分割标注，覆盖写 annotations.jsonl 并导出 MOT / YOLO。

    bbox 由 mask 像素 min/max 直接计算（primary GT），原几何 bbox 保留在
    geometry_bbox_* 字段作为 fallback。原始 mask/*.png 永不修改。

    并行策略：多相机优先相机级并行；单相机（或相机数少于 worker 数）自动按连续
    帧区间分块并行。快速模式示例（仅 MOT + 检测、跳过分割）：
      grf-ue annotate-masks <dir> --formats json,mot,yolo-det --no-segmentation --workers 4
    """
    from .mask_annotator import annotate_masks_dir

    exit_code = annotate_masks_dir(
        annotation_dir.resolve(),
        mask_channel=mask_channel,
        include_ball=include_ball,
        polygon_tolerance_px=polygon_tolerance_px,
        max_polygon_points=max_polygon_points,
        id_scale=id_scale,
        id_offset=id_offset,
        workers=workers,
        chunk_size=chunk_size,
        formats=formats,
        no_segmentation=no_segmentation,
        clean_stale=clean_stale,
    )
    if exit_code != 0:
        typer.echo("ANNOTATE MASKS FAILED", err=True)
        raise typer.Exit(exit_code)
    typer.echo("ANNOTATE MASKS DONE")


@app.command("annotate-pose")
def annotate_pose(
    annotation_dir: Path = typer.Argument(
        ...,
        help="标注输出目录（含多个 camera 子目录，每个含 pose_keypoints.jsonl、annotations.jsonl、mask/）",
    ),
    workers: int = typer.Option(
        0, "--workers",
        help="并行 worker 数：0=自动（min(相机数, cpu//2)），1=串行，>1=多进程",
    ),
    visibility_neighborhood_radius: int = typer.Option(
        2, "--visibility-neighborhood-radius",
        help="Instance-ID Mask 邻域判定半径（像素），用于 keypoint 遮挡判定",
    ),
    no_yaml: bool = typer.Option(
        False, "--no-yaml",
        help="不生成 episode 根的 yolo_pose/ 可训练暂存目录与 futsal_pose.yaml",
    ),
):
    """从 pose_keypoints.jsonl + annotations.jsonl + mask 生成 YOLO Pose 标签（labels_pose/）。

    bbox 复用 annotations.jsonl 的 mask-primary bbox（与 YOLO det 一致）；
    visibility 由 Instance-ID Mask 邻域 + UE 遮挡 trace（occluded 标志）判定。
    推荐经 `grf-ue task postprocess` 自动集成；本命令用于局部重跑/调试。
    """
    from grf_ue_bridge.pose_annotator import annotate_pose_dir

    exit_code = annotate_pose_dir(
        annotation_dir.resolve(),
        pose_cfg={"visibility_neighborhood_radius": visibility_neighborhood_radius},
        workers=workers,
        write_yaml=not no_yaml,
    )
    if exit_code != 0:
        typer.echo("ANNOTATE POSE FAILED", err=True)
        raise typer.Exit(exit_code)
    typer.echo("ANNOTATE POSE DONE")


@app.command("validate-pose")
def validate_pose(
    annotation_dir: Path = typer.Argument(
        ...,
        help="标注输出目录（含多个 camera 子目录，每个含 labels_pose/）",
    ),
    workers: int = typer.Option(
        0, "--workers",
        help="并行 worker 数：0=自动（min(相机数, cpu//2)），1=串行，>1=多进程",
    ),
    validation_level: str = typer.Option(
        "full", "--validation-level",
        help="验证级别：full（结构 + 逐帧 mask 重算比对，默认）/ quick（结构 + 行数比对，快）",
    ),
    visibility_neighborhood_radius: int = typer.Option(
        2, "--visibility-neighborhood-radius",
        help="重算用邻域半径（须与 annotate-pose 一致）",
    ),
):
    """验证 YOLO Pose 标签：56 字段 / 数值范围 / 帧对应 / 左右轴一致性。"""
    from grf_ue_bridge.pose_validator import validate_pose_dir

    exit_code = validate_pose_dir(
        annotation_dir.resolve(),
        workers=workers,
        validation_level=validation_level,
        visibility_neighborhood_radius=visibility_neighborhood_radius,
    )
    if exit_code != 0:
        typer.echo("POSE VALIDATION FAILED", err=True)
        raise typer.Exit(exit_code)
    typer.echo("POSE VALIDATION PASSED")


@app.command("pose-overlay")
def pose_overlay(
    camera_dataset_dir: Path = typer.Argument(
        ...,
        help="单个 camera 的 dataset 目录（含 img1/、pose_keypoints.jsonl、annotations.jsonl、mask/）",
    ),
    frames: Optional[str] = typer.Option(
        None, "--frames",
        help="只处理指定帧号（1 基，逗号分隔，如 '1,2,3'；缺省全部）",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", help="输出目录（默认 <camera_dir>/debug/pose/）"
    ),
    dot_radius: int = typer.Option(
        5, "--dot-radius", min=1, max=30,
        help="关键点半径（像素），放大预览时调大",
    ),
    edge_width: int = typer.Option(
        3, "--edge-width", min=1, max=20, help="骨架连线宽度（像素）",
    ),
    keypoint_names: bool = typer.Option(
        False, "--keypoint-names", help="在每个关键点旁标注 COCO 名（调试用）",
    ),
):
    """把 Pose 关键点/骨架/bbox 画到 img1/ 的 RGB 帧上，输出 debug/pose/。需要 pillow。

    颜色区分 visibility：绿=可见(v=2)、橙=遮挡(v=1)、红=无效(v=0)。
    """
    from grf_ue_bridge.pose_annotator import pose_overlay_dir

    want_frames = None
    if frames:
        want_frames = [int(x) for x in frames.split(",") if x.strip().isdigit()]
    drawn = pose_overlay_dir(
        camera_dataset_dir.resolve(),
        frames=want_frames,
        out_dir=out.resolve() if out else None,
        dot_radius=dot_radius,
        edge_width=edge_width,
        keypoint_names=keypoint_names,
    )
    if drawn == 0:
        typer.echo("Pose overlay: 无输出（检查 img1/、pose_keypoints.jsonl、annotations.jsonl）", err=True)
        raise typer.Exit(1)
    typer.echo(f"Pose overlay 完成: {drawn} 帧")


# ── make-video：把 img1/ 帧（可选 bbox 叠加）编码成 mp4 标注视频 ────────

def _read_seqinfo_fps(cam_dir: Path) -> Optional[int]:
    """从 seqinfo.ini 读取 frameRate（缺省 None）。"""
    p = cam_dir / "seqinfo.ini"
    if not p.exists():
        return None
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().lower().startswith("framerat"):
                return int(line.split("=", 1)[1].strip())
    except Exception:
        return None
    return None


def _png_frame_numbers(img_dir: Path) -> List[int]:
    """从目录里的 PNG 文件名解析帧号集合（有序）。"""
    nums = []
    for p in img_dir.glob("*.png"):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            nums.append(int(digits))
    return sorted(nums)


def _make_video(camera_dataset_dir: Path, fps: Optional[int], out: Optional[Path],
                plain: bool, include_ball: bool, max_frames: Optional[int]) -> int:
    """把 img1/ 帧编码成 mp4 标注视频（默认叠加 bbox；多视角 = 每相机跑一次）。

    帧顺序取 annotations.jsonl 的 frame_index（默认），否则按 img1/ 文件名排序。
    需要 opencv-python（可选依赖，`uv sync --extra video`）。
    """
    import numpy as np

    try:
        import cv2
    except ImportError:
        typer.echo(
            "需要 opencv-python：请运行 `uv sync --extra video` 或 `uv pip install opencv-python`",
            err=True,
        )
        return 1
    from PIL import Image

    img_dir = camera_dataset_dir / "img1"
    ann_path = camera_dataset_dir / "annotations.jsonl"
    if not img_dir.exists():
        typer.echo(f"ERROR: 缺 img1/: {camera_dataset_dir}", err=True)
        return 1

    if fps is None:
        fps = _read_seqinfo_fps(camera_dataset_dir) or 30
    out_path = out or (camera_dataset_dir / f"video_{fps}fps.mp4")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 帧列表：annotations（默认，带 bbox 信息）或 img1 文件名
    frames: List[dict] = []
    if ann_path.exists() and not plain:
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
    else:
        frames = [{"frame_index": n} for n in _png_frame_numbers(img_dir)]
    if max_frames:
        frames = frames[:max_frames]

    first = next((f for f in frames if (img_dir / f"{f['frame_index']:06d}.png").exists()), None)
    if first is None:
        typer.echo(f"ERROR: 无可用的 img1 帧: {img_dir}", err=True)
        return 1
    with Image.open(img_dir / f"{first['frame_index']:06d}.png") as im:
        W, H = im.size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))
    written = 0
    for fr in frames:
        p = img_dir / f"{fr['frame_index']:06d}.png"
        if not p.exists():
            continue
        img = Image.open(p)
        if not plain:
            img = _draw_frame_overlay(img, fr.get("objects", []), include_ball)
        writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        written += 1
    writer.release()
    typer.echo(f"视频已写: {out_path}（{written} 帧 @ {fps}fps，{'bbox 叠加' if not plain else '原图'}）")
    return 0 if written else 1


@app.command()
def make_video(
    camera_dataset_dir: Path = typer.Argument(
        ...,
        help="单个 camera 的 dataset 目录（含 img1/ 与可选 annotations.jsonl）",
    ),
    fps: Optional[int] = typer.Option(
        None, "--fps", help="视频帧率（默认读 seqinfo.ini frameRate，否则 30）"
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", help="输出 mp4 路径（默认 <camera_dir>/video_<fps>fps.mp4）"
    ),
    plain: bool = typer.Option(
        False, "--plain", help="不画 bbox，直接编码 img1/ 原图"
    ),
    include_ball: bool = typer.Option(
        False, "--include-ball", help="叠加球 bbox"
    ),
    max_frames: Optional[int] = typer.Option(
        None, "--max-frames", help="只编码前 N 帧（smoke 用）"
    ),
):
    """把 img1/ 帧编码成 mp4 标注视频（默认叠加 bbox）。多视角 = 对每个 camera 目录各跑一次。"""
    raise typer.Exit(_make_video(camera_dataset_dir.resolve(), fps, out, plain, include_ball, max_frames))


@app.command()
def cryptomatte_to_mask(
    annotation_dir: Path = typer.Argument(
        ...,
        help="标注输出目录（含多个 camera 子目录，每个含 render_mask/*.exr）",
    ),
    mapping: Path = typer.Option(
        None, "--mapping", help="actor 映射 JSON（必需；推荐用 grf-ue task postprocess）",
    ),
    episode: Path = typer.Option(
        None, "--episode", help="episode 目录，读时序（必需；推荐用 grf-ue task postprocess）",
    ),
    workers: int = typer.Option(
        0, "--workers",
        help="并行 worker 数：0=自动，1=串行，>1=帧级多进程并行",
    ),
    chunk_size: int = typer.Option(
        0, "--chunk-size",
        help="进程池批大小（>0 时，传给 executor.map 的 chunksize）",
    ),
    png_compress_level: int = typer.Option(
        1, "--png-compress-level",
        help="实例 ID mask PNG 压缩等级 0–9（1 为性能推荐值，像素不变）",
    ),
):
    """把 Object ID Pass 的 Cryptomatte multilayer EXR 转为 mask/{frame}.png（mask_id 值）。

    转换后即可用 `grf-ue annotate-masks` 生成 mask-primary bbox/分割标注。
    """
    import json as _json

    from .cryptomatte import convert_render_mask_dir  # 先 import（会补 ue/ 到 sys.path）
    from dataset_export import load_episode, load_mapping  # noqa: E402

    # 不再读取根目录隐式 ue_import_config.json：episode/mapping 必须显式提供
    #（推荐直接用 `grf-ue task postprocess <task>`，由 resolver 提供路径）。
    mapping_path = mapping
    episode_path = episode
    if mapping_path is None or episode_path is None \
            or not mapping_path.exists() or not episode_path.exists():
        typer.echo(
            "需要显式 --mapping 与 --episode（不再从根目录 ue_import_config.json 推断）。"
            "推荐：uv run grf-ue task postprocess <task>",
            err=True,
        )
        raise typer.Exit(1)
    meta, _frames = load_episode(episode_path)
    num_steps = int(meta["timing"]["num_steps"])
    step_sec = float(meta["timing"]["source_step_seconds"])
    fps = int(meta["timing"].get("playback_fps", 30))
    mapping_dict = load_mapping(mapping_path)

    cam_dirs = sorted(d.parent for d in annotation_dir.rglob("camera.json"))
    if not cam_dirs:
        typer.echo(f"ERROR: {annotation_dir} 下没有 camera 子目录", err=True)
        raise typer.Exit(1)
    total_ok = 0
    for cam in cam_dirs:
        rmask = cam / "render_mask"
        mdir = cam / "mask"
        if not rmask.exists():
            typer.echo(f"  SKIP {cam.name}: 无 render_mask/")
            continue
        status, per = convert_render_mask_dir(
            rmask, mapping_dict, mdir, num_steps, step_sec, fps,
            png_compress_level=png_compress_level,
            workers=workers,
            chunk_size=chunk_size,
        )
        typer.echo(f"  [{status.upper()}] {cam.name}: {len(per)} 帧 mask 已生成")
        if status == "success":
            total_ok += 1
    typer.echo(f"cryptomatte-to-mask 完成（{total_ok}/{len(cam_dirs)} camera）")
    if total_ok == 0:
        raise typer.Exit(1)


# ── task 工作流 CLI（推荐入口）────────────────────────────────────────

from grf_ue_bridge.config import paths as _cfg_paths
from grf_ue_bridge.config import resolver as _resolver

task_app = typer.Typer(help="基于 dataset task 配置的工作流（推荐入口）")
app.add_typer(task_app, name="task")


def _resolve_task_or_active(task: Optional[Path]) -> Path:
    """显式 task 优先；否则用 active task（无 active 时报错）。"""
    if task is not None:
        return Path(task).expanduser().resolve()
    repo_root = _cfg_paths.default_repo_root()
    active = _resolver.load_active_task(repo_root)
    if active is None:
        typer.echo(
            "未提供 task 且无 active task。请先 `grf-ue task activate <task>` "
            "或显式传入 task 文件。",
            err=True,
        )
        raise typer.Exit(2)
    typer.echo(f"Active task: {active.stem}")
    typer.echo(f"Task file: {active}")
    return active


def _resolve_runtime(task: Optional[Path]):
    task_file = _resolve_task_or_active(task)
    return task_file, _resolver.resolve_task(task_file)


@task_app.command("validate")
def task_validate(
    task: Optional[Path] = typer.Argument(
        None, help="task 文件（缺省用 active task）"
    ),
):
    """只读校验 task（schema/机器路径/相机/帧数）。不生成文件。"""
    task_file = _resolve_task_or_active(task)
    problems = _resolver.validate_task(task_file)
    if problems:
        typer.echo(f"task validate: FAIL ({len(problems)} 项)")
        for p in problems:
            typer.echo(f"  - {p}")
        raise typer.Exit(1)
    typer.echo(f"task validate: PASS  ({task_file})")


@task_app.command("resolve")
def task_resolve(
    task: Optional[Path] = typer.Argument(
        None, help="task 文件（缺省用 active task）"
    ),
):
    """解析 task → 保存 resolved task，打印关键字段。"""
    task_file, resolved = _resolve_runtime(task)
    runtime_file = _resolver.save_resolved_task(
        resolved, Path(resolved.repo_root)
    )
    cam_count = len((resolved.ue_profile.get("annotation_export") or {}).get("cameras") or [])
    typer.echo(f"Task ID: {resolved.task_id}")
    typer.echo(f"Trajectory output: {resolved.trajectory_output}")
    typer.echo(f"Dataset output: {resolved.dataset_episode_dir}")
    typer.echo(f"Export: scenario={resolved.export_profile.get('scenario')} "
               f"steps={resolved.export_profile.get('num_steps')} seed={resolved.export_profile.get('seed')}")
    typer.echo(f"UE: {cam_count} cameras")
    typer.echo(f"Expected frame count: {resolved.audit.get('expected_frames_per_camera')}")
    typer.echo(f"Postprocess formats: {resolved.postprocess.get('formats')}")
    typer.echo(f"Resolved task saved: {runtime_file}")


@task_app.command("export")
def task_export(
    task: Optional[Path] = typer.Argument(
        None, help="task 文件（缺省用 active task）"
    ),
):
    """按 task 导出轨迹（复用现有 exporter），写 provenance。"""
    from grf_ue_bridge.workflows.task_export import run_export

    _task_file, resolved = _resolve_runtime(task)
    rc = run_export(resolved, print_fn=typer.echo)
    if rc != 0:
        raise typer.Exit(rc)


@task_app.command("ue-command")
def task_ue_command(
    task: Optional[Path] = typer.Argument(
        None, help="task 文件（缺省用 active task）"
    ),
):
    """输出可在 Unreal Editor Python Console 复制的命令（先保存 resolved task）。"""
    task_file, resolved = _resolve_runtime(task)
    runtime_file = _resolver.save_resolved_task(
        resolved, Path(resolved.repo_root)
    )
    run_task = Path(resolved.repo_root) / "ue" / "run_task.py"
    typer.echo(
        f'py "{run_task}" --resolved-task "{runtime_file}"'
    )


@task_app.command("postprocess")
def task_postprocess(
    task: Optional[Path] = typer.Argument(
        None, help="task 文件（缺省用 active task）"
    ),
    skip_cryptomatte: bool = typer.Option(False, "--skip-cryptomatte"),
    skip_annotate: bool = typer.Option(False, "--skip-annotate"),
    skip_validate: bool = typer.Option(False, "--skip-validate"),
    skip_pose: bool = typer.Option(False, "--skip-pose"),
):
    """按 task 顺序执行 cryptomatte → annotate → validate →（可选）yolo pose。"""
    from grf_ue_bridge.workflows.task_postprocess import run_postprocess

    _task_file, resolved = _resolve_runtime(task)
    rc = run_postprocess(
        resolved,
        skip_cryptomatte=skip_cryptomatte,
        skip_annotate=skip_annotate,
        skip_validate=skip_validate,
        skip_pose=skip_pose,
        print_fn=typer.echo,
    )
    if rc != 0:
        raise typer.Exit(rc)


@task_app.command("audit")
def task_audit(
    task: Optional[Path] = typer.Argument(
        None, help="task 文件（缺省用 active task）"
    ),
    validation_level: str = typer.Option(
        "quick", "--validation-level", help="进程内 validate 级别（quick/full/none）"
    ),
):
    """对 task 的数据集目录运行完整性审计。"""
    from grf_ue_bridge.workflows.task_audit import main as audit_main

    _task_file, resolved = _resolve_runtime(task)
    audit_cfg = resolved.audit
    rc = audit_main([
        "--input", resolved.dataset_episode_dir,
        "--expected-cameras", str(audit_cfg.get("expected_cameras", 4)),
        "--expected-frames-per-camera", str(audit_cfg.get("expected_frames_per_camera", 300)),
        "--episode", resolved.trajectory_output,
        "--validation-level", validation_level,
    ])
    if rc != 0:
        raise typer.Exit(rc)


@task_app.command("status")
def task_status(
    task: Optional[Path] = typer.Argument(
        None, help="task 文件（缺省用 active task）；空参数时也可只查 active"
    ),
):
    """只读显示任务各产物状态（不修改文件）。"""
    from grf_ue_bridge.workflows.task_status import collect_status, print_status

    _task_file, resolved = _resolve_runtime(task)
    st = collect_status(resolved)
    print_status(resolved, st, print_fn=typer.echo)


@task_app.command("activate")
def task_activate(
    task: Path = typer.Argument(..., help="task 文件"),
):
    """激活 task（可选便利；显式 task 参数始终优先）。"""
    repo_root = _cfg_paths.default_repo_root()
    path = _resolver.save_active_task(task, repo_root)
    typer.echo(f"Active task set: {task}  ->  {path}")


@task_app.command("deactivate")
def task_deactivate():
    """清除 active task。"""
    repo_root = _cfg_paths.default_repo_root()
    _resolver.clear_active_task(repo_root)
    typer.echo("Active task cleared.")


# ── 工具命令 ───────────────────────────────────────────────────────────

@app.command("monitor")
def monitor_cmd(
    task: Optional[Path] = typer.Argument(
        None, help="task 文件（缺省用 active task；监控其 dataset 目录）"
    ),
    interval: float = typer.Option(30.0, "--interval"),
    out: Path = typer.Option(Path("soak_resources.csv"), "--out"),
):
    """渲染期间资源/目录增长监控（按 task 的 dataset 目录）。"""
    from grf_ue_bridge.tools.resource_monitor import main as mon_main

    _task_file, resolved = _resolve_runtime(task)
    rc = mon_main(["--input", resolved.dataset_episode_dir,
                   "--interval", str(interval), "--out", str(out)])
    if rc != 0:
        raise typer.Exit(rc)


@app.command(
    "measure",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def measure_cmd(ctx: typer.Context):
    """运行任意命令并报告墙钟时间 + 进程树峰值 RSS。"""
    from grf_ue_bridge.tools.process_measure import main as meas_main

    rc = meas_main(ctx.args)
    if rc != 0:
        raise typer.Exit(rc)


@app.command(
    "benchmark",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
def benchmark_cmd(ctx: typer.Context):
    """后处理性能基准（透传 benchmark_postprocess 参数）。"""
    from grf_ue_bridge.tools.benchmark_postprocess import main as bench_main

    rc = bench_main(ctx.args)
    if rc != 0:
        raise typer.Exit(rc)


if __name__ == "__main__":
    app()
