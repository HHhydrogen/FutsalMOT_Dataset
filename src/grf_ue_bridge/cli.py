"""GRF-UE 桥接的 CLI。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .config import ExportConfig
from .exporter import export_episode
from .grf_runner import run_episode
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

    typer.echo(f"Export config: {export_cfg.model_dump_json(indent=2)}")
    typer.echo(f"Running episode: scenario={export_cfg.scenario}, steps={export_cfg.num_steps}")

    # 运行 episode
    result = run_episode(
        scenario=export_cfg.scenario,
        seed=export_cfg.seed,
        num_steps=export_cfg.num_steps,
        render=export_cfg.render,
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
        None, "--mapping", help="actor 映射 JSON；缺省读 ue_import_config.json",
    ),
    episode: Path = typer.Option(
        None, "--episode", help="episode 目录（读时序）；缺省读 ue_import_config.json",
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

    cfg = {}
    cfg_path = Path(__file__).resolve().parent.parent.parent / "ue_import_config.json"
    if cfg_path.exists():
        raw = _json.load(open(cfg_path, encoding="utf-8"))
        cfg = {k: v for k, v in raw.items() if not k.startswith("comment_")}
    mapping_path = mapping or Path(cfg.get("mapping", ""))
    episode_path = episode or Path(cfg.get("episode", ""))
    if not mapping_path.exists() or not episode_path.exists():
        typer.echo(
            "需要 --mapping 与 --episode（或 ue_import_config.json 中的映射/episode）",
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


if __name__ == "__main__":
    app()
