"""GRF-UE 桥接的 CLI。"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import ExportConfig
from .exporter import export_episode
from .grf_runner import run_episode
from .validator import validate_episode

app = typer.Typer()


def _draw_overlay(camera_dataset_dir: Path, include_ball: bool) -> int:
    """把标注 bbox 绘制到 img1/ 中对应的 RGB 帧上，输出到 debug/。

    需要 pillow（可选依赖，`uv sync --extra overlay` 安装）。无 RGB 帧时跳过。
    """
    try:
        from PIL import Image, ImageDraw
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
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for obj in frame.get("objects", []):
                if not obj.get("in_frame"):
                    continue
                if obj.get("class") == "ball" and not include_ball:
                    continue
                xmin, ymin, xmax, ymax = obj["bbox_xyxy"]
                color = (0, 255, 0) if obj.get("class") == "player" else (255, 128, 0)
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
                label = f"{obj['entity_id']} #{obj['track_id']}"
                draw.text((xmin, max(0, ymin - 14)), label, fill=color)
            out_path = out_dir / f"{frame_index:06d}_bbox.png"
            img.save(out_path)
            drawn += 1
    typer.echo(f"Overlay 完成: {drawn} 帧 -> {out_dir}")
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
):
    """验证导出的 CV 标注目录。"""
    from .annotation_validator import validate_annotation_dir

    exit_code = validate_annotation_dir(annotation_dir.resolve())
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
):
    """把标注 bbox 绘制到 img1/ 中对应的 RGB 帧上，输出到 debug/。需要 pillow。"""
    raise typer.Exit(_draw_overlay(camera_dataset_dir.resolve(), include_ball))


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
):
    """从 Instance-ID Mask 计算 pixel-tight bbox / 分割标注，覆盖写 annotations.jsonl 并导出 MOT / YOLO。

    bbox 由 mask 像素 min/max 直接计算（primary GT），原几何 bbox 保留在
    geometry_bbox_* 字段作为 fallback。原始 mask/*.png 永不修改。
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
    )
    if exit_code != 0:
        typer.echo("ANNOTATE MASKS FAILED", err=True)
        raise typer.Exit(exit_code)
    typer.echo("ANNOTATE MASKS DONE")


if __name__ == "__main__":
    app()
