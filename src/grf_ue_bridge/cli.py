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

    with open(config) as f:
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


if __name__ == "__main__":
    app()
