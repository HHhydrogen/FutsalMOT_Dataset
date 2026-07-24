"""CLI for the GRF-UE bridge."""

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
        help="Path to export config JSON file",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Output directory for the episode",
    ),
):
    """Run a GRF episode and export to GRF-UE JSONL format."""
    # Load config
    if not config.exists():
        typer.echo(f"ERROR: Config file not found: {config}", err=True)
        raise typer.Exit(1)

    with open(config) as f:
        config_data = json.load(f)
    export_cfg = ExportConfig(**config_data)

    typer.echo(f"Export config: {export_cfg.model_dump_json(indent=2)}")
    typer.echo(f"Running episode: scenario={export_cfg.scenario}, steps={export_cfg.num_steps}")

    # Run episode
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

    # Export
    output_path = output if output.is_absolute() else Path.cwd() / output
    export_episode(export_cfg, result, output_path)

    typer.echo(f"Export complete: {output_path}")
    typer.echo(f"  meta.json: {(output_path / 'meta.json').stat().st_size} bytes")
    typer.echo(f"  frames.jsonl: {(output_path / 'frames.jsonl').stat().st_size} bytes")


@app.command()
def validate(
    episode_dir: Path = typer.Argument(
        ...,
        help="Episode directory containing meta.json and frames.jsonl",
    ),
):
    """Validate an exported episode directory."""
    exit_code = validate_episode(episode_dir.resolve())
    if exit_code != 0:
        typer.echo("VALIDATION FAILED", err=True)
        raise typer.Exit(exit_code)
    typer.echo("VALIDATION PASSED")


if __name__ == "__main__":
    app()
