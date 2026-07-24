"""Export GRF episode data to the GRF-UE JSONL format."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from .config import ExportConfig
from .coordinate_transform import CoordinateTransform
from .grf_runner import EpisodeResult, StepSnapshot
from .schema import (
    BallFrame,
    EntityDefinition,
    FieldInfo,
    Frame,
    Meta,
    PlayerFrame,
    SourceInfo,
    TimingInfo,
    create_ball_entity,
)


def _get_football_commit() -> str:
    """Read the football commit from external_sources.lock.json if available."""
    lock_path = Path("external_sources.lock.json")
    if lock_path.exists():
        try:
            with open(lock_path) as f:
                lock = json.load(f)
            return lock.get("repositories", {})\
                .get("google-research-football", {})\
                .get("commit", "")
        except Exception:
            return ""
    return ""


def _get_grf_marl_commit() -> str:
    """Read the GRF_MARL commit from external_sources.lock.json if available."""
    lock_path = Path("external_sources.lock.json")
    if lock_path.exists():
        try:
            with open(lock_path) as f:
                lock = json.load(f)
            return lock.get("repositories", {})\
                .get("GRF_MARL", {})\
                .get("commit", "")
        except Exception:
            return ""
    return ""


def _build_entities() -> List[EntityDefinition]:
    """Build entity definitions for 5v5 (10 players + ball)."""
    entities: List[EntityDefinition] = []
    # Left team: roles from scenario, defaulting to reasonable GK/DEF/MID/FWD
    left_roles = [0, 7, 9, 2, 1]  # GK, RM, CF, LB, CB
    for i, role in enumerate(left_roles):
        entities.append(EntityDefinition.from_grf_role(role, "L", i))
    # Right team
    right_roles = [0, 7, 9, 2, 1]
    for i, role in enumerate(right_roles):
        entities.append(EntityDefinition.from_grf_role(role, "R", i))
    # Ball
    entities.append(create_ball_entity())
    return entities


def export_episode(
    config: ExportConfig,
    result: EpisodeResult,
    output_dir: Path,
) -> None:
    """Export an EpisodeResult to meta.json and frames.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = CoordinateTransform(
        field_length_m=config.field_length_m,
        field_width_m=config.field_width_m,
    )

    # ── Build entities ──────────────────────────────────────────────
    entities = _build_entities()

    # ── Build meta ──────────────────────────────────────────────────
    source_step_seconds = 0.1  # GRF default: 10 FPS simulation

    meta = Meta(
        schema="grf_ue_episode",
        version=1,
        episode_id=output_dir.name,
        source=SourceInfo(
            environment="google_research_football",
            scenario=config.scenario,
            control_mode="builtin_AI_vs_builtin_AI",
            seed=config.seed,
            football_commit=_get_football_commit(),
            grf_marl_commit=_get_grf_marl_commit(),
        ),
        timing=TimingInfo(
            source_step_seconds=source_step_seconds,
            playback_fps=config.playback_fps,
            num_steps=config.num_steps,
        ),
        field=FieldInfo(
            length_m=config.field_length_m,
            width_m=config.field_width_m,
            origin="center",
            x_range_m=[-config.field_length_m / 2, config.field_length_m / 2],
            y_range_m=[-config.field_width_m / 2, config.field_width_m / 2],
        ),
        entities=entities,
    )

    # ── Write meta.json ─────────────────────────────────────────────
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(meta.model_dump_json(indent=2, by_alias=True))
    print(f"Wrote: {meta_path}")

    # ── Write frames.jsonl ──────────────────────────────────────────
    entity_ids = [e.id for e in entities]
    player_entity_ids = [e.id for e in entities if e.id != "BALL"]
    ball_entity_id = "BALL"

    frames_path = output_dir / "frames.jsonl"
    with open(frames_path, "w", encoding="utf-8") as f:
        for snapshot in result.snapshots:
            ob = snapshot.observation
            step = snapshot.step
            frame = _build_frame(
                step=step,
                ob=ob,
                transform=transform,
                player_entity_ids=player_entity_ids,
                source_step_seconds=source_step_seconds,
            )
            f.write(frame.model_dump_json() + "\n")
    print(f"Wrote: {frames_path} ({len(result.snapshots)} lines)")

    # ── Debug: dump raw observations if configured ──────────────────
    if config.dump_full_raw_observation:
        raw_path = output_dir / "raw_observations.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for snapshot in result.snapshots:
                f.write(json.dumps(snapshot.observation) + "\n")
        print(f"Wrote: {raw_path}")


def _build_frame(
    step: int,
    ob: dict,
    transform: CoordinateTransform,
    player_entity_ids: List[str],
    source_step_seconds: float,
) -> Frame:
    """Build a Frame from a single step observation."""
    left_team = ob["left_team"]  # [(x,y), ...] 5 players
    right_team = ob["right_team"]  # [(x,y), ...] 5 players
    ball_grf = ob["ball"]  # [x, y, z]

    score = [int(ob["score"][0]), int(ob["score"][1])]

    # Ball
    ball_pos_m, source_grf = transform.transform_ball_position(ball_grf)
    ball_frame = BallFrame(position_m=ball_pos_m, source_grf_position=source_grf)

    # Players
    players: List[PlayerFrame] = []
    # Left team (L0-L4)
    for i in range(5):
        grf_x, grf_y = left_team[i][0], left_team[i][1]
        pos = transform.transform_player_position(grf_x, grf_y)
        players.append(PlayerFrame(id=f"L{i}", position_m=pos))
    # Right team (R0-R4)
    for i in range(5):
        grf_x, grf_y = right_team[i][0], right_team[i][1]
        pos = transform.transform_player_position(grf_x, grf_y)
        players.append(PlayerFrame(id=f"R{i}", position_m=pos))

    return Frame(
        step=step,
        time_seconds=step * source_step_seconds,
        score=score,
        ball=ball_frame,
        players=players,
    )
