"""把 GRF episode 数据导出为 GRF-UE JSONL 格式。"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional

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
    RandomnessInfo,
    SourceInfo,
    TimingInfo,
    create_ball_entity,
)
from .seeds import derive_episode_seeds


def _get_football_commit() -> str:
    """如存在，从 external_sources.lock.json 读取 football 提交号。"""
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
    """如存在，从 external_sources.lock.json 读取 GRF_MARL 提交号。"""
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
    """构建 5v5 的实体定义（10 名球员 + 1 个球）。"""
    entities: List[EntityDefinition] = []
    # 左队：角色取自场景，默认使用合理的 GK/DEF/MID/FWD 组合
    left_roles = [0, 7, 9, 2, 1]  # GK, RM, CF, LB, CB
    for i, role in enumerate(left_roles):
        entities.append(EntityDefinition.from_grf_role(role, "L", i))
    # 右队
    right_roles = [0, 7, 9, 2, 1]
    for i, role in enumerate(right_roles):
        entities.append(EntityDefinition.from_grf_role(role, "R", i))
    # 球
    entities.append(create_ball_entity())
    return entities


def export_episode(
    config: ExportConfig,
    result: EpisodeResult,
    output_dir: Path,
) -> None:
    """把 EpisodeResult 导出为 meta.json 和 frames.jsonl。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = CoordinateTransform(
        field_length_m=config.field_length_m,
        field_width_m=config.field_width_m,
    )

    # ── 构建实体 ──────────────────────────────────────────────
    entities = _build_entities()

    # ── 目标帧率：GRF 原生 10fps，可选插值到更高帧率（如 30） ──
    source_step_seconds = 0.1  # GRF 默认：10 FPS 仿真
    factor = 1
    if config.target_fps and config.target_fps != 10:
        if config.target_fps % 10 != 0 or config.target_fps < 10:
            raise ValueError(
                f"target_fps 必须是 10 的倍数且 >= 10（GRF 原生 10fps）：{config.target_fps}"
            )
        factor = config.target_fps // 10
        source_step_seconds = 1.0 / config.target_fps

    # ── 构建 meta ─────────────────────────────────────────────
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
            game_duration=config.game_duration,
            left_team_difficulty=config.left_team_difficulty,
            right_team_difficulty=config.right_team_difficulty,
        ),
        timing=TimingInfo(
            source_step_seconds=source_step_seconds,
            playback_fps=config.playback_fps,
            num_steps=config.num_steps * factor,
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

    # 随机种子体系：优先用 run_episode 实际返回的 seeds，否则按 root seed 重新派生。
    # source.seed 必须等于 randomness.root_seed。
    if result.seeds is not None:
        seeds = result.seeds
    else:
        seeds = derive_episode_seeds(config.seed)
    meta.randomness = RandomnessInfo(**seeds.model_dump())
    if meta.source.seed != seeds.root_seed:
        meta.source.seed = seeds.root_seed

    # ── 写入 meta.json ────────────────────────────────────────
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(meta.model_dump_json(indent=2, by_alias=True))
    print(f"Wrote: {meta_path}")

    # ── 构建 frames（先 GRF 10fps，再按需插值到目标帧率）──────
    entity_ids = [e.id for e in entities]
    player_entity_ids = [e.id for e in entities if e.id != "BALL"]
    ball_entity_id = "BALL"

    frames = []
    for snapshot in result.snapshots:
        frames.append(_build_frame(
            step=snapshot.step,
            ob=snapshot.observation,
            transform=transform,
            player_entity_ids=player_entity_ids,
            source_step_seconds=0.1,
            time_scale=config.trajectory_time_scale,
        ))

    if factor > 1:
        from .interpolate import interpolate_frames

        interpolated = interpolate_frames(
            [f.model_dump() for f in frames], factor, source_step_seconds
        )
        frames = [Frame(**f) for f in interpolated]
        print(
            f"插值到 {config.target_fps}fps：{len(frames)} 帧"
            f"（factor={factor}，source_step_seconds={source_step_seconds:.6f}）"
        )

    # ── 写入 frames.jsonl ─────────────────────────────────────
    frames_path = output_dir / "frames.jsonl"
    with open(frames_path, "w", encoding="utf-8") as f:
        for frame in frames:
            f.write(frame.model_dump_json() + "\n")
    print(f"Wrote: {frames_path} ({len(frames)} lines)")

    # ── provenance：快照实际使用的配置（供 manifest 做确定性 provenance）──
    _write_provenance(config, output_dir)

    # ── 调试：按需导出完整原始观测 ───────────────────────────
    if config.dump_full_raw_observation:
        raw_path = output_dir / "raw_observations.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for snapshot in result.snapshots:
                f.write(json.dumps(snapshot.observation) + "\n")
        print(f"Wrote: {raw_path}")


def _write_provenance(config: ExportConfig, output_dir: Path) -> None:
    """把实际使用的导出配置与外部仓库锁文件快照到 <output>/provenance/。

    供 dataset manifest 读取，避免「hash 当前仓库中看起来对应的配置却声称是
    实际使用配置」的伪确定性。快照仅为记录，不参与 GRF 运行。
    """
    prov = output_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    cfg_path = prov / "export_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=2))
    lock = Path("external_sources.lock.json")
    if lock.exists():
        shutil.copy2(lock, prov / "external_sources.lock.json")
    print(f"Wrote: {cfg_path}（配置快照，供 manifest provenance）")


def _build_frame(
    step: int,
    ob: dict,
    transform: CoordinateTransform,
    player_entity_ids: List[str],
    source_step_seconds: float,
    time_scale: float = 1.0,
) -> Frame:
    """根据单个步的观测构建一个 Frame。

    time_scale: 轨迹时间缩放（速度放大系数），只缩放速度/速率字段，
    不改动 GRF 位置 Ground Truth。
    """
    left_team = ob["left_team"]  # [(x,y), ...] 5 名球员
    right_team = ob["right_team"]  # [(x,y), ...] 5 名球员
    ball_grf = ob["ball"]  # [x, y, z]

    # GRF 方向（每步位移，归一化坐标）与 active / 持球信息（旧观测可能缺失，用 get 兜底）
    left_dir = ob.get("left_team_direction")
    right_dir = ob.get("right_team_direction")
    ball_dir = ob.get("ball_direction")
    left_active = ob.get("left_team_active")
    right_active = ob.get("right_team_active")
    ball_owned_team = ob.get("ball_owned_team")
    ball_owned_player = ob.get("ball_owned_player")
    game_mode = ob.get("game_mode")

    score = [int(ob["score"][0]), int(ob["score"][1])]

    # 球
    ball_pos_m, source_grf = transform.transform_ball_position(ball_grf)
    ball_velocity_mps = None
    if ball_dir is not None and len(ball_dir) >= 3:
        ball_velocity_mps = [
            round(v * time_scale, 6)
            for v in transform.grf_ball_direction_to_velocity_mps(
                float(ball_dir[0]), float(ball_dir[1]), float(ball_dir[2]),
                source_step_seconds,
            )
        ]
    ball_frame = BallFrame(
        position_m=ball_pos_m,
        source_grf_position=source_grf,
        velocity_mps=ball_velocity_mps,
    )

    # 球员
    players: List[PlayerFrame] = []

    def _build_player(team: list, direction: Optional[list], active_list: Optional[list],
                      prefix: str, team_id: int) -> None:
        """构建单队（5 名）球员的 PlayerFrame。"""
        for i in range(5):
            grf_x, grf_y = team[i][0], team[i][1]
            pos = transform.transform_player_position(grf_x, grf_y)

            velocity_mps = None
            if direction is not None and len(direction) > i:
                vx, vy = transform.grf_direction_to_velocity_mps(
                    float(direction[i][0]), float(direction[i][1]),
                    source_step_seconds,
                )
                velocity_mps = [round(vx * time_scale, 6), round(vy * time_scale, 6)]

            speed_mps = None
            heading_deg = None
            if velocity_mps is not None:
                vx, vy = velocity_mps
                speed_mps = round(math.hypot(vx, vy), 6)
                if speed_mps > 1e-9:
                    heading_deg = round(math.degrees(math.atan2(vy, vx)), 6)

            is_active = None
            if active_list is not None and len(active_list) > i:
                is_active = bool(active_list[i])

            has_ball = None
            if ball_owned_team is not None and ball_owned_player is not None:
                has_ball = (
                    int(ball_owned_team) == team_id
                    and int(ball_owned_player) == i
                )

            players.append(PlayerFrame(
                id=f"{prefix}{i}",
                position_m=pos,
                velocity_mps=velocity_mps,
                speed_mps=speed_mps,
                movement_heading_deg=heading_deg,
                active=is_active,
                has_ball=has_ball,
            ))

    _build_player(left_team, left_dir, left_active, "L", 0)
    _build_player(right_team, right_dir, right_active, "R", 1)

    frame_owned_team = int(ball_owned_team) if ball_owned_team is not None else None
    frame_owned_player = int(ball_owned_player) if ball_owned_player is not None else None
    frame_game_mode = int(game_mode) if game_mode is not None else None

    return Frame(
        step=step,
        time_seconds=step * source_step_seconds,
        score=score,
        ball=ball_frame,
        players=players,
        ball_owned_team=frame_owned_team,
        ball_owned_player=frame_owned_player,
        game_mode=frame_game_mode,
    )
