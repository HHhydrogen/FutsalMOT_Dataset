"""UE 侧共享的 Actor 变换 / 查找辅助函数。

本模块与 import_grf_episode.py 的 preview 逻辑共用同一套变换规则，保证
Level Sequence 渲染、preview 预览、annotation 导出三者的 actor 变换一致。
unreal 一律延迟 import，避免模块顶层依赖 UE。
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple


M_TO_CM = 100.0
SPEED_THRESHOLD_CM = 5.0  # cm/s，低于该速度保持上一次的 yaw
BALL_Z_OFFSET_CM = 2.0  # 偏移，使 GRF 球 z（~0.11 * 100）+ 偏移 ≈ 13cm
PLAYER_Z_CM = 90.0  # 球员 actor 的固定地面高度


def build_yaw(dx: float, dy: float, prev_yaw: float) -> float:
    """根据位移增量计算 yaw，并带低速滞回。"""
    speed = math.sqrt(dx * dx + dy * dy)
    if speed < SPEED_THRESHOLD_CM:
        return prev_yaw
    return math.degrees(math.atan2(dy, dx))


def pos_m_to_cm(pos_m: list) -> tuple:
    """把 [x, y, z] 米转换为 (x_cm, y_cm, z_cm)。"""
    return (pos_m[0] * M_TO_CM, pos_m[1] * M_TO_CM, pos_m[2] * M_TO_CM)


# ── Actor 查找 ─────────────────────────────────────────────────────────

def get_actor_subsystem():
    """获取 EditorActorSubsystem。"""
    import unreal
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def find_actor(name: str):
    """按标签或名称（不区分大小写）查找 UE actor。"""
    import unreal
    subsystem = get_actor_subsystem()
    actors = subsystem.get_all_level_actors()
    for actor in actors:
        actor_name = actor.get_actor_label() or actor.get_name()
        if actor_name.lower() == name.lower():
            return actor
        if name.lower() in actor_name.lower():
            return actor
    return None


def find_all_actors(mapping: dict) -> dict:
    """在映射中找到所有 actor。返回 {entity_id: actor} 字典。"""
    actors = {}
    for entity_id, actor_name in mapping.items():
        actor = find_actor(actor_name)
        if actor:
            actors[entity_id] = actor
            print(f"  Found: {entity_id} -> {actor_name}")
        else:
            print(f"  WARNING: Not found: {entity_id} -> {actor_name}")
    if not actors:
        print("ERROR: No actors found. Check your actor mapping and level.")
    return actors


# ── 逐帧变换应用 ───────────────────────────────────────────────────────

def apply_ball_frame(actors: dict, frame: dict):
    """根据帧数据设置球 actor 的位置。"""
    import unreal
    if "BALL" not in actors:
        return
    ball_pos = frame["ball"]["position_m"]
    px, py, pz = pos_m_to_cm(ball_pos)
    actors["BALL"].set_actor_location(
        unreal.Vector(px, py, pz + BALL_Z_OFFSET_CM), False, False
    )


def apply_player_frame(actors: dict, frame: dict, prev_yaws: dict, prev_positions: dict):
    """根据帧数据设置球员 actor 的位置与旋转。"""
    import unreal
    for player_data in frame["players"]:
        pid = player_data["id"]
        if pid not in actors:
            continue
        px, py, _ = pos_m_to_cm(player_data["position_m"])
        pos_cm = unreal.Vector(px, py, PLAYER_Z_CM)

        prev_pos = prev_positions.get(pid)
        if prev_pos is not None:
            dx = pos_cm.x - prev_pos.x
            dy = pos_cm.y - prev_pos.y
            prev_yaw = prev_yaws.get(pid, 0.0)
            yaw = build_yaw(dx, dy, prev_yaw)
        else:
            yaw = 0.0

        prev_yaws[pid] = yaw
        prev_positions[pid] = pos_cm

        actors[pid].set_actor_location_and_rotation(
            pos_cm, unreal.Rotator(0.0, 0.0, yaw), False, False
        )


def apply_preview_frame(actors: dict, frame: dict, prev_yaws: dict, prev_positions: dict):
    """应用单个帧的 actor 变换（球 + 全部球员）。"""
    apply_ball_frame(actors, frame)
    apply_player_frame(actors, frame, prev_yaws, prev_positions)


def apply_preview(meta: dict, frames: list, mapping: dict):
    """在关卡中逐帧设置 actor 变换（preview 模式）。"""
    actors = find_all_actors(mapping)
    if not actors:
        return

    num_steps = meta["timing"]["num_steps"]
    prev_yaws = {}
    prev_positions = {}

    for frame in frames:
        step = frame["step"]
        apply_preview_frame(actors, frame, prev_yaws, prev_positions)

        if step > 0 and step % 50 == 0:
            print(f"  Preview: {step}/{num_steps}")

    print(f"Preview complete: {num_steps} frames applied.")
