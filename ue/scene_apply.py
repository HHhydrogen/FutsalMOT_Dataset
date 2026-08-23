"""UE 侧共享的 Actor 变换 / 查找辅助函数。

本模块与 import_grf_episode.py 的 preview 逻辑共用同一套变换规则，保证
Level Sequence 渲染、preview 预览、annotation 导出三者的 actor 变换一致。
unreal 一律延迟 import，避免模块顶层依赖 UE。

朝向（facing/yaw）统一由 ue/player_motion 的 PlayerMotionTracker 计算：
速度（优先用 frame 的 velocity_mps，缺失时位置差分）→ 朝向，低速保持上一帧、
平滑 + 限速转向。位置 Ground Truth 仍完全由 frame.position_m（米 → 厘米）决定，
本模块绝不改写位置。
"""

from player_motion import (
    DEFAULT_MOTION_CONFIG,
    GK_ENTITY_IDS,
    PlayerMotionTracker,
)


M_TO_CM = 100.0
BALL_Z_OFFSET_CM = 2.0  # 偏移，使 GRF 球 z（~0.11 * 100）+ 偏移 ≈ 13cm
PLAYER_Z_CM = 90.0  # 球员 actor 的固定地面高度


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


def apply_player_frame(actors: dict, frame: dict, trackers: dict,
                       config=DEFAULT_MOTION_CONFIG):
    """根据帧数据设置球员 actor 的位置与朝向（yaw）。

    朝向由 PlayerMotionTracker 统一计算（速度 → 朝向，平滑限速，低速保持）；
    位置仍由 frame.position_m 直接决定（Ground Truth 不被动画层改写）。

    Args:
        actors: {entity_id: actor}。
        frame: 帧 dict（含 players[{id, position_m, velocity_mps?, ...}]、
            time_seconds）。
        trackers: {player_id: PlayerMotionTracker}（跨帧维护，勿复用）。
        config: MotionConfig（默认 DEFAULT_MOTION_CONFIG）。
    """
    import unreal
    time_s = float(frame.get("time_seconds", 0.0))
    for player_data in frame["players"]:
        pid = player_data["id"]
        if pid not in actors:
            continue
        px, py, _ = pos_m_to_cm(player_data["position_m"])
        pos_cm = unreal.Vector(px, py, PLAYER_Z_CM)

        tracker = trackers.setdefault(pid, PlayerMotionTracker(config=config))
        params = tracker.update(
            player_data["position_m"],
            player_data.get("velocity_mps"),
            time_s,
            has_ball=bool(player_data.get("has_ball", False)),
            ball_position_m=(
                frame["ball"]["position_m"] if pid in GK_ENTITY_IDS else None
            ),
            face_ball=(pid in GK_ENTITY_IDS),
        )
        yaw = params["facing_deg"]

        actors[pid].set_actor_location_and_rotation(
            pos_cm, unreal.Rotator(0.0, 0.0, yaw), False, False
        )


def apply_preview_frame(actors: dict, frame: dict, trackers: dict,
                        config=DEFAULT_MOTION_CONFIG):
    """应用单个帧的 actor 变换（球 + 全部球员）。trackers 为 {pid: tracker}。"""
    apply_ball_frame(actors, frame)
    apply_player_frame(actors, frame, trackers, config)


def apply_preview(meta: dict, frames: list, mapping: dict):
    """在关卡中逐帧设置 actor 变换（preview 模式）。"""
    actors = find_all_actors(mapping)
    if not actors:
        return

    num_steps = meta["timing"]["num_steps"]
    trackers = {}

    for frame in frames:
        step = frame["step"]
        apply_preview_frame(actors, frame, trackers)

        if step > 0 and step % 50 == 0:
            print(f"  Preview: {step}/{num_steps}")

    print(f"Preview complete: {num_steps} frames applied.")
