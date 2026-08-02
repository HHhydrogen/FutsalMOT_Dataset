"""
Unreal Engine Python 脚本 —— 把 GRF-UE episode 导入到 Level Sequence。

两种模式（默认 both）：
  --preview    直接在关卡中设置 actor 变换（编辑器预览）。
  --sequence   创建 / 覆盖带关键帧变换的 Level Sequence 资产。

用法（在 Unreal Editor Python Console 中执行）：
    # 最简单（参数在 ue_import_config.json 中）
    py "D:/path/to/code/ue/import_grf_episode.py"

    # 临时覆盖部分参数
    py "D:/path/to/code/ue/import_grf_episode.py" --episode "D:/path/to/other_episode" --replace-existing

依赖：
    - unreal（UE 内置模块）
    - json、math、pathlib（标准库）

本脚本不依赖 gfootball、GRF_MARL 或 .venv。
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

M_TO_CM = 100.0
SPEED_THRESHOLD_CM = 5.0  # cm/s，低于该速度保持上一次的 yaw
BALL_Z_OFFSET_CM = 2.0  # 偏移，使 GRF 球 z（~0.11 * 100）+ 偏移 ≈ 13cm
PLAYER_Z_CM = 90.0  # 球员 actor 的固定地面高度
DEFAULT_SEQUENCE_PACKAGE_PATH = "/Game/FutsalMOT/Sequences"
EXPECTED_CHANNEL_NAMES = [
    "Location.X", "Location.Y", "Location.Z",
    "Rotation.X", "Rotation.Y", "Rotation.Z",
    "Scale.X", "Scale.Y", "Scale.Z",
]


# ── 参数解析 ────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="把 GRF-UE episode 导入到 Unreal Engine"
    )
    parser.add_argument(
        "--episode", default=None,
        help="episode 目录路径（包含 meta.json、frames.jsonl）"
    )
    parser.add_argument(
        "--mapping", default=None,
        help="actor 映射 JSON 文件路径"
    )
    parser.add_argument(
        "--mode", choices=["preview", "sequence", "both"], default="both",
        help="执行模式"
    )
    parser.add_argument(
        "--replace-existing", action="store_true", default=False,
        help="删除并重建已存在的 Level Sequence，不弹出覆盖确认对话框"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="已弃用：配置会自动从脚本同级的 ue_import_config.json 加载。"
    )
    parsed = parser.parse_args()
    return parsed


# ── 加载辅助函数 ────────────────────────────────────────────────────────

def load_episode(episode_dir: Path):
    """从 episode 目录加载 meta.json 和 frames.jsonl。"""
    with open(episode_dir / "meta.json") as f:
        meta = json.load(f)
    frames = []
    with open(episode_dir / "frames.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return meta, frames


def load_mapping(path: Path) -> dict:
    """加载 actor 映射 JSON。"""
    with open(path) as f:
        return json.load(f)


# ── Actor 辅助函数 ─────────────────────────────────────────────────────

def _get_actor_subsystem():
    """获取 EditorActorSubsystem。"""
    import unreal
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def find_actor(name: str):
    """按标签或名称（不区分大小写）查找 UE actor。"""
    import unreal
    subsystem = _get_actor_subsystem()
    actors = subsystem.get_all_level_actors()
    for actor in actors:
        actor_name = actor.get_actor_label() or actor.get_name()
        if actor_name.lower() == name.lower():
            return actor
        if name.lower() in actor_name.lower():
            return actor
    return None


def _find_all_actors(mapping: dict) -> dict:
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


# ── 数学工具 ───────────────────────────────────────────────────────────

def build_yaw(dx: float, dy: float, prev_yaw: float) -> float:
    """根据位移增量计算 yaw，并带低速滞回。"""
    speed = math.sqrt(dx * dx + dy * dy)
    if speed < SPEED_THRESHOLD_CM:
        return prev_yaw
    return math.degrees(math.atan2(dy, dx))


def _pos_m_to_cm(pos_m: list) -> tuple:
    """把 [x, y, z] 米转换为 (x_cm, y_cm, z_cm)。"""
    return (pos_m[0] * M_TO_CM, pos_m[1] * M_TO_CM, pos_m[2] * M_TO_CM)


# ── Sequence 关键帧辅助函数 ─────────────────────────────────────────────

def add_double_channel_key(channel, frame_index: int, value: float, *, interpolation=None):
    """向 MovieSceneDoubleChannel 添加关键帧，并正确包装 FrameNumber。

    Args:
        channel: 要添加关键帧的 MovieSceneDoubleChannel。
        frame_index: 显示速率下的帧索引（普通 int）。内部会包装为 FrameNumber。
        value: 关键帧数值。
        interpolation: MovieSceneKeyInterpolation 枚举值。默认为 LINEAR。

    Raises:
        ValueError: 当 channel 为 None 或 value 非有限时抛出。
    """
    import unreal

    if channel is None:
        raise ValueError("Cannot add a Sequencer key to a null channel.")
    frame_number = unreal.FrameNumber(int(frame_index))
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(
            f"Non-finite Sequencer value: frame={frame_index}, value={value!r}"
        )
    interp = (
        interpolation
        if interpolation is not None
        else unreal.MovieSceneKeyInterpolation.LINEAR
    )
    return channel.add_key(time=frame_number, new_value=numeric_value, interpolation=interp)


def _canonical_channel_name(channel) -> str:
    """获取通道名，去掉末尾的数字后缀（UE 5.8+ 会追加 _NNN）。"""
    raw_name = None
    try:
        raw_name = channel.get_editor_property("channel_name")
    except Exception:
        pass
    if not raw_name:
        try:
            raw_name = channel.get_name()
        except Exception:
            raw_name = str(channel)
    return re.sub(r"_\d+$", "", str(raw_name))


def _channel_name(channel) -> str:
    """安全地获取通道的显示名，兼容不同 UE 版本。"""
    try:
        name = channel.get_name()
        if isinstance(name, str):
            return name
        return str(name)
    except Exception:
        try:
            return str(channel)
        except Exception:
            return "UnnamedChannel"


def _build_channel_map(channels) -> dict:
    """构建“规范通道名（例如 'Location.X'）→ 通道对象”的字典。

    匹配前先去除 UE 5.8+ 的数字后缀（_NNN）。
    如有必要，回退到带长度检查的按位置索引。
    记录原始通道名以便诊断。
    """
    import unreal
    canonical_map = {}
    for ch in channels:
        canonical = _canonical_channel_name(ch)
        canonical_map[canonical] = ch
    unreal.log(
        "Transform channels: "
        + ", ".join(f"{c}" for c in canonical_map.keys())
    )
    if len(channels) < 9:
        raise RuntimeError(
            f"Expected at least 9 transform channels, got {len(channels)}"
        )
    if all(n in canonical_map for n in EXPECTED_CHANNEL_NAMES):
        print("  Transform channel name mapping: PASS")
        return canonical_map
    # 回退：按位置索引并做长度检查
    unreal.log("Warning: canonical channel name matching incomplete, using positional fallback")
    return {
        "Location.X": channels[0],
        "Location.Y": channels[1],
        "Location.Z": channels[2],
        "Rotation.X": channels[3],
        "Rotation.Y": channels[4],
        "Rotation.Z": channels[5],
        "Scale.X": channels[6],
        "Scale.Y": channels[7],
        "Scale.Z": channels[8],
    }


# ── 球滚动 ──────────────────────────────────────────────────────────────

def _unwind_angle(previous, current):
    """展开角度，避免 -180/180 跳变。"""
    while current - previous > 180.0:
        current -= 360.0
    while current - previous < -180.0:
        current += 360.0
    return current


def compute_ball_rotation_quat(
    ball_positions,
    radius_m,
    minimum_move_distance_m,
    roll_sign,
):
    """使用四元数累加计算每帧的球旋转。

    Args:
        ball_positions: 球的米坐标 (x, y, z) 列表。
        radius_m: 球半径（米）。
        minimum_move_distance_m: 视为移动的最小位移。
        roll_sign: 用于修正轴向的符号乘子（1.0 或 -1.0）。

    Returns:
        每帧的 (roll_deg, pitch_deg, yaw_deg) 列表。
        第一帧为 (0, 0, 0)。
    """
    import unreal

    current_quat = unreal.MathLibrary.quat_make_from_euler(unreal.Vector(0.0, 0.0, 0.0))
    rotations = [(0.0, 0.0, 0.0)]  # 第 0 帧

    prev_roll, prev_pitch, prev_yaw = 0.0, 0.0, 0.0

    for i in range(1, len(ball_positions)):
        bx_prev, by_prev, _ = ball_positions[i - 1]
        bx_cur, by_cur, bz_cur = ball_positions[i]

        dx = bx_cur - bx_prev
        dy = by_cur - by_prev
        distance = math.hypot(dx, dy)

        # 地面滚动（高度接近球半径）
        if bz_cur <= radius_m + 0.03 and distance > minimum_move_distance_m:
            angle_degrees = math.degrees(distance / radius_m)
            axis = unreal.Vector(-dy / distance, dx / distance, 0.0)
            rotation_vector = unreal.Vector(
                axis.x * angle_degrees * roll_sign,
                axis.y * angle_degrees * roll_sign,
                0.0,
            )
            delta_quat = unreal.MathLibrary.quat_make_from_rotation_vector(rotation_vector)
            current_quat = unreal.MathLibrary.multiply_quat_quat(delta_quat, current_quat)

        current_quat = unreal.MathLibrary.quat_normalized(current_quat)
        rotator = unreal.MathLibrary.quat_rotator(current_quat)
        roll = _unwind_angle(prev_roll, rotator.roll)
        pitch = _unwind_angle(prev_pitch, rotator.pitch)
        yaw = _unwind_angle(prev_yaw, rotator.yaw)
        rotations.append((roll, pitch, yaw))
        prev_roll, prev_pitch, prev_yaw = roll, pitch, yaw

    return rotations


def _add_ball_rolling(frames, sequence, source_step, playback_fps,
                       actor_bindings, ball_cfg):
    """向球的变换轨道添加滚动旋转关键帧。"""
    import unreal

    if "BALL" not in actor_bindings:
        return

    _, channels, _ = actor_bindings["BALL"]
    cmap = _build_channel_map(channels)

    if "Rotation.X" not in cmap:
        return

    radius_m = ball_cfg.get("radius_m", 0.11)
    minimum_move_distance_m = ball_cfg.get("minimum_move_distance_m", 0.0001)
    roll_sign = ball_cfg.get("roll_sign", 1.0)

    ball_positions = [(f["ball"]["position_m"][0], f["ball"]["position_m"][1], f["ball"]["position_m"][2])
                      for f in frames]

    rotations = compute_ball_rotation_quat(ball_positions, radius_m, minimum_move_distance_m, roll_sign)

    for fi in range(len(frames)):
        kf = int(round(fi * source_step * playback_fps))
        r, p, y = rotations[fi]
        add_double_channel_key(cmap["Rotation.X"], kf, r)
        add_double_channel_key(cmap["Rotation.Y"], kf, p)
        add_double_channel_key(cmap["Rotation.Z"], kf, y)

    print(f"  Ball rolling keys added: {len(frames)} frames")


# ── 预览模式 ────────────────────────────────────────────────────────────

def apply_preview(meta: dict, frames: list, mapping: dict):
    """在关卡中逐帧设置 actor 变换。"""
    actors = _find_all_actors(mapping)
    if not actors:
        return

    num_steps = meta["timing"]["num_steps"]
    prev_yaws = {}
    prev_positions = {}

    for frame in frames:
        step = frame["step"]
        _apply_ball_frame(actors, frame)
        _apply_player_frame(actors, frame, prev_yaws, prev_positions)

        if step > 0 and step % 50 == 0:
            print(f"  Preview: {step}/{num_steps}")

    print(f"Preview complete: {num_steps} frames applied.")


# ── 冒烟测试 ───────────────────────────────────────────────────────────

def _smoke_test_sequencer_api(actors: dict, total_output_frames: int, package_path: str = None):
    """创建一个临时 Sequence，验证 add_key 接受 FrameNumber。

    返回 (temp_sequence, channel_name_map)，失败时抛出异常。
    """
    import unreal

    pkg = package_path or DEFAULT_SEQUENCE_PACKAGE_PATH
    seq_factory = unreal.LevelSequenceFactoryNew()
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    temp_seq = asset_tools.create_asset("_TEMP_SMOKE", pkg, None, seq_factory)
    if not temp_seq:
        raise RuntimeError("Failed to create smoke test sequence.")

    # 绑定第一个 actor 并创建变换轨道
    first_actor = next(iter(actors.values()))
    binding = temp_seq.add_possessable(first_actor)
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    channels = section.get_all_channels()
    cmap = _build_channel_map(channels)

    # 添加两个测试关键帧，再通过 remove_key() 移除它们
    test_ch = cmap["Location.X"]
    test_key_0 = add_double_channel_key(test_ch, 0, 0.0)
    test_key_1 = add_double_channel_key(test_ch, 1, 100.0)
    if test_ch.get_num_keys() != 2:
        raise RuntimeError(
            f"Sequencer smoke test: expected 2 keys, got {test_ch.get_num_keys()}"
        )
    test_ch.remove_key(test_key_0)
    test_ch.remove_key(test_key_1)
    if test_ch.get_num_keys() != 0:
        raise RuntimeError(
            f"Sequencer smoke test cleanup: expected 0 keys, got {test_ch.get_num_keys()}"
        )
    print("  Smoke test PASS: add_key accepts FrameNumber, remove_key works")

    return temp_seq, cmap


# ── Sequence 模式 ───────────────────────────────────────────────────────

def create_sequence(meta: dict, frames: list, mapping: dict, replace_existing: bool = False,
                    package_path: str = None, sequences_cfg: list = None,
                    ball_rolling_cfg: dict = None):
    """创建带关键帧变换的 Level Sequence 资产。

    如果提供了 sequences_cfg，则为每个条目创建一个 Sequence，
    每个都包含相同的球员/球数据并绑定摄像机。
    """
    import unreal

    actors = _find_all_actors(mapping)
    if not actors:
        return

    pkg_path = package_path or DEFAULT_SEQUENCE_PACKAGE_PATH
    num_steps = meta["timing"]["num_steps"]
    source_step = meta["timing"]["source_step_seconds"]
    playback_fps = int(meta["timing"].get("playback_fps", 30))
    total_output_frames = int(math.ceil(num_steps * source_step * playback_fps))

    unreal.log(f"Unreal version: {unreal.SystemLibrary.get_engine_version()}")

    # 确保包目录存在
    if not unreal.EditorAssetLibrary.does_directory_exist(pkg_path):
        unreal.EditorAssetLibrary.make_directory(pkg_path)

    # 冒烟测试一次
    temp_seq, _ = _smoke_test_sequencer_api(actors, total_output_frames, pkg_path)
    unreal.EditorAssetLibrary.delete_asset(f"{pkg_path}/_TEMP_SMOKE")

    # 决定要创建哪些 Sequence
    if sequences_cfg:
        seq_list = sequences_cfg
    else:
        seq_list = [{"name": f"SEQ_{meta.get('episode_id', 'episode_0000')}"}]

    for seq_entry in seq_list:
        seq_name = seq_entry["name"]
        camera_actor_name = seq_entry.get("camera_actor")
        sequence_asset_path = f"{pkg_path}/{seq_name}"

        print(f"\n--- Creating Sequence: {seq_name} ---")

        # 删除已存在的资产
        if unreal.EditorAssetLibrary.does_asset_exist(sequence_asset_path):
            if not replace_existing:
                raise RuntimeError(
                    f"Level Sequence already exists: {sequence_asset_path}. "
                    f"Run again with --replace-existing."
                )
            unreal.EditorAssetLibrary.delete_asset(sequence_asset_path)
            print(f"  Deleted existing: {sequence_asset_path}")

        # 创建新的 Sequence
        seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            seq_name, pkg_path, None, unreal.LevelSequenceFactoryNew()
        )
        if not seq:
            raise RuntimeError(f"Failed to create Level Sequence: {sequence_asset_path}")

        display_rate = unreal.FrameRate(numerator=playback_fps, denominator=1)
        seq.set_display_rate(display_rate)
        seq.set_playback_start(0)
        seq.set_playback_end(total_output_frames)
        print(f"  Timeline: {total_output_frames} frames @ {playback_fps} FPS")

        # ── 绑定所有 actor（球员 + 球）───────────────────────────────
        actor_bindings = {}
        for entity_id, actor in actors.items():
            binding, channels, section = _build_entity_binding(
                seq, entity_id, actor, total_output_frames,
            )
            actor_bindings[entity_id] = (binding, channels, section)

        # ── 如指定则绑定摄像机 ────────────────────────────────────────
        camera_binding = None
        if camera_actor_name:
            cam_actor = find_actor(camera_actor_name)
            if not cam_actor:
                print(f"  WARNING: Camera actor '{camera_actor_name}' not found, skipping camera binding")
            else:
                camera_binding = seq.add_possessable(cam_actor)
                print(f"  Camera bound: {camera_actor_name}")

                # 摄像机以 possessable 方式绑定。UE 5.8 Python API 未暴露
                # CameraCutTrack 的创建——需在 Sequencer 中手动设置：
                # 右键摄像机轨道 → "Set as Camera Cut"。
                unreal.log(f"  Camera bound: {camera_actor_name} (set as Camera Cut manually in Sequencer)")

        # ── 为每个 actor 写入变换关键帧 ──────────────────────────────
        total_transform_keys = 0
        for entity_id, actor in actors.items():
            binding, channels, section = actor_bindings[entity_id]
            cmap = _build_channel_map(channels)

            prev_yaws = {}
            previous_pos = None

            for fi, frame in enumerate(frames):
                frame_time = fi * source_step
                kf = int(round(frame_time * playback_fps))

                if entity_id == "BALL":
                    ball_pos = frame["ball"]["position_m"]
                    px, py, pz = _pos_m_to_cm(ball_pos)
                    pz += BALL_Z_OFFSET_CM
                    add_double_channel_key(cmap["Location.X"], kf, px)
                    add_double_channel_key(cmap["Location.Y"], kf, py)
                    add_double_channel_key(cmap["Location.Z"], kf, pz)
                    if fi == 0:
                        add_double_channel_key(cmap["Scale.X"], 0, 0.5)
                        add_double_channel_key(cmap["Scale.Y"], 0, 0.5)
                        add_double_channel_key(cmap["Scale.Z"], 0, 0.5)
                else:
                    for player_data in frame["players"]:
                        if player_data["id"] != entity_id:
                            continue
                        px, py, _ = _pos_m_to_cm(player_data["position_m"])
                        add_double_channel_key(cmap["Location.X"], kf, px)
                        add_double_channel_key(cmap["Location.Y"], kf, py)
                        add_double_channel_key(cmap["Location.Z"], kf, PLAYER_Z_CM)

                        if previous_pos is not None:
                            dx = px - previous_pos[0]
                            dy = py - previous_pos[1]
                            prev_yaw = prev_yaws.get(entity_id, 0.0)
                            yaw = build_yaw(dx, dy, prev_yaw)
                        else:
                            yaw = 0.0
                        prev_yaws[entity_id] = yaw
                        previous_pos = (px, py)
                        add_double_channel_key(cmap["Rotation.Z"], kf, yaw)

            # 校验关键帧数量
            actual_loc_x = cmap["Location.X"].get_num_keys()
            actual_loc_y = cmap["Location.Y"].get_num_keys()
            actual_loc_z = cmap["Location.Z"].get_num_keys()
            expected = num_steps
            for n, c in [("Location.X", actual_loc_x), ("Location.Y", actual_loc_y),
                         ("Location.Z", actual_loc_z)]:
                if c != expected:
                    raise RuntimeError(f"{entity_id} {n}: expected {expected} keys, got {c}")
            total_transform_keys += actual_loc_x + actual_loc_y + actual_loc_z

            if entity_id != "BALL":
                actual_rot = cmap["Rotation.Z"].get_num_keys()
                if actual_rot != expected:
                    raise RuntimeError(f"{entity_id} Rotation.Z: expected {expected} keys, got {actual_rot}")
                total_transform_keys += actual_rot

        print(f"  Total transform keys: {total_transform_keys}")

        # ── 球滚动 ──────────────────────────────────────────────
        if ball_rolling_cfg and ball_rolling_cfg.get("enabled", True):
            _add_ball_rolling(frames, seq, source_step, playback_fps,
                              actor_bindings, ball_rolling_cfg)

        # ── 保存并打开 ────────────────────────────────────────────────
        saved = unreal.EditorAssetLibrary.save_loaded_asset(seq, only_if_is_dirty=False)
        if not saved:
            raise RuntimeError(f"Failed to save: {sequence_asset_path}")
        print(f"  Saved: {sequence_asset_path}")

        try:
            unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(seq)
            print("  Opened in Sequencer.")
        except Exception as exc:
            unreal.log(f"Warning: could not open in Sequencer: {exc}")

    # ── 汇总 ─────────────────────────────────────────────────────────
    player_count = sum(1 for eid in actors if eid != "BALL")
    print()
    print("GRF UE IMPORT PASS")
    print(f"Sequences created: {len(seq_list)}")
    for s in seq_list:
        cam = s.get("camera_actor", "none")
        print(f"  {pkg_path}/{s['name']} (camera: {cam})")
    print(f"Players: {player_count}, Ball: {'BALL' in actors}")
    print(f"Package path: {pkg_path}")




def _build_entity_binding(sequence, entity_id, actor, total_output_frames):
    """为 entity_id 添加 possessable 绑定 + 变换轨道 + section。

    返回 (binding, channels_list, section)，调用方可检查通道。
    """
    import unreal
    binding = sequence.add_possessable(actor)
    transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
    transform_section = transform_track.add_section()
    transform_section.set_range(0, int(total_output_frames))
    channels = transform_section.get_all_channels()
    return binding, channels, transform_section


# ── 预览辅助函数（共享，模块顶层不依赖 unreal）──────────────────────

def _apply_ball_frame(actors: dict, frame: dict):
    """根据帧数据设置球 actor 的位置。"""
    import unreal
    if "BALL" not in actors:
        return
    ball_pos = frame["ball"]["position_m"]
    px, py, pz = _pos_m_to_cm(ball_pos)
    actors["BALL"].set_actor_location(
        unreal.Vector(px, py, pz + BALL_Z_OFFSET_CM), False, False
    )


def _apply_player_frame(actors: dict, frame: dict, prev_yaws: dict, prev_positions: dict):
    """根据帧数据设置球员 actor 的位置与旋转。"""
    import unreal
    for player_data in frame["players"]:
        pid = player_data["id"]
        if pid not in actors:
            continue
        px, py, _ = _pos_m_to_cm(player_data["position_m"])
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


# ── 入口 ───────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # 从固定路径加载配置（本脚本同级）
    cfg_defaults = {}
    cfg_path = Path(__file__).resolve().parent.parent / "ue_import_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if not k.startswith("comment_"):
                cfg_defaults[k] = v

    episode_dir = Path(
        args.episode if args.episode
        else cfg_defaults.get("episode")
    )
    mapping_path = Path(
        args.mapping if args.mapping
        else cfg_defaults.get("mapping")
    )
    mode = (
        args.mode if args.mode != "both"
        else cfg_defaults.get("mode", "both")
    )
    replace_existing = (
        args.replace_existing if args.replace_existing
        else cfg_defaults.get("replace_existing", False)
    )
    if not episode_dir or not episode_dir.exists():
        print(f"ERROR: Episode directory not found: {episode_dir}", file=sys.stderr)
        sys.exit(1)
    if not mapping_path or not mapping_path.exists():
        print(f"ERROR: Mapping file not found: {mapping_path}", file=sys.stderr)
        sys.exit(1)

    meta, frames = load_episode(episode_dir)
    mapping = load_mapping(mapping_path)

    num_steps = meta["timing"]["num_steps"]
    source_step = meta["timing"]["source_step_seconds"]
    print(f"Episode: {meta.get('episode_id', 'unknown')}")
    print(f"  Frames: {num_steps}")
    print(f"  Source step: {source_step}s")
    print(f"  Field: {meta['field']['length_m']}m x {meta['field']['width_m']}m")

    if mode in ("preview", "both"):
        print("\n--- Preview mode: setting actor transforms ---")
        apply_preview(meta, frames, mapping)

    if mode in ("sequence", "both"):
        print("\n--- Sequence mode: creating Level Sequence ---")
        seq_pkg = cfg_defaults.get("sequence_package_path") or DEFAULT_SEQUENCE_PACKAGE_PATH
        seq_list = cfg_defaults.get("sequences") or None
        ball_rolling_cfg = cfg_defaults.get("ball_rolling", None)
        create_sequence(meta, frames, mapping, replace_existing, seq_pkg, seq_list, ball_rolling_cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
