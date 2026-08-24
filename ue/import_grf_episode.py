"""
Unreal Engine Python 脚本 —— 把 GRF-UE episode 导入到 Level Sequence。

模式（默认 both）：
  --preview    直接在关卡中设置 actor 变换（编辑器预览）。
  --sequence   创建 / 覆盖带关键帧变换的 Level Sequence 资产。
  --both       先 preview 后 sequence。
  --annotations 导出 CV Ground-Truth 标注（见 ue/annotation_exporter.py）。
  --render     用 MRQ 渲染已有 Sequence 的 RGB 帧到各 Camera 的 img1/（见 ue/render_episode.py）。
  --full       一键全流程：创建 Sequence + 导出标注 + 渲染 RGB。

用法（在 Unreal Editor Python Console 中执行）：
    # 最简单（推荐用 ue/run_task.py + resolved task）
    py "D:/path/to/code/ue/import_grf_episode.py"

    # 临时覆盖部分参数
    py "D:/path/to/code/ue/import_grf_episode.py" --episode "D:/path/to/other_episode" --replace-existing

    # 只导出 CV 标注
    py "D:/path/to/code/ue/import_grf_episode.py" --mode annotations

依赖：
    - unreal（UE 内置模块）
    - json、math、pathlib（标准库）

本脚本不依赖 gfootball、GRF_MARL 或 .venv。
"""

import argparse
import importlib
import json
import math
import re
import sys
from pathlib import Path

# 保证在 UE 中运行时能 import 同目录的模块（scene_apply / dataset_export / ...）
sys.path.insert(0, str(Path(__file__).resolve().parent))

# UE Python 会话内，已 import 过的 ue/ 模块会缓存在 sys.modules 中；多次执行
# 本脚本时需先强制重载，否则会运行到磁盘上已修改但会话里仍是旧版本的代码。
_UE_MODULE_NAMES = (
    "camera_projection", "annotation_utils", "dataset_export",
    "scene_apply", "annotation_exporter", "render_preset", "render_episode",
    "pose_bones", "pose_export", "player_motion",
)
for _name in _UE_MODULE_NAMES:
    if _name in sys.modules:
        importlib.reload(sys.modules[_name])

from scene_apply import (  # noqa: E402
    BALL_Z_OFFSET_CM,
    M_TO_CM,
    PLAYER_Z_CM,
    apply_preview,
    find_actor,
    find_all_actors,
    pos_m_to_cm,
)
from player_motion import (  # noqa: E402
    PlayerMotionTracker,
    gk_entity_ids_from_meta,
)
from dataset_export import load_episode, load_mapping  # noqa: E402
from annotation_exporter import export_annotations  # noqa: E402
from render_episode import render_sequences  # noqa: E402

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
        "--mode", choices=["preview", "sequence", "both", "annotations", "render", "full"], default="both",
        help="执行模式"
    )
    parser.add_argument(
        "--replace-existing", action="store_true", default=False,
        help="删除并重建已存在的 Level Sequence，不弹出覆盖确认对话框"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="（已弃用）legacy 配置文件路径；推荐 ue/run_task.py --resolved-task。"
    )
    parsed = parser.parse_args()
    return parsed


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


# ── Camera Cut 辅助（MRQ 渲染必需）─────────────────────────────────────

def _get_camera_binding_id(sequence, camera_binding):
    """多级尝试获取相机的 MovieSceneObjectBindingID。

    不同 UE 版本获取方式不同：binding 自带方法 / MovieSceneSequenceExtensions /
    MovieSceneBindingExtensions。
    """
    import unreal

    # 1) binding 自带方法
    m = getattr(camera_binding, "get_binding_id", None)
    if m is not None:
        try:
            bid = m()
            if bid is not None:
                return bid
        except Exception:
            pass
    # 2) 扩展库
    for lib_name in ("MovieSceneSequenceExtensions", "MovieSceneBindingExtensions"):
        lib = getattr(unreal, lib_name, None)
        if lib is None:
            continue
        m = getattr(lib, "get_binding_id", None)
        if m is None:
            continue
        for args in ((sequence, camera_binding), (camera_binding,)):
            try:
                bid = m(*args)
                if bid is not None:
                    return bid
            except Exception:
                continue

    # 诊断：打印 binding 对象与扩展库的可用成员，便于适配
    print(f"  [CameraCut 诊断] binding 类型: {type(camera_binding)}")
    print(
        "  [CameraCut 诊断] binding 成员:",
        sorted(n for n in dir(camera_binding) if not n.startswith("_")),
    )
    for lib_name in ("MovieSceneSequenceExtensions", "MovieSceneBindingExtensions"):
        lib = getattr(unreal, lib_name, None)
        print(f"  [CameraCut 诊断] {lib_name} 存在: {lib is not None}")
        if lib is not None:
            print(
                f"  [CameraCut 诊断] {lib_name} 成员:",
                sorted(n for n in dir(lib) if "binding" in n.lower() or "id" in n.lower()),
            )
    return None


def _add_camera_cut(sequence, camera_binding, total_output_frames: int) -> bool:
    """为 Sequence 添加 Camera Cut（MRQ 渲染必需，否则无活动相机导致白屏）。

    UE 5.8：LevelSequence.add_track(MovieSceneCameraCutTrack) + add_section() +
    set_camera_binding_id(...) + set_range(...)。

    注意：MRQ 渲染的**第 0 帧**必须落在 Camera Cut 内，否则首帧会用默认视角
    （实测首帧渲成原点视角）。这里除了 set_range，还显式设置起始/结束帧边界
    （FrameNumber）与 is_active，确保覆盖第 0 帧。
    """
    import unreal

    try:
        camera_cut_track = sequence.add_track(unreal.MovieSceneCameraCutTrack)
        section = camera_cut_track.add_section()
        binding_id = _get_camera_binding_id(sequence, camera_binding)
        if binding_id is None:
            print("  WARNING: 无法获取相机 binding id，未设置 Camera Cut（MRQ 渲染会无画面）")
            return False
        section.set_camera_binding_id(binding_id)
        end = int(total_output_frames)
        section.set_range(0, end)
        # 显式帧边界 + 激活（best-effort，兼容 5.8 的 API 差异）
        for setter in (
            lambda: section.set_start_frame(unreal.FrameNumber(0)),
            lambda: section.set_end_frame(unreal.FrameNumber(end)),
            lambda: section.set_start_frame_bounded(True),
            lambda: section.set_end_frame_bounded(True),
            lambda: section.set_is_active(True),
        ):
            try:
                setter()
            except Exception:
                pass
        print(f"  Camera Cut 已设置 [0, {end})")
        return True
    except Exception as e:
        print(f"  WARNING: 设置 Camera Cut 失败: {e}")
        return False


def _add_camera_transform_track(sequence, camera_binding, cam_actor, total_output_frames) -> bool:
    """把相机世界变换烘焙进 Sequence 的 3D Transform 轨道（帧 0 起有效）。

    相机是 possessable；若只有 Camera Cut 而没有任何轨道，MRQ 在第 0 帧可能拿不到
    相机变换而退回默认视角（实测首帧渲成原点视角）。烘焙静态变换（首尾各一帧）
    保证帧 0 起就是相机视角。

    旋转通道沿用项目约定（球员 yaw 写 Rotation.Z）：X=Roll、Y=Pitch、Z=Yaw。
    """
    import unreal

    try:
        transform_track = camera_binding.add_track(unreal.MovieScene3DTransformTrack)
        section = transform_track.add_section()
        section.set_range(0, int(total_output_frames))
        cmap = _build_channel_map(section.get_all_channels())
        loc = cam_actor.get_actor_location()
        rot = cam_actor.get_actor_rotation()
        px, py, pz = float(loc.x), float(loc.y), float(loc.z)
        pitch, yaw, roll = float(rot.pitch), float(rot.yaw), float(rot.roll)
        for kf in (0, max(0, int(total_output_frames) - 1)):
            add_double_channel_key(cmap["Location.X"], kf, px)
            add_double_channel_key(cmap["Location.Y"], kf, py)
            add_double_channel_key(cmap["Location.Z"], kf, pz)
            add_double_channel_key(cmap["Rotation.X"], kf, roll)
            add_double_channel_key(cmap["Rotation.Y"], kf, pitch)
            add_double_channel_key(cmap["Rotation.Z"], kf, yaw)
        print(
            f"  Camera transform baked: loc=({px:.0f},{py:.0f},{pz:.0f})"
            f" rot=(p{pitch:.0f} y{yaw:.0f} r{roll:.0f})"
        )
        return True
    except Exception as e:
        print(f"  WARNING: 烘焙相机变换失败: {e}")
        return False


# ── Sequence 模式 ───────────────────────────────────────────────────────

def create_sequence(meta: dict, frames: list, mapping: dict, replace_existing: bool = False,
                    package_path: str = None, sequences_cfg: list = None,
                    ball_rolling_cfg: dict = None):
    """创建带关键帧变换的 Level Sequence 资产。

    如果提供了 sequences_cfg，则为每个条目创建一个 Sequence，
    每个都包含相同的球员/球数据并绑定摄像机。
    """
    import unreal

    actors = find_all_actors(mapping)
    if not actors:
        return

    # GK 身份来自 meta.entities[].is_goalkeeper（不假设 L0/R0）；缺失时回退
    gk_ids = gk_entity_ids_from_meta(meta)

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

                # 自动设置 Camera Cut（MRQ 渲染必需）。若自动设置失败，
                # 需在 Sequencer 手动右键摄像机轨道 → "Set as Camera Cut"。
                _add_camera_cut(seq, camera_binding, total_output_frames)
                # 烘焙相机变换轨道：保证帧 0 起就是相机视角（否则 MRQ 首帧退回默认视角）
                _add_camera_transform_track(seq, camera_binding, cam_actor, total_output_frames)
                unreal.log(f"  Camera bound: {camera_actor_name} (Camera Cut set)")

        # ── 为每个 actor 写入变换关键帧 ──────────────────────────────
        total_transform_keys = 0
        for entity_id, actor in actors.items():
            binding, channels, section = actor_bindings[entity_id]
            cmap = _build_channel_map(channels)

            # 朝向由 PlayerMotionTracker 统一计算（与 preview/annotation/pose 同一套），
            # 速度优先取 frame 的 velocity_mps，缺失时按位置差分；位置仍由 frame 决定。
            player_tracker = None
            # 写入 Sequence 的连续 yaw：facing_deg 归一到 [-180,180]，若直接写入，
            # 跨 ±180° 边界时 Sequencer 线性插值会沿长路径旋转约 350°；
            # 这里用 _unwind_angle 展开为连续角度（首帧取 facing_deg 为起点）。
            player_yaw_continuous = None

            for fi, frame in enumerate(frames):
                frame_time = fi * source_step
                kf = int(round(frame_time * playback_fps))

                if entity_id == "BALL":
                    ball_pos = frame["ball"]["position_m"]
                    px, py, pz = pos_m_to_cm(ball_pos)
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
                        px, py, _ = pos_m_to_cm(player_data["position_m"])
                        add_double_channel_key(cmap["Location.X"], kf, px)
                        add_double_channel_key(cmap["Location.Y"], kf, py)
                        add_double_channel_key(cmap["Location.Z"], kf, PLAYER_Z_CM)

                        if player_tracker is None:
                            player_tracker = PlayerMotionTracker()
                        params = player_tracker.update(
                            player_data["position_m"],
                            player_data.get("velocity_mps"),
                            float(frame.get("time_seconds", frame_time)),
                            ball_position_m=(
                                frame["ball"]["position_m"]
                                if entity_id in gk_ids else None
                            ),
                            face_ball=entity_id in gk_ids,
                            is_goalkeeper=entity_id in gk_ids,
                        )
                        yaw = params["facing_deg"]
                        if player_yaw_continuous is None:
                            player_yaw_continuous = yaw
                        else:
                            player_yaw_continuous = _unwind_angle(player_yaw_continuous, yaw)
                        add_double_channel_key(cmap["Rotation.Z"], kf, player_yaw_continuous)

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


# ── 入口 ───────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # 加载配置。旧流程（--config / 根目录 ue_import_config.json）已弃用：
    # 推荐 ue/run_task.py --resolved-task <resolved-task.json>。
    cfg_defaults = {}
    if args.config:
        cfg_path = Path(args.config)
        # 相对路径按脚本目录（仓库根）解析，避免依赖 UE Python 的 CWD
        if not cfg_path.is_absolute():
            cfg_path = Path(__file__).resolve().parent.parent / args.config
        print(
            "WARNING: Legacy UE config mode (--config) is deprecated. "
            "Use ue/run_task.py with a resolved task.",
            file=sys.stderr,
        )
    else:
        cfg_path = Path(__file__).resolve().parent.parent / "ue_import_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if not k.startswith("comment_"):
                cfg_defaults[k] = v
    else:
        print(
            f"ERROR: 配置文件不存在: {cfg_path}\n"
            "根目录隐式配置已移除。请改用：\n"
            '  py ".../ue/run_task.py" --resolved-task ".../.futsalmot/runtime/<task>/resolved-task.json"\n'
            "（先运行 uv run grf-ue task ue-command <task> 获取该命令）",
            file=sys.stderr,
        )
        sys.exit(1)

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

    # annotations 模式：只导出 CV 标注，不做 preview / sequence
    if mode == "annotations":
        ann_cfg = cfg_defaults.get("annotation_export") or {}
        if ann_cfg.get("enabled", True) is False:
            print("annotation_export.enabled = false，跳过标注导出")
            return
        output_dir = Path(
            ann_cfg.get("output_dir") or (episode_dir.parent / "dataset")
        )
        export_annotations(episode_dir, mapping_path, output_dir, ann_cfg)
        print("\nDone.")
        return

    # full / render 模式：一键全流程（sequence + annotations + render）或仅渲染
    if mode in ("full", "render"):
        ann_cfg = cfg_defaults.get("annotation_export") or {}
        ann_out = Path(ann_cfg.get("output_dir") or (episode_dir.parent / "dataset"))
        seq_pkg = cfg_defaults.get("sequence_package_path") or DEFAULT_SEQUENCE_PACKAGE_PATH
        seq_list = cfg_defaults.get("sequences") or None
        if mode == "full":
            meta, frames = load_episode(episode_dir)
            mapping = load_mapping(mapping_path)
            ball_rolling_cfg = cfg_defaults.get("ball_rolling", None)
            print("\n--- 全流程：创建 Level Sequence ---")
            create_sequence(meta, frames, mapping, replace_existing, seq_pkg, seq_list, ball_rolling_cfg)
            if ann_cfg.get("enabled", True) is not False:
                export_annotations(episode_dir, mapping_path, ann_out, ann_cfg)
            else:
                print("annotation_export.enabled = false，跳过标注导出")
        render_sequences(seq_list, ann_cfg, seq_pkg, episode_dir, ann_out, mapping_path)
        print("\n已提交。MRQ 渲染为异步执行（不阻塞编辑器），完成后自动复制 RGB 到 img1/（"
              "及 Instance-ID Mask 到 mask/）并写 render_summary.json。")
        return

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
