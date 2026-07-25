"""
Unreal Engine Python script — Import GRF-UE episode into a Level Sequence.

Two modes (default: both):
  --preview    Set actor transforms directly in the level (Editor preview).
  --sequence   Create / overwrite a Level Sequence asset with keyframed transforms.

Usage (in Unreal Editor Python Console):
    # 最简单（参数在 ue_import_config.json 中）
    py "D:/path/to/code/ue/import_grf_episode.py"

    # 临时覆盖部分参数
    py "D:/path/to/code/ue/import_grf_episode.py" --episode "D:/path/to/other_episode" --replace-existing

Dependencies:
    - unreal (UE built-in module)
    - json, math, pathlib (stdlib)

This script does NOT require gfootball, GRF_MARL, or the .venv.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

M_TO_CM = 100.0
SPEED_THRESHOLD_CM = 5.0  # cm/s, below this keep previous yaw
BALL_Z_OFFSET_CM = 2.0  # offset so GRF ball_z (~0.11 * 100) + offset ≈ 13cm
PLAYER_Z_CM = 90.0  # fixed ground level for player actors
DEFAULT_SEQUENCE_PACKAGE_PATH = "/Game/FutsalMOT/Sequences"
EXPECTED_CHANNEL_NAMES = [
    "Location.X", "Location.Y", "Location.Z",
    "Rotation.X", "Rotation.Y", "Rotation.Z",
    "Scale.X", "Scale.Y", "Scale.Z",
]


# ── Argument parsing ────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Import GRF-UE episode into Unreal Engine"
    )
    parser.add_argument(
        "--episode", default=None,
        help="Path to episode directory (contains meta.json, frames.jsonl)"
    )
    parser.add_argument(
        "--mapping", default=None,
        help="Path to actor mapping JSON file"
    )
    parser.add_argument(
        "--mode", choices=["preview", "sequence", "both"], default="both",
        help="Execution mode"
    )
    parser.add_argument(
        "--replace-existing", action="store_true", default=False,
        help="Delete and recreate an existing Level Sequence without showing an overwrite dialog"
    )
    parser.add_argument(
        "--animation-config", type=str, default=None,
        help="Optional animation configuration JSON. When omitted, locomotion animation and ball rolling are not added."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Deprecated: config is auto-loaded from ue_import_config.json next to this script."
    )
    parsed = parser.parse_args()
    return parsed


# ── Load helpers ────────────────────────────────────────────────────────

def load_episode(episode_dir: Path):
    """Load meta.json and frames.jsonl from an episode directory."""
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
    """Load actor mapping JSON."""
    with open(path) as f:
        return json.load(f)


# ── Actor helpers ───────────────────────────────────────────────────────

def _get_actor_subsystem():
    """Get EditorActorSubsystem."""
    import unreal
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def find_actor(name: str):
    """Find a UE actor by label or name, case-insensitive."""
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
    """Find all actors in the mapping. Returns {entity_id: actor} dict."""
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


# ── Maths ───────────────────────────────────────────────────────────────

def build_yaw(dx: float, dy: float, prev_yaw: float) -> float:
    """Compute yaw from movement delta with low-speed hysteresis."""
    speed = math.sqrt(dx * dx + dy * dy)
    if speed < SPEED_THRESHOLD_CM:
        return prev_yaw
    return math.degrees(math.atan2(dy, dx))


def _pos_m_to_cm(pos_m: list) -> tuple:
    """Convert [x, y, z] meters to (x_cm, y_cm, z_cm)."""
    return (pos_m[0] * M_TO_CM, pos_m[1] * M_TO_CM, pos_m[2] * M_TO_CM)


# ── Animation config ────────────────────────────────────────────────────

def load_animation_config(path: Path) -> dict:
    """Load animation configuration JSON and validate required fields."""
    with open(path) as f:
        cfg = json.load(f)

    enabled = cfg.get("enabled", True)

    if enabled:
        anims = cfg.get("animations", {})
        for key in ("idle", "walk", "run"):
            if key not in anims or not anims[key]:
                raise ValueError(f"Animation config missing '{key}' asset path")

    loco = cfg.get("locomotion", {})
    for key in ("idle_max_speed_mps", "run_min_speed_mps", "smoothing_window",
                 "minimum_segment_frames"):
        if key not in loco:
            raise ValueError(f"Locomotion config missing '{key}'")

    ball_cfg = cfg.get("ball", {})
    if "radius_m" not in ball_cfg:
        raise ValueError("Ball config missing 'radius_m'")

    return cfg


# ── Ball rolling ─────────────────────────────────────────────────────────

def compute_ball_rotation_quat(
    ball_positions,
    radius_m,
    minimum_move_distance_m,
    roll_sign,
):
    """Compute ball rotation per output frame using quaternion accumulation.

    Args:
        ball_positions: list of (x, y, z) in meters.
        radius_m: ball radius in meters.
        minimum_move_distance_m: minimum displacement to count as moving.
        roll_sign: sign multiplier (1.0 or -1.0) for axis correction.

    Returns:
        list of (roll_deg, pitch_deg, yaw_deg) for each frame.
        First frame is (0, 0, 0).
    """
    import unreal

    current_quat = unreal.MathLibrary.quat_make_from_euler(unreal.Vector(0.0, 0.0, 0.0))
    rotations = [(0.0, 0.0, 0.0)]  # frame 0

    prev_roll, prev_pitch, prev_yaw = 0.0, 0.0, 0.0

    for i in range(1, len(ball_positions)):
        bx_prev, by_prev, _ = ball_positions[i - 1]
        bx_cur, by_cur, bz_cur = ball_positions[i]

        dx = bx_cur - bx_prev
        dy = by_cur - by_prev
        distance = math.hypot(dx, dy)

        # Check if ball is near ground (within radius + 3cm tolerance)
        if bz_cur <= radius_m + 0.03:
            if distance > minimum_move_distance_m:
                # Rolling on ground
                angle_degrees = math.degrees(distance / radius_m)
                axis = unreal.Vector(-dy / distance, dx / distance, 0.0)
                rotation_vector = unreal.Vector(
                    axis.x * angle_degrees * roll_sign,
                    axis.y * angle_degrees * roll_sign,
                    0.0,
                )
                delta_quat = unreal.MathLibrary.quat_make_from_rotation_vector(rotation_vector)
                current_quat = unreal.MathLibrary.multiply_quat_quat(delta_quat, current_quat)
        # else: airborne, keep current rotation (already set above)

        current_quat = unreal.MathLibrary.quat_normalized(current_quat)
        rotator = unreal.MathLibrary.quat_rotator(current_quat)
        roll = _unwind_angle(prev_roll, rotator.roll)
        pitch = _unwind_angle(prev_pitch, rotator.pitch)
        yaw = _unwind_angle(prev_yaw, rotator.yaw)
        rotations.append((roll, pitch, yaw))
        prev_roll, prev_pitch, prev_yaw = roll, pitch, yaw

    return rotations


def _unwind_angle(previous, current):
    """Unwind angle to avoid -180/180 jumps."""
    while current - previous > 180.0:
        current -= 360.0
    while current - previous < -180.0:
        current += 360.0
    return current


# ── Sequence key helpers ────────────────────────────────────────────────

def add_double_channel_key(channel, frame_index: int, value: float, *, interpolation=None):
    """Add a key to a MovieSceneDoubleChannel with proper FrameNumber wrapping.

    Args:
        channel: The MovieSceneDoubleChannel to add a key to.
        frame_index: Display-rate frame index (plain int). Wrapped in FrameNumber internally.
        value: Numeric key value.
        interpolation: MovieSceneKeyInterpolation enum value. Defaults to LINEAR.

    Raises:
        ValueError: If channel is None or value is non-finite.
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
    """Get channel name with trailing numeric suffix stripped (UE 5.8+ appends _NNN)."""
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
    """Safely get a display name for a channel, tolerant of UE version differences."""
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
    """Build a dict mapping canonical channel names (e.g. 'Location.X') to channel objects.

    Strips UE 5.8+ numeric suffixes (_NNN) before matching.
    Falls back to positional indexing with a length check if needed.
    Logs raw channel names for diagnostics.
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
    # Fallback: positional indexing with length check
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


# ── Preview mode ────────────────────────────────────────────────────────

def apply_preview(meta: dict, frames: list, mapping: dict):
    """Set actor transforms frame-by-frame in the level."""
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


# ── Smoke test ──────────────────────────────────────────────────────────

def _smoke_test_sequencer_api(actors: dict, total_output_frames: int, package_path: str = None):
    """Create a throwaway sequence and validate add_key accepts FrameNumber.

    Returns (temp_sequence, channel_name_map) or raises on failure.
    """
    import unreal

    pkg = package_path or DEFAULT_SEQUENCE_PACKAGE_PATH
    seq_factory = unreal.LevelSequenceFactoryNew()
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    temp_seq = asset_tools.create_asset("_TEMP_SMOKE", pkg, None, seq_factory)
    if not temp_seq:
        raise RuntimeError("Failed to create smoke test sequence.")

    # Bind the first actor and create a transform track
    first_actor = next(iter(actors.values()))
    binding = temp_seq.add_possessable(first_actor)
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    channels = section.get_all_channels()
    cmap = _build_channel_map(channels)

    # Add two test keys and remove them via remove_key()
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


# ── Sequence mode ───────────────────────────────────────────────────────

def create_sequence(meta: dict, frames: list, mapping: dict, replace_existing: bool = False,
                    anim_cfg: dict = None, package_path: str = None, sequences_cfg: list = None):
    """Create Level Sequence assets with keyframed transforms.

    If sequences_cfg is provided, creates one sequence per entry, each with
    the same player/ball data plus a camera binding.
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

    # Ensure package directory
    if not unreal.EditorAssetLibrary.does_directory_exist(pkg_path):
        unreal.EditorAssetLibrary.make_directory(pkg_path)

    # Smoke test once
    temp_seq, _ = _smoke_test_sequencer_api(actors, total_output_frames, pkg_path)
    unreal.EditorAssetLibrary.delete_asset(f"{pkg_path}/_TEMP_SMOKE")

    # Determine which sequences to create
    if sequences_cfg:
        seq_list = sequences_cfg
    else:
        seq_list = [{"name": f"SEQ_{meta.get('episode_id', 'episode_0000')}"}]

    for seq_entry in seq_list:
        seq_name = seq_entry["name"]
        camera_actor_name = seq_entry.get("camera_actor")
        sequence_asset_path = f"{pkg_path}/{seq_name}"

        print(f"\n--- Creating Sequence: {seq_name} ---")

        # Delete existing
        if unreal.EditorAssetLibrary.does_asset_exist(sequence_asset_path):
            if not replace_existing:
                raise RuntimeError(
                    f"Level Sequence already exists: {sequence_asset_path}. "
                    f"Run again with --replace-existing."
                )
            unreal.EditorAssetLibrary.delete_asset(sequence_asset_path)
            print(f"  Deleted existing: {sequence_asset_path}")

        # Create new sequence
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

        # ── Bind all actors (players + ball) ────────────────────────────
        actor_bindings = {}
        for entity_id, actor in actors.items():
            binding, channels, section = _build_entity_binding(
                seq, entity_id, actor, total_output_frames,
            )
            actor_bindings[entity_id] = (binding, channels, section)

        # ── Bind camera if specified ────────────────────────────────────
        camera_binding = None
        if camera_actor_name:
            cam_actor = find_actor(camera_actor_name)
            if not cam_actor:
                print(f"  WARNING: Camera actor '{camera_actor_name}' not found, skipping camera binding")
            else:
                camera_binding = seq.add_possessable(cam_actor)
                print(f"  Camera bound: {camera_actor_name}")

                # Camera is bound as possessable. UE 5.8 Python API does not expose
                # CameraCutTrack creation — manually set camera in Sequencer:
                # right-click camera track → "Set as Camera Cut".
                unreal.log(f"  Camera bound: {camera_actor_name} (set as Camera Cut manually in Sequencer)")

        # ── Write transform keys for each actor ─────────────────────────
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

            # Verify key counts
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

        # ── Ball rolling (if config provided) ───────────────────────────
        _add_ball_rolling(anim_cfg, frames, seq, source_step, playback_fps,
                          total_output_frames, actor_bindings)

        # ── Save & open ────────────────────────────────────────────────
        saved = unreal.EditorAssetLibrary.save_loaded_asset(seq, only_if_is_dirty=False)
        if not saved:
            raise RuntimeError(f"Failed to save: {sequence_asset_path}")
        print(f"  Saved: {sequence_asset_path}")

        try:
            unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(seq)
            print("  Opened in Sequencer.")
        except Exception as exc:
            unreal.log(f"Warning: could not open in Sequencer: {exc}")

    # ── Summary ─────────────────────────────────────────────────────────
    player_count = sum(1 for eid in actors if eid != "BALL")
    print()
    print("GRF UE IMPORT PASS")
    print(f"Sequences created: {len(seq_list)}")
    for s in seq_list:
        cam = s.get("camera_actor", "none")
        print(f"  {pkg_path}/{s['name']} (camera: {cam})")
    print(f"Players: {player_count}, Ball: {'BALL' in actors}")
    print(f"Package path: {pkg_path}")



def _add_ball_rolling(anim_cfg, frames, seq, source_step, playback_fps,
                       total_output_frames, actor_bindings=None):
    """Add ball rolling rotation keys if animation config enables it."""
    import unreal

    if not anim_cfg or not anim_cfg.get("ball", {}).get("enabled", True):
        return

    ball_cfg = anim_cfg["ball"]
    num_steps = len(frames)

    # Find ball's channel map from actor_bindings
    cmap_ref = None
    if actor_bindings and "BALL" in actor_bindings:
        _, _channels, _section = actor_bindings["BALL"]
        cmap_ref = _build_channel_map(_channels)

    if not cmap_ref or "Rotation.X" not in cmap_ref:
        return

    ball_positions = []
    for frame in frames:
        bpos = frame["ball"]["position_m"]
        ball_positions.append((bpos[0], bpos[1], bpos[2]))

    rotations = compute_ball_rotation_quat(
        ball_positions,
        ball_cfg["radius_m"],
        ball_cfg.get("minimum_move_distance_m", 0.0001),
        ball_cfg.get("roll_sign", 1.0),
    )

    if "Rotation.X" not in cmap_ref:
        return

    for fi in range(num_steps):
        kf = int(round(fi * source_step * playback_fps))
        r, p, y = rotations[fi]
        add_double_channel_key(cmap_ref["Rotation.X"], kf, r)
        add_double_channel_key(cmap_ref["Rotation.Y"], kf, p)
        add_double_channel_key(cmap_ref["Rotation.Z"], kf, y)


def _build_entity_binding(sequence, entity_id, actor, total_output_frames):
    """Add a possessable binding + transform track + section for entity_id.

    Returns (binding, channels_list, section) so caller can inspect channels.
    """
    import unreal
    binding = sequence.add_possessable(actor)
    transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
    transform_section = transform_track.add_section()
    transform_section.set_range(0, int(total_output_frames))
    channels = transform_section.get_all_channels()
    return binding, channels, transform_section


# ── Preview helpers (shared, no unreal dependency at module level) ──────

def _apply_ball_frame(actors: dict, frame: dict):
    """Position ball actor from frame data."""
    import unreal
    if "BALL" not in actors:
        return
    ball_pos = frame["ball"]["position_m"]
    px, py, pz = _pos_m_to_cm(ball_pos)
    actors["BALL"].set_actor_location(
        unreal.Vector(px, py, pz + BALL_Z_OFFSET_CM), False, False
    )


def _apply_player_frame(actors: dict, frame: dict, prev_yaws: dict, prev_positions: dict):
    """Position and rotate player actors from frame data."""
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


# ── Entry ───────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # Load config from hardcoded path (next to this script)
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
    anim_config_path = Path(
        args.animation_config if args.animation_config
        else cfg_defaults.get("animation_config", "")
    ) if (args.animation_config or cfg_defaults.get("animation_config")) else None

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

    # Load animation config if provided
    anim_cfg = None
    if anim_config_path:
        if not anim_config_path.exists():
            print(f"ERROR: Animation config not found: {anim_config_path}", file=sys.stderr)
            sys.exit(1)
        anim_cfg = load_animation_config(anim_config_path)

    if mode in ("preview", "both"):
        print("\n--- Preview mode: setting actor transforms ---")
        apply_preview(meta, frames, mapping)

    if mode in ("sequence", "both"):
        print("\n--- Sequence mode: creating Level Sequence ---")
        seq_pkg = cfg_defaults.get("sequence_package_path") or DEFAULT_SEQUENCE_PACKAGE_PATH
        seq_list = cfg_defaults.get("sequences") or None
        create_sequence(meta, frames, mapping, replace_existing, anim_cfg, seq_pkg, seq_list)

    print("\nDone.")


if __name__ == "__main__":
    main()
