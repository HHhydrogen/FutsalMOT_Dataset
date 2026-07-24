"""
Unreal Engine Python script — Import GRF-UE episode into a Level Sequence.

Two modes (default: both):
  --preview    Set actor transforms directly in the level (Editor preview).
  --sequence   Create / overwrite a Level Sequence asset with keyframed transforms.

Usage (in Unreal Editor Python Console):
    py "D:/path/to/ue/import_grf_episode.py" --episode "D:/path/to/outputs/episode_0001" --mapping "D:/path/to/ue/actor_mapping.example.json" --replace-existing

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
from dataclasses import dataclass
from pathlib import Path

M_TO_CM = 100.0
SPEED_THRESHOLD_CM = 5.0  # cm/s, below this keep previous yaw
BALL_Z_OFFSET_CM = 2.0  # offset so GRF ball_z (~0.11 * 100) + offset ≈ 13cm
PLAYER_Z_CM = 90.0  # fixed ground level for player actors
SEQUENCE_PACKAGE_PATH = "/Game/GRF/Sequences"
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
        "--episode", required=True,
        help="Path to episode directory (contains meta.json, frames.jsonl)"
    )
    parser.add_argument(
        "--mapping", required=True,
        help="Path to actor mapping JSON file"
    )
    parser.add_argument(
        "--mode", choices=["preview", "sequence", "both"], default="both",
        help="Execution mode"
    )
    parser.add_argument(
        "--replace-existing", action="store_true",
        help="Delete and recreate an existing Level Sequence without showing an overwrite dialog"
    )
    parser.add_argument(
        "--animation-config", type=str, default=None,
        help="Optional animation configuration JSON. When omitted, locomotion animation and ball rolling are not added."
    )
    return parser.parse_args()


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
    return math.degrees(math.atan2(dx, dy))


def _pos_m_to_cm(pos_m: list) -> tuple:
    """Convert [x, y, z] meters to (x_cm, y_cm, z_cm)."""
    return (pos_m[0] * M_TO_CM, pos_m[1] * M_TO_CM, pos_m[2] * M_TO_CM)


# ── Animation config ────────────────────────────────────────────────────

def load_animation_config(path: Path) -> dict:
    """Load animation configuration JSON and validate required fields."""
    with open(path) as f:
        cfg = json.load(f)

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


def precheck_animation_assets(anim_cfg: dict, actors: dict, mapping: dict):
    """Verify animation assets exist and players have skeletal mesh components."""
    import unreal

    anims = anim_cfg["animations"]
    for state, path in anims.items():
        asset = unreal.load_asset(path)
        if asset is None:
            raise RuntimeError(
                f"Animation asset not found: state={state}, path={path}"
            )

    # Check each player has a SkeletalMeshComponent
    for entity_id, actor_name in mapping.items():
        if entity_id == "BALL":
            continue
        actor = actors.get(entity_id)
        if not actor:
            continue
        skel_comp = actor.get_component_by_class(unreal.SkeletalMeshComponent)
        if not skel_comp:
            raise RuntimeError(
                f"{actor_name} has no SkeletalMeshComponent. "
                "Cannot add animation track."
            )
        # Check skeleton compatibility
        unused = actor.get_skeletal_mesh_asset()
        for state, path in anims.items():
            anim_asset = unreal.load_asset(path)
            if anim_asset and hasattr(anim_asset, "get_skeleton"):
                anim_skeleton = anim_asset.get_skeleton()
                if anim_skeleton and unused:
                    actor_skeleton_asset = unused.get_skeleton()
                    if actor_skeleton_asset and anim_skeleton.get_name() != actor_skeleton_asset.get_name():
                        print(
                            f"  WARNING: Animation skeleton mismatch: "
                            f"actor={actor_name}, animation={path} ({anim_skeleton.get_name()} vs "
                            f"{actor_skeleton_asset.get_name()})"
                        )
    print("  Animation assets: PASS")


# ── Speed and locomotion ────────────────────────────────────────────────

def compute_speeds_mps(positions, step_seconds):
    """Compute per-frame speed from 2D positions using central difference.

    Args:
        positions: list of (x_m, y_m) tuples.
        step_seconds: time between GRF source steps.

    Returns:
        list of speeds in meters per second.
    """
    if step_seconds <= 0:
        raise ValueError(f"step_seconds must be > 0, got {step_seconds}")
    if len(positions) < 2:
        raise ValueError(f"Need at least 2 positions, got {len(positions)}")

    speeds = []
    for index in range(len(positions)):
        if index == 0:
            x0, y0 = positions[0]
            x1, y1 = positions[1]
            dt = step_seconds
        elif index == len(positions) - 1:
            x0, y0 = positions[index - 1]
            x1, y1 = positions[index]
            dt = step_seconds
        else:
            x0, y0 = positions[index - 1]
            x1, y1 = positions[index + 1]
            dt = 2.0 * step_seconds

        distance = math.hypot(x1 - x0, y1 - y0)
        speed = distance / dt
        if not math.isfinite(speed):
            raise ValueError(f"Non-finite speed at index {index}: {speed}")
        speeds.append(speed)

    return speeds


def smooth_values(values, window_size):
    """Simple moving average smoothing.

    Args:
        values: list of floats.
        window_size: odd positive int.

    Returns:
        smoothed list of same length.
    """
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError(f"window_size must be positive odd, got {window_size}")
    n = len(values)
    if n == 0:
        return []
    half = window_size // 2
    smoothed = []
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        smoothed.append(sum(values[left:right]) / (right - left))
    return smoothed


@dataclass
class LocomotionSegment:
    """A contiguous section of a single locomotion state."""
    state: str
    source_start: int
    source_end_exclusive: int
    mean_speed_mps: float


def classify_states(speeds, cfg, source_step, playback_fps):
    """Classify source frames into IDLE/WALK/RUN states with hysteresis.

    Returns list of LocomotionSegment covering all source frames.
    """
    idle_max = cfg["idle_max_speed_mps"]
    run_min = cfg["run_min_speed_mps"]
    min_frames = cfg["minimum_segment_frames"]

    # Convert min_frames from output frames to source frames
    min_source = max(1, int(round(min_frames / (playback_fps * source_step))))

    hysteresis_idle_up = idle_max + 0.10
    hysteresis_run_down = run_min - 0.30

    # Raw classification
    raw = []
    for s in speeds:
        if s <= idle_max:
            raw.append("idle")
        elif s >= run_min:
            raw.append("run")
        else:
            raw.append("walk")

    # Apply hysteresis
    states = list(raw)
    for i in range(1, len(states)):
        if states[i] == "walk" and states[i - 1] == "idle":
            if speeds[i] <= hysteresis_idle_up:
                states[i] = "idle"
        elif states[i] == "run" and states[i - 1] == "walk":
            if speeds[i] <= hysteresis_run_down:
                states[i] = "walk"
        elif states[i] == "walk" and states[i - 1] == "run":
            if speeds[i] <= hysteresis_run_down:
                states[i] = "run"
        elif states[i] == "idle" and states[i - 1] == "walk":
            if speeds[i] <= idle_max:
                states[i] = "walk"

    # Initial segments
    segments = []
    start = 0
    for i in range(1, len(states)):
        if states[i] != states[start]:
            mean_speed = sum(speeds[start:i]) / (i - start)
            segments.append(LocomotionSegment(
                state=states[start],
                source_start=start,
                source_end_exclusive=i,
                mean_speed_mps=mean_speed,
            ))
            start = i
    mean_speed = sum(speeds[start:]) / (len(speeds) - start)
    segments.append(LocomotionSegment(
        state=states[start],
        source_start=start,
        source_end_exclusive=len(states),
        mean_speed_mps=mean_speed,
    ))

    # Merge short segments
    merged = list(segments)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            seg = merged[i]
            seg_frames = int(round((seg.source_end_exclusive - seg.source_start) * source_step * playback_fps))
            if seg_frames >= min_frames:
                i += 1
                continue
            # Short segment: merge with neighbor
            if i == 0 and len(merged) > 1:
                # Merge into next
                nxt = merged[1]
                combined = _merge_segments(seg, nxt, source_step, playback_fps)
                merged[1] = combined
                merged.pop(0)
                changed = True
            elif i == len(merged) - 1 and len(merged) > 1:
                # Merge into previous
                prv = merged[i - 1]
                combined = _merge_segments(prv, seg, source_step, playback_fps)
                merged[i - 1] = combined
                merged.pop(i)
                changed = True
            elif 0 < i < len(merged) - 1:
                prv = merged[i - 1]
                nxt = merged[i + 1]
                prv_frames = int(round((prv.source_end_exclusive - prv.source_start) * source_step * playback_fps))
                nxt_frames = int(round((nxt.source_end_exclusive - nxt.source_start) * source_step * playback_fps))
                if prv_frames >= nxt_frames:
                    combined = _merge_segments(prv, seg, source_step, playback_fps)
                    merged[i - 1] = combined
                    merged.pop(i)
                else:
                    combined = _merge_segments(seg, nxt, source_step, playback_fps)
                    merged[i + 1] = combined
                    merged.pop(i)
                changed = True
            else:
                i += 1
            if changed:
                break

    return merged


def _merge_segments(a, b, source_step, playback_fps):
    """Merge two adjacent LocomotionSegments into one.

    The merged segment takes the state from the longer segment (in output frames).
    If equal, takes state from the first segment.
    """
    a_frames = int(round((a.source_end_exclusive - a.source_start) * source_step * playback_fps))
    b_frames = int(round((b.source_end_exclusive - b.source_start) * source_step * playback_fps))
    if a_frames >= b_frames:
        state = a.state
    else:
        state = b.state
    total_speed_sum = a.mean_speed_mps * (a.source_end_exclusive - a.source_start) + \
                      b.mean_speed_mps * (b.source_end_exclusive - b.source_start)
    total_len = (a.source_end_exclusive - a.source_start) + (b.source_end_exclusive - b.source_start)
    return LocomotionSegment(
        state=state,
        source_start=a.source_start,
        source_end_exclusive=b.source_end_exclusive,
        mean_speed_mps=total_speed_sum / total_len,
    )


def compute_play_rate(segment, cfg):
    """Compute animation play rate for a locomotion segment."""
    if segment.state == "idle":
        rate = cfg.get("idle_play_rate", 1.0)
    elif segment.state == "walk":
        rate = segment.mean_speed_mps / cfg["walk_reference_speed_mps"]
    else:
        rate = segment.mean_speed_mps / cfg["run_reference_speed_mps"]

    rate = max(cfg["minimum_play_rate"], min(cfg["maximum_play_rate"], rate))
    if not math.isfinite(rate) or rate <= 0:
        rate = 1.0
    return rate


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
        euler = unreal.MathLibrary.quat_to_euler(current_quat)
        roll = _unwind_angle(prev_roll, euler.x)
        pitch = _unwind_angle(prev_pitch, euler.y)
        yaw = _unwind_angle(prev_yaw, euler.z)
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

def _smoke_test_sequencer_api(actors: dict, total_output_frames: int):
    """Create a throwaway sequence and validate add_key accepts FrameNumber.

    Returns (temp_sequence, channel_name_map) or raises on failure.
    """
    import unreal

    seq_factory = unreal.LevelSequenceFactoryNew()
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    temp_seq = asset_tools.create_asset("_TEMP_SMOKE", SEQUENCE_PACKAGE_PATH, None, seq_factory)
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
                    anim_cfg: dict = None):
    """Create a Level Sequence asset with keyframed transforms."""
    import unreal

    actors = _find_all_actors(mapping)
    if not actors:
        return

    episode_id = meta.get("episode_id", "episode_0000")
    num_steps = meta["timing"]["num_steps"]
    source_step = meta["timing"]["source_step_seconds"]
    playback_fps = int(meta["timing"].get("playback_fps", 30))

    # Log Unreal version
    unreal.log(f"Unreal version: {unreal.SystemLibrary.get_engine_version()}")

    # Total output frames at display rate
    total_output_frames = int(math.ceil(num_steps * source_step * playback_fps))

    # ── Handle existing asset ──────────────────────────────────────────
    sequence_asset_path = f"{SEQUENCE_PACKAGE_PATH}/SEQ_{episode_id}"
    if unreal.EditorAssetLibrary.does_asset_exist(sequence_asset_path):
        if not replace_existing:
            raise RuntimeError(
                f"Level Sequence already exists: {sequence_asset_path}. "
                f"Run again with --replace-existing."
            )
        deleted = unreal.EditorAssetLibrary.delete_asset(sequence_asset_path)
        if not deleted:
            raise RuntimeError(
                f"Failed to delete existing Level Sequence: {sequence_asset_path}"
            )
        print(f"  Deleted existing: {sequence_asset_path}")

    # ── Ensure directory ────────────────────────────────────────────────
    if not unreal.EditorAssetLibrary.does_directory_exist(SEQUENCE_PACKAGE_PATH):
        unreal.EditorAssetLibrary.make_directory(SEQUENCE_PACKAGE_PATH)

    # ── Create sequence ────────────────────────────────────────────────
    seq_factory = unreal.LevelSequenceFactoryNew()
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    sequence = asset_tools.create_asset(f"SEQ_{episode_id}", SEQUENCE_PACKAGE_PATH, None, seq_factory)
    if not sequence:
        raise RuntimeError(f"Failed to create Level Sequence asset at {sequence_asset_path}")

    # ── Timeline settings ──────────────────────────────────────────────
    display_rate = unreal.FrameRate(numerator=playback_fps, denominator=1)
    sequence.set_display_rate(display_rate)
    sequence.set_playback_start(0)
    sequence.set_playback_end(total_output_frames)
    print(f"  Timeline: {total_output_frames} frames @ {playback_fps} FPS")

    # ── Smoke test on a temporary sequence before batch writing ─────────
    temp_seq, _ = _smoke_test_sequencer_api(actors, total_output_frames)
    # Clean up temp sequence
    unreal.EditorAssetLibrary.delete_asset(f"{SEQUENCE_PACKAGE_PATH}/_TEMP_SMOKE")

    # ── Process each actor: transform tracks ────────────────────────────
    total_transform_keys = 0
    bindings_count = 0
    player_positions = {}  # {entity_id: [(x_m, y_m), ...]} for speed calc
    actor_bindings = {}    # {entity_id: binding} for animation tracks
    actor_cmaps = {}       # {entity_id: channel_map} for ball rotation

    for entity_id, actor in actors.items():
        binding, channels, section = _build_entity_binding(
            sequence, entity_id, actor, total_output_frames,
        )
        bindings_count += 1
        actor_bindings[entity_id] = binding
        cmap = _build_channel_map(channels)
        actor_cmaps[entity_id] = cmap

        prev_yaws = {}
        previous_pos = None
        player_positions[entity_id] = []

        for fi, frame in enumerate(frames):
            frame_time = fi * source_step
            key_frame_index = int(round(frame_time * playback_fps))

            if entity_id == "BALL":
                ball_pos = frame["ball"]["position_m"]
                px, py, pz = _pos_m_to_cm(ball_pos)
                pz += BALL_Z_OFFSET_CM
                add_double_channel_key(cmap["Location.X"], key_frame_index, px)
                add_double_channel_key(cmap["Location.Y"], key_frame_index, py)
                add_double_channel_key(cmap["Location.Z"], key_frame_index, pz)
                if fi == 0:
                    add_double_channel_key(cmap["Scale.X"], 0, 0.5)
                    add_double_channel_key(cmap["Scale.Y"], 0, 0.5)
                    add_double_channel_key(cmap["Scale.Z"], 0, 0.5)
            else:
                for player_data in frame["players"]:
                    if player_data["id"] != entity_id:
                        continue
                    px_m = player_data["position_m"][0]
                    py_m = player_data["position_m"][1]
                    px, py, _ = _pos_m_to_cm(player_data["position_m"])
                    player_positions[entity_id].append((px_m, py_m))
                    add_double_channel_key(cmap["Location.X"], key_frame_index, px)
                    add_double_channel_key(cmap["Location.Y"], key_frame_index, py)
                    add_double_channel_key(cmap["Location.Z"], key_frame_index, PLAYER_Z_CM)

                    # Yaw from delta
                    if previous_pos is not None:
                        dx = px - previous_pos[0]
                        dy = py - previous_pos[1]
                        prev_yaw = prev_yaws.get(entity_id, 0.0)
                        yaw = build_yaw(dx, dy, prev_yaw)
                    else:
                        yaw = 0.0
                    prev_yaws[entity_id] = yaw
                    previous_pos = (px, py)
                    add_double_channel_key(cmap["Rotation.Z"], key_frame_index, yaw)

        # ── Verify key counts ──────────────────────────────────────────
        actual_loc_x = cmap["Location.X"].get_num_keys()
        actual_loc_y = cmap["Location.Y"].get_num_keys()
        actual_loc_z = cmap["Location.Z"].get_num_keys()
        expected = num_steps  # one key per GRF step
        for name, count in [("Location.X", actual_loc_x), ("Location.Y", actual_loc_y),
                            ("Location.Z", actual_loc_z)]:
            if count != expected:
                raise RuntimeError(
                    f"{entity_id} {name}: expected {expected} keys, got {count}"
                )
        total_transform_keys += actual_loc_x + actual_loc_y + actual_loc_z

        if entity_id != "BALL":
            actual_rot = cmap["Rotation.Z"].get_num_keys()
            if actual_rot != expected:
                raise RuntimeError(
                    f"{entity_id} Rotation.Z: expected {expected} keys, got {actual_rot}"
                )
            total_transform_keys += actual_rot

    print(f"  Actor bindings: {bindings_count}")
    print(f"  Total transform keys written: {total_transform_keys}")

    # ── Animation tracks (if config provided) ──────────────────────────
    total_sections = 0
    animation_ok = False

    if anim_cfg and anim_cfg.get("enabled", True):
        anims = anim_cfg["animations"]
        loco = anim_cfg["locomotion"]

        print("\n--- Player locomotion animation ---")
        for entity_id in actors:
            if entity_id == "BALL":
                continue
            binding = actor_bindings[entity_id]
            positions = player_positions.get(entity_id, [])
            if len(positions) < 2:
                print(f"  {entity_id}: insufficient positions ({len(positions)})")
                continue

            speeds = compute_speeds_mps(positions, source_step)
            speeds = smooth_values(speeds, loco["smoothing_window"])

            min_speed = min(speeds)
            mean_speed = sum(speeds) / len(speeds)
            max_speed = max(speeds)
            print(f"  {entity_id}: min={min_speed:.2f} mean={mean_speed:.2f} max={max_speed:.2f} m/s")

            segments = classify_states(speeds, loco, source_step, playback_fps)

            # Build segment count display
            state_counts = {"idle": 0, "walk": 0, "run": 0}
            for seg in segments:
                state_counts[seg.state] = state_counts.get(seg.state, 0) + 1
            print(f"    segments: idle={state_counts.get('idle', 0)} walk={state_counts.get('walk', 0)} run={state_counts.get('run', 0)}")

            # Create animation track
            anim_track = binding.add_track(unreal.MovieSceneSkeletalAnimationTrack)

            for seg in segments:
                start_frame = int(round(seg.source_start * source_step * playback_fps))
                end_frame = int(round(seg.source_end_exclusive * source_step * playback_fps))
                if seg.source_end_exclusive == num_steps:
                    end_frame = total_output_frames

                section = anim_track.add_section()
                section.set_range(start_frame, end_frame)

                asset_path = anims[seg.state]
                anim_asset = unreal.load_asset(asset_path)
                if not anim_asset:
                    raise RuntimeError(f"Failed to load animation: {asset_path}")

                # Set animation via params
                params = section.get_editor_property("params")
                params.set_editor_property("animation", anim_asset)
                params.set_editor_property("force_custom_mode", True)
                params.set_editor_property("skip_anim_notifiers", True)

                play_rate = compute_play_rate(seg, loco)
                play_rate_variant = unreal.MovieSceneTimeWarpVariant()
                play_rate_variant.set_fixed_play_rate(float(play_rate))
                params.set_editor_property("play_rate", play_rate_variant)
                section.set_editor_property("params", params)

                total_sections += 1

        print(f"  Animation tracks created: {bindings_count - (1 if 'BALL' in actors else 0)}")
        print(f"  Animation sections created: {total_sections}")
        animation_ok = True

    # ── Ball rolling (if config provided) ──────────────────────────────
    ball_rolling_ok = False
    if anim_cfg and anim_cfg.get("ball", {}).get("enabled", True) and "BALL" in actors:
        ball_cfg = anim_cfg["ball"]
        cmap = actor_cmaps["BALL"]

        # Collect ball positions in meters
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

        # Count rolling/stationary/airborne
        rolling_frames = 0
        stationary_frames = 0
        airborne_frames = 0
        for i, frame in enumerate(frames):
            bz = frame["ball"]["position_m"][2]
            if bz > ball_cfg["radius_m"] + 0.03:
                airborne_frames += 1
            elif i > 0:
                bpos = frame["ball"]["position_m"]
                bpos_prev = frames[i - 1]["ball"]["position_m"]
                dist = math.hypot(bpos[0] - bpos_prev[0], bpos[1] - bpos_prev[1])
                if dist > ball_cfg.get("minimum_move_distance_m", 0.0001):
                    rolling_frames += 1
                else:
                    stationary_frames += 1
            else:
                stationary_frames += 1

        # Total horizontal travel
        total_horizontal = 0.0
        for i in range(1, len(ball_positions)):
            dx = ball_positions[i][0] - ball_positions[i - 1][0]
            dy = ball_positions[i][1] - ball_positions[i - 1][1]
            total_horizontal += math.hypot(dx, dy)

        # Accumulated roll
        if len(rotations) > 1:
            last_roll, last_pitch, last_yaw = rotations[-1]
            accumulated_roll = math.sqrt(last_roll**2 + last_pitch**2 + last_yaw**2)
        else:
            accumulated_roll = 0.0

        # Write rotation keys to Rotation.X/Y/Z channels
        if "Rotation.X" in cmap:
            for fi in range(len(frames)):
                frame_time = fi * source_step
                kf = int(round(frame_time * playback_fps))
                r, p, y = rotations[fi]
                add_double_channel_key(cmap["Rotation.X"], kf, r)
                add_double_channel_key(cmap["Rotation.Y"], kf, p)
                add_double_channel_key(cmap["Rotation.Z"], kf, y)

        print(f"\n--- Ball rolling ---")
        print(f"  Ball radius: {ball_cfg['radius_m']:.3f} m")
        print(f"  Rolling frames: {rolling_frames}")
        print(f"  Stationary frames: {stationary_frames}")
        print(f"  Airborne frames: {airborne_frames}")
        print(f"  Horizontal travel: {total_horizontal:.2f} m")
        print(f"  Accumulated roll: {accumulated_roll:.1f} degrees")
        ball_rolling_ok = True

    # ── Final validation ───────────────────────────────────────────────
    if animation_ok:
        for entity_id in actors:
            if entity_id == "BALL":
                continue
            binding = actor_bindings.get(entity_id)
            if not binding:
                raise RuntimeError(f"Missing binding for {entity_id}")
            # Should have 1 transform track
            tracks = binding.get_tracks()
            transform_tracks = [t for t in tracks if t.get_class().get_name() == "MovieScene3DTransformTrack"]
            anim_tracks = [t for t in tracks if t.get_class().get_name() == "MovieSceneSkeletalAnimationTrack"]
            if len(transform_tracks) < 1:
                raise RuntimeError(f"{entity_id}: missing TransformTrack")
            if len(anim_tracks) < 1:
                raise RuntimeError(f"{entity_id}: missing SkeletalAnimationTrack")
            # Animation track should have at least 1 section
            anim_sections = anim_tracks[0].get_sections()
            if len(anim_sections) < 1:
                raise RuntimeError(f"{entity_id}: AnimationTrack has no sections")

    print()
    if animation_ok or ball_rolling_ok:
        print("GRF UE ANIMATION IMPORT PASS")

    # ── Save ───────────────────────────────────────────────────────────
    saved = unreal.EditorAssetLibrary.save_loaded_asset(sequence, only_if_is_dirty=False)
    if not saved:
        raise RuntimeError(f"Failed to save Level Sequence: {sequence_asset_path}")
    print(f"  Sequence saved: {sequence_asset_path}")

    # ── Open in Sequencer ──────────────────────────────────────────────
    try:
        unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(sequence)
        print("  Sequence opened in Sequencer.")
    except Exception as exc:
        unreal.log(f"Warning: could not open Level Sequence in Sequencer: {exc}")

    # ── Final report ───────────────────────────────────────────────────
    player_count = sum(1 for eid in actors if eid != "BALL")
    ball_count = 1 if "BALL" in actors else 0
    print()
    print("GRF UE IMPORT PASS")
    print(f"Sequence asset: {sequence_asset_path}")
    print(f"Display rate: {playback_fps} FPS")
    print(f"Playback frames: 0-{total_output_frames}")
    print(f"Bindings: {bindings_count}")
    print(f"Player bindings: {player_count}")
    print(f"Ball bindings: {ball_count}")
    print(f"Total transform keys: {total_transform_keys}")


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

    episode_dir = Path(args.episode)
    mapping_path = Path(args.mapping)
    mode = args.mode
    replace_existing = args.replace_existing
    anim_config_path = Path(args.animation_config) if args.animation_config else None

    if not episode_dir.exists():
        print(f"ERROR: Episode directory not found: {episode_dir}", file=sys.stderr)
        sys.exit(1)
    if not mapping_path.exists():
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
        print(f"  Animation config: {anim_config_path}")
        mapping_check = load_mapping(mapping_path)
        actors = _find_all_actors(mapping_check)
        if actors:
            precheck_animation_assets(anim_cfg, actors, mapping_check)

    if mode in ("preview", "both"):
        print("\n--- Preview mode: setting actor transforms ---")
        apply_preview(meta, frames, mapping)

    if mode in ("sequence", "both"):
        print("\n--- Sequence mode: creating Level Sequence ---")
        create_sequence(meta, frames, mapping, replace_existing, anim_cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
