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
import sys
from pathlib import Path

M_TO_CM = 100.0
SPEED_THRESHOLD_CM = 5.0  # cm/s, below this keep previous yaw
BALL_Z_OFFSET_CM = 50.0  # lift ball so it sits on ground plane
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
    channel.add_key(time=frame_number, new_value=numeric_value, interpolation=interp)


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
    """Build a dict mapping display names like 'Location.X' to channel objects.

    Falls back to positional indexing with a length check when name resolution
    is unreliable. Logs the channel names for diagnostics.
    """
    import unreal
    name_map = {}
    for ch in channels:
        name = _channel_name(ch)
        name_map[name] = ch
    unreal.log(
        "Transform channels: "
        + ", ".join(name_map.keys())
    )
    if len(channels) < 9:
        raise RuntimeError(
            f"Expected at least 9 transform channels, got {len(channels)}"
        )
    # If we got all 9 by name, use named map
    if all(n in name_map for n in EXPECTED_CHANNEL_NAMES):
        return name_map
    # Fallback: positional indexing with length check
    unreal.log("Warning: channel name matching incomplete, using positional fallback")
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

    # Add two test keys
    test_ch = cmap["Location.X"]
    test_ch.remove_all_keys()
    test_ch.add_key(unreal.FrameNumber(0), 0.0)
    test_ch.add_key(unreal.FrameNumber(1), 100.0)

    return temp_seq, cmap


# ── Sequence mode ───────────────────────────────────────────────────────

def create_sequence(meta: dict, frames: list, mapping: dict, replace_existing: bool = False):
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
    temp_seq, temp_channels = _smoke_test_sequencer_api(
        actors, total_output_frames,
    )
    key_count = temp_channels["Location.X"].get_num_keys()
    # Delete temp sequence
    temp_path = f"{SEQUENCE_PACKAGE_PATH}/_TEMP_SMOKE"
    unreal.EditorAssetLibrary.delete_asset(temp_path)
    if key_count < 2:
        raise RuntimeError(
            f"Smoke test failed: expected 2 keys from add_key, got {key_count}. "
            f"Incompatible UE Python Sequencer API."
        )
    print(f"  Smoke test PASS: add_key accepts FrameNumber")

    # ── Process each actor ─────────────────────────────────────────────
    total_transform_keys = 0
    bindings_count = 0

    for entity_id, actor in actors.items():
        binding, channels, section = _build_entity_binding(
            sequence, entity_id, actor, total_output_frames,
        )
        bindings_count += 1

        cmap = _build_channel_map(channels)

        prev_yaws = {}
        previous_pos = None

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
            else:
                for player_data in frame["players"]:
                    if player_data["id"] != entity_id:
                        continue
                    px, py, _ = _pos_m_to_cm(player_data["position_m"])
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
                    add_double_channel_key(cmap["Rotation.X"], key_frame_index, yaw)

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
            actual_rot = cmap["Rotation.X"].get_num_keys()
            if actual_rot != expected:
                raise RuntimeError(
                    f"{entity_id} Rotation.X: expected {expected} keys, got {actual_rot}"
                )
            total_transform_keys += actual_rot

    print(f"  Actor bindings: {bindings_count}")
    print(f"  Total transform keys written: {total_transform_keys}")

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

    if mode in ("preview", "both"):
        print("\n--- Preview mode: setting actor transforms ---")
        apply_preview(meta, frames, mapping)

    if mode in ("sequence", "both"):
        print("\n--- Sequence mode: creating Level Sequence ---")
        create_sequence(meta, frames, mapping, replace_existing)

    print("\nDone.")


if __name__ == "__main__":
    main()
