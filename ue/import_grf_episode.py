"""
Unreal Engine Python script — Import GRF-UE episode.

Two modes:
  --preview    Set actor transforms directly in the level (Editor preview).
  --sequence   Create a Level Sequence asset with keyframed transforms.

Default (no --mode flag): both preview AND sequence.

Usage (in Unreal Editor Python Console):
    py "C:/path/to/ue/import_grf_episode.py" --episode "C:/path/to/outputs/episode_0001" --mapping "C:/path/to/ue/actor_mapping.example.json"

Dependencies:
    - unreal (UE built-in module)
    - json, math, pathlib (stdlib)

This script does NOT require gfootball, GRF_MARL, or the .venv.
"""

import json
import math
import sys
from pathlib import Path

M_TO_CM = 100.0
SPEED_THRESHOLD_CM = 5.0  # cm/s, below this keep previous yaw
BALL_Z_OFFSET_CM = 50.0  # lift ball so it sits on ground plane


def _parse_args():
    """Parse command-line arguments."""
    episode_dir = None
    mapping_path = None
    mode = "both"
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--episode" and i + 1 < len(args):
            episode_dir = Path(args[i + 1])
            i += 2
        elif args[i] == "--mapping" and i + 1 < len(args):
            mapping_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        else:
            i += 1
    return episode_dir, mapping_path, mode


def load_episode(episode_dir: Path):
    """Load meta.json and frames.jsonl."""
    with open(episode_dir / "meta.json") as f:
        meta = json.load(f)
    frames = []
    with open(episode_dir / "frames.jsonl") as f:
        for line in f:
            frames.append(json.loads(line))
    return meta, frames


def load_mapping(path: Path) -> dict:
    """Load actor mapping JSON."""
    with open(path) as f:
        return json.load(f)


def find_actor(name: str):
    """Find a UE actor by label or name, case-insensitive."""
    import unreal

    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    for actor in actors:
        actor_name = actor.get_actor_label() or actor.get_name()
        if actor_name.lower() == name.lower():
            return actor
        if name.lower() in actor_name.lower():
            return actor
    return None


def build_yaw(dx: float, dy: float, prev_yaw: float) -> float:
    """Compute yaw from movement delta with low-speed hysteresis."""
    speed = math.sqrt(dx * dx + dy * dy)
    if speed < SPEED_THRESHOLD_CM:
        return prev_yaw
    return math.degrees(math.atan2(dx, dy))


def _pos_m_to_cm(pos_m: list) -> tuple:
    """Convert [x, y, z] meters to (x_cm, y_cm, z_cm)."""
    return (pos_m[0] * M_TO_CM, pos_m[1] * M_TO_CM, pos_m[2] * M_TO_CM)


# ── Preview mode: direct transform setting ──────────────────────────────

def apply_preview(meta: dict, frames: list, mapping: dict):
    """Set actor transforms frame-by-frame in the level."""
    import unreal

    actors = _find_all_actors(mapping)
    if not actors:
        return

    num_steps = meta["timing"]["num_steps"]
    prev_yaws: dict = {}

    for frame_idx, frame in enumerate(frames):
        step = frame["step"]
        _apply_ball_frame(actors, frame)
        _apply_player_frame(actors, frame, prev_yaws)

        if step > 0 and step % 50 == 0:
            print(f"  Preview: {step}/{num_steps}")

    print(f"Preview complete: {num_steps} frames applied.")


# ── Sequence mode: create Level Sequence ────────────────────────────────

def create_sequence(meta: dict, frames: list, mapping: dict):
    """Create a Level Sequence asset with keyframed transforms."""
    import unreal

    actors = _find_all_actors(mapping)
    if not actors:
        return

    episode_id = meta.get("episode_id", "episode_0000")
    num_steps = meta["timing"]["num_steps"]
    source_step = meta["timing"]["source_step_seconds"]

    # Asset paths
    package_path = "/Game/GRF/Sequences"
    asset_name = f"SEQ_{episode_id}"

    # Ensure directory exists
    if not unreal.EditorAssetLibrary.does_directory_exist(package_path):
        unreal.EditorAssetLibrary.make_directory(package_path)

    # Create Level Sequence asset
    seq_factory = unreal.LevelSequenceFactoryNew()
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    seq = asset_tools.create_asset(asset_name, package_path, None, seq_factory)
    if not seq:
        print(f"ERROR: Failed to create Level Sequence asset at {package_path}/{asset_name}")
        return

    # Set display rate to 30 FPS, duration to the episode length
    seq.set_display_rate(unreal.FrameRate(30, 1))
    total_frames = int(math.ceil(num_steps * source_step * 30))
    seq.set_playback_end(unreal.FrameNumber(total_frames))

    # Bind actors and add transform tracks
    for entity_id, actor in actors.items():
        binding = seq.add_possessable(actor)
        # Add transform track
        transform_track = binding.add_track(unreal.MovieScene3DTransformTrack)
        transform_section = transform_track.add_section()

        # Set section range
        transform_section.set_range(
            unreal.FrameNumber(0),
            unreal.FrameNumber(total_frames),
        )

        # Channel indices in a 3D Transform section:
        # 0=LocationX, 1=LocationY, 2=LocationZ
        # 3=RotationX, 4=RotationY, 5=RotationZ
        # 6=ScaleX, 7=ScaleY, 8=ScaleZ
        loc_x_channel = transform_section.get_channels()[0]
        loc_y_channel = transform_section.get_channels()[1]
        loc_z_channel = transform_section.get_channels()[2]
        rot_z_channel = transform_section.get_channels()[5]

        prev_yaws = {}
        previous_pos = None

        for fi, frame in enumerate(frames):
            frame_time = fi * source_step
            frame_number = int(round(frame_time * 30))

            if entity_id == "BALL":
                ball_pos = frame["ball"]["position_m"]
                px, py, pz = _pos_m_to_cm(ball_pos)
                pz += BALL_Z_OFFSET_CM
                loc_x_channel.add_key(unreal.FrameNumber(frame_number), px)
                loc_y_channel.add_key(unreal.FrameNumber(frame_number), py)
                loc_z_channel.add_key(unreal.FrameNumber(frame_number), pz)
            else:
                for player_data in frame["players"]:
                    if player_data["id"] != entity_id:
                        continue
                    pos_m = player_data["position_m"]
                    px, py, pz = _pos_m_to_cm(pos_m)
                    loc_x_channel.add_key(unreal.FrameNumber(frame_number), px)
                    loc_y_channel.add_key(unreal.FrameNumber(frame_number), py)
                    loc_z_channel.add_key(unreal.FrameNumber(frame_number), pz)

                    # Yaw from delta
                    if previous_pos is not None:
                        dx = px - previous_pos[0]
                        dy = py - previous_pos[1]
                        prev_yaw = prev_yaws.get(entity_id, 0.0)
                        yaw = build_yaw(dx, dy, prev_yaw)
                    else:
                        yaw = 0.0
                    prev_yaws[entity_id] = yaw
                    previous_pos = (px, py, pz)

                    rot_z_channel.add_key(unreal.FrameNumber(frame_number), -yaw)

    # Save the asset
    unreal.EditorAssetLibrary.save_asset(f"{package_path}/{asset_name}", only_if_is_dirty=True)
    print(f"Sequence created: {package_path}/{asset_name}")
    print(f"  Duration: {total_frames} frames @ 30 FPS ({total_frames / 30:.1f}s)")
    print(f"  Actors bound: {len(actors)}")


# ── Shared helpers ──────────────────────────────────────────────────────

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


def _apply_ball_frame(actors: dict, frame: dict):
    """Position ball actor from frame data."""
    import unreal

    if "BALL" not in actors:
        return
    ball_pos = frame["ball"]["position_m"]
    px, py, pz = _pos_m_to_cm(ball_pos)
    actors["BALL"].set_actor_location(
        unreal.Vector(px, py, pz + BALL_Z_OFFSET_CM), False
    )


def _apply_player_frame(actors: dict, frame: dict, prev_yaws: dict):
    """Position and rotate player actors from frame data."""
    import unreal

    for player_data in frame["players"]:
        pid = player_data["id"]
        if pid not in actors:
            continue

        px, py, pz = _pos_m_to_cm(player_data["position_m"])
        pos_cm = unreal.Vector(px, py, pz)

        # Yaw from delta
        prev_pos = getattr(actors[pid], "_prev_pos", None)
        if prev_pos:
            dx = pos_cm.x - prev_pos.x
            dy = pos_cm.y - prev_pos.y
            prev_yaw = prev_yaws.get(pid, 0.0)
            yaw = build_yaw(dx, dy, prev_yaw)
        else:
            yaw = 0.0

        prev_yaws[pid] = yaw
        actors[pid]._prev_pos = pos_cm

        actors[pid].set_actor_location_and_rotation(
            pos_cm, unreal.Rotator(0.0, yaw, 0.0), False
        )


# ── Entry ───────────────────────────────────────────────────────────────

def main():
    episode_dir, mapping_path, mode = _parse_args()
    if not episode_dir or not mapping_path:
        print(f"Usage: py {__file__} --episode <dir> --mapping <path> [--mode preview|sequence|both]")
        print(f"Example: py {__file__} --episode outputs/episode_0001 --mapping ue/actor_mapping.example.json")
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
        create_sequence(meta, frames, mapping)

    print("\nDone.")


if __name__ == "__main__":
    main()
