"""
Unreal Engine Python script — Import GRF-UE episode and create a preview.

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


def _parse_args():
    """Parse --episode and --mapping from sys.argv."""
    episode_dir = None
    mapping_path = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--episode" and i + 1 < len(args):
            episode_dir = Path(args[i + 1])
        elif arg == "--mapping" and i + 1 < len(args):
            mapping_path = Path(args[i + 1])
    return episode_dir, mapping_path


def load_episode(episode_dir: Path):
    """Load meta.json and frames.jsonl from an episode directory."""
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
        # Also try matching just the base name
        if name.lower() in actor_name.lower():
            return actor
    return None


def build_yaw(dx: float, dy: float, prev_yaw: float, speed_threshold: float = 5.0) -> float:
    """Compute yaw from movement delta, with low-speed hysteresis."""
    speed = math.sqrt(dx * dx + dy * dy)
    if speed < speed_threshold:
        return prev_yaw  # keep previous yaw when stationary/slow
    yaw = math.degrees(math.atan2(dx, dy))
    return yaw


def import_episode(episode_dir: Path, mapping_path: Path, fps: int = 30):
    """Main import function.

    Steps:
      1. Load meta, frames, mapping.
      2. Find all actors.
      3. For each frame, set actor transforms.
      4. Linear interpolation between frames.
    """
    import unreal

    meta, frames = load_episode(episode_dir)
    mapping = load_mapping(mapping_path)

    num_steps = meta["timing"]["num_steps"]
    source_step_seconds = meta["timing"]["source_step_seconds"]

    print(f"Episode: {meta.get('episode_id', 'unknown')}")
    print(f"  Steps: {num_steps}")
    print(f"  Source step: {source_step_seconds}s")
    print(f"  Playback FPS: {fps}")

    # Find actors
    actors = {}
    for entity_id, actor_name in mapping.items():
        actor = find_actor(actor_name)
        if actor:
            actors[entity_id] = actor
            print(f"  Found actor: {entity_id} -> {actor_name}")
        else:
            print(f"  WARNING: Actor not found: {entity_id} -> {actor_name}")

    if not actors:
        print("ERROR: No actors found. Check your actor mapping.")
        return

    # Convert meters to centimeters (UE uses cm)
    M_TO_CM = 100.0

    # Track previous yaw for hysteresis
    prev_yaws: dict = {}

    # Process each frame
    for frame_idx, frame in enumerate(frames):
        step = frame["step"]
        time_sec = frame["time_seconds"]

        # Ball
        ball_data = frame["ball"]
        if "BALL" in actors:
            ball_pos_m = ball_data["position_m"]
            ball_pos_cm = unreal.Vector(
                ball_pos_m[0] * M_TO_CM,
                ball_pos_m[1] * M_TO_CM,
                ball_pos_m[2] * M_TO_CM + 50.0,  # offset so ball sits on ground
            )
            actors["BALL"].set_actor_location(ball_pos_cm, False)

        # Players
        for player_data in frame["players"]:
            pid = player_data["id"]
            if pid not in actors:
                continue

            pos_m = player_data["position_m"]
            pos_cm = unreal.Vector(
                pos_m[0] * M_TO_CM,
                pos_m[1] * M_TO_CM,
                pos_m[2] * M_TO_CM,
            )

            # Compute yaw from this frame vs previous
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

            rot = unreal.Rotator(0.0, yaw, 0.0)
            actors[pid].set_actor_location_and_rotation(pos_cm, rot, False)

        # Log progress at intervals
        if step > 0 and step % 50 == 0:
            print(f"  Processed {step}/{num_steps} frames")

    print(f"Import complete: {num_steps} frames applied.")


if __name__ == "__main__":
    episode_dir, mapping_path = _parse_args()
    if not episode_dir or not mapping_path:
        print(f"Usage: py {__file__} --episode <dir> --mapping <path>")
        print(f"Example: py {__file__} --episode outputs/episode_0001 --mapping ue/actor_mapping.example.json")
        sys.exit(1)

    import_episode(episode_dir, mapping_path)
