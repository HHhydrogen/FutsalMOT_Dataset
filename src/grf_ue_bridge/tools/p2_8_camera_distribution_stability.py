"""P2-8 Camera Distribution 多 seed 稳定性分析工具。"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


def trajectory_is_valid(frames: Sequence[Mapping], fps: float = 10.0, max_stationary_s: float = 2.0) -> bool:
    """检查任一球员是否存在超过阈值的连续静止区间。"""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    player_positions: Dict[str, List[Sequence[float]]] = {}
    for frame in frames:
        for player in frame.get("players", []):
            player_positions.setdefault(player["id"], []).append(player["position_m"])
    max_frames = max(1, int(math.ceil(max_stationary_s * fps)))
    for positions in player_positions.values():
        streak = 1
        for current, previous in zip(positions[1:], positions):
            if math.hypot(float(current[0]) - float(previous[0]), float(current[1]) - float(previous[1])) < 1e-6:
                streak += 1
                if streak > max_frames:
                    return False
            else:
                streak = 1
    return True


def _visibility_by_frame(rows: Sequence[Mapping]) -> Dict[int, Dict[str, bool]]:
    result: Dict[int, Dict[str, bool]] = {}
    for row in rows:
        result[int(row["frame_index"])] = {
            obj["entity_id"]: bool(obj.get("in_frame"))
            for obj in row.get("objects", [])
            if obj.get("class") == "player"
        }
    return result


def analyze_visibility_events(camera_rows: Mapping[str, Sequence[Mapping]], target_camera: str) -> Dict:
    """统计目标 Camera 与其它 Camera 的逐帧 visibility pattern 差异。"""
    target = _visibility_by_frame(camera_rows[target_camera])
    references = [_visibility_by_frame(rows) for name, rows in camera_rows.items() if name != target_camera]
    if not references:
        raise ValueError("at least one reference camera is required")
    event_frames = []
    for frame_index, target_visibility in sorted(target.items()):
        anchor_visibility = {
            entity_id: any(reference.get(frame_index, {}).get(entity_id, False) for reference in references)
            for entity_id in target_visibility
        }
        different = sorted(entity_id for entity_id in target_visibility if target_visibility[entity_id] != anchor_visibility[entity_id])
        if different:
            event_frames.append((frame_index, different))
    events = []
    for frame_index, players in event_frames:
        if events and frame_index == events[-1]["end_frame"] + 1 and players == events[-1]["players"]:
            events[-1]["end_frame"] = frame_index
        else:
            events.append({"start_frame": frame_index, "end_frame": frame_index, "players": players})
    durations = [event["end_frame"] - event["start_frame"] + 1 for event in events]
    return {
        "unique_event_count": len(events),
        "players": sorted({player for _, players in event_frames for player in players}),
        "event_durations_frames": durations,
        "events": events,
    }


def _load_jsonl(path: Path) -> List[Mapping]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze_dataset_pair(anchor_dir: Path, partial_dir: Path, seed: int) -> Dict:
    """分析同一 seed 的 A/B episode 产物。"""
    anchor_meta = json.loads((anchor_dir / "meta.json").read_text(encoding="utf-8"))
    partial_meta = json.loads((partial_dir / "meta.json").read_text(encoding="utf-8"))
    anchor_frames = _load_jsonl(anchor_dir / "frames.jsonl")
    partial_frames = _load_jsonl(partial_dir / "frames.jsonl")
    motion = {
        "anchor_valid": trajectory_is_valid(anchor_frames, fps=float(anchor_meta["timing"]["playback_fps"])),
        "partial_valid": trajectory_is_valid(partial_frames, fps=float(partial_meta["timing"]["playback_fps"])),
    }
    anchor_cameras = sorted(path.name for path in anchor_dir.iterdir() if path.is_dir() and (path / "annotations.jsonl").is_file())
    partial_cameras = sorted(path.name for path in partial_dir.iterdir() if path.is_dir() and (path / "annotations.jsonl").is_file())
    camera_rows = {camera: _load_jsonl(partial_dir / camera / "annotations.jsonl") for camera in partial_cameras}
    p01_events = analyze_visibility_events(camera_rows, "CineCam_P01")
    camera_metrics = {}
    for camera in partial_cameras:
        rows = camera_rows[camera]
        observations = [obj for row in rows for obj in row.get("objects", []) if obj.get("class") == "player"]
        visible = [obj for obj in observations if obj.get("in_frame")]
        camera_metrics[camera] = {
            "annotation_frames": len(rows),
            "observation_slots": len(observations),
            "visible_observations": len(visible),
            "invisible_observations": len(observations) - len(visible),
            "average_visible_players_per_frame": len(visible) / len(rows) if rows else 0.0,
            "gt_rows": sum(1 for line in (partial_dir / camera / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines() if line.strip()),
        }
    return {
        "seed": seed,
        "anchor_cameras": anchor_cameras,
        "partial_cameras": partial_cameras,
        "trajectory": motion,
        "camera_metrics": camera_metrics,
        "p01_complementarity": p01_events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for seed in args.seeds:
        base = args.root / ("seed_%d" % seed)
        results.append(analyze_dataset_pair(base / "anchor_only", base / "anchor_plus_one_partial", seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
