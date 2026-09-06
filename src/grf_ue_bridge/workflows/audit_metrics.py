"""Task Audit 的轻量统计，不参与 PASS/FAIL。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _camera_dirs(root: Path) -> List[Path]:
    return sorted(path.parent for path in root.rglob("camera.json"))


def _dataset_metrics(cameras: List[Path]) -> Dict[str, int]:
    image_count = sum(len(list((camera / "img1").glob("*.png"))) for camera in cameras)
    frame_count = 0
    for camera in cameras:
        annotation_path = camera / "annotations.jsonl"
        if annotation_path.is_file():
            frame_count += _count_lines(annotation_path)
    return {
        "frame_count": frame_count,
        "image_count": image_count,
        "camera_count": len(cameras),
    }


def _mot_metrics(cameras: List[Path]) -> Dict[str, float]:
    tracks = set()
    frames = set()
    lengths = {}
    for camera in cameras:
        path = camera / "gt" / "gt.txt"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                fields = line.strip().split(",")
                if len(fields) < 2:
                    raise ValueError(f"invalid MOT row: {line.strip()}")
                frame = int(fields[0])
                track = int(fields[1])
                frames.add(frame)
                tracks.add(track)
                lengths.setdefault(track, set()).add(frame)
    return {
        "track_count": len(tracks),
        "frame_count": len(frames),
        "avg_track_length": round(sum(len(value) for value in lengths.values()) / len(lengths), 3)
        if lengths else 0,
    }


def _mask_metrics(cameras: List[Path]) -> Dict[str, int]:
    from PIL import Image

    instance_ids = set()
    for camera in cameras:
        for path in (camera / "mask").glob("*.png"):
            with Image.open(path) as image:
                instance_ids.update(value for value in image.convert("L").getdata() if value)
    return {"instance_count": len(instance_ids)}


def calculate_metrics(dataset_dir: Path) -> Dict:
    """计算 best-effort 统计；任何单项失败只写入 errors。"""
    root = Path(dataset_dir)
    metrics: Dict = {}
    errors: List[str] = []
    try:
        if not root.is_dir():
            raise OSError(f"dataset directory does not exist: {root}")
        cameras = _camera_dirs(root)
        metrics["dataset"] = _dataset_metrics(cameras)
        try:
            metrics["mot"] = _mot_metrics(cameras)
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"failed to calculate mot statistics: {exc}")
        try:
            metrics["mask"] = _mask_metrics(cameras)
        except (OSError, UnicodeError, ValueError) as exc:
            metrics["mask"] = {"instance_count": None}
            errors.append(f"failed to calculate mask statistics: {exc}")
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"failed to calculate dataset statistics: {exc}")
    if errors:
        metrics["errors"] = errors
    return metrics
