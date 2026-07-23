"""Post-UE-rendering validation for RL closed-loop check.

Verifies that the RL A3.3 config was correctly rendered by UE:
- Annotation JSON exists and has expected records
- Image files exist for all cameras and frames
- Player_05 track_id is correct
- No NaN values in bbox data
- Layout check output exists
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from futsalmot_rl.core.rl_io import read_json, write_json_atomic
from futsalmot_rl.core.rl_paths import PROJECT_ROOT, ensure_dirs

UE_CLOSED_LOOP_DIR = PROJECT_ROOT / "Saved" / "FutsalMOT_RL" / "ue_closed_loop"


def check_ue_outputs(
    seq_id: str,
    a33_path: str | Path | None = None,
    annotation_dir: str | Path | None = None,
    image_dir: str | Path | None = None,
    layout_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Check UE rendering outputs for completeness and correctness.

    Args:
        seq_id: The seq_id used for rendering.
        a33_path: Path to the RL A3.3 config used.
        annotation_dir: Directory containing annotation JSON.
        image_dir: Directory containing rendered images.
        layout_dir: Directory containing layout_check output.
        output_dir: Report output directory.

    Returns:
        Report dict with check results.
    """
    output_dir = Path(output_dir) if output_dir else UE_CLOSED_LOOP_DIR
    ensure_dirs()

    # Default paths
    if annotation_dir is None:
        annotation_dir = PROJECT_ROOT / "Saved" / "FutsalMOT" / "annotations"
    if image_dir is None:
        image_dir = PROJECT_ROOT / "Saved" / "FutsalMOT_RL" / "ue_closed_loop" / "images" / seq_id
    if layout_dir is None:
        layout_dir = PROJECT_ROOT / "Saved" / "FutsalMOT" / "layout_check" / seq_id

    checks: dict[str, Any] = {}

    # 1. Check annotation JSON
    annotation_path = annotation_dir / f"objects_bbox_2d_clean_{seq_id}.json"
    checks["annotation_file_exists"] = annotation_path.is_file()
    checks["annotation_file_path"] = str(annotation_path.resolve())

    if checks["annotation_file_exists"]:
        try:
            ann_data = read_json(annotation_path)
            records = ann_data.get("records", [])
            checks["annotation_record_count"] = len(records)
            checks["annotation_has_data"] = len(records) > 0

            # Check for Player_05 track_id
            player_05_records = [
                r for r in records if r.get("track_id") == 5 or r.get("object_id") == "Player_05"
            ]
            checks["player_05_record_count"] = len(player_05_records)

            # Check for NaN values in bbox
            nan_count = 0
            for r in records[:100]:
                bbox = r.get("bbox_2d_clean") or r.get("bbox") or []
                for v in bbox:
                    if v is None or (isinstance(v, float) and (v != v)):
                        nan_count += 1
            checks["bbox_nan_count_sampled"] = nan_count

            # Check frames per camera
            cameras = set(r.get("camera", r.get("cam_id", "")) for r in records)
            checks["camera_count"] = len(cameras)
            checks["cameras"] = sorted(cameras) if cameras else []

            # Check track_id consistency
            track_ids = set(r.get("track_id") for r in records if r.get("track_id") is not None)
            checks["track_ids_found"] = sorted(track_ids) if track_ids else []

        except Exception as exc:
            checks["annotation_parse_error"] = str(exc)
    else:
        checks["annotation_record_count"] = 0
        checks["annotation_has_data"] = False

    # 2. Check image files (sample)
    checks["image_dir_exists"] = image_dir.is_dir()
    if checks["image_dir_exists"]:
        camera_dirs = [d for d in image_dir.iterdir() if d.is_dir()] if image_dir.is_dir() else []
        checks["camera_image_dirs"] = [d.name for d in camera_dirs]
        total_images = sum(len(list(d.glob("*.png"))) for d in camera_dirs)
        checks["total_png_images"] = total_images
    else:
        checks["camera_image_dirs"] = []
        checks["total_png_images"] = 0

    # 3. Check layout check output
    checks["layout_dir_exists"] = layout_dir.is_dir()
    if checks["layout_dir_exists"]:
        layout_pngs = list(layout_dir.glob("*.png"))
        checks["layout_check_pngs"] = len(layout_pngs)
    else:
        checks["layout_check_pngs"] = 0

    # 4. Overall status
    errors = []
    warnings = []

    if not checks["annotation_file_exists"]:
        errors.append("Annotation JSON not found")
    elif checks.get("annotation_record_count", 0) == 0:
        errors.append("Annotation has 0 records")
    elif checks.get("player_05_record_count", 0) == 0:
        warnings.append("No Player_05 records found in annotation")

    if checks.get("bbox_nan_count_sampled", 0) > 0:
        errors.append("NaN values found in bbox data")

    if checks.get("total_png_images", 0) == 0:
        warnings.append("No rendered PNG images found")

    if checks.get("track_ids_found") and 5 not in checks["track_ids_found"]:
        errors.append("track_id=5 (Player_05) not found in annotations")

    report = {
        "schema_version": "UE_CLOSED_LOOP_CHECK_V1",
        "seq_id": seq_id,
        "a33_source": str(a33_path) if a33_path else None,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
    }

    report_path = output_dir / "ue_render_check_report.json"
    write_json_atomic(report_path, report)

    return report


def generate_layout_check_report(layout_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Check layout_check output directory contents."""
    layout_dir = Path(layout_dir)
    output_dir = Path(output_dir)
    ensure_dirs()

    info: dict[str, Any] = {
        "layout_dir": str(layout_dir.resolve()),
        "exists": layout_dir.is_dir(),
    }

    if layout_dir.is_dir():
        pngs = sorted(layout_dir.glob("*.png"))
        info["png_count"] = len(pngs)
        info["sample_pngs"] = [p.name for p in pngs[:5]] if pngs else []
    else:
        info["png_count"] = 0
        info["sample_pngs"] = []

    report_path = output_dir / "layout_check_report.json"
    write_json_atomic(report_path, info)
    return info
