#!/usr/bin/env python3
"""
rl_07d_verify_rl_render.py — Verify RL rendering directly without 03_check_labels.py.

This script reads the annotation JSON and rendered images directly,
checks their integrity, and generates a verification report.

Usage:
    D:/Anaconda/envs/yolov11/python.exe rl_07d_verify_rl_render.py --seq-id rl_episode_random_0001_t1_p05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_io import write_json_atomic, write_text_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RL rendering outputs.")
    parser.add_argument("--seq-id", default="rl_episode_random_0001_t1_p05")
    parser.add_argument("--annotations-dir", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seq_id = args.seq_id
    project_root = Path("D:/projects/FustalMOT_UEDataset")

    ann_dir = Path(args.annotations_dir) if args.annotations_dir else (
        project_root / "Saved" / "FutsalMOT" / "annotations"
    )
    images_base = Path(args.images_dir) if args.images_dir else (
        project_root / "Saved" / "FutsalMOT_RL" / "ue_closed_loop" / "images"
    )

    ann_file = ann_dir / "objects_bbox_2d_clean_{}.json".format(seq_id)
    images_dir = images_base / seq_id

    checks: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    print("=" * 60)
    print("FutsalMOT-RL Render Verification")
    print("Seq ID: {}".format(seq_id))
    print("=" * 60)

    # 1. Check annotation file
    if not ann_file.is_file():
        errors.append("Annotation JSON not found: {}".format(ann_file))
        print("[FAIL] Annotation file missing")
    else:
        size_mb = ann_file.stat().st_size / 1e6
        checks.append("Annotation file: {:.0f} MB".format(size_mb))
        print("[OK] Annotation: {:.0f} MB".format(size_mb))

        # Load and check structure
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("records", [])
        n_records = len(records)
        checks.append("Records: {}".format(n_records))

        # Check record structure
        if n_records == 0:
            errors.append("No records in annotation")
        else:
            sample = records[0]
            required = ["frame_id", "camera_id", "objects", "rgb_path"]
            for key in required:
                if key not in sample:
                    errors.append("Missing key '{}' in records".format(key))

            # Count frames per camera
            cameras = set()
            frames = set()
            for r in records:
                cameras.add(r.get("camera_id", "?"))
                frames.add(r.get("frame_id", -1))
            checks.append("Cameras: {} {}".format(len(cameras), sorted(cameras)))
            checks.append("Frames: {}..{} ({} total)".format(min(frames), max(frames), len(frames)))

            if n_records == 1200:
                print("[OK] Records: 1200 (4 cameras x 300 frames)")
            else:
                warnings.append("Expected 1200 records, got {}".format(n_records))
                print("[WARN] Records: {} (expected 1200)".format(n_records))

            # Count objects per record
            obj_counts = set(len(r.get("objects", [])) for r in records)
            if obj_counts == {9}:
                print("[OK] Objects per record: 9")
            else:
                warnings.append("Objects per record: {} (expected 9)".format(obj_counts))

            # Check Player_05 is present
            p05_found = 0
            for r in records:
                objects = r.get("objects", [])
                for obj in objects:
                    if obj.get("object_id") == "Player_05" or obj.get("track_id") == 5:
                        p05_found += 1
                        break
            if p05_found > 0:
                print("[OK] Player_05 found in {} records".format(p05_found))
            else:
                errors.append("Player_05 not found in any record")

            # Check bbox data
            nan_bbox = 0
            for r in records[:50]:
                for obj in r.get("objects", []):
                    bbox = obj.get("bbox_2d_clean", obj.get("bbox", []))
                    for v in bbox:
                        if v is None or (isinstance(v, float) and v != v):
                            nan_bbox += 1
            if nan_bbox == 0:
                print("[OK] No NaN bbox values (sampled 50 records)")
            else:
                errors.append("Found {} NaN bbox values".format(nan_bbox))

    # 2. Check rendered images
    cam_dirs = sorted(images_dir.glob("cam_*")) if images_dir.is_dir() else []
    if not cam_dirs:
        errors.append("No camera image directories found at {}".format(images_dir))
    else:
        checks.append("Image base dir: {}".format(images_dir))
        print("[OK] Image directory: {} ({} cameras)".format(images_dir, len(cam_dirs)))

        total_pngs = 0
        missing_frame_count = 0
        for cam_dir in cam_dirs:
            pngs = sorted(cam_dir.glob("*.png"))
            # Prefer 6-digit naming
            pngs_6digit = [p for p in pngs if len(p.stem) == 6]
            pngs_4digit = [p for p in pngs if len(p.stem) == 4]
            actual = pngs_6digit if len(pngs_6digit) >= 300 else pngs_4digit if len(pngs_4digit) >= 300 else pngs
            n = len(actual)
            total_pngs += n

            if n < 300:
                missing_frame_count += 300 - n
                warnings.append("{}: {} images (expected 300)".format(cam_dir.name, n))
            else:
                print("[OK] {}: {} images".format(cam_dir.name, n))

            # Check first frame
            if actual:
                from PIL import Image
                try:
                    img = Image.open(actual[0])
                    w, h = img.size
                    if (w, h) == (1920, 1080):
                        pass  # Correct resolution
                    else:
                        warnings.append("{} first frame: {}x{} (expected 1920x1080)".format(
                            cam_dir.name, w, h))
                except Exception as e:
                    errors.append("{} first frame unreadable: {}".format(cam_dir.name, e))

        if total_pngs >= 1200:
            print("[OK] Total images: {} (expected 1200)".format(total_pngs))
        else:
            warnings.append("Total images: {} (expected 1200)".format(total_pngs))

    # 3. Summary
    print("\n--- Verification Summary ---")
    print("Checks:   {}".format(len(checks)))
    for c in checks:
        print("  [INFO] {}".format(c))
    print("Warnings: {}".format(len(warnings)))
    for w in warnings:
        print("  [WARN] {}".format(w))
    print("Errors:   {}".format(len(errors)))
    for e in errors:
        print("  [FAIL] {}".format(e))

    status = "PASS" if not errors else "FAIL"
    print("\nStatus: {}".format(status))

    # 4. Write report
    report = {
        "schema_version": "RL_RENDER_VERIFY_V1",
        "seq_id": seq_id,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "status": status,
        "annotation_file": str(ann_file),
        "image_dir": str(images_dir),
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "player_05_present": p05_found > 0 if "p05_found" in dir() else None,
    }

    from futsalmot_rl.core.rl_paths import REPORTS_DIR
    report_path = REPORTS_DIR / "rl_render_verify_report.json"
    write_json_atomic(report_path, report)
    print("\nReport: {}".format(report_path))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
