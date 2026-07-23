#!/usr/bin/env python3
"""
rl_07c_check_rl_render.py — Check RL rendering outputs with path compatibility.

Handles mismatches between:
  - Annotation path (images_clean/) vs actual image location (ue_closed_loop/images/)
  - 6-digit frame naming (000000.png) vs 4-digit (0000.png)

Usage:
    python rl_07c_check_rl_render.py --seq-id rl_episode_random_0001_t1_p05
    python rl_07c_check_rl_render.py --seq-id rl_episode_random_0001_t1_p05 --step 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_io import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check RL rendering outputs.")
    parser.add_argument("--seq-id", default="rl_episode_random_0001_t1_p05", help="Sequence ID")
    parser.add_argument("--step", type=int, default=5, help="Layout check frame interval")
    parser.add_argument("--check-only", action="store_true", help="Only run compatibility check, not layout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seq_id = args.seq_id
    project_root = Path("D:/projects/FustalMOT_UEDataset")

    ann_file = project_root / "Saved" / "FutsalMOT" / "annotations" / "objects_bbox_2d_clean_{}.json".format(seq_id)
    actual_images_dir = project_root / "Saved" / "FutsalMOT_RL" / "ue_closed_loop" / "images" / seq_id
    expected_images_dir = project_root / "Saved" / "FutsalMOT" / "images_clean" / seq_id

    print("=" * 60)
    print("RL Render Check — Seq ID: {}".format(seq_id))
    print("=" * 60)

    # ── Step 1: Check annotation file ──────────────────────────
    if not ann_file.is_file():
        print("[ERROR] 标注文件不存在: {}".format(ann_file))
        print("请先在 UE 中运行 rl_07b_ue_render_rl.py")
        return 1
    print("[OK] 标注文件: {} ({:.0f} MB)".format(ann_file.name, ann_file.stat().st_size / 1e6))

    # ── Step 2: Create path compatibility ──────────────────────
    # Create junction: images_clean/<seq_id> → ue_closed_loop/images/<seq_id>
    if not expected_images_dir.is_dir():
        expected_images_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Try junction (Windows) via mklink
            import subprocess
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(expected_images_dir), str(actual_images_dir)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print("[OK] 创建目录链接: {} → {}".format(expected_images_dir, actual_images_dir))
            else:
                # Fallback: copy
                print("[INFO] 无法创建链接，使用复制...")
                import shutil
                shutil.copytree(str(actual_images_dir), str(expected_images_dir), dirs_exist_ok=True)
                print("[OK] 复制图像完成")
        except Exception as exc:
            print("[WARNING] 创建链接失败: {}".format(exc))
            print("  手动复制: robocopy \"{}\" \"{}\" /E".format(actual_images_dir, expected_images_dir))
    else:
        print("[OK] 图像目录已存在: {}".format(expected_images_dir))

    # ── Step 3: Create 6-digit filename symlinks ───────────────
    # The annotation expects 000000.png but files are 0000.png
    link_count = 0
    for cam_dir in actual_images_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        target_cam = expected_images_dir / cam_dir.name
        target_cam.mkdir(parents=True, exist_ok=True)

        for png_file in sorted(cam_dir.glob("*.png")):
            # Create 6-digit symlink: 000000.png → 0000.png
            old_name = png_file.stem  # e.g. "0000"
            try:
                frame_num = int(old_name)
                new_name = "{:06d}.png".format(frame_num)
                link_path = target_cam / new_name
                if not link_path.exists():
                    # Use hardlink (same filesystem) or copy
                    try:
                        os.link(str(png_file), str(link_path))
                    except OSError:
                        import shutil
                        shutil.copy2(str(png_file), str(link_path))
                    link_count += 1
            except ValueError:
                pass

    print("[OK] 创建了 {} 个文件名映射 (4位→6位)".format(link_count))

    if args.check_only:
        print("\n[OK] 兼容性检查完成，图像已就绪。")
        print("运行: python 03_check_labels.py --annotation \"{}\" --step {}".format(ann_file, args.step))
        return 0

    # ── Step 4: Run 03_check_labels.py ─────────────────────────
    print("\n▶ 运行 layout_check...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "03_check_labels.py",
         "--annotation", str(ann_file),
         "--step", str(args.step)],
        cwd=str(_CODE_ROOT),
        capture_output=True, text=True, timeout=600,
    )
    print(result.stdout[-1000:] if result.stdout else "")
    if result.stderr:
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

    if result.returncode == 0:
        print("\n[PASS] Layout check 通过!")
    else:
        print("\n[FAIL] Layout check 返回代码: {}".format(result.returncode))

    # ── Step 5: Run UE closed-loop check ───────────────────────
    print("\n▶ 运行 UE 闭环验证...")
    from futsalmot_rl.ue_bridge.ue_closed_loop_check import check_ue_outputs, UE_CLOSED_LOOP_DIR
    report = check_ue_outputs(seq_id=seq_id)

    print("\n结果:")
    for err in report.get("errors", []):
        print("  ❌ {}".format(err))
    for warn in report.get("warnings", []):
        print("  ⚠ {}".format(warn))
    if report.get("passed"):
        print("\n✅ UE 闭环验证通过!")
    else:
        print("\n❌ 部分检查未通过")

    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
