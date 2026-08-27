"""C5.1 回归：从正式 resolved task 重新生成 camera.json。

分辨率唯一来源 = resolved task 的 render_rgb.output_resolution（经
resolve_output_resolution 解析），不再依赖任何人工 fix 脚本。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../regenerate_camera_c5.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_preset import resolve_output_resolution  # noqa: E402
from annotation_exporter import read_camera_calibration  # noqa: E402
from dataset_export import write_json_atomic  # noqa: E402

RT = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\runtime\bp_frame_sync_30f\resolved-task.json")
CAM_OUT = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync\CineCam_01")


def main():
    import unreal
    print("======== C5.1 回归：从 resolved task 生成 camera.json ========")
    rt = json.loads(RT.read_text(encoding="utf-8"))
    ann_cfg = rt["ue_profile"].get("annotation_export") or {}
    w, h = resolve_output_resolution(ann_cfg)
    print(f"  resolution (唯一源 render_rgb) = {w}x{h}")

    ed = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cam = None
    for a in ed.get_all_level_actors():
        try:
            if a.get_actor_label() == "CineCam_01":
                cam = a
                break
        except Exception:
            continue
    if cam is None:
        print("  ERROR: CineCam_01 未找到")
        return
    intr, extr, meta = read_camera_calibration(cam, w, h)
    CAM_OUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(CAM_OUT / "camera.json", meta)
    print(f"  camera.json written: fx={intr.fx:.6f} fy={intr.fy:.6f} "
          f"cx={intr.cx} cy={intr.cy} @ {w}x{h}")


if __name__ == "__main__":
    main()