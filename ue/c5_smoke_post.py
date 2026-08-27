"""C5.1 smoke 后处理：导出 SaveGame→pose JSONL + 生成 smoke camera.json（1280x720）。

分辨率来自 smoke resolved task（render_rgb，经 resolve_output_resolution）。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../c5_smoke_post.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_preset import resolve_output_resolution  # noqa: E402
from annotation_exporter import read_camera_calibration  # noqa: E402
from dataset_export import write_json_atomic  # noqa: E402

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_smoke_post.log")
RESOLVED_TASK = Path(
    r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\runtime\c5_smoke_1280\resolved-task.json"
)
SMOKE_EP = Path(r"G:\FutsalMOT_Dataset\c5_resolution_smoke_1280\episode_c5_smoke_1280")
CAM = SMOKE_EP / "CineCam_01"
SLOTS = [f"PoseCaptureG{i}" for i in range(5)]


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def main():
    import unreal
    _log("======== C5.1 smoke 后处理：pose 导出 + camera.json ========")
    rt = json.loads(RESOLVED_TASK.read_text(encoding="utf-8"))
    ann_cfg = rt["ue_profile"].get("annotation_export") or {}
    w, h = resolve_output_resolution(ann_cfg)
    _log(f"  resolution = {w}x{h}")

    # 1) camera.json（正式 calibration 流程）
    ed = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cam_actor = None
    for a in ed.get_all_level_actors():
        try:
            if a.get_actor_label() == "CineCam_01":
                cam_actor = a
                break
        except Exception:
            continue
    if cam_actor is None:
        _log("  ERROR: CineCam_01 未找到")
        _flush()
        return
    _, _, cam_meta = read_camera_calibration(cam_actor, w, h)
    CAM.mkdir(parents=True, exist_ok=True)
    write_json_atomic(CAM / "camera.json", cam_meta)
    _log(f"  camera.json: fx={cam_meta['intrinsics']['fx']} cx={cam_meta['intrinsics']['cx']} {w}x{h}")

    # 2) SaveGame → pose JSONL（390 行）
    cap, shot, gt, aid, bone, loc, rot = [], [], [], [], [], [], []
    for sl in SLOTS:
        sg = unreal.GameplayStatics.load_game_from_slot(sl, 0)
        if sg is None:
            _log(f"  ERROR: {sl} 空")
            continue
        cap += list(sg.get_editor_property("capture_indices"))
        shot += list(sg.get_editor_property("shot_frames"))
        gt += list(sg.get_editor_property("game_times"))
        aid += list(sg.get_editor_property("actor_ids"))
        bone += list(sg.get_editor_property("bone_names"))
        loc += list(sg.get_editor_property("world_locations"))
        rot += list(sg.get_editor_property("world_rotations"))

    lib = getattr(unreal, "MathLibrary", None)
    rows = []
    for i in range(len(cap)):
        q = lib.conv_rotator_to_quaternion(rot[i])
        rows.append({
            "root": cap[i], "shot": shot[i], "game_time": round(gt[i], 6),
            "actor_id": aid[i], "bone": bone[i],
            "x": round(loc[i].x, 3), "y": round(loc[i].y, 3), "z": round(loc[i].z, 3),
            "qx": round(q.x, 6), "qy": round(q.y, 6), "qz": round(q.z, 6), "qw": round(q.w, 6),
        })
    (SMOKE_EP / "pose_capture_c4.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    _log(f"  pose 导出 {len(rows)} 行 -> {SMOKE_EP / 'pose_capture_c4.jsonl'}")
    _flush()


if __name__ == "__main__":
    main()