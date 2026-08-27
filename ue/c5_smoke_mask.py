"""C5.1 smoke：重渲染 3 帧 Object ID Mask（1280x720，独立跑，不影响 SaveGame 语义）。

分辨率来自 smoke resolved task（render_rgb）。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../c5_smoke_mask.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_preset import resolve_output_resolution  # noqa: E402

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_smoke_mask.log")
RESOLVED_TASK = Path(
    r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\runtime\c5_smoke_1280\resolved-task.json"
)
MASK_DIR = Path(r"G:\FutsalMOT_Dataset\c5_resolution_smoke_1280\episode_c5_smoke_1280\CineCam_01\mask_c4_render")
SEQ_ASSET = "/Game/FutsalMOT/Sequences/LS_Cam_01.LS_Cam_01"
SEQ_LOAD = "/Game/FutsalMOT/Sequences/LS_Cam_01"
MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"


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
    _log("======== C5.1 smoke：重渲染 Mask ========")
    rt = json.loads(RESOLVED_TASK.read_text(encoding="utf-8"))
    ann_cfg = rt["ue_profile"].get("annotation_export") or {}
    w, h = resolve_output_resolution(ann_cfg)
    _log(f"  resolution = {w}x{h}")

    import shutil
    if MASK_DIR.exists():
        shutil.rmtree(MASK_DIR, ignore_errors=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    seq = unreal.load_asset(SEQ_LOAD)
    seq.set_playback_start(0)
    seq.set_playback_end(3)

    sub = None
    for cls_name in ("MoviePipelineQueueSubsystem", "MovieRenderQueueSubsystem"):
        cls = getattr(unreal, cls_name, None)
        if cls is not None:
            try:
                sub = unreal.get_editor_subsystem(cls)
                break
            except Exception:
                continue
    if sub is None:
        _log("  ERROR: 无 MRQ 子系统")
        _flush()
        return
    queue = sub.get_queue()
    try:
        queue.delete_all_jobs()
    except Exception:
        pass
    job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    job.sequence = unreal.SoftObjectPath(SEQ_ASSET)
    job.map = unreal.SoftObjectPath(MAP)
    try:
        config = job.get_configuration()
    except Exception:
        config = None
    if config is None:
        config = unreal.MoviePipelinePrimaryConfig()
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.output_directory = unreal.DirectoryPath(str(MASK_DIR))
    output.file_name_format = "{frame_number}"
    output.zero_pad_frame_numbers = 4
    output.output_resolution = unreal.IntPoint(w, h)
    try:
        output.frame_rate = unreal.FrameRate(30, 1)
    except Exception:
        pass
    exr = None
    for cls_name in ("MoviePipelineImageSequenceOutput_EXR", "MoviePipelineImageSequenceOutput_OpenEXR"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            exr = config.find_or_add_setting_by_class(cls)
            try:
                exr.set_editor_property("multilayer", True)
            except Exception:
                pass
            break
        except Exception:
            continue
    oid = config.find_or_add_setting_by_class(unreal.MoviePipelineObjectIdRenderPass)
    id_enum = getattr(unreal, "MoviePipelineObjectIdPassIdType", None)
    if id_enum is not None:
        for name in ("ACTOR", "ACTOR_WITH_HIERARCHY"):
            v = getattr(id_enum, name, None)
            if v is not None:
                try:
                    oid.set_editor_property("id_type", v)
                except Exception:
                    pass
                break
    try:
        job.set_configuration(config)
    except Exception:
        try:
            job.configuration = config
        except Exception:
            pass

    executor = None
    for cls_name in ("MoviePipelinePIEExecutor", "MoviePipelineEditorExecutor"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            executor = cls()
            break
        except Exception:
            continue
    if executor is None:
        _log("  ERROR: 无 executor")
        _flush()
        return
    submitted = False
    for m_name in ("render_queue_with_executor_instance", "render_queue_with_executor"):
        m = getattr(sub, m_name, None)
        if m is None:
            continue
        try:
            m(executor)
            submitted = True
            _log(f"  已提交（{m_name}）")
            break
        except Exception as e:
            _log(f"  {m_name} ERR: {type(e).__name__} {e}")
    if not submitted:
        _log("  ERROR: 提交失败")
    _log(f"  异步渲染 3 帧 Mask（{w}x{h}）...")
    _flush()


if __name__ == "__main__":
    main()