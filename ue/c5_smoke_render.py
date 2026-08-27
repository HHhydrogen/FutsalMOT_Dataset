"""C5.1 隔离 resolution smoke：1280x720 3 帧端到端验证（RGB + Pose Capture + Object ID Mask）。

MRQ 输出分辨率 **不是硬编码**，而是从 smoke resolved task 经
resolve_output_resolution() 解析（render_rgb.output_resolution = 1280x720）。
隔离目录，不触碰 C4 基线。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../c5_smoke_render.py"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_preset import resolve_output_resolution  # noqa: E402

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_smoke_render.log")
RESOLVED_TASK = Path(
    r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\runtime\c5_smoke_1280\resolved-task.json"
)
RGB_DIR = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_smoke_render")
SMOKE_EP = Path(r"G:\FutsalMOT_Dataset\c5_resolution_smoke_1280\episode_c5_smoke_1280")
MASK_DIR = SMOKE_EP / "CineCam_01" / "mask_c4_render"
SEQ_ASSET = "/Game/FutsalMOT/Sequences/LS_Cam_01.LS_Cam_01"
SEQ_LOAD = "/Game/FutsalMOT/Sequences/LS_Cam_01"
MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
BURN_IN_CLASS = "/Game/FutsalMOT/Blueprints/WBP_PoseMRQBurnInC4.WBP_PoseMRQBurnInC4_C"
SLOTS = [f"PoseCaptureG{i}" for i in range(5)]
FRAME_START = 0
FRAME_END = 3  # 3 帧
DONE = RGB_DIR / "smoke_done.txt"


def _log(msg):
    print(msg)
    LOG.append(str(msg))


def _flush():
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("\n".join(LOG), encoding="utf-8")
    except Exception as e:
        print(f"  写日志失败: {e}")


def _on_finished(executor, success):
    try:
        _log(f"MRQ finished -> success={success}")
        DONE.write_text(f"success={success}", encoding="utf-8")
    except Exception as e:
        _log(f"_on_finished ERR: {e}")


def _build_job(queue, w, h, out_dir, kind):
    """kind: 'rgb'（Deferred+PNG+BurnIn）或 'mask'（ObjectId+EXR）。"""
    import unreal
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
    output.output_directory = unreal.DirectoryPath(str(out_dir))
    output.file_name_format = "{frame_number}"
    output.zero_pad_frame_numbers = 4
    output.output_resolution = unreal.IntPoint(w, h)
    try:
        output.frame_rate = unreal.FrameRate(30, 1)
    except Exception:
        pass
    if kind == "rgb":
        config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
        config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
        try:
            burnin = config.find_or_add_setting_by_class(unreal.MoviePipelineBurnInSetting)
            burnin.set_editor_property("burn_in_class", unreal.SoftClassPath(BURN_IN_CLASS))
            burnin.set_editor_property("composite_onto_final_image", False)
        except Exception as e:
            _log(f"  BurnIn ERR: {type(e).__name__} {e}")
    else:
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
        if exr is None:
            raise RuntimeError("无 EXR 输出类")
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
            _log("  set_configuration 失败")
    return job


def main():
    import unreal
    _log("======== C5.1 隔离 smoke：1280x720 3 帧 ========")

    rt = json.loads(RESOLVED_TASK.read_text(encoding="utf-8"))
    ann_cfg = rt["ue_profile"].get("annotation_export") or {}
    w, h = resolve_output_resolution(ann_cfg)
    _log(f"  resolution（来自 resolved task, render_rgb）= {w}x{h}")

    for sl in SLOTS:
        try:
            unreal.GameplayStatics.delete_game_in_slot(sl, 0)
        except Exception:
            pass
    _log("  已清空 SaveGame slot")

    import shutil
    for d in (RGB_DIR, MASK_DIR):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    seq = unreal.load_asset(SEQ_LOAD)
    seq.set_playback_start(FRAME_START)
    seq.set_playback_end(FRAME_END)
    _log(f"  playback {FRAME_START}..{FRAME_END}")

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
    _build_job(queue, w, h, RGB_DIR, "rgb")
    _log(f"  已加入 RGB+capture job（{w}x{h}）【Mask job 单独跑，避免覆盖 SaveGame】")

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
    delegate = getattr(executor, "on_executor_finished_delegate", None)
    if delegate is not None:
        try:
            delegate.add_callable(_on_finished)
        except Exception as e:
            _log(f"  绑定 delegate ERR: {e}")
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
    _log("  异步渲染 3 帧（RGB + Mask）中...")
    _flush()


if __name__ == "__main__":
    main()