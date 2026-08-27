"""C5.2 测试渲染：按 env 渲染 N 帧（可选 BurnIn），用于 Test1/Test2/Test3。

env:
  C5_RENDER_FRAMES  帧数（默认 3）
  C5_RENDER_MODE    'pose'（带 BurnIn）| 'nonpose'（RGB-only，无 BurnIn）
  C5_RENDER_DIR     输出目录

pose 模式：清空目标动态 slot 后带 BurnIn 渲染（CaptureOutputFrame 被调用）。
nonpose 模式：不带 BurnIn 渲染（CaptureOutputFrame 不被调用 → recorder 不参与 session）。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../c5_render.py"
"""

import os
from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_render.log")
SEQ_ASSET = "/Game/FutsalMOT/Sequences/LS_Cam_01.LS_Cam_01"
SEQ_LOAD = "/Game/FutsalMOT/Sequences/LS_Cam_01"
MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
BURN_IN_CLASS = "/Game/FutsalMOT/Blueprints/Pose/MRQ/WBP_PoseMRQBurnInC4.WBP_PoseMRQBurnInC4_C"
EPISODE = "c5test"
CAMERA = "CineCam_01"
SLOTS = [f"PoseCapture_{EPISODE}_{CAMERA}_G{i}" for i in range(5)]
DEFAULT_DIR = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\c5_render")


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
    frames = int(os.environ.get("C5_RENDER_FRAMES", "3"))
    mode = os.environ.get("C5_RENDER_MODE", "pose")
    out_dir = Path(os.environ.get("C5_RENDER_DIR", str(DEFAULT_DIR)))
    burnin = (mode == "pose")
    _log(f"======== C5.2 渲染：frames={frames} mode={mode} burnin={burnin} ========")

    if burnin:
        for sl in SLOTS:
            try:
                unreal.GameplayStatics.delete_game_in_slot(sl, 0)
            except Exception:
                pass
        _log("  已清空目标动态 slot")

    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = unreal.load_asset(SEQ_LOAD)
    seq.set_playback_start(0)
    seq.set_playback_end(frames)

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
    output.output_directory = unreal.DirectoryPath(str(out_dir))
    output.file_name_format = "{frame_number}"
    output.zero_pad_frame_numbers = 4
    output.output_resolution = unreal.IntPoint(1920, 1080)
    try:
        output.frame_rate = unreal.FrameRate(30, 1)
    except Exception:
        pass
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    if burnin:
        try:
            b = config.find_or_add_setting_by_class(unreal.MoviePipelineBurnInSetting)
            b.set_editor_property("burn_in_class", unreal.SoftClassPath(BURN_IN_CLASS))
            b.set_editor_property("composite_onto_final_image", False)
        except Exception as e:
            _log(f"  BurnIn ERR: {type(e).__name__} {e}")
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
    _log(f"  异步渲染 {frames} 帧（{mode}）...")
    _flush()


if __name__ == "__main__":
    main()