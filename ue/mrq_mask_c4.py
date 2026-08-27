"""C4.1：渲染 90 帧 Object ID Instance Mask（弱 Ground Truth，用于投影偏移量化）。

输出 multilayer EXR 到 G:\...\CineCam_01\mask_c4_render\。
与 C4 RGB 同序列/同相机/同分辨率（1920x1080），帧号 1:1 对齐。

用法（Unreal Editor Python，FutsalMOTTools.run_python_file）：
    py ".../mrq_mask_c4.py"
"""

from pathlib import Path

LOG = []
LOG_PATH = Path(r"D:\projects\FustalMOT_UEDataset\Content\FutsalMOT\code\.futsalmot\mrq_mask_c4.log")
MASK_DIR = Path(r"G:\FutsalMOT_Dataset\episode_bp_frame_sync\CineCam_01\mask_c4_render")
SEQ_ASSET = "/Game/FutsalMOT/Sequences/LS_Cam_01.LS_Cam_01"
SEQ_LOAD = "/Game/FutsalMOT/Sequences/LS_Cam_01"
MAP = "/Game/FutsalMOT/Maps/L_FutsalCourt"
IMAGE_W = 1920
IMAGE_H = 1080


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
    _log("======== C4.1 掩码渲染（90 帧 Object ID EXR）========")

    import shutil
    if MASK_DIR.exists():
        shutil.rmtree(MASK_DIR, ignore_errors=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    seq = unreal.load_asset(SEQ_LOAD)
    if seq is None:
        _log("  ERROR: 序列加载失败")
        _flush()
        return
    seq.set_playback_start(0)
    seq.set_playback_end(90)
    _log("  playback 0..90")

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
    output.output_resolution = unreal.IntPoint(IMAGE_W, IMAGE_H)
    try:
        output.frame_rate = unreal.FrameRate(30, 1)
    except Exception:
        pass

    # EXR 输出（multilayer）
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
        _log("  ERROR: 无 EXR 输出类")
        _flush()
        return

    # Object ID pass（IdType=Actor）
    oid = config.find_or_add_setting_by_class(unreal.MoviePipelineObjectIdRenderPass)
    id_enum = getattr(unreal, "MoviePipelineObjectIdPassIdType", None)
    if id_enum is not None:
        for name in ("ACTOR", "ACTOR_WITH_HIERARCHY"):
            v = getattr(id_enum, name, None)
            if v is not None:
                try:
                    oid.set_editor_property("id_type", v)
                    _log(f"  ObjectId id_type={name}")
                except Exception as e:
                    _log(f"  id_type set ERR: {e}")
                break

    try:
        job.set_configuration(config)
    except Exception:
        try:
            job.configuration = config
        except Exception:
            _log("  set_configuration 失败")

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
    _log("  异步渲染 90 帧掩码中...")
    _flush()


if __name__ == "__main__":
    main()