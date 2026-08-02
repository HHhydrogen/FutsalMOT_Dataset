"""UE 侧自动渲染 RGB 帧（Movie Render Queue, MRQ）。

职责：
  1. 用 MRQ 渲染每个 Camera 对应的 Level Sequence，输出 PNG 到 <cam>/render/。
  2. 按帧同步契约从渲染结果里选出与 annotation 一一对应的帧，复制为
     <cam>/img1/{frame_index:06d}.png。

帧同步契约（与 annotation_exporter / README 一致）：
  - annotation frame_index（1 基）= 图片 img1/{frame_index:06d}.png。
  - annotation frame N ↔ Sequence 帧 round((N-1)*source_step_seconds*playback_fps)。
  - MRQ 以 Sequence 的 display rate（= playback_fps）渲染全范围，再按上面的映射取帧。

MRQ Python API 在不同 UE 版本命名/结构不同，本模块用多级 fallback + 清晰报错；
纯函数部分（帧选择、映射、复制）不依赖 unreal，可用 pytest 测试。

重要：渲染必须异步。提交 MRQ 后立即返回，后续阶段（复制 RGB / 写完成标记）由
Movie Render Queue 的 finished/error delegate 回调驱动，并有 slate post-tick
watchdog 兜底。delegate 回调须用与委托一致的显式签名（不能用 *args），否则
UE 5.8 绑定会报 "incorrect number of arguments" 而失败。PIE 渲染窗口需要编辑器
主线程持续 tick 才能推进，任何 time.sleep / Event.wait 同步阻塞都会卡死渲染。
"""

import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_export import ensure_dir, load_episode, write_json_atomic  # noqa: E402

# 模块级：当前活跃的异步渲染管线。脚本（main）返回后 MRQ delegate 仍持有它的
# 方法引用，但显式保留模块级引用可防止任何环境下对象被垃圾回收。
_ACTIVE_RENDER = None


# ── 纯函数：帧选择与映射 ────────────────────────────────────────────────

def select_rendered_frame_indices(
    num_steps: int,
    source_step_seconds: float,
    playback_fps: int,
) -> List[int]:
    """返回与 annotation 各帧对应的渲染帧序号（0 基）。

    annotation frame_index = i + 1（i 为 0 基 GRF step）对应的 Sequence 帧：
        round(i * source_step_seconds * playback_fps)
    """
    return [
        int(round(i * source_step_seconds * playback_fps))
        for i in range(num_steps)
    ]


def map_rendered_to_annotation(
    rendered_numbers: Sequence[int],
    keep_indices: Sequence[int],
) -> Dict[int, int]:
    """把渲染输出帧号映射到 annotation frame_index。

    rendered_numbers: render 目录中实际存在的渲染帧号列表（0 基，可为乱序）。
    keep_indices: select_rendered_frame_indices 的返回值。
    返回 {frame_index: rendered_number}，frame_index 从 1 开始。
    若某目标帧在渲染输出中缺失，跳过（不产生非法映射）。
    """
    available = set(rendered_numbers)
    mapping: Dict[int, int] = {}
    for i, num in enumerate(keep_indices):
        if num in available:
            mapping[i + 1] = num
    return mapping


def find_rendered_frame_numbers(render_dir: Path) -> Dict[int, Path]:
    """扫描 render 目录（含子目录），从 PNG 文件名里解析帧号。

    返回 {frame_number: path}。文件名须包含数字（MRQ 输出如 000000.png）。
    """
    result: Dict[int, Path] = {}
    for p in sorted(render_dir.rglob("*.png")):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits:
            result[int(digits)] = p
    return result


def copy_rendered_frames(
    render_dir: Path,
    img1_dir: Path,
    keep_indices: Sequence[int],
) -> int:
    """把渲染帧中与 annotation 对齐的帧复制为 img1/{frame_index:06d}.png。

    返回复制的帧数。
    """
    rendered = find_rendered_frame_numbers(render_dir)
    mapping = map_rendered_to_annotation(sorted(rendered.keys()), keep_indices)
    ensure_dir(img1_dir)
    copied = 0
    for frame_index, num in mapping.items():
        dst = img1_dir / f"{frame_index:06d}.png"
        shutil.copy2(rendered[num], dst)
        copied += 1
    return copied


def recover_render_to_img1(
    sequences_cfg,
    annotation_cfg: dict,
    episode_dir: Path,
    output_dir: Path,
):
    """从已渲染的 render/ 目录恢复 RGB 到 img1/（无需 MRQ / 无需 UE 渲染）。

    场景：上一次 MRQ 渲染已把 PNG 输出到各 camera 的 render/，但完成回调
    （finished delegate / watchdog）未触发，导致 img1/ 为空。用本函数从现有
    render/ 帧复制出 img1/ 并写 render_summary.json，避免重新渲染。

    纯 Python，不依赖 unreal，可在 P1（.venv）或 UE Python 中运行。

    Returns:
        (status, per_camera)。status ∈ {"success", "partial", "failed"}。
    """
    render_cfg = annotation_cfg.get("render_rgb") or {}
    meta, frames = load_episode(episode_dir)
    episode_id = meta.get("episode_id") or episode_dir.name
    source_step = float(meta["timing"].get("source_step_seconds", 0.1))
    frame_rate = int(render_cfg.get("frame_rate") or annotation_cfg.get("playback_fps") or 30)
    keep_indices = select_rendered_frame_indices(len(frames), source_step, frame_rate)

    per_camera = {}
    total_copied = 0
    expected = len(keep_indices)
    for seq_entry in (sequences_cfg or []):
        seq_name = seq_entry.get("name")
        cam_id = seq_entry.get("camera_actor") or seq_name
        if not seq_name:
            continue
        cam_out = Path(output_dir) / episode_id / cam_id
        render_dir, img1_dir = cam_out / "render", cam_out / "img1"
        if not render_dir.exists():
            print(f"  [SKIP] {cam_id}: 无 render/ 目录，跳过")
            continue
        copied = copy_rendered_frames(render_dir, img1_dir, keep_indices)
        total_copied += copied
        per_camera[cam_id] = {
            "sequence": seq_name,
            "img1_frames": copied,
            "expected_frames": expected,
            "ok": copied == expected,
        }
        if copied == 0:
            mark = "MISSING"
        elif copied == expected:
            mark = "OK"
        else:
            mark = "PARTIAL"
        print(f"  [{mark}] {cam_id}: img1/ 写入 {copied}/{expected} 帧")

    if not per_camera:
        print("WARNING: 没有任何可恢复的 camera render/ 目录")
        return "failed", per_camera

    if total_copied == 0:
        status = "failed"
    elif all(e["ok"] for e in per_camera.values()):
        status = "success"
    else:
        status = "partial"
    summary = {
        "episode_id": episode_id,
        "status": status,
        "reason": "从已有 render/ 目录恢复（MRQ 完成回调未触发）",
        "total_img1_frames": total_copied,
        "cameras": per_camera,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "由 recover_render_to_img1 恢复写入。完整标注校验请在 P1 运行："
                "uv run grf-ue validate-annotations <output_dir>",
    }
    summary_path = Path(output_dir) / episode_id / "render_summary.json"
    write_json_atomic(summary_path, summary)
    print(f"\n=== 恢复完成: {status} ===")
    print(f"  render_summary.json -> {summary_path}")
    return status, per_camera


# ── UE 侧：MRQ 渲染 ─────────────────────────────────────────────────────

def _list_mrq_classes():
    """打印当前 unreal 模块里可用的 MRQ/渲染相关类，便于按 UE 版本适配。

    MRQ Python API 在不同 UE 版本命名不同（本工程为 UE 5.8）。当找不到期望
    的类时调用，把输出贴回即可定位正确的类名。
    """
    import unreal

    names = sorted(
        n for n in dir(unreal)
        if "MoviePipeline" in n
        or "RenderQueue" in n
        or "MovieRender" in n
        or "SequencerTools" in n
        or "SceneCapture" in n
    )
    print("  [MRQ 诊断] 相关 unreal 类:", names)
    return names


def _print_mrq_members(obj, label: str) -> None:
    """打印 MRQ 对象的可用成员名，便于适配不同 UE 版本的 API。"""
    names = sorted(
        n for n in dir(obj)
        if not n.startswith("__")
        and any(k in n.lower() for k in ("setting", "config", "shot", "render", "output", "add_", "set_", "get_"))
    )
    print(f"  [MRQ 诊断] {label} 相关成员:", names)


def _get_current_map_path() -> str:
    """获取当前关卡（world）的资源路径，供 MRQ job.map 使用。"""
    import unreal

    world = None
    for getter in (
        lambda: unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world(),
        lambda: unreal.EditorLevelLibrary.get_editor_world(),
        lambda: unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).get_editor_world(),
    ):
        try:
            world = getter()
            if world is not None:
                break
        except Exception:
            continue
    if world is None:
        return ""
    return world.get_path_name().split(".")[0]  # 去掉对象实例后缀


def _get_job_config(job):
    """取 job 的配置对象（MoviePipelinePrimaryConfig/MasterConfig）。

    不同 UE 版本获取方式不同，多级 fallback；取不到则新建。
    """
    import unreal

    for getter_name in ("get_configuration",):
        m = getattr(job, getter_name, None)
        if m is not None:
            try:
                cfg = m()
                if cfg is not None:
                    return cfg
            except Exception:
                pass
    for attr in ("configuration", "primary_config"):
        try:
            cfg = getattr(job, attr, None)
            if cfg is not None:
                return cfg
        except Exception:
            pass
    return unreal.MoviePipelinePrimaryConfig()


def _set_job_config(job, config) -> bool:
    """把配置挂到 job 上。不同 UE 版本 API 不同，多级 fallback。"""
    for method_name in ("set_configuration",):
        m = getattr(job, method_name, None)
        if m is not None:
            try:
                m(config)
                return True
            except Exception:
                pass
    for attr in ("configuration", "primary_config"):
        try:
            setattr(job, attr, config)
            return True
        except Exception:
            pass
    return False


def _mrq_get_queue(subsystem):
    """获取 MRQ 当前队列。"""
    try:
        return subsystem.get_queue()
    except Exception:
        return None


def _clear_queue(queue) -> None:
    """清空队列里的已有 job，避免残留。"""
    if queue is None:
        return
    for method_name in ("delete_all_jobs", "clear"):
        m = getattr(queue, method_name, None)
        if m is not None:
            try:
                m()
                return
            except Exception:
                continue
    try:
        queue.jobs.clear()
    except Exception:
        pass


def _clear_dir(path: Path) -> None:
    """清空一个目录的内容（含子目录），避免上一次渲染的残留帧混入。"""
    import shutil

    if not path.exists():
        return
    for p in path.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def _build_mrq_job(
    queue,
    seq_asset_path: str,
    map_path: str,
    render_dir: Path,
    image_width: int,
    image_height: int,
    frame_rate: int,
    file_name_format: str,
    zero_pad: int,
):
    """从队列分配并配置一个 MRQ job。返回 job，失败时抛错并附 API 说明。"""
    import unreal

    for cls_name in (
        "MoviePipelineExecutorJob",
        "MoviePipelineOutputSetting",
        "MoviePipelineImageSequenceOutput_PNG",
        "MoviePipelinePrimaryConfig",
    ):
        if getattr(unreal, cls_name, None) is None:
            _list_mrq_classes()
            raise RuntimeError(
                f"unreal.{cls_name} 不存在。已打印可用 MRQ 类，请把上面 "
                f"[MRQ 诊断] 输出贴回。"
            )

    # 从队列分配 job，保证它会被渲染队列执行
    try:
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
    except Exception:
        job = unreal.MoviePipelineExecutorJob()
        try:
            queue.jobs.add(job)
        except Exception:
            pass
    job.sequence = unreal.SoftObjectPath(seq_asset_path)
    if map_path:
        job.map = unreal.SoftObjectPath(map_path)

    # UE 5.8 MRQ：配置用 find_or_add_setting_by_class 获取/创建设置并配置。
    # 注意：setup_basic_configuration() 依赖默认预设资产 /Temp/MovieRenderPipeline/
    # BasicConfigDefaults，本项目不存在，无法自动带上渲染 pass，需手动添加。
    config = _get_job_config(job)
    try:
        output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
        output.output_directory = unreal.DirectoryPath(str(render_dir))
        output.file_name_format = file_name_format
        output.zero_pad_frame_numbers = zero_pad
        output.output_resolution = unreal.IntPoint(image_width, image_height)
        try:
            output.frame_rate = unreal.FrameRate(frame_rate, 1)
        except Exception:
            pass

        png = config.find_or_add_setting_by_class(
            unreal.MoviePipelineImageSequenceOutput_PNG
        )
        # PNG 输出无需额外配置

        # 渲染 pass（必须，否则 shot 为 0 passes 不输出任何帧）。
        # 标准 Deferred 渲染 pass；若该基类不可实例化，逐个尝试具体变体。
        pass_added = False
        pass_classes = (
            unreal.MoviePipelineDeferredPassBase,
            unreal.MoviePipelineDeferredPass_PathTracer,
            unreal.MoviePipelineDeferredPass_Unlit,
        )
        for idx, pass_cls in enumerate(pass_classes):
            try:
                config.find_or_add_setting_by_class(pass_cls)
                if idx > 0:
                    print(
                        f"  [MRQ] 使用备选渲染 pass: {pass_cls.__name__}"
                        f"（标准 MoviePipelineDeferredPassBase 不可用）"
                    )
                pass_added = True
                break
            except Exception:
                continue
        if not pass_added:
            raise RuntimeError(
                "无法添加 MRQ 渲染 pass（DeferredPassBase/PathTracer/Unlit 均失败）。"
                "请把 [MRQ 诊断] 输出贴回。"
            )

        # 诊断：打印 config 现有设置类（确认是否含渲染 pass）
        try:
            existing = config.get_all_settings()
            names = sorted(type(s).__name__ for s in existing) if existing else []
            print(f"  [MRQ] config 设置类: {names}")
        except Exception:
            pass
    except Exception as e:
        _print_mrq_members(config, "config")
        _print_mrq_members(job, "job")
        raise RuntimeError(
            f"无法用 find_or_add_setting_by_class 配置 MRQ 设置: {e}\n"
            f"已打印 config/job 相关成员，请把上面 [MRQ 诊断] 输出贴回。"
        )

    if not _set_job_config(job, config):
        _print_mrq_members(job, "job")
        raise RuntimeError(
            "无法把配置挂到 job（set_configuration 失败）。请把 [MRQ 诊断] 输出贴回。"
        )

    return job


def _submit_render(subsystem, executor) -> bool:
    """提交渲染当前队列。不同 UE 版本子系统方法名可能不同。

    注意：UE 5.8 的 render_queue_with_executor_instance(executor) 只接受 1 个参数，
    渲染的是子系统当前队列（job 需先通过 queue.allocate_new_job 加入）。
    """
    for method_name in (
        "render_queue_with_executor_instance",
        "render_queue_with_executor",
        "render_queue_instance_with_executor",
        "render_queue_instance_with_executor_instance",
    ):
        m = getattr(subsystem, method_name, None)
        if m is None:
            continue
        try:
            m(executor)
            return True
        except Exception as e:
            print(f"  [MRQ] {method_name} 调用失败: {e}")
            continue
    _print_mrq_members(subsystem, "subsystem")
    _print_mrq_members(executor, "executor")
    return False


# ── MRQ 异步渲染状态机 ─────────────────────────────────────────────────

class _AsyncRenderPipeline:
    """MRQ 异步渲染状态机：提交后立即返回，由 finished/error delegate 驱动。

    阶段：
      start()    清空队列、把所有 Sequence 加入同一个 MRQ queue、绑定 delegate、
                 提交渲染并立即返回（不阻塞编辑器主线程）。
      finished   MRQ 整批渲染完成后触发：检查结果、复制 RGB 到 img1/、
                 写 render_summary.json 完成标记。
      errored    记录 MRQ 错误信息（最终成败由 finished 委托的 success 决定）。

    关键：绝不在本模块内用 time.sleep / Event.wait 阻塞主线程等待 MRQ 完成。
    MoviePipelinePIEExecutor 需要在编辑器主线程持续 tick 才能推进 PIE 窗口，
    任何同步阻塞都会让渲染卡死（上一版本 _render_and_wait 的根因）。
    """

    def __init__(self, jobs, episode_id, output_dir, keep_indices):
        self.jobs = jobs  # [{name, cam_id, seq_asset_path, cam_out, render_dir, img1_dir, job}]
        self.episode_id = episode_id
        self.output_dir = Path(output_dir)
        self.keep_indices = list(keep_indices)
        self.error_messages = []
        self.executor = None
        self.finished = False
        self._watch_handle = None  # slate post-tick 兜底检测的句柄
        self._watch_state = None

    def summary_path(self) -> Path:
        """渲染完成标记（render_summary.json）的路径。"""
        return self.output_dir / self.episode_id / "render_summary.json"

    # ── 生命周期 ──────────────────────────────────────────────────

    def start(self, subsystem, queue, executor, map_path,
              image_width, image_height, frame_rate,
              file_name_format, zero_pad) -> None:
        """清空队列、构建并加入全部 job、绑定 delegate、提交渲染后立即返回。"""
        _clear_queue(queue)

        prepared = []
        for info in self.jobs:
            try:
                job = _build_mrq_job(
                    queue, info["seq_asset_path"], map_path, info["render_dir"],
                    image_width, image_height, frame_rate, file_name_format, zero_pad,
                )
            except Exception as e:
                print(f"  ERROR: 构建 MRQ job 失败 {info['name']}: {e}")
                self.error_messages.append(f"{info['name']}: {e}")
                continue
            info["job"] = job
            prepared.append(info)
        if not prepared:
            self._finalize(failed=True, reason="没有任何可提交的 MRQ job")
            return
        self.jobs = prepared

        self.executor = executor
        # 诊断：打印 executor 上可用的 delegate/渲染相关成员，便于按 UE 版本适配
        try:
            names = sorted(
                n for n in dir(executor)
                if any(k in n.lower() for k in ("delegate", "finished", "errored", "render"))
            )
            print("  [MRQ] executor 相关成员:", names)
        except Exception:
            pass

        # 绑定 delegate：MRQ 完成 / 出错时驱动后续阶段（而非同步等待）
        for delegate_name, cb in (
            ("on_executor_finished_delegate", self._on_finished),
            ("on_executor_errored_delegate", self._on_errored),
        ):
            delegate = getattr(executor, delegate_name, None)
            if delegate is None:
                print(f"  WARNING: executor 无 {delegate_name}，无法绑定回调")
                continue
            try:
                delegate.add_callable(cb)
                print(f"  已绑定 {delegate_name}")
            except Exception as e:
                print(f"  WARNING: 绑定 {delegate_name} 失败: {e}")

        if not _submit_render(subsystem, executor):
            self._finalize(failed=True,
                           reason="无法提交 MRQ 渲染（render_queue_with_executor_instance 等均失败）")
            return

        print("\n=== MRQ 渲染已异步提交（不阻塞编辑器主线程）===")
        print(f"  Sequence 数: {len(self.jobs)}，全部加入同一个 MRQ queue")
        for info in self.jobs:
            print(f"    - {info['name']} -> {info['img1_dir']}")
        print(f"  渲染完成后自动：复制 RGB -> img1/，写完成标记 {self.summary_path()}")
        print("  请保持编辑器运行；完成后控制台会打印结果汇总。")

        # 兜底：finished delegate 不触发的 UE 版本也能完成收尾（非阻塞轮询）
        self._start_completion_watchdog()

    # ── delegate 回调（游戏线程触发，可安全访问 unreal 与文件系统）──

    def _on_finished(self, executor, success):
        """MRQ 整批渲染结束。签名与 on_executor_finished_delegate 一致：(executor, success)。

        UE 5.8 的 delegate 绑定会检查回调签名：必须是显式参数，不能用 *args（变参
        计为 0 个形参，报 "Callable has the incorrect number of arguments
        (expected 2, got 0)"，导致绑定失败、回调永不触发）。
        """
        if self.finished:
            return
        try:
            print(f"  [MRQ] on_executor_finished_delegate 回调触发（success={success}）")
            if not success:
                self._finalize(failed=True, reason="MRQ finished 委托报告 success=False")
                return
            self._copy_and_finalize()
        except Exception as e:
            # 回调异常必须可见（UE 默认吞掉 delegate 异常）。watchdog / 手动入口
            # 仍可收尾，但打印 traceback 便于定位。
            print(f"  [MRQ 回调异常] _on_finished: {e}")
            traceback.print_exc()

    def _on_errored(self, executor, pipeline, is_fatal, errors):
        """MRQ 渲染出错（不中断最终收尾，成败由 finished 委托决定）。

        签名与 on_executor_errored_delegate 一致：
        (executor, pipeline, is_fatal, errors)。errors 为 MRQ 控制台输出条目列表。
        """
        try:
            try:
                entries = list(errors)
            except TypeError:
                entries = [errors]
            parts = [str(e) for e in entries if str(e)]
            if not parts:
                parts.append(str(errors))
            self.error_messages.append(" | ".join(parts))
            print(f"  [MRQ 错误] {' | '.join(parts)}")
        except Exception as e:
            print(f"  [MRQ 回调异常] _on_errored: {e}")
            traceback.print_exc()

    # ── 完成检测兜底（非阻塞 watchdog）──────────────────────────────

    def _start_completion_watchdog(self):
        """注册 slate post-tick 兜底：finished delegate 不触发时也能收尾。

        兜底：delegate 绑定须用显式签名，若绑定失败（UE 5.8 拒绝 *args 变参，
        报 "incorrect number of arguments"），finished 回调不触发，由 watchdog
        完成收尾。watchdog 每编辑帧 tick 一次，全部为非阻塞轮询（绝不 sleep），
        判定条件（满足其一即收尾）：
          1. executor 不再渲染 且 所有 camera 的目标帧已输出；
          2. is_rendering API 不可用 且 目标帧齐全 且 文件数 ~20s 无变化；
          3. 硬超时 30 分钟（按现有文件收尾）。
        与 delegate 幂等：_copy_and_finalize / _finalize 有 self.finished 守卫。
        """
        import unreal

        self._watch_state = {
            "last_total": -1,
            "stable_ticks": 0,
            "start": time.monotonic(),
            "seen_rendering": False,
        }

        def _on_tick(_delta):
            if self.finished:
                return False  # 已收尾，注销本回调
            st = self._watch_state
            rendering = self._executor_rendering()
            present = self._all_keep_frames_present()
            total = self._total_render_files()
            if total == st["last_total"]:
                st["stable_ticks"] += 1
            else:
                st["stable_ticks"] = 0
            st["last_total"] = total
            if rendering is True:
                st["seen_rendering"] = True
            elapsed = time.monotonic() - st["start"]

            if rendering is False and present:
                print("  [MRQ] watchdog：executor 不再渲染且目标帧齐全，开始收尾")
                self._copy_and_finalize()
                return False
            if rendering is None and present and st["stable_ticks"] >= 1200:
                print("  [MRQ] watchdog：文件数稳定兜底，开始收尾")
                self._copy_and_finalize()
                return False
            if elapsed > 1800.0:
                print("  WARNING: MRQ 渲染等待超时（30 分钟），按当前文件收尾")
                self._copy_and_finalize()
                return False
            return True

        try:
            handle = unreal.register_slate_post_tick_callback(_on_tick)
        except Exception as e:
            print(f"  WARNING: 无法注册 slate post-tick 兜底检测: {e}（依赖 finished delegate）")
            self._watch_handle = None
            return
        self._watch_handle = handle
        print("  已注册非阻塞完成检测（watchdog，每编辑帧检查一次）")

    def _executor_rendering(self) -> Optional[bool]:
        """executor 是否正在渲染。is_rendering API 不可用时返回 None。"""
        m = getattr(self.executor, "is_rendering", None)
        if m is None:
            return None
        return bool(m())

    def _all_keep_frames_present(self) -> bool:
        """所有 camera 的目标渲染帧（keep_indices）是否已输出到 render/。"""
        for info in self.jobs:
            available = find_rendered_frame_numbers(info["render_dir"])
            if not all(n in available for n in self.keep_indices):
                return False
        return True

    def _total_render_files(self) -> int:
        """所有 camera render/ 目录的 PNG 总数。"""
        return sum(
            len(list(info["render_dir"].rglob("*.png"))) for info in self.jobs
        )

    # ── 收尾：复制 RGB + 轻量校验 + 写完成标记 ─────────────────────

    def _copy_and_finalize(self):
        """finished 成功后：逐 camera 复制渲染帧到 img1/，并做帧数一致性校验。"""
        if self.finished:
            return
        per_camera = {}
        total_copied = 0
        expected = len(self.keep_indices)
        for info in self.jobs:
            cam_id = info["cam_id"]
            render_dir, img1_dir = info["render_dir"], info["img1_dir"]
            copied = copy_rendered_frames(render_dir, img1_dir, self.keep_indices)
            total_copied += copied
            entry = {
                "sequence": info["name"],
                "img1_frames": copied,
                "expected_frames": expected,
                "ok": copied == expected,
            }
            ann_count = self._check_annotation_frame_count(info["cam_out"])
            if ann_count is not None:
                entry["annotations_jsonl_frames"] = ann_count
                entry["annotation_img1_match"] = ann_count == copied
            per_camera[cam_id] = entry

            if copied == 0:
                mark = "MISSING"
            elif copied == expected:
                mark = "OK"
            else:
                mark = "PARTIAL"
            suffix = f"（annotations.jsonl {ann_count} 帧）" if ann_count is not None else ""
            print(f"  [{mark}] {cam_id}: img1/ 写入 {copied}/{expected} 帧{suffix}")

        if total_copied == 0:
            status, reason = "failed", "渲染未产生任何可用的 RGB 帧"
        elif all(e["ok"] for e in per_camera.values()):
            status, reason = "success", None
        else:
            status, reason = "partial", "部分 camera 的渲染帧数与预期不符（可能缺帧）"
        self._finalize(failed=(status == "failed"), status=status,
                       per_camera=per_camera, total_copied=total_copied, reason=reason)

    def _check_annotation_frame_count(self, cam_out: Path) -> Optional[int]:
        """轻量校验：annotations.jsonl 行数 vs img1/ PNG 数。

        一致返回 None；不一致返回 annotations.jsonl 帧数；无标注文件返回 None。
        纯文件检查，不导入 P1 代码（保持 UE/P1 运行时隔离）。
        """
        ann_path = cam_out / "annotations.jsonl"
        img1_dir = cam_out / "img1"
        if not ann_path.exists() or not img1_dir.exists():
            return None
        try:
            ann_count = sum(1 for _ in open(ann_path, encoding="utf-8"))
        except OSError:
            return None
        img_count = len(list(img1_dir.glob("*.png")))
        return ann_count if ann_count != img_count else None

    def _finalize(self, failed, status=None, per_camera=None,
                  total_copied=0, reason=None):
        """写完成标记 render_summary.json，并释放模块级引用（只执行一次）。"""
        global _ACTIVE_RENDER
        if self.finished:
            return
        self.finished = True
        _ACTIVE_RENDER = None

        # 注销 watchdog（若有）
        if self._watch_handle is not None:
            try:
                import unreal
                unreal.unregister_slate_post_tick_callback(self._watch_handle)
            except Exception:
                pass
            self._watch_handle = None

        status = status or ("failed" if failed else "success")
        summary = {
            "episode_id": self.episode_id,
            "status": status,
            "reason": reason,
            "total_img1_frames": total_copied,
            "cameras": per_camera or {},
            "mrq_errors": self.error_messages or None,
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "MRQ 渲染完成回调写入。完整标注校验请在 P1 运行："
                    "uv run grf-ue validate-annotations <output_dir>",
        }
        write_json_atomic(self.summary_path(), summary)

        print(f"\n=== MRQ 渲染{'失败' if failed else '完成'}: {status} ===")
        print(f"  完成标记: {self.summary_path()}")
        if failed:
            print(f"  原因: {reason}")
        print(f"  后续步骤（P1）：uv run grf-ue validate-annotations {self.output_dir}")


def render_sequences(
    sequences_cfg,
    annotation_cfg: dict,
    sequence_package_path: str,
    episode_dir: Path,
    output_dir: Path,
) -> None:
    """异步用 MRQ 渲染所有 Sequence 的 RGB 帧，提交后立即返回。

    sequences_cfg: ue_import_config.json 的 sequences 列表（每个含 name/camera_actor）。
    annotation_cfg: annotation_export 配置（含 render_rgb 段与分辨率）。

    渲染完成后（MRQ finished 回调，或 slate post-tick watchdog 兜底）自动：
      - 复制 RGB 帧到每个 camera 的 img1/（按帧同步契约取帧）。
      - 写 <output_dir>/<episode_id>/render_summary.json 完成标记。
    本函数不阻塞编辑器主线程。
    """
    import unreal

    global _ACTIVE_RENDER
    if _ACTIVE_RENDER is not None:
        print("WARNING: 已有 MRQ 渲染正在进行，忽略本次请求（等待上一批完成）")
        return

    render_cfg = annotation_cfg.get("render_rgb") or {}
    if not render_cfg.get("enabled", False):
        print("render_rgb.enabled = false，跳过渲染")
        return

    image_width = int(
        render_cfg.get("output_resolution_x") or annotation_cfg.get("image_width", 1920)
    )
    image_height = int(
        render_cfg.get("output_resolution_y") or annotation_cfg.get("image_height", 1080)
    )
    frame_rate = int(
        render_cfg.get("frame_rate") or annotation_cfg.get("playback_fps") or 30
    )
    file_name_format = render_cfg.get("file_name_format", "{frame_number}")
    zero_pad = int(render_cfg.get("zero_pad_frame_numbers", 6))

    meta, frames = load_episode(episode_dir)
    episode_id = meta.get("episode_id") or episode_dir.name
    source_step = float(meta["timing"].get("source_step_seconds", 0.1))
    keep_indices = select_rendered_frame_indices(len(frames), source_step, frame_rate)

    if not sequences_cfg:
        print("WARNING: sequences 为空，无可渲染的 Sequence")
        return

    # 收集要渲染的 Sequence（资产必须已存在）
    jobs = []
    for seq_entry in sequences_cfg:
        seq_name = seq_entry.get("name")
        cam_id = seq_entry.get("camera_actor") or seq_name
        if not seq_name:
            continue
        seq_asset_path = f"{sequence_package_path}/{seq_name}"
        if not unreal.EditorAssetLibrary.does_asset_exist(seq_asset_path):
            print(f"  WARNING: Sequence 不存在，跳过: {seq_asset_path}（请先 --mode sequence 创建）")
            continue
        cam_out = Path(output_dir) / episode_id / cam_id
        render_dir = cam_out / "render"
        img1_dir = cam_out / "img1"
        ensure_dir(render_dir)
        _clear_dir(render_dir)  # 清空上一次渲染残留，避免旧帧混入
        jobs.append({
            "name": seq_name,
            "cam_id": cam_id,
            "seq_asset_path": seq_asset_path,
            "cam_out": cam_out,
            "render_dir": render_dir,
            "img1_dir": img1_dir,
        })
        print(f"  待渲染: {seq_name} -> {img1_dir}")

    if not jobs:
        print("WARNING: 没有任何可渲染的 Sequence")
        return

    map_path = _get_current_map_path()
    subsystem = _mrq_subsystem()
    queue = _mrq_get_queue(subsystem)
    if queue is None:
        _list_mrq_classes()
        raise RuntimeError(
            "无法获取 MRQ 队列（MoviePipelineQueueSubsystem.get_queue() 返回 None）。"
            "已打印 [MRQ 诊断]，请把输出贴回。"
        )
    executor = _mrq_executor()

    pipeline = _AsyncRenderPipeline(jobs, episode_id, output_dir, keep_indices)
    _ACTIVE_RENDER = pipeline  # 模块级引用：脚本返回后防止 pipeline 被 GC
    pipeline.start(
        subsystem, queue, executor, map_path,
        image_width, image_height, frame_rate,
        file_name_format, zero_pad,
    )


def force_finalize_render() -> bool:
    """手动触发当前活跃渲染管线的收尾（复制 img1/ + 写 render_summary.json）。

    极端情况下（finished delegate 与 watchdog 均未触发），在 UE 控制台执行：
        import render_episode
        render_episode.force_finalize_render()
    """
    global _ACTIVE_RENDER
    if _ACTIVE_RENDER is None:
        print("WARNING: 没有活跃的 MRQ 渲染管线（render_episode._ACTIVE_RENDER 为 None）")
        return False
    _ACTIVE_RENDER._copy_and_finalize()
    return True


def _mrq_subsystem():
    """获取 MRQ 子系统。

    UE 5.8 起类名由 MovieRenderQueueSubsystem 改为 MoviePipelineQueueSubsystem。
    """
    import unreal

    for cls_name in ("MoviePipelineQueueSubsystem", "MovieRenderQueueSubsystem"):
        cls = getattr(unreal, cls_name, None)
        if cls is not None:
            try:
                return unreal.get_editor_subsystem(cls)
            except Exception:
                continue
    _list_mrq_classes()
    raise RuntimeError(
        "找不到 MRQ 子系统（MoviePipelineQueueSubsystem / MovieRenderQueueSubsystem）。"
        "已打印可用 MRQ 类，请把上面 [MRQ 诊断] 输出贴回。"
    )


def _mrq_executor():
    """创建 MRQ 执行器（PIE 优先）。"""
    import unreal

    for cls_name in ("MoviePipelinePIEExecutor", "MoviePipelineEditorExecutor"):
        cls = getattr(unreal, cls_name, None)
        if cls is None:
            continue
        try:
            return cls()
        except Exception:
            continue
    _list_mrq_classes()
    raise RuntimeError(
        "无法创建 MRQ 执行器（MoviePipelinePIEExecutor / MoviePipelineEditorExecutor "
        "均不可用）。已打印可用 MRQ 类，请把上面 [MRQ 诊断] 输出贴回。"
    )
