"""配置数据模型：数据集 task（单 config）、resolved task。

任务 = 一个自包含 JSON（导出 + UE + 机器路径内联，本地路径直接入库）；
resolved task 是 P1↔UE 共享的运行时契约。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .paths import RESOLVED_TASK_SCHEMA, TASK_SCHEMA

VALID_POSTPROCESS_FORMATS = ("json", "mot", "yolo-det", "yolo-seg")


# ── YOLO Pose（人体关键点）后处理配置 ────────────────────────────────────

class YoloPoseTaskConfig(BaseModel):
    """YOLO Pose 标注导出参数（属于 task 的 postprocess.yolo_pose）。

    单一开关同时控制 UE 侧关键点导出（resolved task 经 run_task.py 读到）与
    P1 侧标签生成（postprocess 阶段）。默认关闭，不影响原有 pipeline。
    """

    enabled: bool = Field(
        False,
        description=(
            "是否导出 YOLO Pose（COCO 17 点）。开启后 UE 导出 pose_keypoints.jsonl，"
            "postprocess 生成 labels_pose/ + futsal_pose.yaml"
        ),
    )
    visibility_neighborhood_radius: int = Field(
        2,
        ge=0,
        le=8,
        description="Instance-ID Mask 邻域判定半径（像素），用于 keypoint 遮挡判定",
    )
    write_dataset_yaml: bool = Field(
        True,
        description="是否在 episode 根生成 yolo_pose/ 可训练暂存目录与 futsal_pose.yaml",
    )
    occlusion_trace: bool = Field(
        False,
        description=(
            "UE 侧是否对每个关键点做遮挡 trace（自遮挡 / 非 mask 几何）。"
            "默认关闭：mask 判定已是主要且准确的遮挡信号，trace 是边缘增强，"
            "且 61 万次射线会明显拖慢渲染前导出（大任务建议保持关闭）"
        ),
    )
    trace_tolerance_cm: float = Field(
        20.0,
        gt=0.0,
        le=200.0,
        description="UE 遮挡 trace 容差（cm）：命中距离 < 关键点距离 - 容差 即判遮挡",
    )
    bone_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="UE bone 名覆盖：{COCO 关键点名: UE bone 名}（用于骨骼命名不同的资产）",
    )
    head_offsets_cm: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="脸部五点相对 head 骨骼的局部偏移覆盖：{脸部 COCO 名: [x, y, z] cm}",
    )

    def validate_pose(self) -> None:
        """跨字段一致性：bone_overrides / head_offsets_cm 键名合法。"""
        # 延迟 import：config 层早期加载，避免依赖 ue/ 路径；运行时补 sys.path
        import sys

        _ue_dir = Path(__file__).resolve().parent.parent.parent.parent / "ue"
        if str(_ue_dir) not in sys.path:
            sys.path.insert(0, str(_ue_dir))
        from pose_bones import COCO_KEYPOINT_NAMES, FACE_KEYPOINT_NAMES

        valid = set(COCO_KEYPOINT_NAMES)
        for k in self.bone_overrides:
            if k not in valid:
                raise ValueError(
                    f"yolo_pose.bone_overrides 键 {k!r} 非法（只接受 COCO 关键点名）"
                )
        for k, off in self.head_offsets_cm.items():
            if k not in FACE_KEYPOINT_NAMES:
                raise ValueError(
                    f"yolo_pose.head_offsets_cm 键 {k!r} 非法（只接受脸部五点）"
                )
            if not isinstance(off, list) or len(off) != 3:
                raise ValueError(f"yolo_pose.head_offsets_cm[{k}] 须为 [x, y, z] cm")


# ── 数据集 Task（单 config：导出 + UE + 机器路径内联）──────────────────

class PostprocessTaskConfig(BaseModel):
    """后处理参数（属于 task）。"""

    include_ball: bool = True
    workers: int = Field(4, ge=1, le=32, description="并行 worker 数")
    chunk_size: int = Field(50, ge=0, description="帧分块大小（0=自动）")
    png_compress_level: int = Field(1, ge=0, le=9)
    formats: List[str] = Field(
        default_factory=lambda: ["json", "mot", "yolo-det", "yolo-seg"]
    )
    clean_stale: bool = True
    validation_level: Literal["full", "quick"] = "full"
    yolo_pose: YoloPoseTaskConfig = Field(
        default_factory=YoloPoseTaskConfig,
        description="YOLO Pose 标注导出参数（默认关闭）",
    )
    debug: "DebugTaskConfig" = Field(
        default_factory=lambda: DebugTaskConfig(),
        description="debug 可视化：全量渲染 bbox/彩色 mask/pose 图集并拼接视频（默认关闭）",
    )

    def validate_formats(self) -> None:
        for f in self.formats:
            if f not in VALID_POSTPROCESS_FORMATS:
                raise ValueError(
                    f"不支持的 postprocess 格式: {f!r}（可选 "
                    f"{'/'.join(VALID_POSTPROCESS_FORMATS)}）"
                )
        self.yolo_pose.validate_pose()


class DebugTaskConfig(BaseModel):
    """debug 可视化参数（属于 task 的 postprocess.debug）。

    enabled=true 时 `task postprocess` 全量渲染三套 debug 图集
    （bbox overlay / 彩色 mask / pose 关节点）并自动拼接为三个 mp4。
    """

    enabled: bool = Field(
        False,
        description=(
            "是否全量渲染 debug 可视化：bbox overlay / 彩色 mask / pose 关节点 三套图集，"
            "并把三套图集各拼接为 mp4"
        ),
    )
    include_ball: bool = Field(False, description="bbox overlay 是否绘制球")
    make_videos: bool = Field(True, description="渲染图集后自动拼接视频（bbox/mask/pose 各一个）")
    video_fps: Optional[int] = Field(
        None, ge=1, description="debug 视频帧率（None = 读 seqinfo.ini frameRate，缺省 30）"
    )
    pose_dot_radius: int = Field(3, ge=1, le=30, description="pose 关键点半径（像素，远景默认小一点避免遮挡）")
    pose_edge_width: int = Field(3, ge=1, le=20, description="pose 骨架连线宽度（像素）")


class AuditTaskConfig(BaseModel):
    """审计预期（属于 task）。"""

    expected_cameras: int = Field(4, ge=1)
    expected_frames_per_camera: int = Field(300, ge=1)


class UeProfile(BaseModel):
    """UE 相机/Sequence/渲染参数（内联在 task 的 ue 块，非独立文件）。"""

    actor_mapping: str = Field("ue/actor_mapping.example.json", description="相对仓库根")
    sequence_package_path: str = "/Game/FutsalMOT/Sequences"
    sequences: List[Dict] = Field(default_factory=list, description="相机/Sequence 列表")
    replace_existing: bool = True
    ball_rolling: Dict = Field(default_factory=dict)
    annotation_export: Dict = Field(default_factory=dict, description="标注/渲染配置")

    model_config = {"populate_by_name": True, "extra": "allow"}


class DatasetTaskConfig(BaseModel):
    """一个数据集任务的完整描述（单 config）。

    导出参数（export）+ UE 参数（ue）+ 机器路径（dataset_root / ue_project_root）
    都内联在一个文件里，直接入库；repo_root 自动探测。产出统一落
    <dataset_root>/<episode_name>/ 下自包含。
    """

    schema_: Literal["futsalmot_dataset_task"] = Field(TASK_SCHEMA, alias="schema")
    version: int = 2

    task_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]+$", description="任务唯一 ID")
    episode_name: str = Field(
        ..., pattern=r"^[A-Za-z0-9_]+$", description="episode 名（目录名，无路径分隔符）"
    )

    dataset_root: str = Field(..., description="数据集输出根目录（绝对路径，入库）")
    ue_project_root: str = Field(
        ..., description="Unreal Engine 项目根目录（含 .uproject，绝对路径，入库）"
    )

    export: "ExportConfig" = Field(..., description="导出参数（GRF 侧）")
    ue: UeProfile = Field(..., description="UE 相机/Sequence/渲染参数")

    postprocess: PostprocessTaskConfig = Field(default_factory=PostprocessTaskConfig)
    audit: AuditTaskConfig = Field(default_factory=AuditTaskConfig)

    model_config = {"populate_by_name": True}


# ── Resolved Task（运行时，含绝对路径）──────────────────────────────────

class ResolvedTask(BaseModel):
    """解析后的任务：把单 config 归一化为运行时绝对路径。

    本文件允许含绝对路径，因为它位于被 gitignore 的 `.futsalmot/runtime/`。
    """

    schema_: Literal["futsalmot_resolved_task"] = Field(
        RESOLVED_TASK_SCHEMA, alias="schema"
    )
    version: int = 1

    task_id: str
    episode_name: str
    source_task_file: str = Field(..., description="运行时 source task 绝对路径")

    repo_root: str
    ue_project_root: str
    dataset_root: str

    trajectory_output: str = Field(..., description="轨迹输出目录（绝对）")
    dataset_episode_dir: str = Field(..., description="数据集 episode 目录（绝对）")

    export_profile: Dict = Field(default_factory=dict, description="完整解析后的导出参数")
    ue_profile: Dict = Field(default_factory=dict, description="完整解析后的 UE 参数")
    actor_mapping: str = Field("", description="actor mapping 绝对路径")

    postprocess: Dict = Field(default_factory=dict)
    audit: Dict = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# ── 原有 ExportConfig（从 grf_ue_bridge/config.py 迁入，保持兼容）────────

class ExportConfig(BaseModel):
    """单个 GRF episode 导出的配置。"""

    scenario: str = Field(..., description="GRF 场景名，例如 '5_vs_5'")
    seed: int = Field(42, description="随机种子，用于可复现")
    num_steps: int = Field(300, description="要运行的步数（GRF 10 FPS 仿真）")
    target_fps: int = Field(
        0,
        description=(
            "目标导出帧率：0 或 10 = 保持 GRF 原生 10fps（不插值，每 GRF 步一条标注）；"
            "30 = 把 10fps 位置线性插值到 30fps 导出（渲 900 标 900，1:1）。须为 10 的倍数"
        ),
    )
    trajectory_time_scale: float = Field(
        1.0,
        ge=0.1,
        description=(
            "轨迹时间缩放（时间轴重采样）：source_time = dataset_time × time_scale。"
            "> 1 时按真实 GRF 轨迹做时间型 Hermite 重采样，dataset 位置/速度一致地"
            "加快（速度 = source 速度 × time_scale），score/ownership/game_mode 等"
            "离散状态按 source_time hold/nearest。1.0 = 不缩放（普通 10fps→目标 fps 插值）。"
        ),
    )
    playback_fps: int = Field(30, description="Unreal Engine 目标回放帧率")
    field_length_m: float = Field(40.0, description="UE 场地长度（米）")
    field_width_m: float = Field(20.0, description="UE 场地宽度（米）")
    render: bool = Field(False, description="是否渲染游戏画面")
    write_video: bool = Field(False, description="是否录制视频")
    dump_full_raw_observation: bool = Field(
        False, description="是否导出完整原始观测，用于调试"
    )
    number_of_left_players_agent_controls: int = Field(
        0, description="由 agent 控制的左队球员数（0 = 全部内置 AI）"
    )
    number_of_right_players_agent_controls: int = Field(
        0, description="由 agent 控制的右队球员数（0 = 全部内置 AI）"
    )
    game_duration: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "单个回合的引擎帧数（场景 game_duration，5_vs_5 默认 3000）。"
            "设大避免采集步数耗尽回合→重置瞬移；None = 用场景默认"
        ),
    )
    left_team_difficulty: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "左队 AI 难度（0~1，5_vs_5 默认 0.05）。调高→球更少出界、"
            "减少 set-piece 重摆阵型瞬移；None = 用场景默认"
        ),
    )
    right_team_difficulty: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="右队 AI 难度（0~1，5_vs_5 默认 0.05）；None = 用场景默认",
    )

    model_config = {"populate_by_name": True}
