"""配置数据模型：本地配置、数据集 task、resolved task、profile 结构。

所有模型带 schema/version；task 与 profile 引用分离，不复制完整内容。
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .paths import (
    EXPORT_PROFILE_SCHEMA,
    LOCAL_CONFIG_SCHEMA,
    RESOLVED_TASK_SCHEMA,
    TASK_SCHEMA,
    UE_PROFILE_SCHEMA,
)

VALID_POSTPROCESS_FORMATS = ("json", "mot", "yolo-det", "yolo-seg")


# ── 本地机器配置（.futsalmot.local.json）───────────────────────────────

class LocalConfig(BaseModel):
    """机器相关路径配置。只放路径，不放任务内容（seed/相机/分辨率等）。"""

    schema_: Literal["futsalmot_local_config"] = Field(
        LOCAL_CONFIG_SCHEMA, alias="schema"
    )
    version: int = 1
    repo_root: Optional[str] = Field(
        None, description="仓库根目录（缺省自动探测）"
    )
    ue_project_root: Optional[str] = Field(
        None, description="Unreal Engine 项目根目录（含 .uproject）"
    )
    dataset_root: Optional[str] = Field(
        None, description="数据集输出根目录"
    )
    cache_dir: Optional[str] = Field(None, description="可选缓存目录")
    log_dir: Optional[str] = Field(None, description="可选日志目录")

    model_config = {"populate_by_name": True}


# ── 数据集 Task ─────────────────────────────────────────────────────────

class PostprocessTaskConfig(BaseModel):
    """后处理参数（属于 task，不属于机器配置）。"""

    include_ball: bool = True
    workers: int = Field(4, ge=1, le=32, description="并行 worker 数")
    chunk_size: int = Field(50, ge=0, description="帧分块大小（0=自动）")
    png_compress_level: int = Field(1, ge=0, le=9)
    formats: List[str] = Field(
        default_factory=lambda: ["json", "mot", "yolo-det", "yolo-seg"]
    )
    clean_stale: bool = True
    validation_level: Literal["full", "quick"] = "full"

    def validate_formats(self) -> None:
        for f in self.formats:
            if f not in VALID_POSTPROCESS_FORMATS:
                raise ValueError(
                    f"不支持的 postprocess 格式: {f!r}（可选 "
                    f"{'/'.join(VALID_POSTPROCESS_FORMATS)}）"
                )


class AuditTaskConfig(BaseModel):
    """审计预期（属于 task）。"""

    expected_cameras: int = Field(4, ge=1)
    expected_frames_per_camera: int = Field(300, ge=1)


class TaskPathOverrides(BaseModel):
    """task 内相对路径覆盖（缺省按 episode_name 推导）。"""

    trajectory_output: Optional[str] = Field(
        None, description="轨迹输出相对仓库根（默认 outputs/<episode_name>）"
    )
    dataset_output: Optional[str] = Field(
        None, description="数据集输出相对 dataset_root（默认 <episode_name>）"
    )


class DatasetTaskConfig(BaseModel):
    """一个数据集任务的完整描述。task 只引用 profile，不复制其内容。"""

    schema_: Literal["futsalmot_dataset_task"] = Field(TASK_SCHEMA, alias="schema")
    version: int = 1

    task_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]+$", description="任务唯一 ID")
    episode_name: str = Field(
        ..., pattern=r"^[A-Za-z0-9_]+$", description="episode 名（目录名，无路径分隔符）"
    )

    export_profile: str = Field(
        ..., description="导出 profile 相对路径（相对 task 文件所在目录）"
    )
    ue_profile: str = Field(
        ..., description="UE profile 相对路径（相对 task 文件所在目录）"
    )

    seed: Optional[int] = Field(None, description="root seed（覆盖 export profile）")

    # 可选机器路径：填了就以 task 为准（单一 config 用法），否则回退
    # 环境变量 > .futsalmot.local.json > 默认。
    repo_root: Optional[str] = Field(None, description="仓库根目录（可选）")
    ue_project_root: Optional[str] = Field(None, description="UE 项目根目录（可选）")
    dataset_root: Optional[str] = Field(
        None,
        description=(
            "数据集输出根目录（可选）。默认所有产出（轨迹 meta/frames/provenance + "
            "相机数据）都落于 <dataset_root>/<episode_name>/ 下自包含"
        ),
    )

    paths: TaskPathOverrides = Field(default_factory=TaskPathOverrides)
    postprocess: PostprocessTaskConfig = Field(default_factory=PostprocessTaskConfig)
    audit: AuditTaskConfig = Field(default_factory=AuditTaskConfig)

    model_config = {"populate_by_name": True}


# ── Profile 模型 ────────────────────────────────────────────────────────

class ExportProfile(BaseModel):
    """导出 profile：与 ExportConfig 字段一致的容器，用于解析校验。"""

    schema_: Literal["futsalmot_export_profile"] = Field(
        EXPORT_PROFILE_SCHEMA, alias="schema"
    )
    version: int = 1
    # 允许任意导出配置键（scenario/seed/num_steps/...），用 extra 保留
    model_config = {"populate_by_name": True, "extra": "allow"}


class UeProfile(BaseModel):
    """UE profile：episode 无关的相机/Sequence/渲染/标注参数。

    不含 episode、dataset、output_dir 等本机路径——这些由 resolver 从
    task + 本地配置填充。actor_mapping 用相对仓库根路径。
    """

    schema_: Literal["futsalmot_ue_profile"] = Field(UE_PROFILE_SCHEMA, alias="schema")
    version: int = 1

    actor_mapping: str = Field("ue/actor_mapping.example.json", description="相对仓库根")
    sequence_package_path: str = "/Game/FutsalMOT/Sequences"
    sequences: List[Dict] = Field(default_factory=list, description="相机/Sequence 列表")
    replace_existing: bool = True
    ball_rolling: Dict = Field(default_factory=dict)
    annotation_export: Dict = Field(default_factory=dict, description="episode 无关标注/渲染配置")

    model_config = {"populate_by_name": True, "extra": "allow"}


# ── Resolved Task（运行时，可含绝对路径）───────────────────────────────

class ResolvedTask(BaseModel):
    """解析后的任务：把 task + profile + 本地配置合并为运行时绝对路径。

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

    export_profile: Dict = Field(default_factory=dict, description="完整解析后的导出 profile")
    ue_profile: Dict = Field(default_factory=dict, description="完整解析后的 UE profile")
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
