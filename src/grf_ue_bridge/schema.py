"""GRF-UE episode 格式的数据模型。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EntityDefinition(BaseModel):
    """单个实体（球员或球）在 episode 中的定义。"""

    id: str = Field(..., description="实体标识，例如 'L0'、'R3'、'BALL'")
    team: Optional[str] = Field(None, description="队伍标识：'left'、'right'，球为 null")
    source_index: Optional[int] = Field(None, description="在源观测数组中的下标")
    role: Optional[str] = Field(None, description="球员角色名（如已知），例如 'goalkeeper'")
    is_goalkeeper: bool = Field(False, description="该实体是否为守门员")

    @classmethod
    def from_grf_role(cls, role_id: int, prefix: str, index: int) -> "EntityDefinition":
        """根据 GRF 角色整数创建 EntityDefinition。"""
        role_map = {
            0: "goalkeeper",
            1: "center_back",
            2: "left_back",
            3: "right_back",
            4: "defensive_midfielder",
            5: "central_midfielder",
            6: "left_midfielder",
            7: "right_midfielder",
            8: "attacking_midfielder",
            9: "center_forward",
            10: "left_wing",
            11: "right_wing",
        }
        role_name = role_map.get(role_id, None)
        return cls(
            id=f"{prefix}{index}",
            team="left" if prefix == "L" else "right",
            source_index=index,
            role=role_name,
            is_goalkeeper=(role_id == 0),
        )


class SourceInfo(BaseModel):
    """关于源环境与代码的信息。"""

    environment: str = "google_research_football"
    scenario: str = ""
    control_mode: str = ""
    seed: int = 42
    football_commit: str = ""
    grf_marl_commit: str = ""
    game_duration: Optional[int] = Field(
        None, description="场景覆盖的回合引擎帧数（None = 场景默认）"
    )
    left_team_difficulty: Optional[float] = Field(
        None, description="场景覆盖的左队 AI 难度（None = 场景默认）"
    )
    right_team_difficulty: Optional[float] = Field(
        None, description="场景覆盖的右队 AI 难度（None = 场景默认）"
    )


class RandomnessInfo(BaseModel):
    """episode 的随机种子体系（root seed 派生出的全部命名空间子 seed）。

    只有 `grf_game_engine_seed` 真正传入 GRF 引擎；python/numpy 用于宿主进程
    随机状态；ue_visual_seed 为未来 UE 视觉随机化预留（本轮仅记录）。
    """

    policy: str = Field("futsalmot_seed_v1", description="seed 派生算法版本标识")
    root_seed: int = Field(..., description="用户提供的根 seed，应等于 source.seed")
    grf_game_engine_seed: int = Field(
        ..., description="真正传入 GRF 的 game_engine_random_seed"
    )
    python_seed: int = Field(..., description="Python 标准库 random seed")
    numpy_seed: int = Field(..., description="NumPy random seed")
    ue_visual_seed: int = Field(..., description="预留：未来 UE 视觉随机化 seed")


class TimingInfo(BaseModel):
    """episode 的时序信息。"""

    source_step_seconds: float = Field(..., description="单个 GRF 步的时长（秒）")
    playback_fps: int = Field(30, description="目标回放帧率")
    num_steps: int = Field(300, description="本 episode 的步数")


class FieldInfo(BaseModel):
    """场地尺寸与坐标系。"""

    length_m: float = Field(40.0, description="场地长度（米）")
    width_m: float = Field(20.0, description="场地宽度（米）")
    origin: str = Field("center", description="坐标原点位置")
    x_range_m: List[float] = Field(default_factory=lambda: [-20.0, 20.0])
    y_range_m: List[float] = Field(default_factory=lambda: [-10.0, 10.0])


class CoordinateTransformNote(BaseModel):
    """所应用的坐标变换的说明。"""

    grf_normalized_range: str = "场地局部坐标的 x/y 为 [-1, 1]；z 为引擎单位（Z_FIELD_SCALE=1）"
    conversion: str = (
        "meter_x = grf_x * half_field_length；"
        "meter_y = grf_y * half_field_width；"
        "meter_z = grf_z（原样透传，Z_FIELD_SCALE=1）"
    )
    player_z: str = "固定为 0"
    ball_z_note: str = (
        "GRF 的球 z 直接透传，因为 Z_FIELD_SCALE=1 且引擎 z 已近似为米。"
        "GRF 观测中的源球 z 值 [x, y, z] 会记录在 meta.coordinate_transform 中供参考。"
    )


class Meta(BaseModel):
    """GRF-UE episode 的顶层元数据。"""

    schema_: str = Field("grf_ue_episode", alias="schema")
    version: int = 1
    episode_id: str = ""
    source: SourceInfo = Field(default_factory=SourceInfo)
    randomness: Optional[RandomnessInfo] = Field(
        None,
        description=(
            "随机种子体系（policy/root/子 seed）。旧 episode 可能缺失，"
            "视为 legacy seed metadata，validator 不将其伪装为已验证可复现"
        ),
    )
    timing: TimingInfo
    field: FieldInfo = Field(default_factory=FieldInfo)
    entities: List[EntityDefinition] = Field(default_factory=list)
    coordinate_transform: CoordinateTransformNote = Field(
        default_factory=CoordinateTransformNote
    )

    model_config = {"populate_by_name": True}


class BallFrame(BaseModel):
    """单帧中球的状态。"""

    position_m: List[float] = Field(..., description="球的米坐标 [x, y, z]")
    source_grf_position: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="原始 GRF 观测位置 [grf_x, grf_y, grf_z]，供参考",
    )
    velocity_mps: Optional[List[float]] = Field(
        None,
        description=(
            "球的速度 [vx, vy, vz]，单位 m/s，由 GRF ball_direction（每步位移，"
            "x/y 为归一化、z 为米）换算得到。旧 episode 可能缺失（None），"
            "此时下游用位置差分估算。"
        ),
    )


class PlayerFrame(BaseModel):
    """单帧中球员的状态。"""

    id: str = Field(..., description="实体标识，与 EntityDefinition.id 对应")
    position_m: List[float] = Field(..., description="球员的米坐标 [x, y, z]")
    velocity_mps: Optional[List[float]] = Field(
        None,
        description=(
            "球员水平速度 [vx, vy]，单位 m/s，由 GRF *_team_direction（每步位移，"
            "归一化坐标）除以步长换算。旧 episode 可能缺失（None），下游用位置差分估算。"
        ),
    )
    speed_mps: Optional[float] = Field(
        None,
        description="速率（m/s）= |velocity_mps|。为简化 UE Motion Layer 直接读取而冗余存储。",
    )
    movement_heading_deg: Optional[float] = Field(
        None,
        description="运动朝向（度，-180~180）= atan2(vy, vx)；静止时可能为 None。",
    )
    active: Optional[bool] = Field(
        None,
        description="该球员是否 active（来自 GRF *_team_active）。",
    )
    has_ball: Optional[bool] = Field(
        None,
        description="该球员是否持球（由 ball_owned_team / ball_owned_player 推导）。",
    )


class Frame(BaseModel):
    """episode 数据中的单个帧。"""

    step: int = Field(..., ge=0)
    time_seconds: float = Field(..., ge=0.0)
    score: List[int] = Field(default_factory=lambda: [0, 0])
    ball: BallFrame
    players: List[PlayerFrame] = Field(..., min_length=10, max_length=10)
    ball_owned_team: Optional[int] = Field(
        None,
        description=(
            "持球队：-1 无、0 左队、1 右队（来自 GRF ball_owned_team）。"
            "旧 episode 可能缺失（None）。"
        ),
    )
    ball_owned_player: Optional[int] = Field(
        None,
        description=(
            "持球球员在队内下标（0~4），-1 无（来自 GRF ball_owned_player）。"
            "旧 episode 可能缺失（None）。"
        ),
    )
    game_mode: Optional[int] = Field(
        None,
        description="GRF game_mode（0=normal、1=kickoff、…），旧 episode 可能缺失（None）。",
    )


def create_ball_entity() -> EntityDefinition:
    """创建球的实体定义。"""
    return EntityDefinition(
        id="BALL",
        team=None,
        source_index=None,
        role=None,
        is_goalkeeper=False,
    )
