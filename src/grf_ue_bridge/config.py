"""GRF episode 导出的配置模型。"""

from pydantic import BaseModel, Field


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
