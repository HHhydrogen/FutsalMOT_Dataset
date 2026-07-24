"""Configuration model for GRF episode export."""

from pydantic import BaseModel, Field


class ExportConfig(BaseModel):
    """Configuration for a single GRF episode export."""

    scenario: str = Field(..., description="GRF scenario name, e.g. '5_vs_5'")
    seed: int = Field(42, description="Random seed for reproducibility")
    num_steps: int = Field(300, description="Number of steps to run")
    playback_fps: int = Field(30, description="Target Unreal Engine playback FPS")
    field_length_m: float = Field(40.0, description="UE field length in meters")
    field_width_m: float = Field(20.0, description="UE field width in meters")
    render: bool = Field(False, description="Whether to render game frames")
    write_video: bool = Field(False, description="Whether to write video dumps")
    dump_full_raw_observation: bool = Field(
        False, description="Whether to dump full raw observations for debugging"
    )
    number_of_left_players_agent_controls: int = Field(
        0, description="Number of left players controlled by agent (0 = built-in AI)"
    )
    number_of_right_players_agent_controls: int = Field(
        0, description="Number of right players controlled by agent (0 = built-in AI)"
    )
