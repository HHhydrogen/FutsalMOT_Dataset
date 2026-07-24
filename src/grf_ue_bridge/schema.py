"""Data models for the GRF-UE episode format."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EntityDefinition(BaseModel):
    """Definition of a single entity (player or ball) in the episode."""

    id: str = Field(..., description="Entity identifier, e.g. 'L0', 'R3', 'BALL'")
    team: Optional[str] = Field(None, description="Team identifier: 'left', 'right', or null for ball")
    source_index: Optional[int] = Field(None, description="Index within the source observation array")
    role: Optional[str] = Field(None, description="Player role name if known, e.g. 'goalkeeper'")
    is_goalkeeper: bool = Field(False, description="Whether this entity is a goalkeeper")

    @classmethod
    def from_grf_role(cls, role_id: int, prefix: str, index: int) -> "EntityDefinition":
        """Create an EntityDefinition from a GRF role integer."""
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
    """Information about the source environment and code."""

    environment: str = "google_research_football"
    scenario: str = ""
    control_mode: str = ""
    seed: int = 42
    football_commit: str = ""
    grf_marl_commit: str = ""


class TimingInfo(BaseModel):
    """Timing information for the episode."""

    source_step_seconds: float = Field(..., description="Duration of one GRF step in seconds")
    playback_fps: int = Field(30, description="Target playback FPS")
    num_steps: int = Field(300, description="Number of steps in this episode")


class FieldInfo(BaseModel):
    """Field dimensions and coordinate system."""

    length_m: float = Field(40.0, description="Field length in meters")
    width_m: float = Field(20.0, description="Field width in meters")
    origin: str = Field("center", description="Coordinate origin location")
    x_range_m: List[float] = Field(default_factory=lambda: [-20.0, 20.0])
    y_range_m: List[float] = Field(default_factory=lambda: [-10.0, 10.0])


class CoordinateTransformNote(BaseModel):
    """Documentation of the coordinate transform applied."""

    grf_normalized_range: str = "[-1, 1] for pitch-local x/y; z is engine units (Z_FIELD_SCALE=1)"
    conversion: str = (
        "meter_x = grf_x * half_field_length; "
        "meter_y = grf_y * half_field_width; "
        "meter_z = grf_z (passed through, Z_FIELD_SCALE=1)"
    )
    player_z: str = "fixed to 0"
    ball_z_note: str = (
        "GRF ball_z is passed through directly because Z_FIELD_SCALE=1 "
        "and engine Z is already in approximate meters. "
        "Source grf_ball_z values: [x, y, z] from GRF observation are "
        "recorded in meta.coordinate_transform for reference."
    )


class Meta(BaseModel):
    """Top-level metadata for a GRF-UE episode."""

    schema_: str = Field("grf_ue_episode", alias="schema")
    version: int = 1
    episode_id: str = ""
    source: SourceInfo = Field(default_factory=SourceInfo)
    timing: TimingInfo
    field: FieldInfo = Field(default_factory=FieldInfo)
    entities: List[EntityDefinition] = Field(default_factory=list)
    coordinate_transform: CoordinateTransformNote = Field(
        default_factory=CoordinateTransformNote
    )

    model_config = {"populate_by_name": True}


class BallFrame(BaseModel):
    """Ball state at a single frame."""

    position_m: List[float] = Field(..., description="Ball position in meters [x, y, z]")
    source_grf_position: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Raw GRF observation position [grf_x, grf_y, grf_z] for reference",
    )


class PlayerFrame(BaseModel):
    """Player state at a single frame."""

    id: str = Field(..., description="Entity identifier matching EntityDefinition.id")
    position_m: List[float] = Field(..., description="Player position in meters [x, y, z]")


class Frame(BaseModel):
    """A single frame of episode data."""

    step: int = Field(..., ge=0)
    time_seconds: float = Field(..., ge=0.0)
    score: List[int] = Field(default_factory=lambda: [0, 0])
    ball: BallFrame
    players: List[PlayerFrame] = Field(..., min_length=10, max_length=10)


def create_ball_entity() -> EntityDefinition:
    """Create the ball entity definition."""
    return EntityDefinition(
        id="BALL",
        team=None,
        source_index=None,
        role=None,
        is_goalkeeper=False,
    )
