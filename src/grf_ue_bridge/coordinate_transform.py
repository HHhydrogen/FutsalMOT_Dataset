"""Coordinate transformation from GRF to UE meters.

GRF Observation Coordinate System
----------------------------------
The GRF engine normalizes x/y but not z in the observation:
  - x: [-1, 1]  (left goal line → right goal line)
       engine_x = grf_x * X_FIELD_SCALE  (X_FIELD_SCALE = 54.4)
  - y: [-1, 1]  (bottom sideline → top sideline)
       engine_y = grf_y * Y_FIELD_SCALE  (Y_FIELD_SCALE = -83.6)
  - z: ball height in engine units.
       engine_z = grf_z * Z_FIELD_SCALE  (Z_FIELD_SCALE = 1.0)
       Since Z_FIELD_SCALE = 1, the observation z IS the engine z
       directly, already in approximate meters (~0.11m for ball on ground).

See third_party/gfootball_engine/src/defines.hpp for field scales.

UE Output Coordinate System
----------------------------
  - X: [-20 m, 20 m]  (left goal line → right goal line)
  - Y: [-10 m, 10 m]  (bottom sideline → top sideline)
  - Z: meters (ball only; player Z fixed to 0)

Note: The actual GRF full-size pitch is approximately 110m × 72m in engine
units, but our export maps the normalized [-1, 1] range to the user-configured
field dimensions (default 40m × 20m). Ball Z is passed through from the engine
almost as-is because Z_FIELD_SCALE=1; the value is already in approximate
meters (e.g. a ball on the ground reads ~0.11, matching ball radius).
"""

from typing import List, Tuple

import numpy as np


class CoordinateTransform:
    """Transforms GRF normalized coordinates to UE meter coordinates."""

    def __init__(self, field_length_m: float = 40.0, field_width_m: float = 20.0):
        self._half_length = field_length_m / 2.0  # 20.0
        self._half_width = field_width_m / 2.0  # 10.0

    @property
    def half_length(self) -> float:
        return self._half_length

    @property
    def half_width(self) -> float:
        return self._half_width

    def grf_to_meter(self, grf_x: float, grf_y: float) -> Tuple[float, float]:
        """Convert GRF normalized (x, y) to UE meters."""
        mx = float(grf_x) * self._half_length
        my = float(grf_y) * self._half_width
        return mx, my

    def grf_ball_z_to_meter(self, grf_z: float) -> float:
        """Convert GRF ball Z to UE meters.

        Z_FIELD_SCALE=1 in the engine, so GRF observation z is already
        in approximate engine meters. We pass it through directly.
        For example, a ball on the ground reads ~0.11 (≈ ball radius).
        For precise ball height relative to UE field, tune in UE.
        """
        return float(grf_z)

    def transform_player_position(
        self, grf_x: float, grf_y: float
    ) -> List[float]:
        """Transform a player's GRF position to [x_m, y_m, 0]."""
        mx, my = self.grf_to_meter(grf_x, grf_y)
        return [mx, my, 0.0]

    def transform_ball_position(
        self, grf_pos: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """Transform ball position from GRF.

        Returns (position_m, source_grf) where position_m is [x_m, y_m, z_m]
        and source_grf is the original [grf_x, grf_y, grf_z] for reference.
        """
        grf_x, grf_y, grf_z = float(grf_pos[0]), float(grf_pos[1]), float(grf_pos[2])
        mx, my = self.grf_to_meter(grf_x, grf_y)
        mz = self.grf_ball_z_to_meter(grf_z)
        return [mx, my, mz], [grf_x, grf_y, grf_z]
