"""GRF 到 UE 米坐标的坐标变换。

GRF 观测坐标系
--------------
GRF 引擎对 x/y 做了归一化，但 z 未归一化：
  - x: [-1, 1]  （左球门线 → 右球门线）
       engine_x = grf_x * X_FIELD_SCALE  （X_FIELD_SCALE = 54.4）
  - y: [-1, 1]  （下边线 → 上边线）
       engine_y = grf_y * Y_FIELD_SCALE  （Y_FIELD_SCALE = -83.6）
  - z: 球的高度，单位为引擎单位。
       engine_z = grf_z * Z_FIELD_SCALE  （Z_FIELD_SCALE = 1.0）
       由于 Z_FIELD_SCALE = 1，观测中的 z 就是引擎 z，
       已经近似为米（地面上的球约为 0.11m）。

字段缩放系数参见 third_party/gfootball_engine/src/defines.hpp。

UE 输出坐标系
--------------
  - X: [-20 m, 20 m]  （左球门线 → 右球门线）
  - Y: [-10 m, 10 m]  （下边线 → 上边线）
  - Z: 米（仅球；球员 Z 固定为 0）

注意：GRF 真实全场在引擎单位下约为 110m × 72m，但我们的导出把归一化
的 [-1, 1] 范围映射到用户配置的场地尺寸（默认 40m × 20m）。球的 Z 由于
Z_FIELD_SCALE=1 几乎原样透传，数值已近似为米（例如地面上的球读取约为
0.11，正好匹配球的半径）。
"""

from typing import List, Tuple

import numpy as np


class CoordinateTransform:
    """将 GRF 归一化坐标转换为 UE 米坐标。"""

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
        """把 GRF 归一化 (x, y) 转换为 UE 米。"""
        mx = float(grf_x) * self._half_length
        my = float(grf_y) * self._half_width
        return mx, my

    def grf_ball_z_to_meter(self, grf_z: float) -> float:
        """把 GRF 球的 Z 转换为 UE 米。

        引擎中 Z_FIELD_SCALE=1，所以 GRF 观测的 z 已近似为引擎米。
        我们原样透传。例如地面上的球读取约为 0.11（≈ 球半径）。
        如需精确控制球相对 UE 场地的高度，请在 UE 中调整。
        """
        return float(grf_z)

    def transform_player_position(
        self, grf_x: float, grf_y: float
    ) -> List[float]:
        """把球员的 GRF 位置变换为 [x_m, y_m, 0]。"""
        mx, my = self.grf_to_meter(grf_x, grf_y)
        return [mx, my, 0.0]

    def grf_direction_to_velocity_mps(
        self, grf_dx: float, grf_dy: float, dt_s: float
    ) -> Tuple[float, float]:
        """把 GRF 方向（每 simulation step 的归一化位移）换算为米/秒速度。

        GRF 观测的 `left_team_direction` / `right_team_direction` 语义为
        "Players' movement direction represented as [x, y] distance per step"
        （见 gfootball `football_env_core.py` 注释），即**每个 simulation step
        （0.1s）在归一化坐标空间中的位移**，与 `*_team` 位置共用同一套归一化
        坐标（x ∈ [-1,1]、y ∈ [-1,1]）。

        因此换算到米/秒只需沿用本项目既有的场地坐标变换（不另定义一套 pitch size）：

            vx_mps = grf_dx * half_field_length / dt_s
            vy_mps = grf_dy * half_field_width  / dt_s

        Args:
            grf_dx / grf_dy: GRF 方向的 x / y 分量（每步归一化位移）。
            dt_s: 单个 simulation step 的时长（秒，GRF 固定 0.1s）。

        Returns:
            (vx_mps, vy_mps) 米/秒。
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s 必须为正，got {dt_s}")
        vx = float(grf_dx) * self._half_length / float(dt_s)
        vy = float(grf_dy) * self._half_width / float(dt_s)
        return vx, vy

    def grf_ball_direction_to_velocity_mps(
        self, grf_dx: float, grf_dy: float, grf_dz: float, dt_s: float
    ) -> Tuple[float, float, float]:
        """把 GRF 球的 `ball_direction` 换算为米/秒速度。

        球的 `ball_direction` 为 "Ball's movement direction represented as
        [x, y] distance per step"——x/y 分量与球位置共用归一化坐标（同位置变换），
        z 分量未归一化（Z_FIELD_SCALE=1，观测中已近似为米）。因此：

            vx_mps = grf_dx * half_field_length / dt_s
            vy_mps = grf_dy * half_field_width  / dt_s
            vz_mps = grf_dz / dt_s                 # z 已为米，仅除步长

        Returns:
            (vx_mps, vy_mps, vz_mps)。
        """
        if dt_s <= 0.0:
            raise ValueError(f"dt_s 必须为正，got {dt_s}")
        vx = float(grf_dx) * self._half_length / float(dt_s)
        vy = float(grf_dy) * self._half_width / float(dt_s)
        vz = float(grf_dz) / float(dt_s)
        return vx, vy, vz

    def transform_ball_position(
        self, grf_pos: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """变换球的 GRF 位置。

        返回 (position_m, source_grf)，其中 position_m 为 [x_m, y_m, z_m]，
        source_grf 为原始 [grf_x, grf_y, grf_z] 供参考。
        """
        grf_x, grf_y, grf_z = float(grf_pos[0]), float(grf_pos[1]), float(grf_pos[2])
        mx, my = self.grf_to_meter(grf_x, grf_y)
        mz = self.grf_ball_z_to_meter(grf_z)
        return [mx, my, mz], [grf_x, grf_y, grf_z]
