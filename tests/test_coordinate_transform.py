"""坐标变换的测试。"""

import numpy as np
import pytest

from grf_ue_bridge.coordinate_transform import CoordinateTransform


class TestCoordinateTransform:
    def test_default_field(self):
        ct = CoordinateTransform()
        assert ct.half_length == 20.0
        assert ct.half_width == 10.0

    def test_custom_field(self):
        ct = CoordinateTransform(field_length_m=60.0, field_width_m=30.0)
        assert ct.half_length == 30.0
        assert ct.half_width == 15.0

    def test_center_spot(self):
        ct = CoordinateTransform()
        x, y = ct.grf_to_meter(0.0, 0.0)
        assert x == 0.0
        assert y == 0.0

    def test_left_goal_line(self):
        ct = CoordinateTransform()
        x, y = ct.grf_to_meter(-1.0, 0.0)
        assert x == -20.0
        assert y == 0.0

    def test_right_goal_line(self):
        ct = CoordinateTransform()
        x, y = ct.grf_to_meter(1.0, 0.0)
        assert x == 20.0
        assert y == 0.0

    def test_top_corner(self):
        ct = CoordinateTransform()
        x, y = ct.grf_to_meter(1.0, 1.0)
        assert x == 20.0
        assert y == 10.0

    def test_bottom_corner(self):
        ct = CoordinateTransform()
        x, y = ct.grf_to_meter(-1.0, -1.0)
        assert x == -20.0
        assert y == -10.0

    def test_player_position(self):
        ct = CoordinateTransform()
        pos = ct.transform_player_position(0.5, -0.3)
        assert pos == [10.0, -3.0, 0.0]

    def test_ball_z_passthrough(self):
        ct = CoordinateTransform()
        mz = ct.grf_ball_z_to_meter(0.11)
        assert mz == 0.11  # Z_FIELD_SCALE=1，原样透传

    def test_ball_z_zero(self):
        ct = CoordinateTransform()
        mz = ct.grf_ball_z_to_meter(0.0)
        assert mz == 0.0

    def test_ball_position(self):
        ct = CoordinateTransform()
        grf_pos = np.array([0.5, -0.3, 0.11])
        pos_m, source = ct.transform_ball_position(grf_pos)
        assert pos_m == [10.0, -3.0, 0.11]
        assert source == [0.5, -0.3, 0.11]

    def test_ball_at_center(self):
        ct = CoordinateTransform()
        grf_pos = np.array([0.0, 0.0, 0.0])
        pos_m, source = ct.transform_ball_position(grf_pos)
        assert pos_m == [0.0, 0.0, 0.0]

    def test_direction_to_velocity(self):
        # GRF direction = 每步归一化位移；0.1s 一步，half_length=20, half_width=10
        ct = CoordinateTransform()
        vx, vy = ct.grf_direction_to_velocity_mps(0.01, 0.02, 0.1)
        assert vx == pytest.approx(0.01 * 20 / 0.1)  # 2.0 m/s
        assert vy == pytest.approx(0.02 * 10 / 0.1)  # 2.0 m/s

    def test_direction_zero(self):
        ct = CoordinateTransform()
        vx, vy = ct.grf_direction_to_velocity_mps(0.0, 0.0, 0.1)
        assert vx == 0.0 and vy == 0.0

    def test_direction_invalid_dt(self):
        ct = CoordinateTransform()
        with pytest.raises(ValueError):
            ct.grf_direction_to_velocity_mps(0.01, 0.0, 0.0)

    def test_ball_direction_to_velocity(self):
        ct = CoordinateTransform()
        vx, vy, vz = ct.grf_ball_direction_to_velocity_mps(0.01, 0.02, 0.005, 0.1)
        assert vx == pytest.approx(2.0)
        assert vy == pytest.approx(2.0)
        assert vz == pytest.approx(0.005 / 0.1)  # z 已为米，仅除步长 = 0.05
