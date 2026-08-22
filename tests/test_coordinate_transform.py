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

    def test_y_usable_half_range_to_side_line(self):
        # GRF 可用 Y 范围约为 ±1/2.25，映射到场地宽度 ±half_width
        ct = CoordinateTransform()
        half = 1.0 / 2.25
        x, y = ct.grf_to_meter(0.0, half)
        assert x == 0.0
        assert y == pytest.approx(10.0)  # 10 / (1/2.25) * (1/2.25) = 10

    def test_y_scale_factor(self):
        # y_scale = half_width / (1/2.25) = 22.5（40x20m 默认场地）
        ct = CoordinateTransform()
        assert ct.y_scale == pytest.approx(22.5)

    def test_y_beyond_usable_range_is_out_of_court(self):
        # grf_y=1.0 超出可用范围 → 映射到 22.5m（超出场地，属越界，非有效位置）
        ct = CoordinateTransform()
        x, y = ct.grf_to_meter(1.0, 1.0)
        assert x == 20.0
        assert y == pytest.approx(22.5)

    def test_player_position(self):
        ct = CoordinateTransform()
        pos = ct.transform_player_position(0.5, -0.3)
        assert pos[0] == 10.0
        assert pos[1] == pytest.approx(-0.3 * 22.5)  # -6.75
        assert pos[2] == 0.0

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
        assert pos_m[0] == 10.0
        assert pos_m[1] == pytest.approx(-6.75)
        assert pos_m[2] == 0.11
        assert source == [0.5, -0.3, 0.11]

    def test_ball_at_center(self):
        ct = CoordinateTransform()
        grf_pos = np.array([0.0, 0.0, 0.0])
        pos_m, source = ct.transform_ball_position(grf_pos)
        assert pos_m == [0.0, 0.0, 0.0]

    def test_direction_to_velocity(self):
        # GRF direction = 每步归一化位移；0.1s 一步，half_length=20, y_scale=22.5
        ct = CoordinateTransform()
        vx, vy = ct.grf_direction_to_velocity_mps(0.01, 0.02, 0.1)
        assert vx == pytest.approx(0.01 * 20 / 0.1)  # 2.0 m/s
        assert vy == pytest.approx(0.02 * 22.5 / 0.1)  # 4.5 m/s

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
        assert vy == pytest.approx(0.02 * 22.5 / 0.1)  # 4.5
        assert vz == pytest.approx(0.005 / 0.1)  # z 已为米，仅除步长 = 0.05
