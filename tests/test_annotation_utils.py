"""标注工具纯函数的测试。"""

from annotation_utils import (
    analyze_bbox,
    entity_class,
    entity_id_to_track_id,
    entity_team,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


class TestTrackIds:
    def test_players(self):
        assert entity_id_to_track_id("L0") == 1
        assert entity_id_to_track_id("L4") == 5
        assert entity_id_to_track_id("R0") == 6
        assert entity_id_to_track_id("R4") == 10

    def test_ball(self):
        assert entity_id_to_track_id("BALL") == 100

    def test_stable_across_frames(self):
        # 同一个 entity 在所有帧永远映射到同一个 track
        for _ in range(3):
            assert entity_id_to_track_id("L3") == 4

    def test_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError):
            entity_id_to_track_id("X0")


class TestClassTeam:
    def test_class(self):
        assert entity_class("L0") == "player"
        assert entity_class("BALL") == "ball"

    def test_team(self):
        assert entity_team("L0") == "left"
        assert entity_team("R4") == "right"
        assert entity_team("BALL") is None


class TestBboxConversion:
    def test_xyxy_to_xywh(self):
        assert xyxy_to_xywh((10, 20, 130, 450)) == (10, 20, 120, 430)

    def test_xywh_to_xyxy(self):
        assert xywh_to_xyxy((10, 20, 120, 430)) == (10, 20, 130, 450)


class TestClipping:
    W, H = 1920, 1080

    def test_fully_inside(self):
        res = analyze_bbox(100, 100, 300, 400, self.W, self.H)
        assert res["in_frame"] is True
        assert res["truncated"] is False
        assert res["clipped_xyxy"] == (100.0, 100.0, 300.0, 400.0)

    def test_left_overflow(self):
        res = analyze_bbox(-35, 200, 85, 630, self.W, self.H)
        assert res["in_frame"] is True
        assert res["truncated"] is True
        assert res["clipped_xyxy"] == (0.0, 200.0, 85.0, 630.0)

    def test_right_overflow(self):
        res = analyze_bbox(1900, 200, 2100, 630, self.W, self.H)
        assert res["in_frame"] is True
        assert res["truncated"] is True
        assert res["clipped_xyxy"] == (1900.0, 200.0, 1920.0, 630.0)

    def test_top_overflow(self):
        res = analyze_bbox(100, -50, 300, 100, self.W, self.H)
        assert res["in_frame"] is True
        assert res["truncated"] is True
        assert res["clipped_xyxy"] == (100.0, 0.0, 300.0, 100.0)

    def test_bottom_overflow(self):
        res = analyze_bbox(100, 1000, 300, 1200, self.W, self.H)
        assert res["in_frame"] is True
        assert res["truncated"] is True
        assert res["clipped_xyxy"] == (100.0, 1000.0, 300.0, 1080.0)

    def test_fully_outside(self):
        res = analyze_bbox(-500, -500, -100, -100, self.W, self.H)
        assert res["in_frame"] is False
        assert res["truncated"] is False

    def test_degenerate_zero_area(self):
        res = analyze_bbox(100, 100, 100, 100, self.W, self.H)
        assert res["in_frame"] is False

    def test_fully_beyond_right_edge(self):
        res = analyze_bbox(2000, 2000, 2100, 2100, self.W, self.H)
        assert res["in_frame"] is False
