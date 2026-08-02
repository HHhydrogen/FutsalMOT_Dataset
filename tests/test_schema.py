"""schema 模型的测试。"""

from grf_ue_bridge.schema import (
    BallFrame,
    EntityDefinition,
    Frame,
    Meta,
    PlayerFrame,
    TimingInfo,
    create_ball_entity,
)


class TestEntityDefinition:
    def test_from_grf_role_goalkeeper(self):
        e = EntityDefinition.from_grf_role(0, "L", 0)
        assert e.id == "L0"
        assert e.team == "left"
        assert e.source_index == 0
        assert e.role == "goalkeeper"
        assert e.is_goalkeeper is True

    def test_from_grf_role_forward(self):
        e = EntityDefinition.from_grf_role(9, "R", 3)
        assert e.id == "R3"
        assert e.team == "right"
        assert e.role == "center_forward"
        assert e.is_goalkeeper is False

    def test_create_ball_entity(self):
        e = create_ball_entity()
        assert e.id == "BALL"
        assert e.team is None
        assert e.is_goalkeeper is False


class TestFrame:
    def test_minimal_frame(self):
        players = []
        for i in range(5):
            players.append(PlayerFrame(id=f"L{i}", position_m=[float(-10 + i), 0.0, 0.0]))
        for i in range(5):
            players.append(PlayerFrame(id=f"R{i}", position_m=[float(10 - i), 0.0, 0.0]))

        frame = Frame(
            step=0,
            time_seconds=0.0,
            score=[0, 0],
            ball=BallFrame(position_m=[0.0, 0.0, 0.11]),
            players=players,
        )
        assert frame.step == 0
        assert len(frame.players) == 10
        assert frame.players[0].id == "L0"
        assert frame.players[9].id == "R4"
        assert frame.ball.source_grf_position == [0.0, 0.0, 0.0]

    def test_ball_with_source_position(self):
        frame = Frame(
            step=1,
            time_seconds=0.1,
            score=[1, 0],
            ball=BallFrame(
                position_m=[5.0, 2.0, 0.11],
                source_grf_position=[0.25, 0.2, 0.11],
            ),
            players=[PlayerFrame(id=f"L{i}", position_m=[0.0, 0.0, 0.0]) for i in range(5)]
            + [PlayerFrame(id=f"R{i}", position_m=[0.0, 0.0, 0.0]) for i in range(5)],
        )
        assert frame.ball.source_grf_position == [0.25, 0.2, 0.11]
        assert frame.step == 1
        assert frame.time_seconds == 0.1


class TestMeta:
    def test_meta_defaults(self):
        meta = Meta(timing=TimingInfo(source_step_seconds=0.1, playback_fps=30, num_steps=300))
        assert meta.schema_ == "grf_ue_episode"
        assert meta.version == 1
        assert meta.timing.source_step_seconds == 0.1
        assert meta.field.length_m == 40.0

    def test_meta_serialization(self):
        meta = Meta(timing=TimingInfo(source_step_seconds=0.1, playback_fps=30, num_steps=300))
        d = meta.model_dump(by_alias=True)
        assert d["schema"] == "grf_ue_episode"
        assert d["version"] == 1
        assert "timing" in d
