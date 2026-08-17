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

    def test_player_frame_motion_fields(self):
        """新增运动字段的序列化/反序列化（velocity/speed/heading/active/has_ball）。"""
        pf = PlayerFrame(
            id="L1",
            position_m=[1.0, 2.0, 0.0],
            velocity_mps=[1.5, 0.0],
            speed_mps=1.5,
            movement_heading_deg=0.0,
            active=True,
            has_ball=False,
        )
        d = pf.model_dump()
        assert d["velocity_mps"] == [1.5, 0.0]
        assert d["speed_mps"] == 1.5
        assert d["has_ball"] is False
        # 往返一致
        assert PlayerFrame(**d) == pf

    def test_player_frame_motion_fields_default_none(self):
        """旧 episode（无运动字段）反序列化时字段为 None，不报错。"""
        pf = PlayerFrame(id="L0", position_m=[0.0, 0.0, 0.0])
        assert pf.velocity_mps is None
        assert pf.speed_mps is None
        assert pf.movement_heading_deg is None
        assert pf.active is None
        assert pf.has_ball is None

    def test_frame_ownership_fields(self):
        """Frame 级 ball ownership / game_mode 字段的序列化/反序列化。"""
        frame = Frame(
            step=2,
            time_seconds=0.2,
            ball=BallFrame(position_m=[0.0, 0.0, 0.11]),
            players=[PlayerFrame(id=f"L{i}", position_m=[0.0, 0.0, 0.0]) for i in range(5)]
            + [PlayerFrame(id=f"R{i}", position_m=[0.0, 0.0, 0.0]) for i in range(5)],
            ball_owned_team=0,
            ball_owned_player=2,
            game_mode=1,
        )
        d = frame.model_dump()
        assert d["ball_owned_team"] == 0
        assert d["ball_owned_player"] == 2
        assert d["game_mode"] == 1
        assert Frame(**d) == frame

    def test_old_frame_backward_compatible(self):
        """旧 episode 帧（无任何新字段）仍可正常读取。"""
        raw = {
            "step": 0,
            "time_seconds": 0.0,
            "score": [0, 0],
            "ball": {"position_m": [0.0, 0.0, 0.11], "source_grf_position": [0.0, 0.0, 0.11]},
            "players": [{"id": f"L{i}", "position_m": [0.0, 0.0, 0.0]} for i in range(5)]
            + [{"id": f"R{i}", "position_m": [0.0, 0.0, 0.0]} for i in range(5)],
        }
        frame = Frame(**raw)
        assert frame.ball.velocity_mps is None
        assert frame.ball_owned_team is None
        assert frame.players[0].velocity_mps is None
        assert len(frame.players) == 10


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
