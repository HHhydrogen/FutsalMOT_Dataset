"""import_grf_episode.py 的 Rotation.Z yaw 展开（_unwind_angle）测试。

验证写入 Sequencer 前的连续 yaw 在跨 ±180° 边界时保持最短路径旋转。
"""

import import_grf_episode as ig
from types import SimpleNamespace


def _unwrap_sequence(facings):
    """模拟 create_sequence 里对 Rotation.Z 的展开：首帧取原始值，后续累加最短角度差。"""
    continuous = None
    out = []
    for f in facings:
        if continuous is None:
            continuous = f
        else:
            continuous = ig._unwind_angle(continuous, f)
        out.append(continuous)
    return out


def test_unwrap_forward_crossing_180():
    facings = [170.0, 179.0, -176.0, -165.0]
    assert _unwrap_sequence(facings) == [170.0, 179.0, 184.0, 195.0]


def test_unwrap_backward_crossing_neg180():
    facings = [-170.0, -179.0, 176.0, 165.0]
    assert _unwrap_sequence(facings) == [-170.0, -179.0, -184.0, -195.0]


def test_unwrap_multiple_crossings():
    # 单调增方向连续两次跨 ±180°：160→174→183→195→200→210
    facings = [160.0, 174.0, -177.0, -165.0, -160.0, -150.0]
    assert _unwrap_sequence(facings) == [160.0, 174.0, 183.0, 195.0, 200.0, 210.0]


def test_unwrap_follows_shortest_reversal():
    # 跨 180° 后反向（最短路径是转回而非继续绕圈）
    facings = [174.0, -177.0, 178.0]
    assert _unwrap_sequence(facings) == [174.0, 183.0, 178.0]


def test_unwrap_keeps_non_crossing_unchanged():
    facings = [10.0, 30.0, 45.0, -10.0, -60.0]
    assert _unwrap_sequence(facings) == [10.0, 30.0, 45.0, -10.0, -60.0]


def test_player_sequence_motion_samples_use_tracker_speed_values(monkeypatch):
    calls = []

    class Tracker:
        def update(self, position, velocity, time_s, **kwargs):
            calls.append((position, velocity, time_s, kwargs))
            return {
                "speed_mps": 1.25 + len(calls),
                "velocity_mps": [3.0, 4.0],
                "facing_deg": 90.0,
            }

    class Channel:
        def __init__(self):
            self.keys = []

        def add_key(self, frame, value):
            self.keys.append((frame, value))

    monkeypatch.setattr(ig, "PlayerMotionTracker", Tracker)
    channels = {name: Channel() for name in (
        "Location.X", "Location.Y", "Location.Z", "Rotation.Z"
    )}
    monkeypatch.setattr(
        ig, "add_double_channel_key",
        lambda channel, frame, value, **_kwargs: channel.add_key(frame, value),
    )
    frames = [
        {
            "time_seconds": i * 0.1,
            "players": [
                    {"id": f"{team}{slot}", "position_m": [i, slot, 0.0], "velocity_mps": [0.3, 0.4]}
                for team in ("L", "R") for slot in range(5)
            ],
        }
        for i in range(3)
    ]

    samples = []
    yaw = None
    tracker = Tracker()
    for i, frame in enumerate(frames):
        player = next(p for p in frame["players"] if p["id"] == "R0")
        params = tracker.update(
            player["position_m"], player.get("velocity_mps"), frame["time_seconds"]
        )
        yaw, sample = ig._write_player_frame(channels, player, params, i * 3, yaw)
        samples.append(sample)

    assert [sample["frame"] for sample in samples] == [0, 3, 6]
    assert [sample["speed_mps"] for sample in samples] == [2.25, 3.25, 4.25]
    assert len(calls) == 3
    assert all(call[1] == [0.3, 0.4] for call in calls)
    assert channels["Rotation.Z"].keys == [(0, 90.0), (3, 90.0), (6, 90.0)]


def test_external_motion_track_writer_targets_only_canonical_property_names(monkeypatch):
    class Channel:
        def __init__(self):
            self.keys = []

        def add_key(self, frame=None, value=None, *args, **kwargs):
            frame = kwargs.get("time", frame)
            value = kwargs.get("new_value", value)
            self.keys.append((frame.value, value))

        def get_num_keys(self):
            return len(self.keys)

    class Section:
        def __init__(self, channels):
            self.channels = channels

        def set_range(self, start, end):
            self.range = (start, end)

        def get_all_channels(self):
            return self.channels

    class Track:
        def __init__(self, channels):
            self.section = Section(channels)

        def set_property_name_and_path(self, name, path):
            self.property = (name, path)

        def add_section(self):
            return self.section

    class Binding:
        def __init__(self):
            self.tracks = []

        def add_track(self, track_class):
            track = track_class()
            self.tracks.append(track)
            return track

    class FrameNumber:
        def __init__(self, value):
            self.value = value

    class BoolTrack(Track):
        def __init__(self):
            super().__init__([Channel()])

    class FloatTrack(Track):
        def __init__(self):
            super().__init__([Channel()])

    fake_unreal = SimpleNamespace(
        MovieSceneBoolTrack=BoolTrack,
        MovieSceneFloatTrack=FloatTrack,
        FrameNumber=FrameNumber,
        MovieSceneKeyInterpolation=SimpleNamespace(LINEAR=0),
    )
    monkeypatch.setitem(__import__("sys").modules, "unreal", fake_unreal)
    binding = Binding()
    samples = [
        {"frame": 0, "speed_mps": 0.0},
        {"frame": 3, "speed_mps": 3.0},
        {"frame": 6, "speed_mps": 6.0},
    ]

    ig._add_external_motion_tracks(binding, samples, 10)

    assert [track.property for track in binding.tracks] == [
        ("ExternalMotionActive", "ExternalMotionActive"),
        ("ExternalMotionSpeedMps", "ExternalMotionSpeedMps"),
    ]
    assert binding.tracks[0].section.channels[0].keys == [(0, True), (9, True)]
    assert binding.tracks[1].section.channels[0].keys == [
        (0, 0.0), (3, 3.0), (6, 6.0)
    ]


def test_all_ten_player_bindings_get_motion_tracks_and_ball_does_not(monkeypatch):
    class Channel:
        def __init__(self):
            self.keys = []

        def add_key(self, frame=None, value=None, **kwargs):
            frame = kwargs.get("time", frame)
            value = kwargs.get("new_value", value)
            self.keys.append((frame.value, value))

        def get_num_keys(self):
            return len(self.keys)

    class Section:
        def __init__(self):
            self.channels = [Channel()]

        def set_range(self, *_args):
            pass

        def get_all_channels(self):
            return self.channels

    class Track:
        def __init__(self):
            self.section = Section()

        def set_property_name_and_path(self, name, path):
            self.property = (name, path)

        def add_section(self):
            return self.section

    class Binding:
        def __init__(self):
            self.tracks = []

        def add_track(self, track_class):
            track = track_class()
            self.tracks.append(track)
            return track

    class FrameNumber:
        def __init__(self, value):
            self.value = value

    class BoolTrack(Track):
        pass

    class FloatTrack(Track):
        pass

    fake_unreal = SimpleNamespace(
        MovieSceneBoolTrack=BoolTrack,
        MovieSceneFloatTrack=FloatTrack,
        FrameNumber=FrameNumber,
        MovieSceneKeyInterpolation=SimpleNamespace(LINEAR=0),
    )
    monkeypatch.setitem(__import__("sys").modules, "unreal", fake_unreal)

    class LocationChannel(Channel):
        def __init__(self, name):
            super().__init__()
            self.channel_name = name

    channels = {
        name: LocationChannel(name)
        for name in (
            "Location.X", "Location.Y", "Location.Z", "Rotation.Z"
        )
    }
    monkeypatch.setattr(
        ig, "add_double_channel_key",
        lambda channel, frame, value, **_kwargs: channel.add_key(
            FrameNumber(frame), float(value)
        ),
    )
    frame_samples = []
    yaw, sample = ig._write_player_frame(
        channels,
        {"position_m": [1.0, 2.0, 0.0]},
        {"facing_deg": 179.0, "speed_mps": 3.0},
        0,
        None,
    )
    frame_samples.append(sample)
    yaw, sample = ig._write_player_frame(
        channels,
        {"position_m": [1.5, 2.0, 0.0]},
        {"facing_deg": -179.0, "speed_mps": 4.0},
        3,
        yaw,
    )
    frame_samples.append(sample)
    assert channels["Location.X"].keys == [(0, 100.0), (3, 150.0)]
    assert channels["Rotation.Z"].keys == [(0, 179.0), (3, 181.0)]
    assert frame_samples == [
        {"frame": 0, "speed_mps": 3.0},
        {"frame": 3, "speed_mps": 4.0},
    ]

    player_bindings = [Binding() for _ in range(10)]
    samples = [{"frame": 0, "speed_mps": 0.0}, {"frame": 3, "speed_mps": 2.5}]
    for binding in player_bindings:
        ig._add_external_motion_tracks(binding, samples, 10)
    ball_binding = Binding()

    assert all(len(binding.tracks) == 2 for binding in player_bindings)
    assert ball_binding.tracks == []


def test_external_motion_track_names_are_canonical_and_not_retired():
    assert ig.EXTERNAL_MOTION_ACTIVE_PROPERTY == "ExternalMotionActive"
    assert ig.EXTERNAL_MOTION_SPEED_PROPERTY == "ExternalMotionSpeedMps"
    assert ig.EXTERNAL_MOTION_ACTIVE_PROPERTY != "MotionSpeedMps"
    assert ig.EXTERNAL_MOTION_SPEED_PROPERTY != "MotionSpeedMps"
