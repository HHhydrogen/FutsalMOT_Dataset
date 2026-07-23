"""Gymnasium environment for FutsalDefenderFollow task.

Controls Player_05 (defender) while replaying all other agents from a
rule-generated A3.3 trajectory. Uses a dense reward function and a
structured observation space.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from futsalmot_rl.core.rl_paths import (
    COURT_X_MAX,
    COURT_X_MIN,
    COURT_Y_MAX,
    COURT_Y_MIN,
    FPS,
    RUNS_DIR,
)
from futsalmot_rl.data.a33_reader import (
    get_ball_positions_2d,
    get_event_frame_map,
    get_player_positions_2d,
    get_player_velocities,
    get_player_yaws,
    get_possession_timeline,
    load_a33_config,
)
from futsalmot_rl.features.action_builder import (
    PLAYER_05_MAX_ACCEL_CM_S2,
    PLAYER_05_MAX_SPEED_CM_S,
    apply_motion_constraints,
)
from futsalmot_rl.features.obs_builder import OBS_DIM, build_observation
from futsalmot_rl.rewards.defender_rewards import DefenderReward


class FutsalDefenderFollowEnv(gym.Env):
    """A 4v4 futsal environment where Player_05 learns to follow Player_01.

    All other players and the ball follow their rule-generated trajectories.
    """

    metadata: dict = {"render_modes": ["rgb_array"], "render_fps": 15}  # noqa: RUF012

    def __init__(
        self,
        source_episode_path: str | Path | None = None,
        a33_config: dict[str, Any] | None = None,
        agent_id: str = "Player_05",
        target_id: str = "Player_01",
        fps: int = FPS,
        episode_length_frames: int = 300,
        reward_config: dict[str, Any] | None = None,
        seed: int = 42,
    ):
        super().__init__()

        self.agent_id = agent_id
        self.target_id = target_id
        self.fps = fps
        self.episode_length = episode_length_frames
        self.seed = seed
        self.own_goal_pos = (COURT_X_MIN, 0.0)  # Team B defends -X

        # ── Reward function ──────────────────────────────────────
        reward_kwargs = reward_config or {}
        self.reward_fn = DefenderReward(**reward_kwargs)

        # ── Action / Observation spaces ──────────────────────────
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # ── Load source trajectory data ───────────────────────────
        self._source_path = None
        if a33_config is not None:
            self._cfg = a33_config
        elif source_episode_path is not None:
            self._source_path = Path(source_episode_path)
            self._cfg = load_a33_config(self._source_path)
        else:
            # Try to find a default episode
            candidate = RUNS_DIR / "production_run" / "episode_random_0001_t1_a33.json"
            if candidate.is_file():
                self._source_path = candidate
                self._cfg = load_a33_config(candidate)
            else:
                raise ValueError("Either source_episode_path or a33_config must be provided")

        self._load_source_data()

        # ── Internal state ────────────────────────────────────────
        self.current_frame = 0
        self.agent_pos: tuple[float, float] = (0.0, 0.0)
        self.agent_vel: tuple[float, float] = (0.0, 0.0)
        self.prev_vel: tuple[float, float] = (0.0, 0.0)
        self.trail: list[tuple[float, float]] = []
        self._last_yaw: float = 0.0
        self._yaw_speed_threshold: float = 5.0  # cm/s
        self.last_info: dict[str, Any] = {}

    def _load_source_data(self) -> None:
        """Load all rule trajectory data from the A3.3 config."""
        self.cfg = self._cfg

        # Player positions (2D)
        self.all_positions = get_player_positions_2d(self._cfg)

        # Player yaws
        self.all_yaws = get_player_yaws(self._cfg)

        # Ball positions
        self.ball_positions = get_ball_positions_2d(self._cfg)

        # Velocities
        all_3d = {pid: [(x, y, 0.0) for x, y in pts] for pid, pts in self.all_positions.items()}
        self.all_velocities = get_player_velocities(all_3d, self.fps)

        # Possession timeline
        self.possession_segments = get_possession_timeline(self._cfg)
        self._possession_by_frame: dict[int, str | None] = {}
        for seg in self.possession_segments:
            start = int(seg.get("start_frame", 0))
            end = int(seg.get("end_frame", 0))
            owner = seg.get("owner")
            for f in range(start, end + 1):
                self._possession_by_frame[f] = owner

        # Event frame map
        self.event_map = get_event_frame_map(self._cfg)
        self._frame_event_type: dict[int, str] = {}
        for _evt_id, evt_data in self.event_map.items():
            evt_type = evt_data.get("type", "")
            actor = evt_data.get("actor", "")
            start = int(evt_data.get("start_frame", 0))
            end_exc = int(evt_data.get("end_frame_exclusive", 0))
            if actor == self.agent_id:
                for f in range(start, end_exc):
                    self._frame_event_type[f] = evt_type

        # Agent and target trajectories
        self.agent_rule_positions = self.all_positions.get(self.agent_id, [])
        self.target_positions = self.all_positions.get(self.target_id, [])

        # Validate
        total_frames = len(self.agent_rule_positions)
        if total_frames < self.episode_length:
            raise ValueError(f"Episode has {total_frames} frames, expected {self.episode_length}")

        self.total_frames = total_frames
        self.seq_id = str(self._cfg.get("seq_id", "unknown"))

    def _get_possession_owner(self, frame: int) -> str | None:
        """Get possession owner at a given frame."""
        return self._possession_by_frame.get(frame)

    def _get_event_type(self, frame: int) -> str | None:
        """Get event type for the agent at a given frame."""
        return self._frame_event_type.get(frame)

    def _get_other_positions(self, frame: int) -> dict[str, tuple[float, float]]:
        """Get positions of all non-agent players at a given frame."""
        result: dict[str, tuple[float, float]] = {}
        for pid, positions in self.all_positions.items():
            if pid != self.agent_id and frame < len(positions):
                result[pid] = (float(positions[frame][0]), float(positions[frame][1]))
        return result

    def _check_collisions(
        self,
        pos: tuple[float, float],
    ) -> list[str]:
        """Check if the agent collides with any other player."""
        collisions: list[str] = []
        for pid, positions in self.all_positions.items():
            if pid == self.agent_id:
                continue
            frame = min(self.current_frame, len(positions) - 1)
            other_pos = positions[frame]
            dist = math.hypot(
                pos[0] - float(other_pos[0]),
                pos[1] - float(other_pos[1]),
            )
            if dist < self.reward_fn.collision_distance_cm:
                collisions.append(pid)
        return collisions

    def _check_out_of_bounds(self, pos: tuple[float, float]) -> bool:
        """Check if the agent is out of court bounds."""
        margin = 10.0  # small tolerance
        return not (
            COURT_X_MIN + margin <= pos[0] <= COURT_X_MAX - margin
            and COURT_Y_MIN + margin <= pos[1] <= COURT_Y_MAX - margin
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the start of an episode."""
        super().reset(seed=seed)

        self.current_frame = 0
        self.agent_pos = (
            float(self.agent_rule_positions[0][0]),
            float(self.agent_rule_positions[0][1]),
        )
        self.agent_vel = (0.0, 0.0)
        self.prev_vel = (0.0, 0.0)
        self.trail = [self.agent_pos]
        self._last_yaw = 0.0

        obs = self._build_observation()
        info = self._build_info()
        self.last_info = info

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Array of shape (2,) with values in [-1, 1].

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Update velocity with motion constraints
        self.prev_vel = self.agent_vel
        self.agent_vel = apply_motion_constraints(
            action,
            self.agent_vel,
            max_speed=PLAYER_05_MAX_SPEED_CM_S,
            max_accel=PLAYER_05_MAX_ACCEL_CM_S2,
            fps=self.fps,
        )

        # Update position
        dt = 1.0 / self.fps
        new_x = self.agent_pos[0] + self.agent_vel[0] * dt
        new_y = self.agent_pos[1] + self.agent_vel[1] * dt

        # Check bounds
        out_of_bounds = self._check_out_of_bounds((new_x, new_y))

        # Clip to court
        new_x = max(COURT_X_MIN + 10.0, min(COURT_X_MAX - 10.0, new_x))
        new_y = max(COURT_Y_MIN + 10.0, min(COURT_Y_MAX - 10.0, new_y))

        self.agent_pos = (new_x, new_y)
        self.trail.append(self.agent_pos)
        if len(self.trail) > 100:
            self.trail = self.trail[-100:]

        # Advance frame
        self.current_frame += 1
        terminated = self.current_frame >= self.total_frames - 1
        truncated = False

        # Compute target position at current frame
        target_idx = min(self.current_frame, len(self.target_positions) - 1)
        target_pos = (
            float(self.target_positions[target_idx][0]),
            float(self.target_positions[target_idx][1]),
        )

        # Check collisions
        collisions = self._check_collisions(self.agent_pos)

        # Compute reward
        ball_idx = min(self.current_frame, len(self.ball_positions) - 1)
        ball_pos = (
            (
                float(self.ball_positions[ball_idx][0]),
                float(self.ball_positions[ball_idx][1]),
            )
            if self.ball_positions
            else (0.0, 0.0)
        )

        reward, reward_components = self.reward_fn.compute(
            player_pos=self.agent_pos,
            target_pos=target_pos,
            own_goal_pos=self.own_goal_pos,
            ball_pos=ball_pos,
            all_player_positions=self._get_other_positions(self.current_frame),
            agent_id=self.agent_id,
            prev_velocity=self.prev_vel,
            current_velocity=self.agent_vel,
            fps=self.fps,
            out_of_bounds=out_of_bounds,
            collisions=collisions,
        )

        # Build observation
        obs = self._build_observation()

        # Build info
        dist_to_target = math.hypot(
            self.agent_pos[0] - target_pos[0],
            self.agent_pos[1] - target_pos[1],
        )

        info = {
            "all_positions": self._get_all_positions(),
            "ball_pos": ball_pos,
            "agent_velocity": self.agent_vel,
            "agent_trail": list(self.trail),
            "ghost_positions": self._get_all_positions(),  # Rule ghost = rule positions
            "distance_to_target": dist_to_target,
            "collision": len(collisions) > 0,
            "collisions": collisions,
            "out_of_bounds": out_of_bounds,
            "possession_owner": self._get_possession_owner(self.current_frame),
            "event_type": self._get_event_type(self.current_frame),
            "reward_components": reward_components,
            "frame": self.current_frame,
            "seq_id": self.seq_id,
        }

        # Update agent_rule_positions reference for video
        if self.current_frame < len(self.agent_rule_positions):
            rule_pos = self.agent_rule_positions[self.current_frame]
            info["agent_rule_position"] = (float(rule_pos[0]), float(rule_pos[1]))

        # Rule ghost: all players at their rule positions
        ghost: dict[str, tuple[float, float]] = {}
        for pid, positions in self.all_positions.items():
            idx = min(self.current_frame, len(positions) - 1)
            ghost[pid] = (float(positions[idx][0]), float(positions[idx][1]))
        info["ghost_positions"] = ghost

        self.last_info = info

        return obs, float(reward), terminated, truncated, info

    def _update_yaw(self, vx: float, vy: float) -> float:
        """Update and return yaw from actual velocity. Below threshold keeps last yaw."""
        import numpy as np
        speed = np.hypot(vx, vy)
        if speed > self._yaw_speed_threshold:
            self._last_yaw = float(np.degrees(np.arctan2(vy, vx)))
        return self._last_yaw

    def _compute_ball_velocity(self, frame: int) -> tuple[float, float]:
        """Compute ball velocity from consecutive ball positions."""
        if frame <= 0 or frame >= len(self.ball_positions):
            return (0.0, 0.0)
        dx = self.ball_positions[frame][0] - self.ball_positions[frame - 1][0]
        dy = self.ball_positions[frame][1] - self.ball_positions[frame - 1][1]
        return (dx * self.fps, dy * self.fps)

    def _build_observation(self) -> np.ndarray:
        """Build the current observation from ACTUAL agent state (not rule replay)."""
        frame = self.current_frame
        target_idx = min(frame, len(self.target_positions) - 1)

        # Self velocity: use actual agent velocity from physics
        av = self.agent_vel

        # Self yaw: compute from actual velocity (persists when stopped)
        self_yaw = self._update_yaw(av[0], av[1])

        # Target velocity: use rule trajectory velocity (target follows rule)
        vel_idx = min(frame, len(self.all_velocities.get(self.target_id, [(0, 0)])) - 1)
        target_vel_arr = self.all_velocities.get(self.target_id, [(0.0, 0.0)])
        tv = target_vel_arr[vel_idx] if vel_idx < len(target_vel_arr) else (0.0, 0.0)

        # Ball velocity: compute from ball positions independently
        bv = self._compute_ball_velocity(frame)

        # Ball position
        ball_idx = min(frame, len(self.ball_positions) - 1)
        ball_pos = self.ball_positions[ball_idx] if self.ball_positions else (0.0, 0.0)

        # 300 frames = 299 transitions
        num_transitions = max(0, self.total_frames - 1)
        remaining_transitions = max(0, num_transitions - frame)

        obs = build_observation(
            self_pos=self.agent_pos,
            self_vel=av,
            self_yaw_deg=self_yaw,
            target_pos=(
                float(self.target_positions[target_idx][0]),
                float(self.target_positions[target_idx][1]),
            ),
            target_vel=tv,
            ball_pos=(
                float(ball_pos[0]),
                float(ball_pos[1]),
            ),
            ball_vel=bv,
            own_goal_pos=self.own_goal_pos,
            possession_owner=self._get_possession_owner(frame),
            current_event_type=self._get_event_type(frame),
            steps_left=remaining_transitions,
            total_steps=num_transitions,
        )
        return obs

    def _get_all_positions(self) -> dict[str, tuple[float, float]]:
        """Get positions of all players at current frame."""
        result: dict[str, tuple[float, float]] = {}
        result[self.agent_id] = self.agent_pos
        result.update(self._get_other_positions(self.current_frame))
        return result

    def _build_info(self) -> dict[str, Any]:
        """Build initial info dict."""
        return {
            "all_positions": self._get_all_positions(),
            "ball_pos": self.ball_positions[0] if self.ball_positions else (0.0, 0.0),
            "agent_velocity": self.agent_vel,
            "agent_trail": list(self.trail),
            "ghost_positions": self._get_all_positions(),
            "distance_to_target": 0.0,
            "collision": False,
            "collisions": [],
            "out_of_bounds": False,
            "possession_owner": self._get_possession_owner(0),
            "event_type": self._get_event_type(0),
            "reward_components": {},
            "frame": 0,
            "seq_id": self.seq_id,
        }
