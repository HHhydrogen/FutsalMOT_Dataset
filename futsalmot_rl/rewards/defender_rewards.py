"""Reward function for the FutsalDefenderFollow environment."""

from __future__ import annotations

import math
from typing import Any


class DefenderReward:
    """Dense reward function for the defender follow task.

    Components:
    - r_marking_point: Negative distance to the ideal marking position.
    - r_distance_band: Penalty for being too far or too close to target.
    - r_goal_side: Bonus for being between target and own goal.
    - r_smoothness: Penalty for high acceleration.
    - r_boundary: Large penalty for going out of bounds.
    - r_collision: Large penalty for player collision.
    """

    def __init__(
        self,
        marking_point_weight: float = -0.004,
        distance_band_weight: float = -0.003,
        goal_side_bonus: float = 0.5,
        goal_side_penalty: float = -0.5,
        acceleration_penalty: float = -0.002,
        out_of_bounds_penalty: float = -10.0,
        collision_penalty: float = -5.0,
        boundary_proximity_weight: float = -0.02,
        boundary_proximity_margin_cm: float = 300.0,
        ideal_mark_distance_cm: float = 180.0,
        collision_distance_cm: float = 50.0,
        court_x_min: float = -1950.0,
        court_x_max: float = 1950.0,
        court_y_min: float = -950.0,
        court_y_max: float = 950.0,
    ):
        self.marking_point_weight = marking_point_weight
        self.distance_band_weight = distance_band_weight
        self.goal_side_bonus = goal_side_bonus
        self.goal_side_penalty = goal_side_penalty
        self.acceleration_penalty = acceleration_penalty
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.collision_penalty = collision_penalty
        self.boundary_proximity_weight = boundary_proximity_weight
        self.boundary_proximity_margin_cm = boundary_proximity_margin_cm
        self.ideal_mark_distance_cm = ideal_mark_distance_cm
        self.collision_distance_cm = collision_distance_cm
        self.court_x_min = court_x_min
        self.court_x_max = court_x_max
        self.court_y_min = court_y_min
        self.court_y_max = court_y_max

    def compute(
        self,
        *,
        player_pos: tuple[float, float],
        target_pos: tuple[float, float],
        own_goal_pos: tuple[float, float],
        ball_pos: tuple[float, float],
        all_player_positions: dict[str, tuple[float, float]],
        agent_id: str,
        prev_velocity: tuple[float, float],
        current_velocity: tuple[float, float],
        fps: int = 30,
        out_of_bounds: bool = False,
        collisions: list[str] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Compute the dense reward for the current timestep.

        Returns:
            (total_reward, component_dict)
        """
        components: dict[str, float] = {}

        # ── Marking point reward ─────────────────────────────────
        # The ideal marking position is along the line from own_goal
        # to target, at ideal_mark_distance_cm behind the target
        # (between target and own goal)

        dx = target_pos[0] - own_goal_pos[0]
        dy = target_pos[1] - own_goal_pos[1]
        dist_goal_to_target = math.hypot(dx, dy)
        if dist_goal_to_target < 1e-6:
            # Target at goal line; mark directly
            marking_pos = (
                target_pos[0] + own_goal_pos[0],
                target_pos[1] + own_goal_pos[1],
            )
        else:
            # Marking position = own_goal + (target - own_goal) normalized
            # at a distance of (dist_goal_to_target - ideal_mark_distance)
            # behind the target (closer to own goal)
            dir_to_target = (dx / dist_goal_to_target, dy / dist_goal_to_target)
            # Position behind target toward own goal
            marking_pos = (
                target_pos[0] - dir_to_target[0] * self.ideal_mark_distance_cm,
                target_pos[1] - dir_to_target[1] * self.ideal_mark_distance_cm,
            )

        dist_to_marking = math.hypot(
            player_pos[0] - marking_pos[0],
            player_pos[1] - marking_pos[1],
        )
        components["r_marking_point"] = self.marking_point_weight * dist_to_marking

        # ── Distance band reward ─────────────────────────────────
        dist_to_target = math.hypot(
            player_pos[0] - target_pos[0],
            player_pos[1] - target_pos[1],
        )
        components["r_distance_band"] = (
            self.distance_band_weight
            * abs(dist_to_target - self.ideal_mark_distance_cm)
        )

        # ── Goal-side positioning ─────────────────────────────────
        # Check if player is between target and own goal
        # Using dot product: if (target - player) · (own_goal - player) < 0,
        # the player is between them (the two vectors point in different directions)
        vec_to_target = (target_pos[0] - player_pos[0], target_pos[1] - player_pos[1])
        vec_to_goal = (own_goal_pos[0] - player_pos[0], own_goal_pos[1] - player_pos[1])
        dot_product = (
            vec_to_target[0] * vec_to_goal[0]
            + vec_to_target[1] * vec_to_goal[1]
        )

        # Also check that player is closer to own goal than target is
        dist_player_to_goal = math.hypot(
            player_pos[0] - own_goal_pos[0],
            player_pos[1] - own_goal_pos[1],
        )

        is_goal_side = dot_product > 0 and dist_player_to_goal < dist_goal_to_target
        components["r_goal_side"] = (
            self.goal_side_bonus if is_goal_side else self.goal_side_penalty
        )

        # ── Smoothness (acceleration penalty) ────────────────────
        acc_x = (current_velocity[0] - prev_velocity[0]) * fps
        acc_y = (current_velocity[1] - prev_velocity[1]) * fps
        acc_mag = math.hypot(acc_x, acc_y)
        components["r_smoothness"] = self.acceleration_penalty * (acc_mag / 100.0)

        # ── Boundary penalty ─────────────────────────────────────
        components["r_boundary"] = self.out_of_bounds_penalty if out_of_bounds else 0.0

        # ── Boundary proximity penalty (gradual near edges) ──────
        px, py = player_pos
        dist_to_left = px - self.court_x_min
        dist_to_right = self.court_x_max - px
        dist_to_bottom = py - self.court_y_min
        dist_to_top = self.court_y_max - py
        min_edge_dist = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)

        if min_edge_dist < self.boundary_proximity_margin_cm and not out_of_bounds:
            proximity = 1.0 - min_edge_dist / self.boundary_proximity_margin_cm
            components["r_boundary_proximity"] = self.boundary_proximity_weight * proximity * proximity
        else:
            components["r_boundary_proximity"] = 0.0

        # ── Collision penalty ───────────────────────────────────
        if collisions and len(collisions) > 0:
            components["r_collision"] = self.collision_penalty * len(collisions)
        else:
            components["r_collision"] = 0.0

        total = sum(components.values())
        return total, components

    def get_config(self) -> dict[str, Any]:
        """Return reward config for serialization."""
        return {
            "marking_point_weight": self.marking_point_weight,
            "distance_band_weight": self.distance_band_weight,
            "goal_side_bonus": self.goal_side_bonus,
            "goal_side_penalty": self.goal_side_penalty,
            "acceleration_penalty": self.acceleration_penalty,
            "out_of_bounds_penalty": self.out_of_bounds_penalty,
            "collision_penalty": self.collision_penalty,
            "boundary_proximity_weight": self.boundary_proximity_weight,
            "boundary_proximity_margin_cm": self.boundary_proximity_margin_cm,
            "ideal_mark_distance_cm": self.ideal_mark_distance_cm,
            "collision_distance_cm": self.collision_distance_cm,
            "court_x_min": self.court_x_min,
            "court_x_max": self.court_x_max,
            "court_y_min": self.court_y_min,
            "court_y_max": self.court_y_max,
        }
