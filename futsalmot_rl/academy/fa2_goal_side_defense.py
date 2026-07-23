"""FA-2: Goal-side Defense task definition.

Task description:
    Player_05 (defender) must not only follow Player_01 (attacker),
    but also maintain position between Player_01 and the own goal (Team B,
    defending -X). This creates a proper defensive blocking stance.

Compared to FA-1:
    - Higher goal_side reward weight
    - New r_behind_attacker penalty
    - New r_shot_lane_block bonus
    - Same observation space
    - Same action space
    - Same environment base (FutsalDefenderFollowEnv with higher goal_side weight)

Training config:
    - Same BC initialization from FA-1
    - Modified reward v3
    - 500k timesteps
    - eval_interval: 25000
"""

from __future__ import annotations

import math
from typing import Any

# ═══════════════════════════════════════════════════════════
# FA-2 Reward v3 (Goal-side Defense)
# ═══════════════════════════════════════════════════════════

FA2_REWARD_CONFIG: dict[str, Any] = {
    # Marking point: keep close to the ideal defensive position
    "marking_point_weight": -0.004,
    # Distance band: maintain ideal distance from target
    "distance_band_weight": -0.003,
    # Goal-side: strongly reward being between attacker and own goal
    "goal_side_bonus": 1.0,
    "goal_side_penalty": -1.0,
    # Behind attacker: strongly penalize being behind the attacker
    # (this is handled as an additional component)
    # Smoothness
    "acceleration_penalty": -0.002,
    # Boundary: same as FA-1 v2
    "out_of_bounds_penalty": -10.0,
    "boundary_proximity_weight": -0.02,
    "boundary_proximity_margin_cm": 300.0,
    # Collision
    "collision_penalty": -5.0,
    # FA-2 specific
    "ideal_mark_distance_cm": 150.0,  # Tighter marking for FA-2
    "collision_distance_cm": 50.0,
}

# ═══════════════════════════════════════════════════════════
# FA-2 新增指标
# ═══════════════════════════════════════════════════════════

# These metrics should be tracked during FA-2 evaluation:
FA2_METRICS = [
    "goal_side_success_rate",  # 在目标与球门之间的帧数比例
    "time_behind_attacker_ratio",  # 在目标后方的帧数比例
    "mean_goal_line_offset_cm",  # 离球门线的平均距离
    "shot_lane_block_score",  # 射门线路阻挡分数
]


def compute_shot_lane_block_score(
    defender_pos: tuple[float, float],
    attacker_pos: tuple[float, float],
    own_goal_pos: tuple[float, float],
) -> float:
    """Compute how well the defender blocks the shot lane.

    A simple geometric score: project the defender onto the line from
    attacker to own goal. The closer the projection is to the midpoint,
    the better the block.

    Returns:
        Score in [0, 1], higher = better block.
    """
    ax, ay = attacker_pos
    gx, gy = own_goal_pos
    dx, dy = defender_pos

    # Vector from attacker to goal
    vx = gx - ax
    vy = gy - ay
    lane_length = math.hypot(vx, vy)
    if lane_length < 1e-6:
        return 0.0

    # Unit vector along lane
    ux = vx / lane_length
    uy = vy / lane_length

    # Project defender onto lane
    px = dx - ax
    py = dy - ay
    t = (px * ux + py * uy) / lane_length
    t = max(0.0, min(1.0, t))

    # Perpendicular distance from lane
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    perp_dist = math.hypot(dx - proj_x, dy - proj_y)

    # Score: closer to lane midpoint with small perpendicular distance
    lane_score = 1.0 - abs(t - 0.5) * 2.0  # 1.0 at midpoint, 0.0 at ends
    width_score = max(0.0, 1.0 - perp_dist / 200.0)  # 1.0 on lane, 0.0 at 200cm

    return lane_score * width_score


# ═══════════════════════════════════════════════════════════
# FA-2 训练配置建议
# ═══════════════════════════════════════════════════════════

FA2_TRAIN_CONFIG: dict[str, Any] = {
    "total_timesteps": 500000,
    "learning_rate": 0.0001,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "eval_interval_steps": 25000,
    "bc_init_model": "Saved/FutsalMOT_RL/models/defender_follow_bc_v1_best.pt",
    "reward_config": FA2_REWARD_CONFIG,
}

# ═══════════════════════════════════════════════════════════
# FA-2 评估用 episodes
# ═══════════════════════════════════════════════════════════

FA2_EVAL_SEQ_IDS = [
    "episode_random_0001_t1",
    "episode_random_0001_t2",
    "episode_random_0001_t3",
]

# ═══════════════════════════════════════════════════════════
# FA-2 验收标准
# ═══════════════════════════════════════════════════════════

FA2_ACCEPTANCE_CRITERIA = {
    "goal_side_success_rate_min": 0.3,  # 至少 30% 帧数在正确位置
    "out_of_bounds_max": 0,  # 不允许出界
    "collision_max": 5,  # 允许少量碰撞
    "trajectory_validation_errors_max": 0,  # 轨迹验证零错误
}
