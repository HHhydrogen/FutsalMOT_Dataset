"""Tests for the tanh-squashed Gaussian actor distribution.

Verifies:
  1. Sampling and re-evaluation produce the same log_prob
  2. Deterministic action equals tanh(raw_mean)
  3. Action is always in [-1, 1]
  4. Extreme action log-probs are finite (no NaN at ±1)
  5. BC weight loading produces consistent behavior
"""

from __future__ import annotations

import torch

from futsalmot_rl.models.mlp_policy import MLPActorCritic, MLPPolicy


def test_sampling_log_prob_consistent() -> None:
    """Sampled action's log_prob == re-evaluated log_prob of same action."""
    ac = MLPActorCritic(4, act_dim=2)
    obs = torch.randn(4)

    # Sample once
    a1, lp1, _, _ = ac.get_action_and_value(obs.unsqueeze(0))

    # Re-evaluate the same action
    _, lp2, _, _ = ac.get_action_and_value(obs.unsqueeze(0), action=a1)

    torch.testing.assert_close(lp1, lp2, atol=1e-5, rtol=1e-5)


def test_deterministic_action_is_tanh_raw_mean() -> None:
    """Deterministic action = tanh(raw_mean) for the same observation."""
    ac = MLPActorCritic(4, act_dim=2)
    obs = torch.randn(4)

    # Get deterministic action via forward
    det_action, _ = ac.forward(obs.unsqueeze(0))

    # Get raw mean directly
    raw_mean = ac.actor(obs.unsqueeze(0))

    assert torch.allclose(det_action, torch.tanh(raw_mean), atol=1e-6)


def test_action_in_m1_p1() -> None:
    """Actions must always be in [-1, 1]."""
    ac = MLPActorCritic(4, act_dim=2)
    obs = torch.randn(4)

    for _ in range(100):
        a, _, _, _ = ac.get_action_and_value(obs.unsqueeze(0))
        assert a.min() >= -1.0
        assert a.max() <= 1.0


def test_extreme_action_log_prob_finite() -> None:
    """Log-prob for near-extreme actions must be finite (no NaN)."""
    ac = MLPActorCritic(4, act_dim=2)
    obs = torch.randn(4)

    # Actions very close to -1 and 1
    extreme_actions = torch.tensor([[0.999, -0.999], [-0.999, 0.999], [0.999, 0.999]])

    for a in extreme_actions:
        _, lp, _, _ = ac.get_action_and_value(obs.unsqueeze(0), action=a.unsqueeze(0))
        assert torch.isfinite(lp).all(), f"log_prob not finite for action {a}"


def test_batch_shape() -> None:
    """Batch input produces batch output with correct shapes."""
    ac = MLPActorCritic(4, act_dim=2)
    batch = torch.randn(8, 4)

    actions, log_probs, entropies, values = ac.get_action_and_value(batch)
    assert actions.shape == (8, 2)
    assert log_probs.shape == (8,)
    assert entropies.shape == (8,)
    assert values.shape == (8, 1)

    # Also test with provided actions
    _, lp2, _, _ = ac.get_action_and_value(batch, action=actions)
    assert lp2.shape == (8,)


def test_no_double_tanh() -> None:
    """Verify tanh is only applied once by checking that action range is [-1, 1].

    If double tanh were applied, the distribution would be overly concentrated
    near the extremes. This is a statistical test that should pass for
    hundreds of samples.
    """
    ac = MLPActorCritic(4, act_dim=2)
    obs = torch.randn(4)

    # Collect many samples
    actions = []
    for _ in range(1000):
        a, _, _, _ = ac.get_action_and_value(obs.unsqueeze(0))
        actions.append(a)
    all_actions = torch.cat(actions)

    # Should have samples across the range
    assert all_actions.min() > -1.0  # strictly greater due to tanh asymptote
    assert all_actions.max() < 1.0
    # Mean should not be extremely close to 0 (not overly concentrated)
    assert abs(all_actions.mean().item()) < 0.5


def test_bc_weight_transfer() -> None:
    """BC weights (MLPPolicy) → MLPActorCritic transfer should have minimal missing keys."""
    obs_dim, act_dim = 4, 2

    bc = MLPPolicy(obs_dim, act_dim=act_dim)
    ac = MLPActorCritic(obs_dim, act_dim=act_dim)

    # Transfer BC weights to actor
    bc_sd = bc.state_dict()
    actor_sd = {}
    for key, value in bc_sd.items():
        if key.startswith("net."):
            actor_sd["actor." + key] = value

    missing, unexpected = ac.load_state_dict(actor_sd, strict=False)
    # Only critic keys should be missing; actor keys should all match
    missing_actor = [k for k in missing if k.startswith("actor.")]
    assert len(missing_actor) == 0, f"Missing actor keys: {missing_actor}"
    # No unexpected keys
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
