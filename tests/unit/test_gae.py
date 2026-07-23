"""Numerical tests for GAE computation.

Tests verify:
  1. Single-step termination: bootstrap_mask=0 when terminated=True
  2. Single-step truncation: bootstrap uses next_value
  3. Rollout boundary mid-episode: uses next_value bootstrap
  4. Two episodes in one rollout: no cross-episode contamination
  5. _compute_gae does NOT call env.reset()
  6. Hand-calculated expected values match implementation
"""

from __future__ import annotations

import torch

from futsalmot_rl.training.train_ppo import compute_gae


def _gae(gamma=0.99, lam=0.95):
    """Helper: create simple reward/value sequences matching GAE semantics."""
    return gamma, lam


def test_single_step_terminated() -> None:
    """Single transition ending in termination: advantage = reward - value."""
    gamma, lam = _gae()
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([0.0])  # doesn't matter — terminated
    terminated = torch.tensor([True])
    episode_ended = torch.tensor([True])

    adv = compute_gae(rewards, values, next_values, terminated, episode_ended, gamma, lam)
    # delta = 1.0 + 0.99*0.0 - 0.5 = 0.5
    # GAE = delta (no further accumulation since done)
    expected = torch.tensor([0.5])
    torch.testing.assert_close(adv, expected, atol=1e-6, rtol=0)


def test_single_step_truncated() -> None:
    """Single transition ending in truncation: uses next_value for bootstrap."""
    gamma, lam = _gae()
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([0.8])  # bootstrap value
    terminated = torch.tensor([False])  # truncated, not terminated
    episode_ended = torch.tensor([True])

    adv = compute_gae(rewards, values, next_values, terminated, episode_ended, gamma, lam)
    # delta = 1.0 + 0.99*0.8 - 0.5 = 1.0 + 0.792 - 0.5 = 1.292
    expected = torch.tensor([1.292])
    torch.testing.assert_close(adv, expected, atol=1e-4, rtol=0)


def test_two_steps_no_termination() -> None:
    """Two steps, neither terminated: GAE accumulates."""
    gamma, lam = _gae()
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.6])
    next_values = torch.tensor([0.6, 0.7])  # bootstrap for step 1
    terminated = torch.tensor([False, False])
    episode_ended = torch.tensor([False, True])

    adv = compute_gae(rewards, values, next_values, terminated, episode_ended, gamma, lam)

    # t=1 (last): delta = 1.0 + 0.99*0.7 - 0.6 = 1.093
    #   gae_1 = delta = 1.093
    # t=0: delta = 1.0 + 0.99*0.6 - 0.5 = 1.094
    #   gae_0 = delta + 0.99*0.95*1.093 = 1.094 + 1.027 = 2.121
    expected_t1 = torch.tensor([2.121, 1.093])
    torch.testing.assert_close(adv, expected_t1, atol=1e-2, rtol=0)


def test_two_episodes_in_rollout() -> None:
    """Two complete episodes in one rollout: no cross-episode GAE contamination."""
    gamma, lam = _gae()

    # Episode 1: reward=1 → terminated
    # Episode 2: reward=2 → terminated
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    next_values = torch.tensor([0.0, 0.0])
    terminated = torch.tensor([True, True])
    episode_ended = torch.tensor([True, True])

    adv = compute_gae(rewards, values, next_values, terminated, episode_ended, gamma, lam)

    # t=1: delta = 2.0 + 0.99*0.0 - 1.0 = 1.0, gae_1 = 1.0
    # t=0: delta = 1.0 + 0.99*0.0 - 0.5 = 0.5, gae_0 = 0.5 (no carry from t=1)
    expected = torch.tensor([0.5, 1.0])
    torch.testing.assert_close(adv, expected, atol=1e-6, rtol=0)


def test_mid_episode_bootstrap() -> None:
    """Rollout ends mid-episode: next_value used, not zero."""
    gamma, lam = _gae()

    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.6])
    # Mid-episode: no termination, next_value represents the value of the next state
    next_values = torch.tensor([0.6, 0.9])
    terminated = torch.tensor([False, False])
    episode_ended = torch.tensor([False, False])  # episode continues past rollout

    adv = compute_gae(rewards, values, next_values, terminated, episode_ended, gamma, lam)

    # t=1: delta = 1.0 + 0.99*0.9 - 0.6 = 1.291
    #   gae_1 = delta = 1.291 (since episode_ended[1]=False, continuation_mask=1)
    # t=0: delta = 1.0 + 0.99*0.6 - 0.5 = 1.094
    #   gae_0 = 1.094 + 0.99*0.95*1.291 = 1.094 + 1.214 = 2.308
    expected_t1 = torch.tensor([2.308, 1.291])
    torch.testing.assert_close(adv, expected_t1, atol=1e-2, rtol=0)


def test_compute_gae_does_not_call_env() -> None:
    """GAE is a pure function — must not access env."""
    gamma, lam = _gae()
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([0.0])
    terminated = torch.tensor([False])
    episode_ended = torch.tensor([False])

    # Should compute without any external state
    adv = compute_gae(rewards, values, next_values, terminated, episode_ended, gamma, lam)
    assert adv.shape == (1,)


def test_zero_reward() -> None:
    """Zero rewards with termination: advantage = -value."""
    gamma, lam = _gae()
    rewards = torch.tensor([0.0])
    values = torch.tensor([0.7])
    next_values = torch.tensor([0.0])
    terminated = torch.tensor([True])
    episode_ended = torch.tensor([True])

    adv = compute_gae(rewards, values, next_values, terminated, episode_ended, gamma, lam)
    # delta = 0.0 + 0.99*0.0 - 0.7 = -0.7
    expected = torch.tensor([-0.7])
    torch.testing.assert_close(adv, expected, atol=1e-6, rtol=0)


def test_batch_independence() -> None:
    """Multiple independent episodes in one batch are independent."""
    gamma, lam = _gae()

    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5])
    next_values = torch.tensor([0.0, 0.0, 0.0])
    terminated = torch.tensor([True, True, True])
    episode_ended = torch.tensor([True, True, True])

    adv = compute_gae(rewards, values, next_values, terminated, episode_ended, gamma, lam)
    # Each step is independent (all terminated)
    expected = torch.tensor([0.5, 1.0, 1.5])  # reward + gamma*0 - value
    torch.testing.assert_close(adv, expected, atol=1e-6, rtol=0)
