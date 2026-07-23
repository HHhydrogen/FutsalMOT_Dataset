"""Tests for model device handling — tensors on correct device."""

from __future__ import annotations

import numpy as np
import torch
from futsalmot_rl.models.mlp_policy import MLPActorCritic, MLPPolicy


def test_cpu_mlppolicy_get_action():
    """MLPPolicy.get_action works with CPU model and numpy input."""
    policy = MLPPolicy(4, act_dim=2)
    policy.to("cpu")
    obs = np.random.randn(4).astype(np.float32)
    action = policy.get_action(obs, deterministic=True)
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


def test_cpu_actorcritic_get_action():
    """MLPActorCritic.get_action works with CPU model and numpy input."""
    ac = MLPActorCritic(4, act_dim=2)
    ac.to("cpu")
    obs = np.random.randn(4).astype(np.float32)
    action = ac.get_action(obs, deterministic=True)
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


def test_cpu_actorcritic_get_value():
    """MLPActorCritic.get_value works with CPU tensor."""
    ac = MLPActorCritic(4, act_dim=2).to("cpu")
    obs = torch.randn(1, 4)
    val = ac.get_value(obs)
    assert val.shape == (1, 1)
    assert torch.isfinite(val).all()


def test_cpu_actorcritic_get_action_and_value():
    """Full get_action_and_value cycle on CPU."""
    ac = MLPActorCritic(4, act_dim=2).to("cpu")
    obs = torch.randn(1, 4)
    a, lp, ent, val = ac.get_action_and_value(obs)
    assert torch.isfinite(a).all()
    assert torch.isfinite(lp).all()
    assert torch.isfinite(ent).all()
    assert torch.isfinite(val).all()


def test_policy_batch_cpu():
    """Batch inference on CPU."""
    ac = MLPActorCritic(4, act_dim=2).to("cpu")
    batch = torch.randn(8, 4)
    actions, log_probs, entropies, values = ac.get_action_and_value(batch)
    assert actions.shape == (8, 2)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probs).all()


def test_deterministic_stochastic_consistency():
    """Deterministic action should be close to tanh(raw_mean)."""
    ac = MLPActorCritic(4, act_dim=2).to("cpu")
    obs = torch.randn(4)
    raw_mean, _ = ac._raw_mean_and_value(obs.unsqueeze(0))
    det_action, _ = ac.forward(obs.unsqueeze(0))
    assert torch.allclose(det_action, torch.tanh(raw_mean), atol=1e-6)
