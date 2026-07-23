"""Tests for shared / non-shared backbone consistency."""

from __future__ import annotations

import torch
import numpy as np
from futsalmot_rl.models.mlp_policy import MLPActorCritic


def _make_ac(shared: bool) -> MLPActorCritic:
    return MLPActorCritic(4, act_dim=2, shared_backbone=shared)


def test_shared_forward():
    ac = _make_ac(True)
    obs = torch.randn(4)
    action, value = ac.forward(obs.unsqueeze(0))
    assert action.shape == (1, 2)
    assert value.shape == (1, 1)
    assert torch.all(action >= -1) and torch.all(action <= 1)


def test_non_shared_forward():
    ac = _make_ac(False)
    obs = torch.randn(4)
    action, value = ac.forward(obs.unsqueeze(0))
    assert action.shape == (1, 2)
    assert value.shape == (1, 1)
    assert torch.all(action >= -1) and torch.all(action <= 1)


def test_shared_get_action_and_value():
    ac = _make_ac(True)
    obs = torch.randn(4)
    a, lp, ent, val = ac.get_action_and_value(obs.unsqueeze(0))
    assert a.shape == (1, 2)
    assert lp.shape == (1,)
    assert ent.shape == (1,)
    assert val.shape == (1, 1)


def test_shared_get_value():
    ac = _make_ac(True)
    obs = torch.randn(4)
    val = ac.get_value(obs.unsqueeze(0))
    assert val.shape == (1, 1)


def test_shared_get_action_numpy():
    ac = _make_ac(True)
    obs = np.random.randn(4).astype(np.float32)
    action = ac.get_action(obs, deterministic=True)
    assert action.shape == (2,)
    assert np.all(action >= -1) and np.all(action <= 1)


def test_non_shared_get_action_numpy():
    ac = _make_ac(False)
    obs = np.random.randn(4).astype(np.float32)
    action = ac.get_action(obs, deterministic=True)
    assert action.shape == (2,)
    assert np.all(action >= -1) and np.all(action <= 1)


def test_shared_batch():
    ac = _make_ac(True)
    batch = torch.randn(8, 4)
    actions, log_probs, entropies, values = ac.get_action_and_value(batch)
    assert actions.shape == (8, 2)
    assert log_probs.shape == (8,)
    assert entropies.shape == (8,)
    assert values.shape == (8, 1)


def test_non_shared_batch():
    ac = _make_ac(False)
    batch = torch.randn(8, 4)
    actions, log_probs, entropies, values = ac.get_action_and_value(batch)
    assert actions.shape == (8, 2)
    assert log_probs.shape == (8,)
    assert entropies.shape == (8,)
    assert values.shape == (8, 1)
