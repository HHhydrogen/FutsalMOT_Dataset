"""Deterministic seed utilities for reproducible RL experiments."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def deterministic_seed(base_seed: int, *extra: int) -> int:
    """Derive a deterministic RNG seed from a base seed and extra context.

    Uses SHA-256 so different extra values produce uncorrelated seeds.
    """
    payload = "|".join(str(int(v)) for v in (base_seed, *extra)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seed_all(seed: int) -> None:
    """Seed Python random, numpy, and (if available) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
