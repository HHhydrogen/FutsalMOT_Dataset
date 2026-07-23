"""PyTorch Dataset for demonstration data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from futsalmot_rl.core.rl_io import read_json


class DemoDataset(Dataset):
    """Dataset for behavior cloning from demonstration data.

    Loads demo-index.json and provides (obs, action) pairs.
    Episodes are split at episode boundaries — never across them.
    """

    def __init__(
        self,
        demo_index_path: str | Path,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            demo_index_path: Path to demo_index.json.
            split: One of 'train', 'val', 'test'.
            train_ratio: Fraction of episodes for training.
            val_ratio: Fraction of episodes for validation.
            seed: Random seed for deterministic split.
        """
        self.demo_index_path = Path(demo_index_path)
        self.index = read_json(self.demo_index_path)
        demos = list(self.index.get("demos", []))

        if not demos:
            raise ValueError(f"No demos found in {demo_index_path}")

        # Shuffle and split episodes deterministically
        rng = np.random.RandomState(seed)
        rng.shuffle(demos)

        n = len(demos)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if split == "train":
            selected = demos[:n_train]
        elif split == "val":
            selected = demos[n_train : n_train + n_val]
        elif split == "test":
            selected = demos[n_train + n_val :]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        # Load all transitions from selected episodes
        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []

        self._episode_ranges: list[tuple[int, int]] = []
        self._episode_ids: list[str] = []
        offset = 0

        for demo_entry in selected:
            path_str = demo_entry.get("path", "")
            demo_path = Path(path_str)
            if not demo_path.is_absolute():
                demo_path = self.demo_index_path.parent / demo_path.name

            if not demo_path.is_file():
                continue

            data = np.load(str(demo_path), allow_pickle=True)
            obs_arr = data["obs"]
            actions_arr = data["actions"]
            seq_id = str(data["seq_id"]) if data["seq_id"].ndim == 0 else "unknown"

            n_trans = len(obs_arr)
            if n_trans > 0:
                all_obs.append(obs_arr)
                all_actions.append(actions_arr)
                self._episode_ranges.append((offset, offset + n_trans))
                self._episode_ids.append(seq_id)
                offset += n_trans

            data.close()

        if not all_obs:
            raise ValueError(f"No valid demo data loaded for split '{split}'")

        self.obs = np.concatenate(all_obs, axis=0).astype(np.float32)
        self.actions = np.concatenate(all_actions, axis=0).astype(np.float32)
        self.n_episodes = len(self._episode_ranges)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
        )

    def get_episode_info(self) -> list[dict]:
        """Return info about episodes in this dataset."""
        return [
            {
                "seq_id": eid,
                "start": start,
                "end": end,
                "transitions": end - start,
            }
            for eid, (start, end) in zip(self._episode_ids, self._episode_ranges)
        ]
