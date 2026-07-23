"""Training callbacks for FutsalMOT-RL (video eval, model checkpoint)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from futsalmot_rl.core.rl_io import ensure_dir


class RLVideoEvalCallback:
    """Callback that records evaluation videos during PPO training.

    At a specified interval, runs a fixed set of eval episodes and records
    a 2D pitch video showing the current policy's behavior.
    """

    def __init__(
        self,
        eval_env: Any,
        video_dir: str | Path,
        eval_freq: int = 25000,
        video_fps: int = 15,
        n_eval_episodes: int = 1,
    ):
        """
        Args:
            eval_env: A separate eval environment (same class as train env).
            video_dir: Directory to save videos to.
            eval_freq: Evaluate every N training steps.
            video_fps: Video frame rate.
            n_eval_episodes: Number of eval episodes (only 1st is recorded).
        """
        self.eval_env = eval_env
        self.video_dir = Path(video_dir)
        self.eval_freq = eval_freq
        self.video_fps = video_fps
        self.n_eval_episodes = n_eval_episodes
        self._last_eval_step = 0

    def __call__(
        self,
        policy: torch.nn.Module,
        step: int,
        device: torch.device,
    ) -> dict[str, float] | None:
        """Evaluate policy and record video at the given step.

        Returns metrics dict if evaluation ran, None otherwise.
        """
        if step - self._last_eval_step < self.eval_freq and step > 0:
            return None

        self._last_eval_step = step

        try:
            metrics = self._evaluate(policy, device)
            self._record_video(policy, step, device)
            return metrics
        except Exception as exc:
            import traceback

            print(f"    [Eval] Failed: {exc}")
            traceback.print_exc()
            return None

    def _evaluate(self, policy: torch.nn.Module, device: torch.device) -> dict[str, float]:
        """Run evaluation episodes and return metrics."""
        total_rewards: list[float] = []
        episode_lengths: list[int] = []

        for _ep in range(self.n_eval_episodes):
            obs, _info = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            ep_len = 0

            while not done:
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    if hasattr(policy, "get_action"):
                        action = policy.get_action(obs, deterministic=True)
                    else:
                        action = policy(obs_tensor).squeeze(0).cpu().numpy()

                next_obs, reward, terminated, truncated, _info = self.eval_env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                done = terminated or truncated
                obs = next_obs

            total_rewards.append(ep_reward)
            episode_lengths.append(ep_len)

        return {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
        }

    def _record_video(self, policy: torch.nn.Module, step: int, device: torch.device) -> None:
        """Record a video of the current policy."""
        from futsalmot_rl.viz.video_recorder import record_episode_video

        seq_id = getattr(self.eval_env, "seq_id", "eval")
        output_path = self.video_dir / f"ppo_step_{step:06d}_{seq_id}.mp4"
        ensure_dir(output_path.parent)

        def policy_fn(obs, deterministic=True):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                if hasattr(policy, "get_action"):
                    return policy.get_action(obs, deterministic=deterministic)
                return policy(obs_tensor).squeeze(0).cpu().numpy()

        result = record_episode_video(
            self.eval_env,
            policy_fn,
            output_path,
            fps=self.video_fps,
            title=f"PPO Step {step}",
        )
        if result:
            print(f"    [Video] Saved {result}")
