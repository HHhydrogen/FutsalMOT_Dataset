"""PPO (Proximal Policy Optimization) training for FutsalMOT-RL.

Pure PyTorch implementation — no dependency on stable-baselines3.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from futsalmot_rl.core.rl_io import write_json_atomic
from futsalmot_rl.core.rl_paths import MODELS_DIR, TRAIN_LOGS_DIR, ensure_dirs
from futsalmot_rl.models.mlp_policy import MLPActorCritic
from futsalmot_rl.models.policy_io import save_policy


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    episode_ended: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Generalized Advantage Estimation — pure function, no env access.

    Args:
        rewards: (n,) tensor of rewards.
        values: (n,) tensor of value estimates V(s_t).
        next_values: (n,) tensor of value estimates V(s_{t+1}) for bootstrap.
        terminated: (n,) bool — True where episode ended naturally.
        episode_ended: (n,) bool — True where episode ended (terminated or truncated
                      or rollout boundary). Used to stop GAE accumulation.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter.

    Returns:
        (n,) tensor of raw (un-normalized) advantages.
    """
    n = len(rewards)
    advantages = torch.zeros(n, device=rewards.device)
    last_gae = 0.0

    for t in reversed(range(n)):
        bootstrap_mask = 1.0 - terminated[t].float()
        continuation_mask = 1.0 - episode_ended[t].float()

        delta = rewards[t] + gamma * next_values[t] * bootstrap_mask - values[t]
        last_gae = delta + gamma * gae_lambda * continuation_mask * last_gae
        advantages[t] = last_gae

    return advantages


class PPOTrainer:
    """PPO trainer for continuous control in FutsalDefenderFollowEnv.

    Uses an MLPActorCritic network with a fixed-log-std Gaussian policy.
    """

    def __init__(
        self,
        env: Any,
        eval_env: Any | None = None,
        config: dict[str, Any] | None = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.env = env
        self.eval_env = eval_env or env

        # ── Training config ──────────────────────────────────────
        self.cfg: dict[str, Any] = {
            "total_timesteps": 500000,
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": 0.02,
        }
        if config is not None:
            self.cfg.update(config)

        # ── Policy ───────────────────────────────────────────────
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.policy = MLPActorCritic(
            obs_dim,
            hidden_sizes=[128, 128],
            act_dim=act_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.cfg["learning_rate"],
        )

        # ── Video callback ───────────────────────────────────────
        self.video_callback: Callable | None = None

        # ── Rollout state (avoids resetting mid-episode) ────────
        self._current_obs: np.ndarray | None = None
        self._current_episode_reward: float = 0.0
        self._current_episode_length: int = 0
        self._rollout_started: bool = False

    def set_video_callback(self, callback: Callable) -> None:
        """Set a callback for recording evaluation videos."""
        self.video_callback = callback

    def load_pretrained(self, model_path: str | Path) -> None:
        """Load a pre-trained BC model as initialization.

        Transfers weights from an MLPPolicy (BC) to the actor part of the
        MLPActorCritic (PPO). The critic is initialized from scratch.
        """
        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=True)
        state_dict = checkpoint["model_state_dict"]

        # MLPPolicy stores weights under "net.K.weight" — map to "actor.net.K.weight"
        actor_state = {}
        for key, value in state_dict.items():
            if key.startswith("net."):
                actor_state["actor." + key] = value
            else:
                actor_state[key] = value

        # Load only matching keys (actor part); skip critic
        missing, unexpected = self.policy.load_state_dict(actor_state, strict=False)
        if missing:
            print(
                "  Critic layers initialized from scratch: {}".format(
                    [k for k in missing if k.startswith("critic.")]
                )
            )
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")
        print(f"  Loaded BC pretrained weights from {model_path}")

    def collect_rollout(self, n_steps: int) -> dict[str, torch.Tensor]:
        """Collect a rollout of n_steps from the environment.

        Each transition stores:
          obs, action, reward, value, log_prob, terminated, truncated,
          next_obs, next_value

        next_value is computed by the critic from the *real* next_obs
        before any env.reset() call, ensuring truncation bootstrap uses
        the correct state.

        Maintains rollout state across calls so episodes are not
        artificially truncated at rollout boundaries.
        """
        obs_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        rewards_list: list[float] = []
        terminated_list: list[bool] = []
        truncated_list: list[bool] = []
        values_list: list[float] = []
        log_probs_list: list[float] = []
        next_obs_list: list[np.ndarray] = []
        next_values_list: list[float] = []

        episode_rewards: list[float] = []

        # Reset only if needed
        if not self._rollout_started or self._current_obs is None:
            self._current_obs, info = self.env.reset()
            self._current_episode_reward = 0.0
            self._current_episode_length = 0
            self._rollout_started = True

        obs = self._current_obs

        for step in range(n_steps):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, _, value = self.policy.get_action_and_value(obs_tensor)

            obs_list.append(obs)
            actions_list.append(action.squeeze(0).cpu().numpy())
            values_list.append(value.squeeze(0).cpu().item())
            log_probs_list.append(log_prob.squeeze(0).cpu().item())

            next_obs, reward, terminated, truncated, info = self.env.step(
                action.squeeze(0).cpu().numpy()
            )
            rewards_list.append(float(reward))
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))

            # Compute next_value from the REAL next_observation
            # before any env.reset() — critical for truncation bootstrap
            next_obs_tensor = torch.from_numpy(next_obs).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.policy.get_value(next_obs_tensor).squeeze(0).cpu().item()
            next_obs_list.append(next_obs)
            next_values_list.append(next_value)

            self._current_episode_reward += float(reward)
            self._current_episode_length += 1

            if terminated or truncated:
                episode_rewards.append(self._current_episode_reward)
                self._current_episode_reward = 0.0
                self._current_episode_length = 0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        # Save state for next rollout call
        self._current_obs = obs

        # Build tensors
        terminated_t = torch.BoolTensor(terminated_list).to(self.device)
        truncated_t = torch.BoolTensor(truncated_list).to(self.device)

        data = {
            "obs": torch.FloatTensor(np.array(obs_list)).to(self.device),
            "actions": torch.FloatTensor(np.array(actions_list)).to(self.device),
            "rewards": torch.FloatTensor(rewards_list).to(self.device),
            "terminated": terminated_t,
            "truncated": truncated_t,
            "values": torch.FloatTensor(values_list).to(self.device),
            "log_probs": torch.FloatTensor(log_probs_list).to(self.device),
            "next_values": torch.FloatTensor(next_values_list).to(self.device),
            # Combined dones (kept for backward compat with train_step logging)
            "dones": terminated_t | truncated_t,
        }

        # Compute raw advantages with GAE using real next_values
        raw_advantages = self._compute_gae(
            data["rewards"],
            data["values"],
            data["next_values"],
            data["terminated"],
            data["truncated"],
            last_obs_tensor=torch.FloatTensor(obs).to(self.device),
            last_terminated=terminated_list[-1] if terminated_list else False,
        )
        data["raw_advantages"] = raw_advantages
        data["returns"] = raw_advantages + data["values"]

        return data, episode_rewards

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        last_obs_tensor: torch.Tensor,
        last_terminated: bool,
    ) -> torch.Tensor:
        """Compute GAE — delegates to standalone compute_gae().

        Uses the REAL next_values from the rollout (computed from the
        actual next_observation before any env.reset()).

        For the final transition, if terminated → bootstrap=0;
        otherwise bootstrap from the critic using last_obs.
        """
        gamma = self.cfg["gamma"]
        lam = self.cfg["gae_lambda"]
        n = len(rewards)

        with torch.no_grad():
            if last_terminated:
                final_value = 0.0
            else:
                final_value = self.policy.get_value(last_obs_tensor.unsqueeze(0)).squeeze(0).item()

        # Append bootstrap value for the final step's next_value
        next_values_full = torch.cat([next_values, torch.tensor([final_value], device=self.device)])

        episode_ended = terminated | truncated

        return compute_gae(
            rewards=rewards,
            values=values,
            next_values=next_values_full,
            terminated=terminated,
            episode_ended=episode_ended,
            gamma=gamma,
            gae_lambda=lam,
        )

    def train_step(self, data: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform one PPO update step on collected rollout data.

        Uses raw_advantages and returns from collect_rollout.
        Advantages are normalized per mini-batch for policy loss only.
        Critic loss uses raw (un-normalized) returns.
        """
        n = len(data["obs"])
        batch_size = self.cfg["batch_size"]
        n_epochs = self.cfg["n_epochs"]
        clip_range = self.cfg["clip_range"]
        target_kl = self.cfg.get("target_kl", 0.02)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        total_clip_frac = 0.0
        _kl_exceeded = False
        _last_approx_kl = 0.0

        raw_advantages = data["raw_advantages"]
        returns = data["returns"]

        # For explained variance at the end
        all_value_preds: list[torch.Tensor] = []

        # Create indices for mini-batches
        indices = np.arange(n)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                batch_indices_tensor = torch.LongTensor(batch_indices).to(self.device)

                obs_batch = data["obs"][batch_indices_tensor]
                actions_batch = data["actions"][batch_indices_tensor]
                # Critic target: raw returns (un-normalized)
                returns_batch = returns[batch_indices_tensor]
                old_log_probs_batch = data["log_probs"][batch_indices_tensor]

                # Policy advantages: normalized per batch (unbiased=False
                # prevents NaN when batch has 1 sample)
                batch_raw = raw_advantages[batch_indices_tensor]
                batch_mean = batch_raw.mean()
                batch_std = batch_raw.std(unbiased=False)
                if batch_raw.numel() > 1 and batch_std > 1e-8:
                    advantages_batch = (batch_raw - batch_mean) / (batch_std + 1e-8)
                else:
                    advantages_batch = batch_raw - batch_mean

                # Get new action log probs and values
                _, log_probs, entropy, values = self.policy.get_action_and_value(
                    obs_batch, actions_batch
                )

                # Ratio for PPO clipping
                ratio = torch.exp(log_probs - old_log_probs_batch)

                # Clipped surrogate objective
                pi_loss1 = -advantages_batch * ratio
                pi_loss2 = -advantages_batch * torch.clamp(
                    ratio, 1.0 - clip_range, 1.0 + clip_range
                )
                pi_loss = torch.mean(torch.max(pi_loss1, pi_loss2))

                # Value function loss — uses raw returns (un-normalized)
                v_loss = F.mse_loss(values.squeeze(), returns_batch)

                # Entropy bonus
                entropy_mean = entropy.mean()

                # Total loss
                loss = (
                    pi_loss
                    + self.cfg["value_coef"] * v_loss
                    - self.cfg["entropy_coef"] * entropy_mean
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg["max_grad_norm"])
                self.optimizer.step()

                total_pi_loss += pi_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy_mean.item()
                total_clip_frac += ((ratio - 1.0).abs() > clip_range).float().mean().item()
                all_value_preds.append(values.detach().squeeze())
                n_updates += 1

                # Early stopping via KL divergence — exits both loops
                if target_kl > 0.0:
                    with torch.no_grad():
                        _last_approx_kl = ((ratio - 1.0) - ratio.log()).mean().item()
                        if _last_approx_kl > target_kl:
                            _kl_exceeded = True
                            break
            if _kl_exceeded:
                break
        else:
            _kl_exceeded = False
        # Reset for next iteration
        kl_stopped = _kl_exceeded

        # Explained variance
        if all_value_preds:
            vpred = torch.cat(all_value_preds)
            with torch.no_grad():
                resid = returns - vpred
                var_resid = resid.pow(2).sum()
                var_returns = (returns - returns.mean()).pow(2).sum().clamp(min=1e-8)
                explained_variance = 1.0 - var_resid / var_returns
        else:
            explained_variance = 0.0

        # Gradient norm
        total_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        grad_norm = total_norm**0.5

        return {
            "pi_loss": total_pi_loss / max(1, n_updates),
            "v_loss": total_v_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
            "approx_kl": float(_last_approx_kl),
            "clip_fraction": total_clip_frac / max(1, n_updates),
            "explained_variance": float(explained_variance),
            "gradient_norm": grad_norm,
            "kl_stopped": kl_stopped,
        }

    def train(
        self,
        total_timesteps: int | None = None,
        log_dir: str | Path = TRAIN_LOGS_DIR / "ppo",
        model_dir: str | Path = MODELS_DIR,
        eval_interval: int = 25000,
    ) -> dict[str, Any]:
        """Run the full PPO training loop.

        Args:
            total_timesteps: Total environment steps to train for.
            log_dir: Directory for training logs.
            model_dir: Directory for saving models.
            eval_interval: Evaluate every N steps.

        Returns:
            Training summary.
        """
        total_timesteps = total_timesteps or self.cfg["total_timesteps"]
        n_steps = self.cfg["n_steps"]

        log_dir = Path(log_dir)
        model_dir = Path(model_dir)
        ensure_dirs()
        log_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / "train_log.jsonl"
        best_model_path = model_dir / "defender_follow_ppo_v1_best.pt"
        latest_model_path = model_dir / "defender_follow_ppo_v1_latest.pt"

        # Training state
        global_step = 0
        best_reward = -float("inf")
        summary: dict[str, Any] = {
            "config": self.cfg,
            "iterations": [],
            "best_mean_reward": None,
            "best_iteration": None,
            "total_train_time_s": 0.0,
        }

        train_start = time.time()
        iteration = 0

        print("=" * 60)
        print("PPO Training")
        print(
            f"Total timesteps: {total_timesteps}  Steps per iter: {n_steps}  Device: {self.device}"
        )
        print("=" * 60)

        while global_step < total_timesteps:
            iteration += 1

            # Collect rollout
            rollout_data, ep_rewards = self.collect_rollout(n_steps)
            global_step += n_steps

            # PPO update
            loss_metrics = self.train_step(rollout_data)

            # Logging
            mean_ep_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            log_entry = {
                "iteration": iteration,
                "global_step": global_step,
                "mean_episode_reward": mean_ep_reward,
                "pi_loss": loss_metrics["pi_loss"],
                "v_loss": loss_metrics["v_loss"],
                "entropy": loss_metrics["entropy"],
                "n_episodes": len(ep_rewards),
            }

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            summary["iterations"].append(log_entry)

            if mean_ep_reward > best_reward:
                best_reward = mean_ep_reward
                summary["best_mean_reward"] = best_reward
                summary["best_iteration"] = iteration
                save_policy(self.policy, best_model_path, config=self.cfg)

            # Print progress
            if iteration % 5 == 0 or iteration == 1:
                print(
                    "  Iter {}: step={} mean_reward={:.3f} pi_loss={:.4f} v_loss={:.4f}".format(
                        iteration,
                        global_step,
                        mean_ep_reward,
                        loss_metrics["pi_loss"],
                        loss_metrics["v_loss"],
                    )
                )

            # Evaluation + video
            if self.video_callback is not None and global_step % eval_interval < n_steps:
                try:
                    self.video_callback(self.policy, global_step, self.device)
                except Exception as exc:
                    print(f"    [Eval] Failed: {exc}")

        # Save final model
        save_policy(self.policy, latest_model_path, config=self.cfg)

        total_time = time.time() - train_start
        summary["total_train_time_s"] = total_time
        summary["total_iterations"] = iteration
        summary["total_steps"] = global_step

        print(f"\nTraining complete ({total_time:.1f}s)")
        print(
            "Best mean reward: {:.3f} (iteration {})".format(
                best_reward, summary.get("best_iteration", 0)
            )
        )
        print(f"Best model: {best_model_path}")
        print(f"Latest model: {latest_model_path}")

        # Generate reward curve
        try:
            _plot_reward_curve(log_path, log_dir / "reward_curve.png")
        except Exception as exc:
            print(f"  [WARNING] Reward curve plot failed: {exc}")

        # Save summary
        summary_path = log_dir / "ppo_summary.json"
        write_json_atomic(summary_path, summary)
        print(f"Summary: {summary_path}")

        return summary


def _plot_reward_curve(log_path: Path, output_path: Path) -> None:
    """Plot training reward curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps: list[int] = []
    rewards: list[float] = []
    pi_losses: list[float] = []
    v_losses: list[float] = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                steps.append(entry["global_step"])
                rewards.append(entry["mean_episode_reward"])
                pi_losses.append(entry["pi_loss"])
                v_losses.append(entry["v_loss"])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Reward
    ax = axes[0]
    ax.plot(steps, rewards, label="Mean Episode Reward", color="#4CAF50")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title("PPO Training Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Losses
    ax = axes[1]
    ax.plot(steps, pi_losses, label="Policy Loss", color="#2196F3", alpha=0.7)
    ax.plot(steps, v_losses, label="Value Loss", color="#FF6F00", alpha=0.7)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("PPO Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
