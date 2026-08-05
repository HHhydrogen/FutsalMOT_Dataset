"""稳定的、版本化的随机种子派生体系。

GRF 引擎只接受一个 `game_engine_random_seed`。为建立可验证的随机种子体系，
从用户提供的 root_seed 通过 SHA-256 派生一组命名空间子 seed：

    grf_game_engine — 真正传入 GRF 的 game_engine_random_seed
    python          — Python 标准库 random.seed
    numpy           — numpy.random.seed
    ue_visual       — 预留：供未来 UE 视觉随机化使用（本轮仅记录，不消费）

派生算法带 policy 前缀（SEED_POLICY）。改变派生算法时必须使用新的 policy
名称，不得静默改变既有 v1 语义。不使用 Python 内置 hash()——其结果跨进程、
跨平台不稳定，不适合作为持久协议。

确定性保证：相同 root_seed + namespace + policy ⇒ 相同子 seed，跨进程、跨机器一致。
"""

from __future__ import annotations

import hashlib
from typing import Dict

from pydantic import BaseModel, Field

SEED_POLICY = "futsalmot_seed_v1"

# 支持的命名空间。新增命名空间时需同时扩展 EpisodeSeeds 与派生。
SEED_NAMESPACES = (
    "grf_game_engine",
    "python",
    "numpy",
    "ue_visual",
)

_MAX_SEED = 0x7FFFFFFF


def derive_seed(root_seed: int, namespace: str) -> int:
    """从 root_seed 派生指定 namespace 的稳定子 seed。

    Args:
        root_seed: 用户提供的主 seed（整数）。
        namespace: 命名空间名，如 "grf_game_engine"。

    Returns:
        位于 [0, 2^31) 的非负整数（目标库安全范围）。
    """
    root_seed = int(root_seed)
    payload = f"{SEED_POLICY}:{root_seed}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) & _MAX_SEED


class EpisodeSeeds(BaseModel):
    """一个 episode 的完整随机种子集合（写入 meta.randomness）。"""

    policy: str = Field(SEED_POLICY, description="seed 派生算法版本标识")
    root_seed: int = Field(..., description="用户提供的根 seed")
    grf_game_engine_seed: int = Field(
        ..., description="真正传入 GRF 的 game_engine_random_seed"
    )
    python_seed: int = Field(..., description="Python 标准库 random.seed")
    numpy_seed: int = Field(..., description="NumPy random.seed")
    ue_visual_seed: int = Field(
        ..., description="预留：未来 UE 视觉随机化用（本轮仅记录）"
    )

    def sub_seeds(self) -> Dict[str, int]:
        """返回 {namespace: seed} 的映射（不含 policy/root）。"""
        return {
            "grf_game_engine_seed": self.grf_game_engine_seed,
            "python_seed": self.python_seed,
            "numpy_seed": self.numpy_seed,
            "ue_visual_seed": self.ue_visual_seed,
        }


def derive_episode_seeds(root_seed: int) -> EpisodeSeeds:
    """从一个 root_seed 派生全部命名空间子 seed。"""
    return EpisodeSeeds(
        policy=SEED_POLICY,
        root_seed=int(root_seed),
        grf_game_engine_seed=derive_seed(root_seed, "grf_game_engine"),
        python_seed=derive_seed(root_seed, "python"),
        numpy_seed=derive_seed(root_seed, "numpy"),
        ue_visual_seed=derive_seed(root_seed, "ue_visual"),
    )
