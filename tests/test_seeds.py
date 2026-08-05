"""seed 派生体系单元测试（默认套件，不运行 GRF 引擎）。

覆盖 6.1 派生函数（相同/不同 namespace/不同 root/范围/跨进程）以及
validator 的 meta.randomness 一致性检查与 legacy 兼容。
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from grf_ue_bridge.seeds import (
    SEED_NAMESPACES,
    SEED_POLICY,
    EpisodeSeeds,
    derive_episode_seeds,
    derive_seed,
)
from grf_ue_bridge.validator import _check_randomness


# ── derive_seed 单元测试 ─────────────────────────────────────────────

class TestDeriveSeed:
    def test_same_input_same_result(self):
        assert derive_seed(1001, "python") == derive_seed(1001, "python")

    def test_different_namespace_different_result(self):
        vals = {derive_seed(1001, ns) for ns in SEED_NAMESPACES}
        assert len(vals) == len(SEED_NAMESPACES)  # 全部互异

    def test_different_root_different_result(self):
        assert derive_seed(1001, "grf_game_engine") != derive_seed(1002, "grf_game_engine")

    def test_result_in_safe_range(self):
        for root in (0, 1, 42, 1001, 2**31 - 1, 2**40):
            for ns in SEED_NAMESPACES:
                v = derive_seed(root, ns)
                assert isinstance(v, int)
                assert 0 <= v <= 0x7FFFFFFF

    def test_nonnegative_root_normalized(self):
        # 负 root 仍应得到确定性、合法的结果
        assert derive_seed(-5, "python") == derive_seed(-5, "python")
        assert 0 <= derive_seed(-5, "python") <= 0x7FFFFFFF

    def test_cross_process_stable(self):
        # 独立 Python 进程计算结果一致（跨进程稳定协议）
        code = (
            "import sys; sys.path.insert(0, 'src'); "
            "from grf_ue_bridge.seeds import derive_seed; "
            "print(derive_seed(1001, 'grf_game_engine'), derive_seed(1001, 'numpy'))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd="."
        )
        assert proc.returncode == 0, proc.stderr
        p1, p2 = (int(x) for x in proc.stdout.split())
        assert (p1, p2) == (
            derive_seed(1001, "grf_game_engine"),
            derive_seed(1001, "numpy"),
        )


class TestDeriveEpisodeSeeds:
    def test_model_fields(self):
        s = derive_episode_seeds(1001)
        assert isinstance(s, EpisodeSeeds)
        assert s.policy == SEED_POLICY
        assert s.root_seed == 1001
        assert s.grf_game_engine_seed == derive_seed(1001, "grf_game_engine")
        assert s.python_seed == derive_seed(1001, "python")
        assert s.numpy_seed == derive_seed(1001, "numpy")
        assert s.ue_visual_seed == derive_seed(1001, "ue_visual")

    def test_serializable(self):
        s = derive_episode_seeds(1001)
        d = s.model_dump()
        assert set(d) == {
            "policy", "root_seed", "grf_game_engine_seed",
            "python_seed", "numpy_seed", "ue_visual_seed",
        }
        assert d["root_seed"] == 1001


# ── validator meta.randomness 一致性 ─────────────────────────────────

def _randomness_meta(root=1001, **over):
    """构造一份含正确 randomness 的 meta dict。"""
    s = derive_episode_seeds(root)
    r = s.model_dump()
    r.update(over)
    return {
        "source": {"seed": root},
        "randomness": r,
    }


class TestValidatorRandomness:
    def test_legacy_no_randomness_allowed(self):
        errors: list = []
        _check_randomness({"source": {"seed": 42}}, errors)
        assert errors == []  # legacy 不报错（validator 仅提示）

    def test_valid_randomness_passes(self):
        errors: list = []
        _check_randomness(_randomness_meta(1001), errors)
        assert errors == []

    def test_source_seed_mismatch(self):
        errors: list = []
        m = _randomness_meta(1001)
        m["source"]["seed"] = 999
        _check_randomness(m, errors)
        assert any("root_seed" in e for e in errors)

    def test_empty_policy_rejected(self):
        errors: list = []
        _check_randomness(_randomness_meta(1001, policy=""), errors)
        assert any("policy" in e for e in errors)

    def test_non_int_root_rejected(self):
        errors: list = []
        _check_randomness(_randomness_meta(1001, root_seed="abc"), errors)
        assert any("root_seed" in e for e in errors)

    def test_bad_sub_seed_rejected(self):
        errors: list = []
        _check_randomness(_randomness_meta(1001, grf_game_engine_seed=12345), errors)
        assert any("grf_game_engine_seed" in e for e in errors)

    def test_foreign_policy_skips_derivation_check(self):
        # 非当前 policy 版本：无法用当前算法复验，不报派生错误（字段合法性仍查）
        errors: list = []
        _check_randomness(_randomness_meta(1001, policy="futsalmot_seed_v9"), errors)
        assert errors == []


# ── CLI seed 优先级辅助（纯逻辑，不依赖 typer 运行时）───────────────

class TestSeedPriority:
    def test_cli_overrides_config(self):
        # 与 cli.export 一致：CLI --seed 优先于配置文件的 seed
        from grf_ue_bridge.config import ExportConfig

        cfg = ExportConfig(scenario="5_vs_5", seed=42)
        effective = cfg.model_copy(update={"seed": 1001})
        assert effective.seed == 1001
        assert cfg.seed == 42  # 磁盘配置不被修改
