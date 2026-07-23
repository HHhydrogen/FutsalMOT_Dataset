"""FutsalMOT-RL unified command-line interface.

Usage:
    futsalmot-rl demos export [--max-episodes N]
    futsalmot-rl demos check
    futsalmot-rl train bc [--epochs N] [--batch-size N]
    futsalmot-rl train ppo [--total-timesteps N]
    futsalmot-rl evaluate bc
    futsalmot-rl evaluate rl
    futsalmot-rl benchmark build
    futsalmot-rl export a33 [--model PATH]
    futsalmot-rl comparison videos
    futsalmot-rl ablation ppo-scratch
    futsalmot-rl fa2 prepare
    futsalmot-rl ue prepare
    futsalmot-rl ue verify [--seq-id ID]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure code/ is on sys.path when run as `python -m futsalmot_rl.cli`
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from futsalmot_rl.core.rl_paths import set_project_root


def add_global_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root", type=str, default=None,
        help="Override UE project root path (default: auto-detect)",
    )


def apply_global_args(args: argparse.Namespace) -> None:
    if args.project_root:
        set_project_root(args.project_root)


def register_demos(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="demos_cmd")

    p = sub_cmds.add_parser("export", help="Export demonstration data from rule A3.3 trajectories")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--source-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--agent", type=str, default="Player_05")
    p.add_argument("--target", type=str, default="Player_01")

    p = sub_cmds.add_parser("check", help="Check exported demo integrity")
    p.add_argument("--demo-dir", type=str, default=None)
    p.add_argument("--sample-seq-id", type=str, default=None)


def run_demos(args: argparse.Namespace) -> int:
    if args.demos_cmd == "export":
        from rl_01_export_demos import main as fn
        return fn(args)
    elif args.demos_cmd == "check":
        from rl_01b_check_demos import main as fn
        return fn(args)
    print("Unknown demos command: {}".format(args.demos_cmd))
    return 1


def register_train(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="train_cmd")

    p = sub_cmds.add_parser("bc", help="Train Behavior Cloning policy")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-video", action="store_true")

    p = sub_cmds.add_parser("ppo", help="Train PPO policy")
    p.add_argument("--total-timesteps", type=int, default=500000)
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--eval-interval", type=int, default=25000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-video", action="store_true")


def run_train(args: argparse.Namespace) -> int:
    if args.train_cmd == "bc":
        from rl_02_train_bc import main as fn
        return fn(args)
    elif args.train_cmd == "ppo":
        from rl_04_train_ppo import main as fn
        return fn(args)
    print("Unknown train command: {}".format(args.train_cmd))
    return 1


def register_evaluate(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="eval_cmd")

    p = sub_cmds.add_parser("bc", help="Evaluate BC policy")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--max-episodes", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")

    p = sub_cmds.add_parser("rl", help="Evaluate RL (PPO) policy")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--device", type=str, default="auto")

    p = sub_cmds.add_parser("sanitize-env", help="Run environment sanity check")
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--n-episodes", type=int, default=3)


def run_evaluate(args: argparse.Namespace) -> int:
    if args.eval_cmd == "bc":
        from rl_03_eval_bc import main as fn
        return fn(args)
    elif args.eval_cmd == "rl":
        from rl_05_eval_rl import main as fn
        return fn(args)
    elif args.eval_cmd == "sanitize-env":
        from rl_03_env_sanity_check import main as fn
        return fn(args)
    print("Unknown evaluate command: {}".format(args.eval_cmd))
    return 1


def register_benchmark(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="bench_cmd")
    p = sub_cmds.add_parser("build", help="Build benchmark CSV/JSON/MD tables")
    p = sub_cmds.add_parser("ablation", help="Run PPO from scratch ablation")


def run_benchmark(args: argparse.Namespace) -> int:
    if args.bench_cmd == "build":
        from rl_10_build_benchmark_table import main as fn
        return fn(args)
    elif args.bench_cmd == "ablation":
        from rl_09_ablation_ppo_from_scratch import main as fn
        return fn(args)
    print("Unknown benchmark command: {}".format(args.bench_cmd))
    return 1


def register_export(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="export_cmd")

    p = sub_cmds.add_parser("a33", help="Export RL trajectory as A3.3-compatible JSON")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")

    p = sub_cmds.add_parser("results", help="Export paper-ready results tables")


def run_export(args: argparse.Namespace) -> int:
    if args.export_cmd == "a33":
        from rl_06_export_rl_a33 import main as fn
        return fn(args)
    elif args.export_cmd == "results":
        from rl_15_export_experiment_results import main as fn
        return fn(args)
    print("Unknown export command: {}".format(args.export_cmd))
    return 1


def register_comparison(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="comp_cmd")
    p = sub_cmds.add_parser("videos", help="Generate Rule/BC/PPO comparison videos")


def run_comparison(args: argparse.Namespace) -> int:
    if args.comp_cmd == "videos":
        from rl_11_make_comparison_videos import main as fn
        return fn(args)
    print("Unknown comparison command: {}".format(args.comp_cmd))
    return 1


def register_ablation(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="abl_cmd")
    p = sub_cmds.add_parser("ppo-scratch", help="Train PPO from scratch (no BC init)")
    p.add_argument("--total-timesteps", type=int, default=500000)
    p.add_argument("--device", type=str, default="auto")


def run_ablation(args: argparse.Namespace) -> int:
    if args.abl_cmd == "ppo-scratch":
        from rl_09_ablation_ppo_from_scratch import main as fn
        return fn(args)
    print("Unknown ablation command: {}".format(args.abl_cmd))
    return 1


def register_fa2(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="fa2_cmd")
    p = sub_cmds.add_parser("prepare", help="Prepare FA-2 Goal-side Defense task definition")


def run_fa2(args: argparse.Namespace) -> int:
    if args.fa2_cmd == "prepare":
        from rl_12_prepare_fa2_goal_side import main as fn
        return fn(args)
    print("Unknown fa2 command: {}".format(args.fa2_cmd))
    return 1


def register_ue(sub: argparse.ArgumentParser) -> None:
    sub_cmds = sub.add_subparsers(dest="ue_cmd")

    p = sub_cmds.add_parser("prepare", help="Prepare RL A3.3 for UE rendering")
    p.add_argument("--rl-a33", type=str, default=None)

    p = sub_cmds.add_parser("verify", help="Verify UE rendering outputs")
    p.add_argument("--seq-id", type=str, default="rl_episode_random_0001_t1_p05")


def run_ue(args: argparse.Namespace) -> int:
    if args.ue_cmd == "prepare":
        from rl_07_validate_ue_closed_loop import main as fn
        # Force --prepare flag
        if not hasattr(args, 'prepare'):
            args.prepare = True
            args.check = False
        return fn(args)
    elif args.ue_cmd == "verify":
        from rl_07d_verify_rl_render import main as fn
        return fn(args)
    print("Unknown ue command: {}".format(args.ue_cmd))
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FutsalMOT-RL: Reinforcement learning pipeline for futsal player control.",
    )
    add_global_args(parser)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("demos", help="Demo data export and check")
    register_demos(p)

    p = sub.add_parser("train", help="Train BC or PPO policy")
    register_train(p)

    p = sub.add_parser("evaluate", help="Evaluate BC or RL policy")
    register_evaluate(p)

    p = sub.add_parser("benchmark", help="Build benchmark tables")
    register_benchmark(p)

    p = sub.add_parser("export", help="Export A3.3 or paper results")
    register_export(p)

    p = sub.add_parser("comparison", help="Comparison videos")
    register_comparison(p)

    p = sub.add_parser("ablation", help="Ablation experiments")
    register_ablation(p)

    p = sub.add_parser("fa2", help="FA-2 Goal-side Defense preparation")
    register_fa2(p)

    p = sub.add_parser("ue", help="UE closed-loop operations")
    register_ue(p)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_global_args(args)

    dispatch = {
        "demos": run_demos,
        "train": run_train,
        "evaluate": run_evaluate,
        "benchmark": run_benchmark,
        "export": run_export,
        "comparison": run_comparison,
        "ablation": run_ablation,
        "fa2": run_fa2,
        "ue": run_ue,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        return 1

    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
