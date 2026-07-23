#!/usr/bin/env python3
"""
Train BC policy — thin wrapper that calls the package implementation.

Usage:
    python scripts/train_bc.py --epochs 20
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from futsalmot_rl.training.train_bc import train_bc
from futsalmot_rl.core.local_config import get_repo_root, get_ue_project_root, load_local_paths

if __name__ == "__main__":
    cfg = load_local_paths()
    project_root = cfg.get("ue_project_root", "")
    if project_root:
        saved = Path(project_root) / "Saved" / "FutsalMOT_RL"
    else:
        saved = get_repo_root().parent.parent.parent / "Saved" / "FutsalMOT_RL"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    demo_index = saved / "demos" / "demo_index.json"
    model_out = saved / "models" / "defender_follow_bc_v1.pt"

    summary = train_bc(
        demo_index_path=demo_index,
        model_out=model_out,
        config={"epochs": args.epochs, "batch_size": args.batch_size},
    )
    print(f"BC training complete. Best loss: {summary.get('best_val_loss', 'N/A')}")
