#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vanilla SAE and JEPA-SAE.")
    parser.add_argument("--hidden-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--vanilla-lr", type=float, default=None)
    parser.add_argument("--jepa-lr", type=float, default=None)
    parser.add_argument(
        "--objective",
        choices=["jepa_direct", "jepa_cosine", "tube_mse", "combined"],
        default=None,
        help=(
            "JEPA-SAE objective. jepa_direct is the default direct prediction "
            "MSE loss; jepa_cosine mirrors the original cosine-style alignment; "
            "tube_mse is a hidden-pair tube ablation, not faithful "
            "upstream STP; combined adds both explicitly."
        ),
    )
    parser.add_argument("--stp-coeff", type=float, default=None)
    parser.add_argument(
        "--enable-tube-mse",
        action="store_true",
        help="Enable the simplified hidden-pair tube MSE ablation.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.vanilla_lr is not None:
        cfg.training.vanilla_lr = args.vanilla_lr
    if args.jepa_lr is not None:
        cfg.training.jepa_lr = args.jepa_lr
    if args.objective is not None:
        cfg.training.objective = args.objective
    if args.stp_coeff is not None:
        cfg.training.stp_coeff = args.stp_coeff
    if args.enable_tube_mse:
        cfg.stp.enabled = True
    from persjepa.training import train_dual_saes

    result = train_dual_saes(args.hidden_pairs, args.output_dir, cfg, device=args.device)
    print(f"Saved checkpoints and history to {result['output_dir']}")


if __name__ == "__main__":
    main()
