#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from persjepa.models import JEPASAE, StandardSAE


def select_indices(
    n_items: int,
    *,
    split: str,
    eval_fraction: float,
    seed: int,
) -> torch.Tensor:
    if split == "all" or eval_fraction <= 0 or n_items < 2:
        return torch.arange(n_items)
    eval_size = max(1, int(round(n_items * eval_fraction)))
    train_size = n_items - eval_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_items, generator=generator)
    if split == "train":
        return indices[:train_size]
    if split == "eval":
        return indices[train_size:]
    raise ValueError(f"Unknown split: {split}")


def load_standard_sae(path: str | Path, *, device: torch.device) -> StandardSAE:
    payload = torch.load(path, map_location=device)
    model = StandardSAE(
        input_dim=int(payload["input_dim"]),
        latent_dim=int(payload["latent_dim"]),
        top_k=int(payload["top_k"]),
        decoder_bias=payload["state_dict"].get("decoder.bias") is not None,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def load_jepa_sae(path: str | Path, *, device: torch.device) -> JEPASAE:
    payload = torch.load(path, map_location=device)
    model = JEPASAE(
        input_dim=int(payload["input_dim"]),
        latent_dim=int(payload["latent_dim"]),
        top_k=int(payload["top_k"]),
        decoder_bias=payload["state_dict"].get("delta_decoder.bias") is not None,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model


def mean_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1).mean()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hidden-space Pers-JEPA metrics against generic and vanilla SAE baselines."
    )
    parser.add_argument("--hidden-pairs", required=True)
    parser.add_argument("--jepa-checkpoint", required=True)
    parser.add_argument("--vanilla-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["all", "train", "eval"], default="eval")
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    payload = torch.load(args.hidden_pairs, map_location="cpu")
    h_gen = payload["h_gen"].float()
    h_pers = payload["h_pers"].float()
    indices = select_indices(
        h_gen.shape[0],
        split=args.split,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    h_gen = h_gen[indices]
    h_pers = h_pers[indices]

    jepa = load_jepa_sae(args.jepa_checkpoint, device=device)
    vanilla = load_standard_sae(args.vanilla_checkpoint, device=device)

    totals = {
        "generic_mse": 0.0,
        "jepa_mse": 0.0,
        "vanilla_recon_mse": 0.0,
        "generic_cosine": 0.0,
        "jepa_cosine": 0.0,
        "vanilla_recon_cosine": 0.0,
        "delta_norm": 0.0,
        "target_delta_norm": 0.0,
        "jepa_active_latents": 0.0,
        "vanilla_active_latents": 0.0,
    }
    seen = 0
    with torch.no_grad():
        for start in range(0, h_gen.shape[0], args.batch_size):
            gen = h_gen[start:start + args.batch_size].to(device)
            pers = h_pers[start:start + args.batch_size].to(device)
            batch = gen.shape[0]
            j_out = jepa(gen)
            v_out = vanilla(pers)
            target_delta = pers - gen
            batch_values = {
                "generic_mse": F.mse_loss(gen, pers),
                "jepa_mse": F.mse_loss(j_out.reconstruction, pers),
                "vanilla_recon_mse": F.mse_loss(v_out.reconstruction, pers),
                "generic_cosine": mean_cosine(gen, pers),
                "jepa_cosine": mean_cosine(j_out.reconstruction, pers),
                "vanilla_recon_cosine": mean_cosine(v_out.reconstruction, pers),
                "delta_norm": j_out.delta.norm(dim=-1).mean(),
                "target_delta_norm": target_delta.norm(dim=-1).mean(),
                "jepa_active_latents": j_out.latent.ne(0).float().sum(dim=-1).mean(),
                "vanilla_active_latents": v_out.latent.ne(0).float().sum(dim=-1).mean(),
            }
            for key, value in batch_values.items():
                totals[key] += float(value.detach().cpu()) * batch
            seen += batch

    metrics = {key: value / max(seen, 1) for key, value in totals.items()}
    baseline = metrics["generic_mse"]
    metrics["jepa_gap_recovery"] = (
        (baseline - metrics["jepa_mse"]) / baseline if baseline else 0.0
    )
    metrics["vanilla_recon_gap_recovery_upper_bound"] = (
        (baseline - metrics["vanilla_recon_mse"]) / baseline if baseline else 0.0
    )
    metrics["split"] = args.split
    metrics["n_examples"] = int(seen)
    metrics["hidden_pairs"] = args.hidden_pairs
    metrics["jepa_checkpoint"] = args.jepa_checkpoint
    metrics["vanilla_checkpoint"] = args.vanilla_checkpoint

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
