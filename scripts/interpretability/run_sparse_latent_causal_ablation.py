#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from persjepa.intervention import load_jepa_sae, load_standard_sae


def pick_examples(activation: torch.Tensor, top_examples: int) -> torch.Tensor:
    k = min(top_examples, activation.numel())
    return torch.topk(activation, k=k).indices


def mse_and_cosine(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    mse = float(F.mse_loss(pred, target).item())
    cosine = float(F.cosine_similarity(pred, target, dim=-1).mean().item())
    return mse, cosine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hidden-space sparse latent zero/amplify/isolate ablations.")
    parser.add_argument("--hidden-pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sae-type", choices=["jepa", "standard"], default="jepa")
    parser.add_argument("--top-latents", type=int, default=5)
    parser.add_argument("--top-examples", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--amplify-factor", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payload = torch.load(args.hidden_pairs, map_location="cpu")
    h_gen = payload["h_gen"].float()[: args.max_examples]
    h_pers = payload["h_pers"].float()[: args.max_examples]

    if args.sae_type == "jepa":
        model, ckpt = load_jepa_sae(args.checkpoint, device=torch.device(args.device))
        with torch.no_grad():
            latents, _pre = model.encode(h_gen.to(args.device))
            base_pred = model(h_gen.to(args.device)).reconstruction
        source = h_gen.to(args.device)
    else:
        model, ckpt = load_standard_sae(args.checkpoint, device=torch.device(args.device))
        with torch.no_grad():
            latents, _pre = model.encode(h_pers.to(args.device))
            base_pred = model(h_pers.to(args.device)).reconstruction
        source = h_pers.to(args.device)

    target = h_pers.to(args.device)
    latents = latents.float().cpu()
    base_pred = base_pred.float().cpu()
    mean_activation = latents.mean(dim=0)
    top_latent_ids = torch.topk(mean_activation, k=min(args.top_latents, latents.shape[1])).indices.tolist()

    rows: list[dict[str, object]] = []
    base_mse, base_cos = mse_and_cosine(base_pred, h_pers[: args.max_examples])
    for latent_id in top_latent_ids:
        activation = latents[:, latent_id]
        example_idx = pick_examples(activation, args.top_examples)
        z = latents.index_select(0, example_idx).to(args.device)
        src = source.index_select(0, example_idx.to(args.device))
        tgt = target.index_select(0, example_idx.to(args.device))
        base_subset = base_pred.index_select(0, example_idx).to(args.device)
        subset_base_mse, subset_base_cos = mse_and_cosine(base_subset, tgt)

        interventions = {}

        z_zero = z.clone()
        z_zero[:, latent_id] = 0
        z_amp = z.clone()
        z_amp[:, latent_id] = z_amp[:, latent_id] * args.amplify_factor
        z_only = torch.zeros_like(z)
        z_only[:, latent_id] = z[:, latent_id]

        for name, latent_tensor in [("zero", z_zero), ("amplify", z_amp), ("latent_only", z_only)]:
            with torch.no_grad():
                if args.sae_type == "jepa":
                    pred = src + model.delta_decoder(latent_tensor)
                else:
                    pred = model.decoder(latent_tensor)
            mse, cosine = mse_and_cosine(pred, tgt)
            interventions[name] = {
                "mse": mse,
                "cosine": cosine,
                "delta_vs_base_mse": mse - subset_base_mse,
                "delta_vs_base_cosine": cosine - subset_base_cos,
            }

        rows.append(
            {
                "latent_id": latent_id,
                "mean_activation": float(mean_activation[latent_id].item()),
                "top_example_count": int(example_idx.numel()),
                "base_subset_mse": subset_base_mse,
                "base_subset_cosine": subset_base_cos,
                "interventions": interventions,
            }
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "hidden_pairs": args.hidden_pairs,
                "checkpoint": args.checkpoint,
                "sae_type": args.sae_type,
                "checkpoint_meta": {
                    "latent_dim": ckpt.get("latent_dim"),
                    "top_k": ckpt.get("top_k"),
                },
                "global_base_mse": base_mse,
                "global_base_cosine": base_cos,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# Sparse Latent Causal Ablation",
        "",
        f"- Hidden pairs: `{args.hidden_pairs}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- SAE type: `{args.sae_type}`",
        f"- Global base MSE / cosine: `{base_mse:.4f} / {base_cos:.4f}`",
        "",
        "| Latent | Mean act | Base subset MSE / cosine | Zero delta | Amplify delta | Latent-only delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        zero = row["interventions"]["zero"]
        amp = row["interventions"]["amplify"]
        only = row["interventions"]["latent_only"]
        md_lines.append(
            f"| {row['latent_id']} | {row['mean_activation']:.4f} | "
            f"{row['base_subset_mse']:.4f} / {row['base_subset_cosine']:.4f} | "
            f"{zero['delta_vs_base_mse']:+.4f} / {zero['delta_vs_base_cosine']:+.4f} | "
            f"{amp['delta_vs_base_mse']:+.4f} / {amp['delta_vs_base_cosine']:+.4f} | "
            f"{only['delta_vs_base_mse']:+.4f} / {only['delta_vs_base_cosine']:+.4f} |"
        )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "latents": top_latent_ids}, indent=2))


if __name__ == "__main__":
    main()

