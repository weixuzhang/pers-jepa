#!/usr/bin/env python
"""Train the group-routed JEPA-SAE (Pers-JEPA v2 main method).

Pipeline
  1. residuals: answer-span (default) or anchor residuals from a span-hidden
     file produced by scripts/training/extract_span_hidden.py at the
     injection layer (e.g. --layer 14).
  2. user latents z_u (task-centered mean residual per user/persona).
  3. groups: given labels (--num-groups 0, e.g. personas) or k-means on z_u.
  4. global JEPA-SAE on all pairs (or --global-checkpoint), then one expert per
     group initialised from it and fine-tuned on the group's pairs; per-group
     offsets initialised to (group mean - global mean) so the routed model
     starts exactly at the per-group constant baseline.
  5. routing table for every calibration user (hard or soft), saved with the
     model, plus mean_delta checkpoints for the constant baselines
     (global + per group) so baselines and method share one extraction.

Outputs in --output-dir:
  routed_jepa_sae.pt   (model_type=routed_jepa_sae, includes routing_table)
  global_jepa_sae.pt   (model_type=jepa_sae)
  global_mean_delta.pt, group_<name>_mean_delta.pt   (model_type=mean_delta)
  manifest.json        (groups, users, counts, training history)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from persjepa.config import ExperimentConfig
from persjepa.intervention import load_jepa_sae
from persjepa.models import RoutedJEPASAE
from persjepa.routing import (
    RoutingTable, anchor_residuals, answer_span_residuals, build_groups, get_field, group_mean_residuals,
    group_of, routing_weights, task_centered_user_latents,
)
from persjepa.training import train_jepa_sae_on_pairs


def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--span-hidden", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--residual", choices=["answer_span", "anchor"], default="answer_span")
    ap.add_argument("--group-field", default=None, help="dotted field for user/persona identity (default: auto)")
    ap.add_argument("--condition-field", default=None, help="dotted field for task-side condition centering (optional)")
    ap.add_argument("--exclude-groups", default=None, help="comma-separated identities to hold out entirely (cold-start evaluation)")
    ap.add_argument("--num-groups", type=int, default=0, help="0 = use given identities as groups; >0 = k-means on z_u")
    ap.add_argument("--routing-mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--global-checkpoint", default=None, help="reuse a trained global JEPA-SAE instead of training one")
    ap.add_argument("--config", default=None)
    ap.add_argument("--latent-dim", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=80, help="global pretraining epochs")
    ap.add_argument("--finetune-epochs", type=int, default=40, help="per-group fine-tuning epochs")
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--min-group-size", type=int, default=20, help="groups smaller than this keep the global expert (no fine-tune)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    cfg.training.seed = args.seed
    if args.latent_dim: cfg.sae.latent_dim = args.latent_dim
    if args.top_k: cfg.sae.top_k = args.top_k
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.span_hidden, map_location="cpu")
    h_gen, r = (answer_span_residuals if args.residual == "answer_span" else anchor_residuals)(payload)
    records = payload["examples"]
    users = [group_of(rec, args.group_field) for rec in records]
    if args.exclude_groups:
        excl = {g.strip() for g in args.exclude_groups.split(",") if g.strip()}
        keep = [i for i, u in enumerate(users) if u not in excl]
        print(f"[routed] holding out {sorted(excl)}: {len(records) - len(keep)} examples removed")
        h_gen, r = h_gen[keep], r[keep]; records = [records[i] for i in keep]; users = [users[i] for i in keep]
    conditions = [str(get_field(rec, args.condition_field, "none")) for rec in records] if args.condition_field else None
    print(f"[routed] n={len(records)} users={len(set(users))} residual={args.residual} |r|={r.norm(dim=1).mean():.2f}")

    # ---- user latents + groups
    latents = task_centered_user_latents(r, users, conditions)
    given = {u: u for u in latents.users} if args.num_groups <= 0 else None
    group_names, centroids, assignment = build_groups(latents, num_groups=args.num_groups, given_groups=given, seed=args.seed)
    k = len(group_names)
    group_means = group_mean_residuals(r, users, assignment, k)
    global_mean = r.mean(0)
    print(f"[routed] groups={k} ({'given' if args.num_groups <= 0 else 'kmeans'}): " +
          ", ".join(f"{g}:{sum(1 for u in users if assignment.get(u) == i)}" for i, g in enumerate(group_names)))

    # ---- global expert
    device = args.device
    if args.global_checkpoint:
        global_sae, _ = load_jepa_sae(args.global_checkpoint, device="cpu")
        global_hist = []
        print(f"[routed] loaded global expert from {args.global_checkpoint}")
    else:
        global_sae, global_hist = train_jepa_sae_on_pairs(h_gen, h_gen + r, cfg, device=device, epochs=args.epochs, desc="global")
        print(f"[routed] global expert: train_mse={global_hist[-1]['train_mse']:.4f} eval_mse={global_hist[-1].get('eval_mse', float('nan')):.4f}")
    global_sae = global_sae.cpu()
    torch.save({**{"state_dict": global_sae.state_dict(), "model_type": "jepa_sae", "input_dim": global_sae.input_dim,
                   "latent_dim": global_sae.latent_dim, "top_k": global_sae.top_k, "config": cfg.to_dict()}}, out / "global_jepa_sae.pt")

    # ---- routed experts
    routed = RoutedJEPASAE.from_global(global_sae, group_means=group_means, global_mean=global_mean, group_names=group_names)
    hist = {}
    idx_by_group = {i: [j for j, u in enumerate(users) if assignment.get(u) == i] for i in range(k)}
    for i, name in enumerate(group_names):
        idx = idx_by_group[i]
        if len(idx) < args.min_group_size:
            print(f"[routed] group {name}: n={len(idx)} < {args.min_group_size}, keeping global expert")
            continue
        expert = routed.experts[i]
        # the expert learns the residual minus its constant offset, so the offset stays the group-mean anchor
        target = h_gen[idx] + r[idx] - routed.group_offsets[i].detach()[None]
        _, h = train_jepa_sae_on_pairs(h_gen[idx], target, cfg, model=expert, device=device,
                                       epochs=args.finetune_epochs, lr=args.finetune_lr, desc=f"expert:{name}", verbose=False)
        routed.experts[i] = expert.cpu()
        hist[name] = h
        print(f"[routed] expert {name}: n={len(idx)} train_mse {h[0]['train_mse']:.4f} -> {h[-1]['train_mse']:.4f}")
    routed = routed.cpu()

    # ---- routing table
    table = RoutingTable(centroids=centroids, group_names=group_names, tau=args.tau, mode=args.routing_mode,
                         weights={}, condition_means=latents.condition_means)
    for u in latents.users:
        table.weights[u] = routing_weights(latents.get(u), centroids, tau=args.tau, mode=args.routing_mode)
    # routing accuracy w.r.t. the assignment (informative when groups are given)
    acc = sum(int(table.weights[u].argmax()) == assignment[u] for u in latents.users) / max(len(latents.users), 1)

    pay = routed.export_payload()
    pay["routing_table"] = table.to_payload()
    pay["residual"] = args.residual
    pay["source_span_hidden"] = args.span_hidden
    torch.save(pay, out / "routed_jepa_sae.pt")
    torch.save({"model_type": "mean_delta", "mean_delta": global_mean, "n": int(r.shape[0])}, out / "global_mean_delta.pt")
    torch.save({"model_type": "routed_mean_delta", "group_means": group_means, "group_names": group_names,
                "routing_table": table.to_payload()}, out / "routed_mean_delta.pt")
    for i, name in enumerate(group_names):
        torch.save({"model_type": "mean_delta", "mean_delta": group_means[i], "n": len(idx_by_group[i]), "group": name},
                   out / f"group_{slug(name)}_mean_delta.pt")
    manifest = {
        "residual": args.residual, "num_groups": k, "group_names": group_names, "routing_mode": args.routing_mode,
        "tau": args.tau, "routing_accuracy_vs_assignment": acc,
        "groups": {name: {"slug": slug(name), "n_pairs": len(idx_by_group[i]),
                          "users": [u for u in latents.users if assignment[u] == i]} for i, name in enumerate(group_names)},
        "user_counts": latents.counts, "global_history": global_hist, "expert_history": hist,
        "config": cfg.to_dict(), "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[routed] wrote {out}/routed_jepa_sae.pt (+ global and group mean_delta checkpoints); routing acc={acc:.3f}")


if __name__ == "__main__":
    main()
