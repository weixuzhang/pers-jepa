#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from persjepa.models import ClusterMeanDeltaPredictor, JEPASAE, MeanDeltaPredictor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def split_tensors(
    h_gen: torch.Tensor,
    h_pers: torch.Tensor,
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_examples = h_gen.shape[0]
    eval_size = max(1, int(round(n_examples * eval_fraction)))
    train_size = n_examples - eval_size
    if train_size <= 0:
        raise ValueError("eval_fraction leaves no training examples.")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_examples, generator=generator)
    train_idx = permutation[:train_size]
    eval_idx = permutation[train_size:]
    return h_gen[train_idx], h_pers[train_idx], h_gen[eval_idx], h_pers[eval_idx]


def make_loader(h_gen: torch.Tensor, h_pers: torch.Tensor, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(h_gen.float(), h_pers.float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def align_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    raise ValueError(f"Unknown loss_type: {loss_type}")


def eval_prediction(
    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        "mse": 0.0,
        "cosine": 0.0,
        "residual_mse": 0.0,
        "delta_norm": 0.0,
        "target_delta_norm": 0.0,
    }
    seen = 0
    with torch.no_grad():
        for h_gen, h_pers in loader:
            h_gen = h_gen.to(device)
            h_pers = h_pers.to(device)
            pred = predict(h_gen, h_pers)
            batch = h_gen.shape[0]
            pred_delta = pred - h_gen
            target_delta = h_pers - h_gen
            totals["mse"] += float(F.mse_loss(pred, h_pers).detach().cpu()) * batch
            totals["cosine"] += float(
                F.cosine_similarity(pred, h_pers, dim=-1).mean().detach().cpu()
            ) * batch
            totals["residual_mse"] += float(F.mse_loss(pred_delta, target_delta).detach().cpu()) * batch
            totals["delta_norm"] += float(pred_delta.norm(dim=-1).mean().detach().cpu()) * batch
            totals["target_delta_norm"] += float(target_delta.norm(dim=-1).mean().detach().cpu()) * batch
            seen += batch
    return {key: value / max(seen, 1) for key, value in totals.items()} | {"n_examples": seen}


def train_residual_jepa(
    loader: DataLoader,
    *,
    input_dim: int,
    center_delta: torch.Tensor | None,
    loss_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> JEPASAE:
    model = JEPASAE(input_dim, latent_dim=args.latent_dim, top_k=args.top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.jepa_lr, weight_decay=args.weight_decay)
    if center_delta is not None:
        center_delta = center_delta.to(device)
    for _epoch in tqdm(range(args.epochs), desc=f"residual_jepa_{loss_type}"):
        model.train()
        for h_gen, h_pers in loader:
            h_gen = h_gen.to(device)
            h_pers = h_pers.to(device)
            target_delta = h_pers - h_gen
            if center_delta is not None:
                target_delta = target_delta - center_delta
            out = model(h_gen)
            loss = align_loss(out.delta, target_delta, loss_type) + args.l1_coeff * out.latent.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
    model.eval()
    return model


def fit_cluster_mean_delta(
    h_gen: torch.Tensor,
    residual: torch.Tensor,
    *,
    num_clusters: int,
    iters: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = h_gen.float().to(device)
    y = residual.float().to(device)
    n_examples = x.shape[0]
    if num_clusters > n_examples:
        raise ValueError("num_clusters cannot exceed number of training examples.")
    generator = torch.Generator(device=device).manual_seed(seed)
    init_idx = torch.randperm(n_examples, generator=generator, device=device)[:num_clusters]
    centroids = x.index_select(0, init_idx).clone()
    assignments = torch.zeros(n_examples, dtype=torch.long, device=device)
    for _ in tqdm(range(iters), desc=f"kmeans_{num_clusters}"):
        distances = torch.cdist(x, centroids)
        assignments = distances.argmin(dim=-1)
        for cluster_id in range(num_clusters):
            mask = assignments.eq(cluster_id)
            if torch.any(mask):
                centroids[cluster_id] = x[mask].mean(dim=0)
    cluster_deltas = torch.empty_like(centroids)
    global_delta = y.mean(dim=0)
    for cluster_id in range(num_clusters):
        mask = assignments.eq(cluster_id)
        cluster_deltas[cluster_id] = y[mask].mean(dim=0) if torch.any(mask) else global_delta
    return centroids.detach().cpu(), cluster_deltas.detach().cpu(), assignments.detach().cpu()


def save_jepa_checkpoint(
    path: Path,
    *,
    model: JEPASAE,
    model_type: str,
    input_dim: int,
    args: argparse.Namespace,
    objective: str,
    extra: dict,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_type": model_type,
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "top_k": args.top_k,
        "objective_proxy": objective,
    }
    payload.update(extra)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run residual-only, residual-centered, and conditional mean residual ablations."
    )
    parser.add_argument("--hidden-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--l1-coeff", type=float, default=1e-4)
    parser.add_argument("--jepa-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--losses", nargs="+", default=["mse"], choices=["mse", "cosine"])
    parser.add_argument("--cluster-counts", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--kmeans-iters", type=int, default=25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    payload = torch.load(args.hidden_pairs, map_location="cpu")
    h_gen = payload["h_gen"].float()
    h_pers = payload["h_pers"].float()
    input_dim = int(h_gen.shape[-1])
    train_h_gen, train_h_pers, eval_h_gen, eval_h_pers = split_tensors(
        h_gen,
        h_pers,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    train_loader = make_loader(train_h_gen, train_h_pers, batch_size=args.batch_size, shuffle=True)
    eval_loader = make_loader(eval_h_gen, eval_h_pers, batch_size=args.batch_size, shuffle=False)
    train_residual = train_h_pers - train_h_gen
    mean_delta = train_residual.mean(dim=0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {
        "metadata": {
            "hidden_pairs": args.hidden_pairs,
            "input_dim": input_dim,
            "train_examples": int(train_h_gen.shape[0]),
            "eval_examples": int(eval_h_gen.shape[0]),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_fraction": args.eval_fraction,
            "seed": args.seed,
            "cluster_counts": args.cluster_counts,
        }
    }

    results["generic_identity"] = eval_prediction(lambda h_gen, _h_pers: h_gen, eval_loader, device=device)
    mean_model = MeanDeltaPredictor(mean_delta).to(device)
    results["mean_delta_train_split"] = eval_prediction(
        lambda h_gen, _h_pers, m=mean_model: m(h_gen),
        eval_loader,
        device=device,
    )
    torch.save(
        {
            "model_type": "mean_delta",
            "mean_delta": mean_delta,
            "input_dim": input_dim,
            "source_hidden": args.hidden_pairs,
            "train_examples": int(train_h_gen.shape[0]),
            "split_seed": args.seed,
        },
        output_dir / "mean_delta_train_split.pt",
    )

    for loss_type in args.losses:
        residual_only = train_residual_jepa(
            train_loader,
            input_dim=input_dim,
            center_delta=None,
            loss_type=loss_type,
            args=args,
            device=device,
        )
        residual_only_name = f"residual_only_jepa_{loss_type}"
        results[residual_only_name] = eval_prediction(
            lambda h_gen, _h_pers, m=residual_only: m(h_gen).reconstruction,
            eval_loader,
            device=device,
        )
        save_jepa_checkpoint(
            output_dir / f"{residual_only_name}.pt",
            model=residual_only,
            model_type="jepa_sae",
            input_dim=input_dim,
            args=args,
            objective=f"residual_only_{loss_type}",
            extra={"source_hidden": args.hidden_pairs},
        )

        residual_centered = train_residual_jepa(
            train_loader,
            input_dim=input_dim,
            center_delta=mean_delta,
            loss_type=loss_type,
            args=args,
            device=device,
        )
        centered_name = f"residual_centered_jepa_{loss_type}"
        mean_delta_device = mean_delta.to(device)
        results[centered_name] = eval_prediction(
            lambda h_gen, _h_pers, m=residual_centered, c=mean_delta_device: h_gen + c + m(h_gen).delta,
            eval_loader,
            device=device,
        )
        save_jepa_checkpoint(
            output_dir / f"{centered_name}.pt",
            model=residual_centered,
            model_type="offset_delta",
            input_dim=input_dim,
            args=args,
            objective=f"residual_centered_{loss_type}",
            extra={
                "base_model_type": "jepa_sae",
                "base_delta": mean_delta,
                "mean_delta": mean_delta,
                "source_hidden": args.hidden_pairs,
            },
        )

    for num_clusters in args.cluster_counts:
        centroids, cluster_deltas, assignments = fit_cluster_mean_delta(
            train_h_gen,
            train_residual,
            num_clusters=num_clusters,
            iters=args.kmeans_iters,
            seed=args.seed + num_clusters,
            device=device,
        )
        cluster_model = ClusterMeanDeltaPredictor(
            centroids=centroids,
            cluster_deltas=cluster_deltas,
        ).to(device)
        name = f"cluster_mean_delta_k{num_clusters}"
        counts = torch.bincount(assignments, minlength=num_clusters)
        results[name] = eval_prediction(
            lambda h_gen, _h_pers, m=cluster_model: m(h_gen),
            eval_loader,
            device=device,
        )
        results[name]["min_cluster_size"] = int(counts.min().item())
        results[name]["max_cluster_size"] = int(counts.max().item())
        torch.save(
            {
                "model_type": "cluster_mean_delta",
                "centroids": centroids,
                "cluster_deltas": cluster_deltas,
                "input_dim": input_dim,
                "num_clusters": num_clusters,
                "source_hidden": args.hidden_pairs,
                "train_examples": int(train_h_gen.shape[0]),
                "split_seed": args.seed,
                "kmeans_iters": args.kmeans_iters,
                "cluster_counts": counts,
            },
            output_dir / f"{name}.pt",
        )

    baseline_mse = results["generic_identity"]["mse"]
    for value in results.values():
        if isinstance(value, dict) and "mse" in value:
            value["gap_recovery"] = (baseline_mse - value["mse"]) / baseline_mse if baseline_mse else 0.0

    (output_dir / "variant_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
