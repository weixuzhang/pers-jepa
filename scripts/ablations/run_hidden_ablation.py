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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from persjepa.models import JEPASAE, StandardSAE


class LinearPredictor(nn.Module):
    """Upstream-style linear predictor over source embeddings."""

    def __init__(self, dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(h_gen, h_pers, *, batch_size: int, eval_fraction: float, seed: int):
    dataset = TensorDataset(h_gen.float(), h_pers.float())
    eval_size = max(1, int(round(len(dataset) * eval_fraction)))
    train_size = len(dataset) - eval_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, eval_ds = random_split(dataset, [train_size, eval_size], generator=generator)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(eval_ds, batch_size=batch_size, shuffle=False),
    )


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
            totals["mse"] += float(F.mse_loss(pred, h_pers).detach().cpu()) * batch
            totals["cosine"] += float(
                F.cosine_similarity(pred, h_pers, dim=-1).mean().detach().cpu()
            ) * batch
            totals["delta_norm"] += float((pred - h_gen).norm(dim=-1).mean().detach().cpu()) * batch
            totals["target_delta_norm"] += float((h_pers - h_gen).norm(dim=-1).mean().detach().cpu()) * batch
            seen += batch
    return {key: value / max(seen, 1) for key, value in totals.items()} | {"n_examples": seen}


def train_vanilla(loader, *, input_dim: int, args, device: torch.device):
    model = StandardSAE(input_dim, latent_dim=args.latent_dim, top_k=args.top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.vanilla_lr, weight_decay=args.weight_decay)
    for _epoch in tqdm(range(args.epochs), desc="vanilla_sae"):
        model.train()
        for _h_gen, h_pers in loader:
            h_pers = h_pers.to(device)
            out = model(h_pers)
            loss = F.mse_loss(out.reconstruction, h_pers) + args.l1_coeff * out.latent.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
    model.eval()
    return model


def train_jepa(loader, *, input_dim: int, loss_type: str, args, device: torch.device):
    model = JEPASAE(input_dim, latent_dim=args.latent_dim, top_k=args.top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.jepa_lr, weight_decay=args.weight_decay)
    for _epoch in tqdm(range(args.epochs), desc=f"jepa_sae_{loss_type}"):
        model.train()
        for h_gen, h_pers in loader:
            h_gen = h_gen.to(device)
            h_pers = h_pers.to(device)
            out = model(h_gen)
            loss = align_loss(out.reconstruction, h_pers, loss_type) + args.l1_coeff * out.latent.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
    model.eval()
    return model


def train_linear_predictor(loader, *, input_dim: int, loss_type: str, args, device: torch.device):
    model = LinearPredictor(input_dim, bias=args.linear_bias).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.linear_lr, weight_decay=args.weight_decay)
    for _epoch in tqdm(range(args.epochs), desc=f"stp_linear_{loss_type}"):
        model.train()
        for h_gen, h_pers in loader:
            h_gen = h_gen.to(device)
            h_pers = h_pers.to(device)
            pred = model(h_gen)
            loss = align_loss(pred, h_pers, loss_type)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hidden-pair objective ablations for JEPA-SAE and upstream-style linear predictor proxies."
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
    parser.add_argument("--vanilla-lr", type=float, default=1e-3)
    parser.add_argument("--jepa-lr", type=float, default=3e-4)
    parser.add_argument("--linear-lr", type=float, default=3e-4)
    parser.add_argument("--linear-bias", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    payload = torch.load(args.hidden_pairs, map_location="cpu")
    h_gen = payload["h_gen"].float()
    h_pers = payload["h_pers"].float()
    input_dim = h_gen.shape[-1]
    train_loader, eval_loader = make_loaders(
        h_gen,
        h_pers,
        batch_size=args.batch_size,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {
        "metadata": {
            "hidden_pairs": args.hidden_pairs,
            "input_dim": int(input_dim),
            "train_examples": len(train_loader.dataset),
            "eval_examples": len(eval_loader.dataset),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_fraction": args.eval_fraction,
            "seed": args.seed,
        }
    }
    generic = eval_prediction(lambda h_gen, _h_pers: h_gen, eval_loader, device=device)
    results["generic_identity"] = generic

    vanilla = train_vanilla(train_loader, input_dim=input_dim, args=args, device=device)
    results["standard_sae"] = eval_prediction(
        lambda _h_gen, h_pers: vanilla(h_pers).reconstruction,
        eval_loader,
        device=device,
    )
    torch.save(
        {
            "state_dict": vanilla.state_dict(),
            "model_type": "standard_sae",
            "input_dim": input_dim,
            "latent_dim": args.latent_dim,
            "top_k": args.top_k,
        },
        output_dir / "standard_sae.pt",
    )

    for loss_type in ["mse", "cosine"]:
        jepa = train_jepa(train_loader, input_dim=input_dim, loss_type=loss_type, args=args, device=device)
        name = f"jepa_sae_{loss_type}"
        results[name] = eval_prediction(lambda h_gen, _h_pers, m=jepa: m(h_gen).reconstruction, eval_loader, device=device)
        torch.save(
            {
                "state_dict": jepa.state_dict(),
                "model_type": "jepa_sae",
                "input_dim": input_dim,
                "latent_dim": args.latent_dim,
                "top_k": args.top_k,
                "objective_proxy": loss_type,
            },
            output_dir / f"{name}.pt",
        )

    for loss_type in ["mse", "cosine"]:
        name = f"stp_identity_{loss_type}"
        results[name] = dict(generic)
        results[name]["note"] = "No learned predictor in the hidden-pair setup; this is the fixed source baseline."

        predictor = train_linear_predictor(train_loader, input_dim=input_dim, loss_type=loss_type, args=args, device=device)
        pred_name = f"stp_linear_predictor_{loss_type}"
        results[pred_name] = eval_prediction(lambda h_gen, _h_pers, m=predictor: m(h_gen), eval_loader, device=device)
        torch.save({"state_dict": predictor.state_dict(), "input_dim": input_dim}, output_dir / f"{pred_name}.pt")

    baseline_mse = results["generic_identity"]["mse"]
    for key, value in results.items():
        if not isinstance(value, dict) or "mse" not in value:
            continue
        value["gap_recovery"] = (baseline_mse - value["mse"]) / baseline_mse if baseline_mse else 0.0

    (output_dir / "ablation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
