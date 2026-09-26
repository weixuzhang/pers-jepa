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
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

from persjepa.models import LinearPredictor, MeanDeltaPredictor


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
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_examples = source.shape[0]
    eval_size = max(1, int(round(n_examples * eval_fraction)))
    train_size = n_examples - eval_size
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_examples, generator=generator)
    train_idx = permutation[:train_size]
    eval_idx = permutation[train_size:]
    return source[train_idx], target[train_idx], source[eval_idx], target[eval_idx]


def make_loader(source: torch.Tensor, target: torch.Tensor, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(source.float(), target.float()), batch_size=batch_size, shuffle=shuffle)


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
        for source, target in loader:
            source = source.to(device)
            target = target.to(device)
            pred = predict(source, target)
            batch = source.shape[0]
            totals["mse"] += float(F.mse_loss(pred, target).detach().cpu()) * batch
            totals["cosine"] += float(
                F.cosine_similarity(pred, target, dim=-1).mean().detach().cpu()
            ) * batch
            totals["delta_norm"] += float((pred - source).norm(dim=-1).mean().detach().cpu()) * batch
            totals["target_delta_norm"] += float((target - source).norm(dim=-1).mean().detach().cpu()) * batch
            seen += batch
    return {key: value / max(seen, 1) for key, value in totals.items()} | {"n_examples": seen}


def mode_tensors(payload: dict, mode: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    if mode == "anchor":
        return payload["source_anchor"], payload["target_anchor"], "Source anchor -> personalized anchor."
    if mode == "mean":
        return payload["source_span_mean"], payload["target_task_span_mean"], "Source task-span mean -> target task-span mean."
    if mode == "e2e":
        source = torch.cat([payload["source_span_start"], payload["source_span_end"]], dim=0)
        target = torch.cat([payload["target_task_span_start"], payload["target_task_span_end"]], dim=0)
        return source, target, "Endpoint STP: start/end source states -> start/end target task states."
    if mode == "random_summary":
        source = torch.cat(
            [payload["source_span_start"], payload["source_span_end"], payload["source_span_mean"]],
            dim=0,
        )
        target = torch.cat(
            [
                payload["target_task_span_start"],
                payload["target_task_span_end"],
                payload["target_task_span_mean"],
            ],
            dim=0,
        )
        return source, target, "Approximate random-span STP over available start/end/mean summaries."
    if mode == "profile_mean":
        return (
            payload["source_span_mean"],
            payload["target_profile_span_mean"],
            "Source task-span mean -> target profile-context span mean.",
        )
    if mode == "answer_mean":
        return (
            payload["source_answer_span_mean"],
            payload["target_answer_span_mean"],
            "Faithful answer-span STP: generic prompt+answer mean -> personalized prompt+answer mean.",
        )
    if mode == "answer_e2e":
        source = torch.cat(
            [payload["source_answer_span_start"], payload["source_answer_span_end"]],
            dim=0,
        )
        target = torch.cat(
            [payload["target_answer_span_start"], payload["target_answer_span_end"]],
            dim=0,
        )
        return source, target, "Faithful answer-span STP over answer start/end token states."
    if mode == "answer_tokens":
        source_tokens = payload["source_answer_tokens"]
        target_tokens = payload["target_answer_tokens"]
        source_mask = payload["source_answer_token_mask"].bool()
        target_mask = payload["target_answer_token_mask"].bool()
        mask = source_mask & target_mask
        return (
            source_tokens[mask],
            target_tokens[mask],
            "Faithful token-level answer-span STP over all gold answer tokens.",
        )
    raise ValueError(f"Unknown span STP mode: {mode}")


def train_linear_predictor(
    loader: DataLoader,
    *,
    input_dim: int,
    loss_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> LinearPredictor:
    model = LinearPredictor(input_dim, bias=args.linear_bias).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.linear_lr, weight_decay=args.weight_decay)
    for _epoch in tqdm(range(args.epochs), desc=f"span_stp_linear_{loss_type}"):
        model.train()
        for source, target in loader:
            source = source.to(device)
            target = target.to(device)
            pred = model(source)
            loss = align_loss(pred, target, loss_type)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent span-level STP objective ablations.")
    parser.add_argument("--span-hidden", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modes", nargs="+", default=["anchor", "mean", "e2e", "random_summary"])
    parser.add_argument("--losses", nargs="+", default=["mse"], choices=["mse", "cosine"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-lr", type=float, default=3e-4)
    parser.add_argument("--linear-bias", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    payload = torch.load(args.span_hidden, map_location="cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {
        "metadata": {
            "span_hidden": args.span_hidden,
            "config": payload.get("config", {}),
            "modes": args.modes,
            "losses": args.losses,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_fraction": args.eval_fraction,
            "seed": args.seed,
        }
    }

    for mode in args.modes:
        source, target, note = mode_tensors(payload, mode)
        source = source.float()
        target = target.float()
        input_dim = int(source.shape[-1])
        train_source, train_target, eval_source, eval_target = split_tensors(
            source,
            target,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
        )
        train_loader = make_loader(train_source, train_target, batch_size=args.batch_size, shuffle=True)
        eval_loader = make_loader(eval_source, eval_target, batch_size=args.batch_size, shuffle=False)
        mean_delta = (train_target - train_source).mean(dim=0)
        mean_model = MeanDeltaPredictor(mean_delta).to(device)
        identity_name = f"{mode}_identity"
        mean_name = f"{mode}_mean_delta"
        results[identity_name] = eval_prediction(lambda source, _target: source, eval_loader, device=device)
        results[identity_name]["note"] = f"No STP predictor. {note}"
        results[mean_name] = eval_prediction(
            lambda source, _target, m=mean_model: m(source),
            eval_loader,
            device=device,
        )
        results[mean_name]["note"] = f"Constant residual over this span mode. {note}"
        torch.save(
            {
                "model_type": "mean_delta",
                "mean_delta": mean_delta,
                "input_dim": input_dim,
                "source_hidden": args.span_hidden,
                "span_mode": mode,
                "train_examples": int(train_source.shape[0]),
                "split_seed": args.seed,
            },
            output_dir / f"{mode}_mean_delta.pt",
        )

        for loss_type in args.losses:
            predictor = train_linear_predictor(
                train_loader,
                input_dim=input_dim,
                loss_type=loss_type,
                args=args,
                device=device,
            )
            predictor_name = f"{mode}_linear_predictor_{loss_type}"
            results[predictor_name] = eval_prediction(
                lambda source, _target, m=predictor: m(source),
                eval_loader,
                device=device,
            )
            results[predictor_name]["note"] = note
            torch.save(
                {
                    "state_dict": predictor.state_dict(),
                    "model_type": "stp_linear_predictor",
                    "input_dim": input_dim,
                    "span_mode": mode,
                    "objective_proxy": loss_type,
                    "source_hidden": args.span_hidden,
                },
                output_dir / f"{predictor_name}.pt",
            )

        baseline_mse = results[identity_name]["mse"]
        for key, value in results.items():
            if key.startswith(mode) and isinstance(value, dict) and "mse" in value:
                value["gap_recovery"] = (baseline_mse - value["mse"]) / baseline_mse if baseline_mse else 0.0

    (output_dir / "span_stp_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
