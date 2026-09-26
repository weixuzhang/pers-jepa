from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import math
import random

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from persjepa.config import ExperimentConfig
from persjepa.losses import jepa_sae_loss, standard_sae_loss
from persjepa.models import JEPASAE, StandardSAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hidden_pairs(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "h_gen" not in payload or "h_pers" not in payload:
        raise KeyError("Hidden-pair file must contain h_gen and h_pers tensors.")
    if payload["h_gen"].shape != payload["h_pers"].shape:
        raise ValueError(
            f"h_gen and h_pers shape mismatch: "
            f"{payload['h_gen'].shape} vs {payload['h_pers'].shape}"
        )
    return payload


def build_loaders(
    h_gen: torch.Tensor,
    h_pers: torch.Tensor,
    *,
    batch_size: int,
    eval_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader | None]:
    dataset = TensorDataset(h_gen.float(), h_pers.float())
    if eval_fraction <= 0 or len(dataset) < 2:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), None
    eval_size = max(1, int(round(len(dataset) * eval_fraction)))
    train_size = len(dataset) - eval_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, eval_ds = random_split(dataset, [train_size, eval_size], generator=generator)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(eval_ds, batch_size=batch_size, shuffle=False),
    )


def evaluate_losses(
    vanilla: StandardSAE,
    jepa: JEPASAE,
    loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    vanilla.eval()
    jepa.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for h_gen, h_pers in loader:
            h_gen = h_gen.to(device)
            h_pers = h_pers.to(device)
            batch = h_gen.shape[0]
            v_loss = standard_sae_loss(
                vanilla(h_pers),
                h_pers,
                l1_coeff=cfg.sae.l1_coeff,
            )
            j_loss = jepa_sae_loss(
                jepa(h_gen),
                h_gen,
                h_pers,
                l1_coeff=cfg.sae.l1_coeff,
                lambda_coeff=cfg.training.lambda_coeff,
                objective=cfg.training.objective,
                stp_coeff=cfg.training.stp_coeff,
                stp_enabled=cfg.stp.enabled,
                stp_num_points=cfg.stp.num_points,
                stp_detach_target=cfg.stp.detach_target,
            )
            for prefix, loss in [("vanilla", v_loss), ("jepa", j_loss)]:
                for key, value in loss.scalars().items():
                    totals[f"{prefix}_{key}"] = totals.get(f"{prefix}_{key}", 0.0) + value * batch
            count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_dual_saes(
    hidden_pairs_path: str | Path,
    output_dir: str | Path,
    cfg: ExperimentConfig,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    set_seed(cfg.training.seed)
    payload = load_hidden_pairs(hidden_pairs_path)
    h_gen = payload["h_gen"]
    h_pers = payload["h_pers"]
    input_dim = cfg.sae.input_dim or h_gen.shape[-1]
    if input_dim != h_gen.shape[-1]:
        raise ValueError(f"Configured input_dim {input_dim} != tensor dim {h_gen.shape[-1]}")

    torch_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    vanilla = StandardSAE(
        input_dim=input_dim,
        latent_dim=cfg.sae.latent_dim,
        top_k=cfg.sae.top_k,
        decoder_bias=cfg.sae.decoder_bias,
    ).to(torch_device)
    jepa = JEPASAE(
        input_dim=input_dim,
        latent_dim=cfg.sae.latent_dim,
        top_k=cfg.sae.top_k,
        decoder_bias=cfg.sae.decoder_bias,
    ).to(torch_device)

    train_loader, eval_loader = build_loaders(
        h_gen,
        h_pers,
        batch_size=cfg.training.batch_size,
        eval_fraction=cfg.training.eval_fraction,
        seed=cfg.training.seed,
    )
    vanilla_opt = torch.optim.AdamW(
        vanilla.parameters(),
        lr=cfg.training.vanilla_lr,
        weight_decay=cfg.training.weight_decay,
    )
    jepa_opt = torch.optim.AdamW(
        jepa.parameters(),
        lr=cfg.training.jepa_lr,
        weight_decay=cfg.training.weight_decay,
    )

    history = []
    for epoch in range(1, cfg.training.epochs + 1):
        vanilla.train()
        jepa.train()
        epoch_totals: dict[str, float] = {}
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.training.epochs}")
        for h_gen_batch, h_pers_batch in progress:
            h_gen_batch = h_gen_batch.to(torch_device)
            h_pers_batch = h_pers_batch.to(torch_device)
            batch = h_gen_batch.shape[0]

            vanilla_opt.zero_grad(set_to_none=True)
            v_loss = standard_sae_loss(
                vanilla(h_pers_batch),
                h_pers_batch,
                l1_coeff=cfg.sae.l1_coeff,
            )
            v_loss.total.backward()
            if cfg.training.grad_clip_norm and math.isfinite(cfg.training.grad_clip_norm):
                torch.nn.utils.clip_grad_norm_(vanilla.parameters(), cfg.training.grad_clip_norm)
            vanilla_opt.step()

            jepa_opt.zero_grad(set_to_none=True)
            j_loss = jepa_sae_loss(
                jepa(h_gen_batch),
                h_gen_batch,
                h_pers_batch,
                l1_coeff=cfg.sae.l1_coeff,
                lambda_coeff=cfg.training.lambda_coeff,
                objective=cfg.training.objective,
                stp_coeff=cfg.training.stp_coeff,
                stp_enabled=cfg.stp.enabled,
                stp_num_points=cfg.stp.num_points,
                stp_detach_target=cfg.stp.detach_target,
            )
            j_loss.total.backward()
            if cfg.training.grad_clip_norm and math.isfinite(cfg.training.grad_clip_norm):
                torch.nn.utils.clip_grad_norm_(jepa.parameters(), cfg.training.grad_clip_norm)
            jepa_opt.step()

            for prefix, loss in [("vanilla", v_loss), ("jepa", j_loss)]:
                for key, value in loss.scalars().items():
                    epoch_totals[f"{prefix}_{key}"] = (
                        epoch_totals.get(f"{prefix}_{key}", 0.0) + value * batch
                    )
            seen += batch
            progress.set_postfix({
                "v_mse": epoch_totals["vanilla_mse"] / seen,
                "j_mse": epoch_totals["jepa_mse"] / seen,
            })

        epoch_record = {
            "epoch": epoch,
            **{key: value / max(seen, 1) for key, value in epoch_totals.items()},
        }
        if eval_loader is not None:
            epoch_record.update({
                f"eval_{key}": value
                for key, value in evaluate_losses(vanilla, jepa, eval_loader, cfg, torch_device).items()
            })
        history.append(epoch_record)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": vanilla.state_dict(),
            "model_type": "standard_sae",
            "input_dim": input_dim,
            "latent_dim": cfg.sae.latent_dim,
            "top_k": cfg.sae.top_k,
            "config": cfg.to_dict(),
        },
        output_dir / "vanilla_sae.pt",
    )
    torch.save(
        {
            "state_dict": jepa.state_dict(),
            "model_type": "jepa_sae",
            "input_dim": input_dim,
            "latent_dim": cfg.sae.latent_dim,
            "top_k": cfg.sae.top_k,
            "config": cfg.to_dict(),
        },
        output_dir / "jepa_sae.pt",
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "history": history,
        "config": asdict(cfg),
    }



def train_jepa_sae_on_pairs(
    h_gen: torch.Tensor,
    h_target: torch.Tensor,
    cfg: ExperimentConfig,
    *,
    model: JEPASAE | None = None,
    device: str = "auto",
    epochs: int | None = None,
    lr: float | None = None,
    desc: str = "jepa",
    verbose: bool = True,
) -> tuple[JEPASAE, list[dict[str, float]]]:
    """Train (or fine-tune) a single JEPA-SAE so that h_gen + delta ~ h_target.

    `h_target` may be h_pers (anchor pairs) or h_gen + answer-span residual.
    When `model` is given it is fine-tuned in place (per-group experts)."""
    set_seed(cfg.training.seed)
    torch_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    input_dim = h_gen.shape[-1]
    if model is None:
        model = JEPASAE(
            input_dim=input_dim, latent_dim=cfg.sae.latent_dim, top_k=cfg.sae.top_k,
            decoder_bias=cfg.sae.decoder_bias,
        )
    model.to(torch_device)
    train_loader, eval_loader = build_loaders(
        h_gen, h_target, batch_size=cfg.training.batch_size,
        eval_fraction=cfg.training.eval_fraction, seed=cfg.training.seed,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr or cfg.training.jepa_lr, weight_decay=cfg.training.weight_decay)
    history: list[dict[str, float]] = []
    n_epochs = epochs or cfg.training.epochs
    for epoch in range(1, n_epochs + 1):
        model.train()
        total, seen = 0.0, 0
        iterator = tqdm(train_loader, desc=f"{desc} {epoch}/{n_epochs}", leave=False) if verbose else train_loader
        for hg, ht in iterator:
            hg, ht = hg.to(torch_device), ht.to(torch_device)
            opt.zero_grad(set_to_none=True)
            loss = jepa_sae_loss(
                model(hg), hg, ht, l1_coeff=cfg.sae.l1_coeff, lambda_coeff=cfg.training.lambda_coeff,
                objective="jepa_direct", stp_coeff=cfg.training.stp_coeff, stp_enabled=False,
                stp_num_points=cfg.stp.num_points, stp_detach_target=cfg.stp.detach_target,
            )
            loss.total.backward()
            if cfg.training.grad_clip_norm and math.isfinite(cfg.training.grad_clip_norm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            opt.step()
            total += float(loss.scalars()["mse"]) * hg.shape[0]
            seen += hg.shape[0]
        rec = {"epoch": epoch, "train_mse": total / max(seen, 1)}
        if eval_loader is not None:
            model.eval()
            et, es = 0.0, 0
            with torch.no_grad():
                for hg, ht in eval_loader:
                    hg, ht = hg.to(torch_device), ht.to(torch_device)
                    et += float(((model(hg).reconstruction - ht) ** 2).mean()) * hg.shape[0]
                    es += hg.shape[0]
            rec["eval_mse"] = et / max(es, 1)
        history.append(rec)
    model.eval()
    return model, history
