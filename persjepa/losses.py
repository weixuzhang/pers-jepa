from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from persjepa.models import SAEOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    mse: torch.Tensor
    l1: torch.Tensor
    stp: torch.Tensor | None = None

    def scalars(self) -> dict[str, float]:
        values = {
            "total": float(self.total.detach().cpu()),
            "mse": float(self.mse.detach().cpu()),
            "l1": float(self.l1.detach().cpu()),
        }
        if self.stp is not None:
            values["stp"] = float(self.stp.detach().cpu())
        return values


def l1_sparsity(z: torch.Tensor) -> torch.Tensor:
    return z.abs().mean()


def standard_sae_loss(
    output: SAEOutput,
    target: torch.Tensor,
    *,
    l1_coeff: float,
) -> LossBreakdown:
    mse = F.mse_loss(output.reconstruction, target)
    l1 = l1_sparsity(output.latent)
    total = mse + l1_coeff * l1
    return LossBreakdown(total=total, mse=mse, l1=l1)


def semantic_tube_mse_loss(
    h_gen: torch.Tensor,
    h_pers: torch.Tensor,
    delta_pred: torch.Tensor,
    *,
    num_points: int = 4,
    detach_target: bool = True,
) -> torch.Tensor:
    """Simplified hidden-pair tube MSE.

    The target tube is the straight semantic trajectory from h_gen to h_pers.
    The predicted tube is h_gen plus the same fractions of the JEPA-predicted
    Delta.

    This is not a faithful port of llm-jepa's STP span modes. Upstream STP uses
    span-level embeddings such as e2e/mean/random_span plus an optional learned
    linear predictor. This hidden-pair approximation is kept only as an explicit
    ablation for already-extracted (h_gen, h_pers) pairs.
    """
    if num_points <= 0:
        return h_gen.new_zeros(())
    target_delta = h_pers - h_gen
    if detach_target:
        target_delta = target_delta.detach()
    alphas = torch.linspace(
        1.0 / num_points,
        1.0,
        steps=num_points,
        device=h_gen.device,
        dtype=h_gen.dtype,
    )
    view_shape = (1, num_points) + (1,) * (h_gen.ndim - 1)
    alphas = alphas.view(view_shape)
    pred_tube = h_gen.unsqueeze(1) + alphas * delta_pred.unsqueeze(1)
    target_tube = h_gen.unsqueeze(1) + alphas * target_delta.unsqueeze(1)
    return F.mse_loss(pred_tube, target_tube)


def jepa_sae_loss(
    output: SAEOutput,
    h_gen: torch.Tensor,
    h_pers: torch.Tensor,
    *,
    l1_coeff: float,
    lambda_coeff: float,
    stp_coeff: float,
    objective: str = "jepa_direct",
    stp_enabled: bool = True,
    stp_num_points: int = 4,
    stp_detach_target: bool = True,
) -> LossBreakdown:
    if output.delta is None:
        raise ValueError("JEPA output must include delta.")
    mse = F.mse_loss(output.reconstruction, h_pers)
    cosine = 1.0 - F.cosine_similarity(output.reconstruction, h_pers, dim=-1).mean()
    l1 = l1_sparsity(output.latent)
    if stp_enabled and stp_coeff > 0:
        stp = semantic_tube_mse_loss(
            h_gen,
            h_pers,
            output.delta,
            num_points=stp_num_points,
            detach_target=stp_detach_target,
        )
    else:
        stp = h_gen.new_zeros(())

    if objective == "jepa_direct":
        total = lambda_coeff * mse + l1_coeff * l1
    elif objective == "jepa_cosine":
        total = lambda_coeff * cosine + l1_coeff * l1
    elif objective == "tube_mse":
        total = stp_coeff * stp + l1_coeff * l1
    elif objective == "combined":
        total = lambda_coeff * mse + l1_coeff * l1 + stp_coeff * stp
    else:
        raise ValueError(
            "Unknown objective "
            f"'{objective}'. Choices: jepa_direct, jepa_cosine, tube_mse, combined."
        )
    return LossBreakdown(total=total, mse=mse, l1=l1, stp=stp)
