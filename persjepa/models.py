from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    pre_activations: torch.Tensor
    delta: torch.Tensor | None = None


def topk_relu(pre_activations: torch.Tensor, k: int) -> torch.Tensor:
    activations = F.relu(pre_activations)
    if k <= 0 or k >= activations.shape[-1]:
        return activations
    values, indices = torch.topk(activations, k=k, dim=-1)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(dim=-1, index=indices, src=values)
    return sparse


class StandardSAE(nn.Module):
    """Vanilla reconstruction baseline over personalized hidden states."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        top_k: int = 16,
        decoder_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.top_k = top_k
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=decoder_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.encoder(x)
        return topk_relu(pre, self.top_k), pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, h_pers: torch.Tensor) -> SAEOutput:
        z, pre = self.encode(h_pers)
        rec = self.decode(z)
        return SAEOutput(reconstruction=rec, latent=z, pre_activations=pre)


class JEPASAE(nn.Module):
    """Predictive SAE that maps h_gen to h_pers through a sparse Delta."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        top_k: int = 16,
        decoder_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.top_k = top_k
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.delta_decoder = nn.Linear(latent_dim, input_dim, bias=decoder_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.delta_decoder.weight)
        if self.delta_decoder.bias is not None:
            nn.init.zeros_(self.delta_decoder.bias)

    def encode(self, h_gen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.encoder(h_gen)
        return topk_relu(pre, self.top_k), pre

    def predict_delta(self, h_gen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, pre = self.encode(h_gen)
        delta = self.delta_decoder(z)
        return delta, z, pre

    def forward(self, h_gen: torch.Tensor) -> SAEOutput:
        delta, z, pre = self.predict_delta(h_gen)
        pred = h_gen + delta
        return SAEOutput(
            reconstruction=pred,
            latent=z,
            pre_activations=pre,
            delta=delta,
        )


class LinearPredictor(nn.Module):
    """Linear source-to-target predictor used by STP-style proxy ablations."""

    def __init__(self, input_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim, input_dim, bias=bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, h_gen: torch.Tensor) -> torch.Tensor:
        return self.proj(h_gen)


class MeanDeltaPredictor(nn.Module):
    """Constant train-set average residual baseline."""

    def __init__(self, mean_delta: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean_delta", mean_delta.detach().clone())
        self.input_dim = int(mean_delta.numel())

    def forward(self, h_gen: torch.Tensor) -> torch.Tensor:
        return h_gen + self.mean_delta.to(dtype=h_gen.dtype, device=h_gen.device)


class OffsetDeltaPredictor(nn.Module):
    """Add a fixed residual offset plus a learned residual correction."""

    def __init__(self, *, base_delta: torch.Tensor, learned_model: nn.Module) -> None:
        super().__init__()
        self.register_buffer("base_delta", base_delta.detach().clone())
        self.learned_model = learned_model
        self.input_dim = int(base_delta.numel())

    def forward(self, h_gen: torch.Tensor) -> torch.Tensor:
        learned_out = self.learned_model(h_gen)
        if hasattr(learned_out, "delta") and learned_out.delta is not None:
            learned_delta = learned_out.delta
        elif hasattr(learned_out, "reconstruction"):
            learned_delta = learned_out.reconstruction - h_gen
        else:
            learned_delta = learned_out - h_gen
        base_delta = self.base_delta.to(dtype=h_gen.dtype, device=h_gen.device)
        return h_gen + base_delta + learned_delta


class ClusterMeanDeltaPredictor(nn.Module):
    """Conditional mean residual baseline using nearest source-state clusters."""

    def __init__(self, *, centroids: torch.Tensor, cluster_deltas: torch.Tensor) -> None:
        super().__init__()
        if centroids.ndim != 2 or cluster_deltas.ndim != 2:
            raise ValueError("centroids and cluster_deltas must be rank-2 tensors.")
        if centroids.shape != cluster_deltas.shape:
            raise ValueError("centroids and cluster_deltas must have matching shapes.")
        self.register_buffer("centroids", centroids.detach().clone().float())
        self.register_buffer("cluster_deltas", cluster_deltas.detach().clone().float())
        self.input_dim = int(centroids.shape[-1])
        self.num_clusters = int(centroids.shape[0])

    def forward(self, h_gen: torch.Tensor) -> torch.Tensor:
        h_float = h_gen.float()
        centroids = self.centroids.to(device=h_gen.device)
        distances = torch.cdist(h_float, centroids)
        cluster_ids = distances.argmin(dim=-1)
        deltas = self.cluster_deltas.to(device=h_gen.device).index_select(0, cluster_ids)
        return h_gen + deltas.to(dtype=h_gen.dtype)


class BlendDeltaPredictor(nn.Module):
    """Interpolate between a constant residual and a learned residual."""

    def __init__(
        self,
        *,
        mean_delta: torch.Tensor,
        learned_model: nn.Module,
        learned_delta_weight: float,
    ) -> None:
        super().__init__()
        self.register_buffer("mean_delta", mean_delta.detach().clone())
        self.learned_model = learned_model
        self.learned_delta_weight = float(learned_delta_weight)
        self.input_dim = int(mean_delta.numel())

    def forward(self, h_gen: torch.Tensor) -> torch.Tensor:
        learned_out = self.learned_model(h_gen)
        if hasattr(learned_out, "delta") and learned_out.delta is not None:
            learned_delta = learned_out.delta
        elif hasattr(learned_out, "reconstruction"):
            learned_delta = learned_out.reconstruction - h_gen
        else:
            learned_delta = learned_out - h_gen
        mean_delta = self.mean_delta.to(dtype=h_gen.dtype, device=h_gen.device)
        delta = (1.0 - self.learned_delta_weight) * mean_delta + self.learned_delta_weight * learned_delta
        return h_gen + delta


class DirectionalScaleDeltaPredictor(nn.Module):
    """Learn only a scalar gate over a fixed residual direction."""

    def __init__(
        self,
        *,
        direction: torch.Tensor,
        max_scale: float = 4.0,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("direction", direction.detach().clone())
        self.input_dim = int(direction.numel())
        self.max_scale = float(max_scale)
        self.scale_head = nn.Linear(self.input_dim, 1)
        nn.init.zeros_(self.scale_head.weight)
        init_scale = min(max(float(init_scale), 1e-4), self.max_scale - 1e-4)
        init_prob = init_scale / self.max_scale
        init_logit = torch.logit(torch.tensor(init_prob)).item()
        nn.init.constant_(self.scale_head.bias, init_logit)

    def scale(self, h_gen: torch.Tensor) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(self.scale_head(h_gen.float())).to(dtype=h_gen.dtype)

    def forward(self, h_gen: torch.Tensor) -> torch.Tensor:
        scale = self.scale(h_gen)
        direction = self.direction.to(dtype=h_gen.dtype, device=h_gen.device)
        return h_gen + scale * direction


class BoundedOrthogonalDeltaPredictor(nn.Module):
    """Fixed residual direction plus a learned orthogonal correction.

    The learned term is projected away from the base direction and hard-clipped
    to a fraction of the base direction norm. This keeps generation steering
    close to the strong answer-token mean direction while allowing bounded
    per-example adaptation.
    """

    def __init__(
        self,
        *,
        base_delta: torch.Tensor,
        hidden_dim: int = 256,
        correction_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.register_buffer("base_delta", base_delta.detach().clone())
        self.input_dim = int(base_delta.numel())
        self.hidden_dim = int(hidden_dim)
        self.correction_scale = float(correction_scale)
        self.correction = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        first = self.correction[0]
        second = self.correction[2]
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        nn.init.zeros_(second.weight)
        nn.init.zeros_(second.bias)

    def orthogonal_correction(self, h_gen: torch.Tensor) -> torch.Tensor:
        raw = self.correction(h_gen.float()).to(dtype=h_gen.dtype)
        base = self.base_delta.to(dtype=h_gen.dtype, device=h_gen.device)
        base_float = base.float()
        denom = base_float.dot(base_float).clamp_min(1e-8).to(dtype=h_gen.dtype)
        projection = raw.matmul(base) / denom
        orthogonal = raw - projection.unsqueeze(-1) * base
        max_norm = self.correction_scale * base.norm().clamp_min(1e-8)
        norm = orthogonal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        factor = (max_norm / norm).clamp(max=1.0)
        return orthogonal * factor

    def delta(self, h_gen: torch.Tensor) -> torch.Tensor:
        base = self.base_delta.to(dtype=h_gen.dtype, device=h_gen.device)
        return base + self.orthogonal_correction(h_gen)

    def forward(self, h_gen: torch.Tensor) -> torch.Tensor:
        return h_gen + self.delta(h_gen)


class RoutedJEPASAE(nn.Module):
    """Group-routed mixture of JEPA-SAE experts with user-level gating.

    delta(h, w) = sum_g w_g * ( expert_g(h).delta + offset_g )

    * `experts`: one independent JEPASAE per group (initialised from a shared
      global JEPASAE and fine-tuned on the group's pairs).
    * `offset_g`: a trainable per-group constant, initialised to
      (group mean residual - global mean residual) so that at initialisation
      the routed model equals the per-group constant-vector baseline
      (group mean) whenever the global expert predicts the global mean.
    * `w`: routing weights over groups. They come from a RoutingTable
      (behavioral user latents); the module itself is stateless w.r.t. users
      except for `current_weights`, a convenience slot set per example so the
      module can be used through the existing single-input steering code.
    """

    def __init__(
        self,
        input_dim: int,
        num_groups: int,
        latent_dim: int = 128,
        top_k: int = 16,
        decoder_bias: bool = True,
        group_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.latent_dim = latent_dim
        self.top_k = top_k
        self.group_names = list(group_names) if group_names else [f"group_{i}" for i in range(num_groups)]
        self.experts = nn.ModuleList([
            JEPASAE(input_dim, latent_dim=latent_dim, top_k=top_k, decoder_bias=decoder_bias)
            for _ in range(num_groups)
        ])
        self.group_offsets = nn.Parameter(torch.zeros(num_groups, input_dim))
        self.current_weights: torch.Tensor | None = None

    @classmethod
    def from_global(
        cls,
        global_sae: JEPASAE,
        *,
        group_means: torch.Tensor,
        global_mean: torch.Tensor,
        group_names: list[str] | None = None,
    ) -> "RoutedJEPASAE":
        k = int(group_means.shape[0])
        model = cls(
            global_sae.input_dim, k, latent_dim=global_sae.latent_dim, top_k=global_sae.top_k,
            decoder_bias=global_sae.delta_decoder.bias is not None, group_names=group_names,
        )
        for expert in model.experts:
            expert.load_state_dict(global_sae.state_dict())
        with torch.no_grad():
            model.group_offsets.copy_(group_means.float() - global_mean.float()[None])
        return model

    def set_weights(self, weights: torch.Tensor | None) -> None:
        self.current_weights = None if weights is None else weights.detach()

    def expert_deltas(self, h_gen: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        deltas, latents = [], []
        for g, expert in enumerate(self.experts):
            d, z, _ = expert.predict_delta(h_gen)
            deltas.append(d + self.group_offsets[g].to(dtype=d.dtype))
            latents.append(z)
        return torch.stack(deltas, dim=1), latents  # [B, K, D]

    def forward(self, h_gen: torch.Tensor, weights: torch.Tensor | None = None) -> SAEOutput:
        if weights is None:
            weights = self.current_weights
        if weights is None:
            raise ValueError("RoutedJEPASAE needs routing weights (set_weights or forward(h, weights)).")
        w = weights.to(device=h_gen.device, dtype=h_gen.dtype)
        if w.ndim == 1:
            w = w[None].expand(h_gen.shape[0], -1)
        deltas, latents = self.expert_deltas(h_gen)
        delta = (w[..., None] * deltas).sum(dim=1)
        top = int(w[0].argmax())
        return SAEOutput(reconstruction=h_gen + delta, latent=latents[top], pre_activations=latents[top], delta=delta)

    def export_payload(self) -> dict:
        return {
            "model_type": "routed_jepa_sae",
            "input_dim": self.input_dim,
            "num_groups": self.num_groups,
            "latent_dim": self.latent_dim,
            "top_k": self.top_k,
            "decoder_bias": self.experts[0].delta_decoder.bias is not None,
            "group_names": list(self.group_names),
            "state_dict": self.state_dict(),
        }


class RoutedMeanDelta(nn.Module):
    """Per-group constant-vector baseline routed exactly like RoutedJEPASAE:
    delta(h, w) = sum_g w_g * m_g  (m_g = group mean residual)."""

    def __init__(self, group_means: torch.Tensor, group_names: list[str] | None = None) -> None:
        super().__init__()
        self.register_buffer("group_means", group_means.detach().clone().float())
        self.num_groups = int(group_means.shape[0])
        self.input_dim = int(group_means.shape[-1])
        self.group_names = list(group_names) if group_names else [f"group_{i}" for i in range(self.num_groups)]
        self.current_weights: torch.Tensor | None = None

    def set_weights(self, weights: torch.Tensor | None) -> None:
        self.current_weights = None if weights is None else weights.detach()

    def forward(self, h_gen: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        w = self.current_weights if weights is None else weights
        if w is None:
            raise ValueError("RoutedMeanDelta needs routing weights.")
        w = w.to(device=h_gen.device, dtype=torch.float32)
        if w.ndim == 1:
            w = w[None].expand(h_gen.shape[0], -1)
        delta = w @ self.group_means.to(h_gen.device)
        return h_gen + delta.to(h_gen.dtype)


class RoutedConstants(nn.Module):
    """Genuine learned group constants, routed like RoutedJEPASAE:
    delta(h, w) = sum_g w_g * v_g with v_g TRAINABLE (init: group mean residual).
    The prediction is independent of h by construction (control E3: does the
    input-dependent expert add anything beyond likelihood-trained constants?).
    Exports as `routed_mean_delta`, so generation loads it as RoutedMeanDelta."""

    def __init__(self, group_means: torch.Tensor, group_names: list[str] | None = None) -> None:
        super().__init__()
        self.group_vectors = nn.Parameter(group_means.detach().clone().float())
        self.num_groups = int(group_means.shape[0])
        self.input_dim = int(group_means.shape[-1])
        self.group_names = list(group_names) if group_names else [f"group_{i}" for i in range(self.num_groups)]
        self.current_weights: torch.Tensor | None = None

    def set_weights(self, weights: torch.Tensor | None) -> None:
        self.current_weights = None if weights is None else weights.detach()

    def forward(self, h_gen: torch.Tensor, weights: torch.Tensor | None = None) -> SAEOutput:
        w = self.current_weights if weights is None else weights
        if w is None:
            raise ValueError("RoutedConstants needs routing weights.")
        w = w.to(device=h_gen.device, dtype=torch.float32)
        if w.ndim == 1:
            w = w[None].expand(h_gen.shape[0], -1)
        delta = (w @ self.group_vectors).to(h_gen.dtype)
        return SAEOutput(reconstruction=h_gen + delta, latent=w, pre_activations=w, delta=delta)

    def export_payload(self) -> dict:
        return {"model_type": "routed_mean_delta", "group_means": self.group_vectors.detach().cpu().clone(),
                "group_names": list(self.group_names), "trained_constants": True}
