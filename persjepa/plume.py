"""PLUME baseline (arXiv 2609.04715, Sep 2026): parameter-efficient LLM
personalization with a shared task subspace and per-user subspace mixers.
Equal-conditions re-implementation (no prompt-side retrieval, same identities,
same calibration data and CE objective as the routed SAE).

For every adapted linear map W0 (layer l, module m):
    y = W0 x + s * B^(l) c_u^(l) A^(l) x + s' * Bs_u^(m) As_u^(m) x + beta_u^(l) (alpha_u^(l) . x)
  - (A^(l), B^(l)): shared task LoRA (rank r), trained on the pooled data, then frozen.
  - c_u^(l) in R^{r x r}: per-user subspace mixer, init = identity.
  - (As_u^(m), Bs_u^(m)): per-user cross-layer-shared low-rank pair (rank r_sh), one per module type.
  - (alpha_u^(l), beta_u^(l)): per-user rank-1 residual.
Unknown user: c = I and the user-specific terms are zero (shared task adapter only).
"""
from __future__ import annotations

import math
import re

import torch
from torch import nn

from persjepa.intervention import decoder_layers

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _key(layer_idx: int, name: str) -> str:
    return f"l{layer_idx}_{name}"


class PlumeAdapter(nn.Module):
    """All PLUME parameters (shared + per user) and the currently active user."""

    def __init__(self, shapes: dict[str, tuple[int, int]], *, rank: int = 8, rank_sh: int = 4,
                 scale: float = 2.0, scale_sh: float = 1.0, users: list[str] | None = None) -> None:
        super().__init__()
        self.shapes = dict(shapes)             # key -> (in_features, out_features)
        self.rank, self.rank_sh, self.scale, self.scale_sh = rank, rank_sh, scale, scale_sh
        self.A = nn.ParameterDict(); self.B = nn.ParameterDict()
        for k, (fi, fo) in self.shapes.items():
            self.A[k] = nn.Parameter(torch.empty(rank, fi)); nn.init.kaiming_uniform_(self.A[k], a=math.sqrt(5))
            self.B[k] = nn.Parameter(torch.zeros(fo, rank))
        self.user_params = nn.ModuleDict()
        self.active: str | None = None
        self.enabled = True
        for u in users or []:
            self.add_user(u)

    # ------------------------------------------------------------- users
    @staticmethod
    def _uid(user: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]", "_", user)

    def module_types(self) -> dict[str, tuple[int, int]]:
        out = {}
        for k, shp in self.shapes.items():
            out.setdefault(k.split("_", 1)[1], shp)
        return out

    def add_user(self, user: str) -> None:
        uid = self._uid(user)
        if uid in self.user_params:
            return
        pd = nn.ParameterDict()
        dev = next(self.A.parameters()).device if len(self.A) else "cpu"
        for k, (fi, fo) in self.shapes.items():
            pd["c_" + k] = nn.Parameter(torch.eye(self.rank, device=dev))
            pd["alpha_" + k] = nn.Parameter(torch.randn(fi, device=dev) / math.sqrt(fi))
            pd["beta_" + k] = nn.Parameter(torch.zeros(fo, device=dev))
        for m, (fi, fo) in self.module_types().items():
            As = torch.empty(self.rank_sh, fi, device=dev); nn.init.kaiming_uniform_(As, a=math.sqrt(5))
            pd["As_" + m] = nn.Parameter(As)
            pd["Bs_" + m] = nn.Parameter(torch.zeros(fo, self.rank_sh, device=dev))
        self.user_params[uid] = pd

    def has_user(self, user: str) -> bool:
        return self._uid(user) in self.user_params

    def set_user(self, user: str | None) -> bool:
        self.active = self._uid(user) if user is not None and self.has_user(user) else None
        return self.active is not None

    def user_parameters(self, user: str):
        return list(self.user_params[self._uid(user)].parameters())

    def shared_parameters(self):
        return list(self.A.parameters()) + list(self.B.parameters())

    def per_user_numel(self) -> int:
        if not len(self.user_params):
            return 0
        return sum(p.numel() for p in next(iter(self.user_params.values())).parameters())

    # ------------------------------------------------------------- delta
    def delta(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros((), dtype=x.dtype, device=x.device)
        A, B = self.A[key], self.B[key]
        z = x @ A.t().to(x.dtype)                                   # [..., r]
        if self.active is None:
            return self.scale * (z @ B.t().to(x.dtype))
        pd = self.user_params[self.active]
        out = self.scale * ((z @ pd["c_" + key].t().to(x.dtype)) @ B.t().to(x.dtype))
        m = key.split("_", 1)[1]
        out = out + self.scale_sh * ((x @ pd["As_" + m].t().to(x.dtype)) @ pd["Bs_" + m].t().to(x.dtype))
        out = out + (x @ pd["alpha_" + key].to(x.dtype))[..., None] * pd["beta_" + key].to(x.dtype)
        return out

    # ------------------------------------------------------------- io
    def export(self) -> dict:
        return {"model_type": "plume", "shapes": self.shapes, "rank": self.rank, "rank_sh": self.rank_sh,
                "scale": self.scale, "scale_sh": self.scale_sh, "users": list(self.user_params.keys()),
                "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}}

    @classmethod
    def from_payload(cls, pay: dict) -> "PlumeAdapter":
        m = cls(pay["shapes"], rank=int(pay["rank"]), rank_sh=int(pay["rank_sh"]), scale=float(pay["scale"]), scale_sh=float(pay["scale_sh"]))
        for u in pay["users"]:
            m.add_user(u)
        m.load_state_dict(pay["state_dict"])
        return m


class PlumeLinear(nn.Module):
    def __init__(self, base: nn.Linear, adapter: PlumeAdapter, key: str) -> None:
        super().__init__()
        self.base, self.adapter, self.key = base, adapter, key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.adapter.delta(self.key, x)


def linear_shapes(model, targets=DEFAULT_TARGETS) -> dict[str, tuple[int, int]]:
    shapes = {}
    for li, layer in enumerate(decoder_layers(model)):
        for name, mod in layer.named_modules():
            leaf = name.split(".")[-1]
            if leaf in targets and isinstance(mod, nn.Linear):
                shapes[_key(li, leaf)] = (mod.in_features, mod.out_features)
    return shapes


def attach_plume(model, adapter: PlumeAdapter, targets=DEFAULT_TARGETS) -> int:
    """Wrap the target linear maps of every decoder layer in place. Returns #wrapped."""
    n = 0
    for li, layer in enumerate(decoder_layers(model)):
        for name, mod in list(layer.named_modules()):
            leaf = name.split(".")[-1]
            if leaf in targets and isinstance(mod, nn.Linear) and _key(li, leaf) in adapter.shapes:
                parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
                setattr(parent, leaf, PlumeLinear(mod, adapter, _key(li, leaf))); n += 1
    return n
