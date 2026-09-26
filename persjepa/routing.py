"""User-level routing for Pers-JEPA: behavioral user latents (z_u), k-means
groups, hard/soft routing weights, and cold-start routing from a few examples.

All routing lives in *residual* space (no profile text), so the pipeline stays
profile-free at inference: a user's latent is the mean of their calibration
residuals after task-condition centering, groups are k-means clusters of user
latents, and routing weights are softmax(-||z_u - c_g||^2 / tau) (tau -> 0 is
hard routing). New users are routed from whatever residual examples exist.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


# --------------------------------------------------------------------------- #
# record helpers
# --------------------------------------------------------------------------- #
def get_field(record: dict, dotted: str, default: Any = None) -> Any:
    """Resolve a dotted path like ``metadata.metadata.persona`` in a record."""
    cur: Any = record
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


DEFAULT_GROUP_FIELDS = (
    "metadata.metadata.persona",
    "metadata.persona",
    "metadata.user_id",
    "metadata.metadata.user_id",
    "user_id",
)


def group_of(record: dict, field_path: str | None = None) -> str:
    """User / persona / group identity of a record.

    With ``field_path`` given, only that dotted path is used. Otherwise the
    usual SynPer (double-nested ``metadata.metadata.persona``) and
    Amazon/LaMP (``metadata.user_id``) locations are tried in order.
    """
    if field_path:
        value = get_field(record, field_path)
        if value in (None, ""):
            # normalize_record() stores the raw record's own ``metadata`` dict under
            # ``metadata`` again, so a field given for the raw jsonl
            # (``metadata.user_id``) lives at ``metadata.metadata.user_id`` in
            # extraction payloads / PersonaExample.metadata. Try that nesting too.
            value = get_field(record, "metadata." + field_path)
        return "unknown" if value in (None, "") else str(value)
    for candidate in DEFAULT_GROUP_FIELDS:
        value = get_field(record, candidate)
        if value not in (None, ""):
            return str(value)
    return "unknown"


# --------------------------------------------------------------------------- #
# residuals from span-hidden files
# --------------------------------------------------------------------------- #
def answer_span_residuals(payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(h_gen anchor states [N, D], answer-span mean residuals [N, D]) from a
    ``extract_span_hidden.py --include-answer-spans`` payload."""
    h_gen = payload["source_anchor"].float()
    src = payload["source_answer_tokens"].float()
    tgt = payload["target_answer_tokens"].float()
    mask = (payload["source_answer_token_mask"].bool() & payload["target_answer_token_mask"].bool()).float()
    residual = ((tgt - src) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
    return h_gen, residual


def anchor_residuals(payload: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(h_gen anchor, h_pers anchor - h_gen anchor) from a span-hidden payload."""
    h_gen = payload["source_anchor"].float()
    return h_gen, payload["target_anchor"].float() - h_gen


# --------------------------------------------------------------------------- #
# user latents
# --------------------------------------------------------------------------- #
@dataclass
class UserLatents:
    users: list[str]
    z: torch.Tensor  # [U, D]
    counts: dict[str, int]
    condition_means: dict[str, torch.Tensor] = field(default_factory=dict)

    def index(self, user: str) -> int:
        return self.users.index(user)

    def get(self, user: str) -> torch.Tensor:
        return self.z[self.index(user)]


def task_centered_user_latents(
    residuals: torch.Tensor,
    users: list[str],
    conditions: list[str] | None = None,
) -> UserLatents:
    """z_u = mean_i (r_i - b_{c(i)}) over the user's examples, where b_c is the
    mean residual of task-side condition c (None -> no centering)."""
    cond_means: dict[str, torch.Tensor] = {}
    centered = residuals.clone()
    if conditions is not None:
        by_cond: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(conditions):
            by_cond[c].append(i)
        for c, idx in by_cond.items():
            cond_means[c] = residuals[idx].mean(0)
            centered[idx] = residuals[idx] - cond_means[c]
    by_user: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(users):
        by_user[u].append(i)
    names = sorted(by_user)
    z = torch.stack([centered[by_user[u]].mean(0) for u in names])
    return UserLatents(users=names, z=z, counts={u: len(by_user[u]) for u in names}, condition_means=cond_means)


def latent_from_examples(residuals: torch.Tensor, condition_means: dict[str, torch.Tensor] | None = None,
                         conditions: list[str] | None = None) -> torch.Tensor:
    """Cold start: a new user's latent from k residual examples."""
    r = residuals.float()
    if condition_means and conditions:
        r = torch.stack([r[i] - condition_means.get(c, torch.zeros_like(r[i])) for i, c in enumerate(conditions)])
    return r.mean(0)


# --------------------------------------------------------------------------- #
# clustering + routing
# --------------------------------------------------------------------------- #
def kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain k-means (k-means++ init). Returns (centroids [k, D], assignment [N])."""
    x = x.float()
    n = x.shape[0]
    k = min(k, n)
    g = torch.Generator().manual_seed(seed)
    centroids = x[torch.randint(n, (1,), generator=g)]
    for _ in range(1, k):
        d2 = torch.cdist(x, centroids).min(dim=1).values ** 2
        probs = d2 / d2.sum().clamp(min=1e-12)
        centroids = torch.cat([centroids, x[torch.multinomial(probs, 1, generator=g)]])
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        new = torch.stack([x[assign == j].mean(0) if (assign == j).any() else centroids[j] for j in range(k)])
        if torch.allclose(new, centroids):
            break
        centroids = new
    return centroids, assign


def routing_weights(z: torch.Tensor, centroids: torch.Tensor, *, tau: float = 1.0, mode: str = "soft") -> torch.Tensor:
    """Routing weights [.., K] for latents z [.., D] against centroids [K, D]."""
    z = z.float()
    single = z.ndim == 1
    if single:
        z = z[None]
    d2 = torch.cdist(z, centroids.float()) ** 2
    if mode == "hard":
        w = torch.zeros_like(d2)
        w.scatter_(1, d2.argmin(dim=1, keepdim=True), 1.0)
    elif mode == "soft":
        # scale distances by their median so tau is unit-free across datasets
        scale = d2.median().clamp(min=1e-6)
        w = torch.softmax(-d2 / (scale * max(tau, 1e-6)), dim=1)
    else:
        raise ValueError(f"unknown routing mode {mode!r}")
    return w[0] if single else w


@dataclass
class RoutingTable:
    """user -> routing weights over K groups, plus the geometry to route new users."""
    centroids: torch.Tensor          # [K, D]
    group_names: list[str]
    tau: float
    mode: str
    weights: dict[str, torch.Tensor]  # user -> [K]
    condition_means: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def num_groups(self) -> int:
        return int(self.centroids.shape[0])

    def get(self, user: str) -> torch.Tensor | None:
        return self.weights.get(user)

    def route_latent(self, z_u: torch.Tensor) -> torch.Tensor:
        return routing_weights(z_u, self.centroids, tau=self.tau, mode=self.mode)

    def add_user_from_examples(self, user: str, residuals: torch.Tensor,
                               conditions: list[str] | None = None) -> torch.Tensor:
        z = latent_from_examples(residuals, self.condition_means, conditions)
        w = self.route_latent(z)
        self.weights[user] = w
        return w

    def to_payload(self) -> dict:
        return {
            "centroids": self.centroids.cpu(),
            "group_names": list(self.group_names),
            "tau": float(self.tau),
            "mode": self.mode,
            "weights": {u: w.cpu() for u, w in self.weights.items()},
            "condition_means": {c: m.cpu() for c, m in self.condition_means.items()},
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "RoutingTable":
        return cls(
            centroids=payload["centroids"].float(),
            group_names=list(payload["group_names"]),
            tau=float(payload.get("tau", 1.0)),
            mode=str(payload.get("mode", "hard")),
            weights={u: w.float() for u, w in payload.get("weights", {}).items()},
            condition_means={c: m.float() for c, m in payload.get("condition_means", {}).items()},
        )


def build_groups(
    latents: UserLatents,
    *,
    num_groups: int = 0,
    given_groups: dict[str, str] | None = None,
    iters: int = 50,
    seed: int = 42,
) -> tuple[list[str], torch.Tensor, dict[str, int]]:
    """Return (group_names, centroids [K, D], user -> group index).

    ``num_groups == 0`` uses the given user -> group mapping (e.g. persona
    labels) and its per-group mean latent as the centroid; otherwise k-means
    over user latents defines the groups."""
    if num_groups <= 0:
        if not given_groups:
            raise ValueError("num_groups=0 requires given_groups (user -> group name)")
        names = sorted(set(given_groups[u] for u in latents.users))
        idx = {g: i for i, g in enumerate(names)}
        assignment = {u: idx[given_groups[u]] for u in latents.users}
        centroids = torch.stack([
            latents.z[[latents.index(u) for u in latents.users if assignment[u] == i]].mean(0)
            for i in range(len(names))
        ])
        return names, centroids, assignment
    centroids, assign = kmeans(latents.z, num_groups, iters=iters, seed=seed)
    names = [f"group_{i}" for i in range(centroids.shape[0])]
    return names, centroids, {u: int(assign[i]) for i, u in enumerate(latents.users)}


def group_mean_residuals(residuals: torch.Tensor, users: list[str], assignment: dict[str, int], k: int) -> torch.Tensor:
    """Per-group mean residual [K, D] (the per-group constant baseline)."""
    out = torch.zeros(k, residuals.shape[-1])
    counts = torch.zeros(k)
    for i, u in enumerate(users):
        g = assignment.get(u)
        if g is None:
            continue
        out[g] += residuals[i]
        counts[g] += 1
    return out / counts.clamp(min=1.0)[:, None]


def iter_records(payload: dict) -> Iterable[dict]:
    return payload.get("examples", [])
