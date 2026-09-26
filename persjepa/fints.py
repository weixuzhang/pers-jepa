"""FinTS baseline (Du et al., 2025, arXiv 2510.27206): fine-grained
instance-tailored steering. Training-free.

Offline (per calibration example i of user u):
    X+ = query_i with the user's own (relevant) context     -> h+_attn, h+_mlp at layer L
    X- = query_i with an irrelevant context from another user -> h-_attn, h-_mlp
    d_attn = h+_attn - h-_attn,  d_mlp = h+_mlp - h-_mlp   (last prompt token)
stored together with an embedding of query_i.
Inference for (user u, query q): retrieve the top-K stored samples of u by
cosine similarity of query embeddings, weight them by (1-d_j)/sum(1-d_k),
and add alpha*s_attn after the attention sub-layer and beta*s_mlp after the
MLP sub-layer of layer L at every position (prompt pass + decoding).

Equal-conditions notes for our comparison: the same calibration pairs and
the same identity information (the user's own history) as the routed SAE;
no gradient training; alpha/beta grid-searched like our scale.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from persjepa.intervention import decoder_layers


def sublayers(layer):
    attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None) or getattr(layer, "attn", None)
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    if attn is None or mlp is None:
        raise AttributeError("could not find attention / mlp sub-modules on the decoder layer")
    return attn, mlp


def _first(output):
    return output[0] if isinstance(output, tuple) else output


@dataclass
class FinTSStore:
    layer: int
    users: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)  # user -> {q: [n, dq], attn: [n, D], mlp: [n, D]}
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, user: str, q: torch.Tensor, d_attn: torch.Tensor, d_mlp: torch.Tensor) -> None:
        cur = self.users.get(user)
        if cur is None:
            self.users[user] = {"q": q, "attn": d_attn, "mlp": d_mlp}
        else:
            for k, v in (("q", q), ("attn", d_attn), ("mlp", d_mlp)):
                cur[k] = torch.cat([cur[k], v])

    def retrieve(self, user: str, q: torch.Tensor, *, top_k: int = 5, weighting: str = "attn") -> tuple[torch.Tensor, torch.Tensor] | None:
        st = self.users.get(user)
        if st is None or st["q"].shape[0] == 0:
            return None
        sims = torch.nn.functional.cosine_similarity(q.float()[None], st["q"].float(), dim=-1)
        k = min(top_k, sims.shape[0])
        top = sims.topk(k).indices
        if weighting == "mean":
            w = torch.full((k,), 1.0 / k)
        else:  # (1 - d_j) / sum(1 - d_k) with d = 1 - cos  ->  cos_j / sum cos_k (clamped)
            c = sims[top].clamp(min=1e-6); w = c / c.sum()
        return (w[:, None] * st["attn"][top].float()).sum(0), (w[:, None] * st["mlp"][top].float()).sum(0)

    def to_payload(self) -> dict:
        return {"model_type": "fints_store", "layer": self.layer, "meta": self.meta,
                "users": {u: {k: v.to(torch.float16) for k, v in d.items()} for u, d in self.users.items()}}

    @classmethod
    def from_payload(cls, pay: dict) -> "FinTSStore":
        return cls(layer=int(pay["layer"]), users={u: {k: v.float() for k, v in d.items()} for u, d in pay["users"].items()},
                   meta=pay.get("meta", {}))


class SublayerCapture:
    """Capture attention-sublayer and MLP-sublayer outputs at layer L (last position)."""

    def __init__(self, model, layer: int) -> None:
        self.attn_mod, self.mlp_mod = sublayers(decoder_layers(model)[layer - 1])
        self.out: dict[str, torch.Tensor] = {}
        self.handles = [
            self.attn_mod.register_forward_hook(lambda m, i, o: self.out.__setitem__("attn", _first(o).detach())),
            self.mlp_mod.register_forward_hook(lambda m, i, o: self.out.__setitem__("mlp", _first(o).detach())),
        ]

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


class FinTSSteerer:
    def __init__(self, model, tokenizer, store: FinTSStore, *, layer: int | None = None, top_k: int = 5,
                 alpha: float = 1.0, beta: float | None = None, weighting: str = "attn", max_length: int = 1024, device=None) -> None:
        self.model, self.tokenizer, self.store = model, tokenizer, store
        self.layer = layer or store.layer
        self.top_k, self.alpha, self.beta, self.weighting = top_k, alpha, (alpha if beta is None else beta), weighting
        self.max_length = max_length
        self.device = device or next(model.parameters()).device
        self.attn_mod, self.mlp_mod = sublayers(decoder_layers(model)[self.layer - 1])

    def tokenize(self, text: str):
        side = self.tokenizer.truncation_side; self.tokenizer.truncation_side = "left"
        try:
            return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        finally:
            self.tokenizer.truncation_side = side

    def query_embedding(self, prompt: str) -> torch.Tensor:
        """Mean-pooled layer-L residual of the prompt (same encoder as the store)."""
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer][0].float().mean(0).cpu()

    @contextmanager
    def injecting(self, s_attn: torch.Tensor, s_mlp: torch.Tensor):
        a = (self.alpha * s_attn).to(self.device); b = (self.beta * s_mlp).to(self.device)

        def h_attn(m, i, o):
            return (o[0] + a.to(o[0].dtype), *o[1:]) if isinstance(o, tuple) else o + a.to(o.dtype)

        def h_mlp(m, i, o):
            return (o[0] + b.to(o[0].dtype), *o[1:]) if isinstance(o, tuple) else o + b.to(o.dtype)

        hs = [self.attn_mod.register_forward_hook(h_attn), self.mlp_mod.register_forward_hook(h_mlp)]
        try:
            yield
        finally:
            for h in hs:
                h.remove()

    def generate(self, user: str, prompt: str, *, max_new_tokens: int = 64) -> tuple[str, bool]:
        q = self.query_embedding(prompt)
        vec = self.store.retrieve(user, q, top_k=self.top_k, weighting=self.weighting)
        inputs = self.tokenize(prompt)
        if vec is None:  # unknown user: no steering
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip(), False
        with self.injecting(*vec), torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip(), True
