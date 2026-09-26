"""TAP-PER baseline (arXiv 2606.04547, Jun 2026): compact user representations
for LLM personalization. Equal-conditions re-implementation.

Per user u: a learnable user-state prefix P_u in R^{L x d} (embedding table E).
Per query q: a query-aware temporal record prefix P_q built from the user's
history records h_j (mean-pooled frozen input embeddings) with DIN-style
attention  s_j = MLP([z_q ; z_hj ; z_q - z_hj ; z_q * z_hj]) minus a temporal /
order-gap decay, softmax-normalised, aggregated and projected to L x d.
A shared "bridge" LoRA on q/k/v/o projections integrates the prefixes.
Training: CE of the personalized target given [P_u ; P_q ; query] with the
backbone frozen (bridge LoRA, E, DIN MLP and projection trained).

Our mapping: users = the same identities as the routed SAE (persona / user id);
records = the user's calibration targets (their own past outputs), which is
exactly the information our z_u is built from; objective and data identical to
the likelihood-trained routed SAE. Stage-1 task LoRA of the paper is omitted
(our method has no task adaptation either).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class TapperPrefix(nn.Module):
    def __init__(self, users: list[str], d_model: int, *, prefix_len: int = 8, din_hidden: int = 256,
                 max_records: int = 32, lambda_o: float = 0.1) -> None:
        super().__init__()
        self.users = list(users)
        self.user_index = {u: i for i, u in enumerate(self.users)}
        self.d, self.L, self.max_records = d_model, prefix_len, max_records
        self.E = nn.Embedding(len(self.users), prefix_len * d_model)
        nn.init.normal_(self.E.weight, std=0.02)
        self.din = nn.Sequential(nn.Linear(4 * d_model, din_hidden), nn.GELU(), nn.Linear(din_hidden, 1))
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, prefix_len * d_model))
        self.lambda_o = nn.Parameter(torch.tensor(float(lambda_o)))
        self.records: dict[str, torch.Tensor] = {}  # user -> [n, d] frozen record embeddings (most recent last)

    # ----------------------------------------------------------------- prefixes
    def user_prefix(self, user: str) -> torch.Tensor:
        idx = self.user_index.get(user)
        if idx is None:  # unknown user: mean user state
            return self.E.weight.mean(0).view(self.L, self.d)
        return self.E(torch.tensor(idx, device=self.E.weight.device)).view(self.L, self.d)

    def record_prefix(self, user: str, z_q: torch.Tensor, exclude: int | None = None) -> torch.Tensor:
        recs = self.records.get(user)
        if recs is not None and exclude is not None and 0 <= exclude < recs.shape[0]:
            recs = torch.cat([recs[:exclude], recs[exclude + 1:]])       # leave the current target out (training)
        if recs is None or recs.shape[0] == 0:
            return torch.zeros(self.L, self.d, device=z_q.device, dtype=z_q.dtype)
        recs = recs[-self.max_records:].to(z_q.device, z_q.dtype)          # [n, d], most recent last
        n = recs.shape[0]
        zq = z_q[None].expand(n, -1)
        s = self.din(torch.cat([zq, recs, zq - recs, zq * recs], dim=-1)).squeeze(-1)   # [n]
        order_gap = torch.arange(n - 1, -1, -1, device=z_q.device, dtype=z_q.dtype)   # 0 for the most recent
        s = s - self.lambda_o * torch.log1p(order_gap)
        w = torch.softmax(s, dim=0)
        h = (w[:, None] * recs).sum(0)
        return self.proj(h).view(self.L, self.d)

    def prefixes(self, user: str, z_q: torch.Tensor, exclude: int | None = None) -> torch.Tensor:
        return torch.cat([self.user_prefix(user).to(z_q.dtype), self.record_prefix(user, z_q, exclude)], dim=0)  # [2L, d]

    def export(self) -> dict:
        return {"model_type": "tapper", "users": self.users, "d_model": self.d, "prefix_len": self.L,
                "max_records": self.max_records, "state_dict": self.state_dict(),
                "records": {u: v.to(torch.float16) for u, v in self.records.items()}}

    @classmethod
    def from_payload(cls, pay: dict) -> "TapperPrefix":
        m = cls(pay["users"], int(pay["d_model"]), prefix_len=int(pay["prefix_len"]), max_records=int(pay["max_records"]))
        m.load_state_dict(pay["state_dict"])
        m.records = {u: v.float() for u, v in pay["records"].items()}
        return m


@torch.no_grad()
def embed_text_mean(model, tokenizer, text: str, *, device, max_length: int = 256) -> torch.Tensor:
    """Mean-pooled frozen input-embedding of a text (the paper's record encoder)."""
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(device)
    emb = model.get_input_embeddings()(ids)[0]
    return emb.float().mean(0)


def build_inputs_with_prefix(model, prefix: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """[prefix ; token embeddings], extended attention mask. prefix: [P, d]; input_ids: [1, T]."""
    tok_emb = model.get_input_embeddings()(input_ids)                       # [1, T, d]
    pre = prefix[None].to(tok_emb.dtype)                                    # [1, P, d]
    inputs_embeds = torch.cat([pre, tok_emb], dim=1)
    attn = torch.cat([torch.ones(1, pre.shape[1], dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=1)
    return inputs_embeds, attn


def add_bridge_lora(model, *, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
    from peft import LoraConfig, get_peft_model
    targets = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj") if any(n in name for name, _ in model.named_modules())]
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)
