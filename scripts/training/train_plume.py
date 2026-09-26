#!/usr/bin/env python
"""Train the PLUME baseline: (A) shared task LoRA on the pooled calibration data,
then (B) per-user subspace mixers + cross-layer-shared low-rank pair + rank-1
residuals on each user's own examples. Same CE objective / data as the routed
SAE likelihood stage; backbone frozen. Writes a `plume` checkpoint."""
from __future__ import annotations

import argparse, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

from persjepa.chat import apply_chat_formatting
from persjepa.config import ExperimentConfig
from persjepa.data import load_persona_jsonl
from persjepa.intervention import load_causal_lm
from persjepa.persistent import target_nll
from persjepa.plume import DEFAULT_TARGETS, PlumeAdapter, attach_plume, linear_shapes
from persjepa.routing import group_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="calibration jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--rank-sh", type=int, default=4)
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--shared-epochs", type=int, default=1)
    ap.add_argument("--user-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--user-lr", type=float, default=1e-3)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--max-target-tokens", type=int, default=64)
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    cfg = ExperimentConfig()
    ex = load_persona_jsonl(a.input, profile_field=cfg.data.profile_field, prompt_field=cfg.data.text_field,
                            target_field=cfg.data.target_field, id_field=cfg.data.id_field, profile_template=cfg.data.profile_template)
    random.Random(a.seed).shuffle(ex)
    if a.max_examples: ex = ex[: a.max_examples]
    model, tok, device = load_causal_lm(a.model_name, pred_token=cfg.model.pred_token, device=a.device, dtype=a.dtype)
    apply_chat_formatting(ex, tok, a.model_name, a.chat_template)
    for p in model.parameters(): p.requires_grad_(False)
    users = [group_of({"metadata": e.metadata or {}}, a.group_field) for e in ex]
    names = sorted(set(users))
    targets = tuple(t for t in a.targets.split(",") if t)
    adapter = PlumeAdapter(linear_shapes(model, targets), rank=a.rank, rank_sh=a.rank_sh, users=names).to(device)
    n = attach_plume(model, adapter, targets)
    print(f"[plume] users={len(names)} wrapped={n} shared={sum(p.numel() for p in adapter.shared_parameters())/1e6:.2f}M per-user={adapter.per_user_numel()/1e6:.3f}M")

    def encode(e):
        side = tok.truncation_side; tok.truncation_side = "left"
        try: p_ids = tok(e.generic_prompt, truncation=True, max_length=a.max_length - a.max_target_tokens)["input_ids"]
        finally: tok.truncation_side = side
        t_ids = tok(" " + e.target.strip(), truncation=True, max_length=a.max_target_tokens, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = torch.tensor([p_ids + t_ids], device=device); attn = torch.ones_like(ids)
        tm = torch.tensor([[0] * len(p_ids) + [1] * len(t_ids)], device=device)
        return ids, attn, tm

    def run_epochs(idx, params, lr, epochs, desc, log):
        for p in params: p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); step = 0
        for epoch in range(1, epochs + 1):
            order = list(idx); random.Random(a.seed + epoch).shuffle(order)
            pbar = tqdm(range(0, len(order), a.batch_size), desc=f"{desc} ep{epoch}", leave=False)
            for start in pbar:
                opt.zero_grad(set_to_none=True); tot = 0.0
                batch = order[start: start + a.batch_size]
                for i in batch:
                    ids, attn, tm = encode(ex[i])
                    logits = model(input_ids=ids, attention_mask=attn).logits
                    loss = target_nll(logits, ids, attn, tm) / len(batch)
                    loss.backward(); tot += float(loss)
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
                log.append(tot); pbar.set_postfix(nll=f"{tot:.3f}")
        for p in params: p.requires_grad_(False)
        return step

    # (A) shared task subspace on pooled data (no active user)
    log_shared: list[float] = []
    adapter.set_user(None)
    run_epochs(range(len(ex)), adapter.shared_parameters(), a.lr, a.shared_epochs, "plume shared", log_shared)
    print(f"[plume] shared stage: first nll {sum(log_shared[:20])/max(1,len(log_shared[:20])):.3f} -> last {sum(log_shared[-20:])/max(1,len(log_shared[-20:])):.3f}")
    # (B) per-user modulation, each user on its own examples, shared frozen
    by_user: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(users): by_user[u].append(i)
    log_users: dict[str, list[float]] = {}
    for u in tqdm(names, desc="plume users"):
        adapter.set_user(u); lg: list[float] = []
        run_epochs(by_user[u], adapter.user_parameters(u), a.user_lr, a.user_epochs, f"plume {u[:12]}", lg)
        log_users[u] = lg
    adapter.set_user(None)
    lasts = [sum(v[-10:]) / max(1, len(v[-10:])) for v in log_users.values() if v]
    print(f"[plume] user stage: mean last nll {sum(lasts)/max(1,len(lasts)):.3f} over {len(lasts)} users")
    pay = adapter.export(); pay["model_name"] = a.model_name; pay["targets"] = list(targets)
    pay["training"] = {"args": vars(a), "log_shared": log_shared, "log_users": log_users}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); torch.save(pay, a.output)
    print(f"[plume] wrote {a.output}")


if __name__ == "__main__":
    main()
