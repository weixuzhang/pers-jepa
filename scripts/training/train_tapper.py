#!/usr/bin/env python
"""Train the TAP-PER baseline (user-state prefix + query-aware record prefix +
shared bridge LoRA) with the same likelihood objective and calibration data as
the routed SAE. Writes a `tapper` checkpoint (prefix module + LoRA weights +
per-user record embeddings) that scripts/eval/persistent_steer.py can run."""
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
from persjepa.routing import group_of
from persjepa.tapper import TapperPrefix, add_bridge_lora, build_inputs_with_prefix, embed_text_mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="calibration jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--prefix-len", type=int, default=8)
    ap.add_argument("--max-records", type=int, default=32)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8, help="examples per optimizer step (accumulated; forward is per example)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--max-target-tokens", type=int, default=64)
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-only", action="store_true", help="control: bridge LoRA only, no user-state / record prefixes (pure task adaptation)")
    ap.add_argument("--prompt-only", action="store_true", help="baseline: per-user soft prompt (prompt tuning) with the backbone frozen: no LoRA, no record prefix")
    ap.add_argument("--micro-batch", type=int, default=4, help="prompt-only: examples per padded forward pass (token-mean NLL over the step, as in the likelihood trainer)")
    ap.add_argument("--prompt-shared", choices=["on", "off"], default="on",
                    help="prompt-only (OPPU-style base + personal PEFT): user prompt = shared prompt (trained on all users) + per-user residual (zero init)")
    ap.add_argument("--prompt-init", choices=["vocab", "random"], default="vocab", help="prompt-only: shared prompt initialised from sampled vocabulary embeddings (Lester et al. 2021)")
    ap.add_argument("--user-lr-scale", type=float, default=0.1, help="prompt-only + shared: learning rate of the per-user residuals relative to --lr")
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
    d_model = model.get_input_embeddings().weight.shape[1]
    prefix = TapperPrefix(names, d_model, prefix_len=a.prefix_len, max_records=a.max_records).to(device)
    # records = the user's own calibration targets (past outputs), frozen mean-pooled embeddings
    recs: dict[str, list[torch.Tensor]] = defaultdict(list); rec_pos: list[int] = []
    if not a.prompt_only:
        for e, u in tqdm(list(zip(ex, users)), desc="records"):
            rec_pos.append(len(recs[u])); recs[u].append(embed_text_mean(model, tok, e.target, device=device).cpu())
        prefix.records = {u: torch.stack(v) for u, v in recs.items()}
    shared = None
    if a.prompt_only:
        params = [prefix.E.weight]
        emb_w = model.get_input_embeddings().weight
        if a.prompt_init == "vocab":   # sampled vocabulary embeddings: the prompt starts inside the token-embedding distribution
            g = torch.Generator().manual_seed(a.seed)
            init = emb_w[torch.randint(0, min(tok.vocab_size, emb_w.shape[0]), (a.prefix_len,), generator=g).to(emb_w.device)].detach().float()
        else:
            init = torch.randn(a.prefix_len, d_model, device=device) * 0.02
        if a.prompt_shared == "on":
            shared = torch.nn.Parameter(init.clone().to(device))
            with torch.no_grad(): prefix.E.weight.zero_()
            params = [shared, prefix.E.weight]
        elif a.prompt_init == "vocab":
            with torch.no_grad(): prefix.E.weight.copy_(init.reshape(1, -1).expand_as(prefix.E.weight))
    else:
        model = add_bridge_lora(model, rank=a.lora_rank)
        params = ([] if a.lora_only else [p for p in prefix.parameters()]) + [p for n, p in model.named_parameters() if "lora_" in n]
    for p in params: p.requires_grad_(True)
    print(f"[tapper] users={len(names)} records={sum(v.shape[0] for v in prefix.records.values())} trainable={sum(p.numel() for p in params)/1e6:.2f}M")
    if shared is not None:
        opt = torch.optim.AdamW([{"params": [shared], "lr": a.lr}, {"params": [prefix.E.weight], "lr": a.lr * a.user_lr_scale}], weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    def upre(u):  # the user's soft prompt [L, d] (float32)
        p = prefix.user_prefix(u).to(torch.float32)
        return p if shared is None else p + shared

    def encode(e):
        side = tok.truncation_side; tok.truncation_side = "left"
        try: p_ids = tok(e.generic_prompt, truncation=True, max_length=a.max_length - a.max_target_tokens)["input_ids"]
        finally: tok.truncation_side = side
        t_ids = tok(" " + e.target.strip(), truncation=True, max_length=a.max_target_tokens, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = torch.tensor([p_ids + t_ids], device=device); attn = torch.ones_like(ids)
        tm = torch.tensor([[0] * len(p_ids) + [1] * len(t_ids)], device=device)
        return ids, attn, tm

    step, log = 0, []
    for epoch in range(1, a.epochs + 1):
        order = list(range(len(ex))); random.Random(a.seed + epoch).shuffle(order)
        pbar = tqdm(range(0, len(order), a.batch_size), desc=f"tapper epoch {epoch}")
        for start in pbar:
            opt.zero_grad(set_to_none=True); tot = 0.0
            batch = order[start: start + a.batch_size]
            if a.prompt_only:  # batched, right-padded forward; token-mean NLL over the whole step
                items = []
                for i in batch:
                    ids, attn, tm = encode(ex[i]); pre = upre(users[i])
                    emb, attn2 = build_inputs_with_prefix(model, pre, ids, attn); P = pre.shape[0]
                    lab = torch.cat([torch.zeros(1, P, device=device, dtype=ids.dtype), ids], dim=1)
                    tm2 = torch.cat([torch.zeros(1, P, device=device, dtype=tm.dtype), tm], dim=1)
                    items.append((emb[0], attn2[0], lab[0], tm2[0]))
                n_tgt = sum(int(t[3][1:].sum()) for t in items)
                for c0 in range(0, len(items), a.micro_batch):
                    chunk = items[c0: c0 + a.micro_batch]; T = max(t[0].shape[0] for t in chunk)
                    E = torch.zeros(len(chunk), T, chunk[0][0].shape[-1], device=device, dtype=chunk[0][0].dtype)
                    A = torch.zeros(len(chunk), T, device=device, dtype=chunk[0][1].dtype)
                    Lb = torch.zeros(len(chunk), T, device=device, dtype=chunk[0][2].dtype)
                    M = torch.zeros(len(chunk), T, device=device, dtype=chunk[0][3].dtype)
                    for r, (e_, a_, l_, m_) in enumerate(chunk):
                        n_ = e_.shape[0]; E[r, :n_] = e_; A[r, :n_] = a_; Lb[r, :n_] = l_; M[r, :n_] = m_
                    logits = model(inputs_embeds=E, attention_mask=A).logits
                    n_c = int((M[:, 1:] * A[:, 1:]).sum())
                    loss = target_nll(logits, Lb, A, M) * (n_c / max(n_tgt, 1))
                    loss.backward(); tot += float(loss)
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
                if step % 20 == 0: log.append({"step": step, "nll": tot}); pbar.set_postfix(nll=f"{tot:.3f}")
                continue
            for i in batch:
                e, u = ex[i], users[i]
                ids, attn, tm = encode(e)
                if a.prompt_only:
                    pre = upre(u)
                elif a.lora_only:
                    pre = torch.zeros(0, d_model, device=device)
                else:  # the current target is left out of its own record set (no label leakage)
                    z_q = embed_text_mean(model, tok, e.generic_prompt, device=device).to(torch.float32)
                    pre = prefix.prefixes(u, z_q, exclude=rec_pos[i])
                emb, attn2 = build_inputs_with_prefix(model, pre, ids, attn)
                logits = model(inputs_embeds=emb, attention_mask=attn2).logits
                P = pre.shape[0]
                labels = torch.cat([torch.full((1, P), -100, device=device, dtype=ids.dtype), ids], dim=1)
                tm2 = torch.cat([torch.zeros(1, P, device=device, dtype=tm.dtype), tm], dim=1)
                loss = target_nll(logits, labels.clamp(min=0), attn2, tm2) / len(batch)
                loss.backward(); tot += float(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
            if step % 20 == 0: log.append({"step": step, "nll": tot}); pbar.set_postfix(nll=f"{tot:.3f}")
    if shared is not None:  # bake the shared prompt into every user row; unknown users get the mean row (= shared + mean residual)
        with torch.no_grad(): prefix.E.weight.add_(shared.reshape(1, -1).to(prefix.E.weight.dtype))
    pay = prefix.export()
    if not a.prompt_only:
        from peft import get_peft_model_state_dict
        pay["lora_state_dict"] = {k: v.cpu() for k, v in get_peft_model_state_dict(model).items()}
    pay["prompt_only"] = bool(a.prompt_only)
    if a.prompt_only: pay["prompt_recipe"] = {"shared": a.prompt_shared, "init": a.prompt_init, "lr": a.lr, "user_lr_scale": a.user_lr_scale}
    pay["lora_rank"] = a.lora_rank; pay["model_name"] = a.model_name; pay["lora_only"] = bool(a.lora_only); pay["training"] = {"args": vars(a), "log": log}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); torch.save(pay, a.output)
    print(f"[tapper] wrote {a.output} after {step} steps; last nll {log[-1]['nll'] if log else float('nan'):.3f}")


if __name__ == "__main__":
    main()
