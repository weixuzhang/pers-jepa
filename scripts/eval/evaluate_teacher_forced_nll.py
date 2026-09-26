#!/usr/bin/env python
"""Held-out teacher-forced NLL of the personalized target under persistent
injection, for one or more steering checkpoints (plus the no-injection
baseline). This is the quantity the likelihood objective optimises, measured
on data it never saw — the cleanest check that the objective moved."""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

from persjepa.chat import apply_chat_formatting
from persjepa.config import ExperimentConfig
from persjepa.data import load_persona_jsonl
from persjepa.intervention import load_causal_lm, load_steering_sae
from persjepa.persistent import PersistentSteerer, predictor_delta, target_nll
from persjepa.routing import group_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--predictor-tokens", type=int, default=3)
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--max-examples", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--max-target-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = ExperimentConfig()
    ex = load_persona_jsonl(a.input, profile_field=cfg.data.profile_field, prompt_field=cfg.data.text_field,
                            target_field=cfg.data.target_field, id_field=cfg.data.id_field, profile_template=cfg.data.profile_template)
    random.Random(a.seed).shuffle(ex); ex = ex[: a.max_examples]
    model, tok, device = load_causal_lm(a.model_name, pred_token=cfg.model.pred_token, device=a.device, dtype=a.dtype)
    apply_chat_formatting(ex, tok, a.model_name, a.chat_template)
    ck = {}
    for spec in a.checkpoints:
        name, path = spec.split("=", 1); ck[name] = load_steering_sae(path, device=device)[0]

    def encode(e):
        side = tok.truncation_side; tok.truncation_side = "left"
        try: p = tok(e.generic_prompt, truncation=True, max_length=a.max_length - a.max_target_tokens)["input_ids"]
        finally: tok.truncation_side = side
        t = tok(" " + e.target.strip(), truncation=True, max_length=a.max_target_tokens, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = torch.tensor([p + t], device=device); attn = torch.ones_like(ids); tm = torch.tensor([[0] * len(p) + [1] * len(t)], device=device)
        return ids, attn, tm

    steerers = {n: PersistentSteerer(model, tok, m, layer=a.layer, pred_token=cfg.model.pred_token, predictor_tokens=a.predictor_tokens,
                                     max_length=a.max_length, residual_scale=a.residual_scale, device=device) for n, m in ck.items()}
    tot = {n: 0.0 for n in ck}; tot["none"] = 0.0; n_ex = 0
    with torch.no_grad():
        for e in tqdm(ex, desc="tf-nll"):
            ids, attn, tm = encode(e)
            tot["none"] += float(target_nll(model(input_ids=ids, attention_mask=attn).logits, ids, attn, tm))
            for n, m in ck.items():
                if hasattr(m, "set_weights"):
                    t = getattr(m, "routing_table", None); w = t.get(group_of({"metadata": e.metadata or {}}, a.group_field)) if t else None
                    m.set_weights(w if w is not None else torch.full((m.num_groups,), 1.0 / m.num_groups))
                delta = steerers[n].compute_delta(e.generic_prompt)
                with steerers[n].injecting(delta):
                    logits = model(input_ids=ids, attention_mask=attn).logits
                tot[n] += float(target_nll(logits, ids, attn, tm))
            n_ex += 1
    res = {n: v / max(n_ex, 1) for n, v in tot.items()}; res["n"] = n_ex; res["scale"] = a.residual_scale
    Path(a.output).write_text(json.dumps(res, indent=2)); print(json.dumps(res))


if __name__ == "__main__":
    main()
