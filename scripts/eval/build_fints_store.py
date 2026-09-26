#!/usr/bin/env python
"""Build the FinTS per-user steering-vector store from calibration pairs.

For each calibration record (profile, prompt, target, metadata):
  X+ = personalized prompt (the user's own profile/history + query)
  X- = the query with another user's profile (irrelevant context)
  d_attn/d_mlp = sublayer outputs at layer L, last prompt token, X+ minus X-
  q  = mean-pooled layer-L residual of the generic query (retrieval key)
Stored per user identity (group_of). Chat template applied like everywhere else.
"""
from __future__ import annotations

import argparse, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

from persjepa.chat import render_chat_prompt, wants_chat_template
from persjepa.config import ExperimentConfig
from persjepa.data import make_personalized_prompt, read_jsonl
from persjepa.fints import FinTSStore, SublayerCapture
from persjepa.intervention import load_causal_lm
from persjepa.routing import group_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="calibration jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-per-user", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = ExperimentConfig()
    recs = list(read_jsonl(a.input))
    rng = random.Random(a.seed); rng.shuffle(recs)
    if a.max_examples: recs = recs[: a.max_examples]
    users = [group_of(r, a.group_field) for r in recs]
    by_user: dict[str, list[int]] = {}
    for i, u in enumerate(users): by_user.setdefault(u, []).append(i)
    if a.max_per_user:
        keep = sorted(i for idx in by_user.values() for i in idx[: a.max_per_user]); recs = [recs[i] for i in keep]; users = [users[i] for i in keep]
    model, tok, device = load_causal_lm(a.model_name, pred_token=cfg.model.pred_token, device=a.device, dtype=a.dtype)
    chat = wants_chat_template(a.model_name, tok, a.chat_template)
    fmt = (lambda t: render_chat_prompt(tok, t)) if chat else (lambda t: t)
    cap = SublayerCapture(model, a.layer)
    store = FinTSStore(layer=a.layer, meta={"model": a.model_name, "source": a.input, "chat": chat, "n": len(recs)})

    def run(text: str):
        side = tok.truncation_side; tok.truncation_side = "left"
        try:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=a.max_length).to(device)
        finally:
            tok.truncation_side = side
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[a.layer][0].float().mean(0).cpu(), cap.out["attn"][0, -1].float().cpu(), cap.out["mlp"][0, -1].float().cpu()

    for i, (r, u) in enumerate(tqdm(list(zip(recs, users)), desc="fints store")):
        prompt = str(r.get(cfg.data.text_field, r.get("prompt", "")))
        profile = str(r.get(cfg.data.profile_field, ""))
        j = rng.randrange(len(recs))
        while len(recs) > 1 and users[j] == u:
            j = rng.randrange(len(recs))
        other_profile = str(recs[j].get(cfg.data.profile_field, ""))
        x_pos = fmt(make_personalized_prompt(profile, prompt, cfg.data.profile_template))
        x_neg = fmt(make_personalized_prompt(other_profile, prompt, cfg.data.profile_template))
        q, _, _ = run(fmt(prompt))
        _, ap_, mp_ = run(x_pos)
        _, an_, mn_ = run(x_neg)
        store.add(u, q[None], (ap_ - an_)[None], (mp_ - mn_)[None])
    cap.remove()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(store.to_payload(), a.output)
    print(f"[fints] wrote {a.output}: {len(store.users)} users, {sum(v['q'].shape[0] for v in store.users.values())} samples, layer {a.layer}")


if __name__ == "__main__":
    main()
