#!/usr/bin/env python
"""SteerX baseline (arXiv 2510.22256, Oct 2025) — style-vector (SV) variant,
training-free, equal-conditions re-implementation.

Per user u with history H_u = the user's calibration responses:
  1. preference-driven tokens: for each history item h_j, per-token attribution
       delta_t = log p(h_t | H^-_j, C) - log p(h_t | none, C)
     with H^-_j = other items of the same user in the prompt (leave-one-out) and
     C = the item's request; anchor tokens = {t : delta_t > tau}.
  2. style vector at layer L = mean over anchor positions of
       0.5 * [ h_L(h_j | H^-_j) - h_L(h_j | none) ]          (user-authentic vs neutral)
     + 0.5 * [ h_L(h_j | S_u)   - h_L(h_j | none) ]          (style description vs neutral)
     where S_u (the paper's LLM "Coherence" description) is approximated by the
     user's most frequent anchor tokens (no external LLM; noted in the paper).
  3. inference: gamma * SV_u added at layer L at all positions (persistent
     injection, the same mechanism as ours); gamma = the scale grid.
Written as a `routed_mean_delta` checkpoint with one-hot per-user routing, so
scripts/eval/persistent_steer.py runs it unchanged.
"""
from __future__ import annotations

import argparse, random, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

from persjepa.chat import render_chat_prompt, wants_chat_template
from persjepa.config import ExperimentConfig
from persjepa.data import read_jsonl
from persjepa.intervention import load_causal_lm
from persjepa.routing import RoutingTable, group_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="calibration jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--max-per-user", type=int, default=20)
    ap.add_argument("--context-items", type=int, default=3)
    ap.add_argument("--tau", type=float, default=1.0, help="attribution threshold (nats)")
    ap.add_argument("--min-anchors", type=int, default=3, help="fallback: top-k tokens when fewer exceed tau")
    ap.add_argument("--desc-tokens", type=int, default=30)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-target-tokens", type=int, default=96)
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = ExperimentConfig()
    recs = list(read_jsonl(a.input)); random.Random(a.seed).shuffle(recs)
    by_user: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        u = group_of(r, a.group_field)
        if len(by_user[u]) < a.max_per_user: by_user[u].append(r)
    model, tok, device = load_causal_lm(a.model_name, pred_token=cfg.model.pred_token, device=a.device, dtype=a.dtype)
    chat = wants_chat_template(a.model_name, tok, a.chat_template)
    fmt = (lambda t: render_chat_prompt(tok, t)) if chat else (lambda t: t)

    def item(r):
        return str(r.get(cfg.data.text_field, "")).strip(), str(r.get(cfg.data.target_field, "")).strip()

    def prompt_text(request: str, context: str | None) -> str:
        head = (context.strip() + "\n\n") if context else ""
        return fmt(f"{head}Request: {request}\nUser's response:")

    def run(prompt: str, target: str):
        """token log-probs of the target and layer-L hidden states at target positions."""
        side = tok.truncation_side; tok.truncation_side = "left"
        try: p_ids = tok(prompt, truncation=True, max_length=a.max_length - a.max_target_tokens)["input_ids"]
        finally: tok.truncation_side = side
        t_ids = tok(" " + target, truncation=True, max_length=a.max_target_tokens, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + t_ids], device=device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        logp = torch.log_softmax(out.logits[0, len(p_ids) - 1: len(p_ids) - 1 + len(t_ids)].float(), -1)
        tok_lp = logp.gather(-1, torch.tensor(t_ids, device=device)[:, None])[:, 0]
        h = out.hidden_states[a.layer][0, len(p_ids): len(p_ids) + len(t_ids)].float()
        return tok_lp.cpu(), h.cpu(), t_ids

    users = sorted(by_user); vectors, n_anchor_stats = [], {}
    for u in tqdm(users, desc="steerx users"):
        items = by_user[u]; term1, term2, anchor_counter, cache = [], [], Counter(), []
        for j, r in enumerate(items):
            req, tgt = item(r)
            if not tgt: continue
            others = [item(o)[1] for k, o in enumerate(items) if k != j][: a.context_items]
            ctx = "Examples of this user's writing:\n" + "\n".join(f"- {o[:300]}" for o in others) if others else None
            lp_ctx, h_ctx, t_ids = run(prompt_text(req, ctx), tgt)
            lp_none, h_none, _ = run(prompt_text(req, None), tgt)
            delta = lp_ctx - lp_none
            anchors = (delta > a.tau).nonzero()[:, 0]
            if anchors.numel() < a.min_anchors:
                anchors = delta.topk(min(a.min_anchors, delta.numel())).indices
            anchor_counter.update(tok.decode([t_ids[i]]).strip().lower() for i in anchors.tolist())
            term1.append((h_ctx[anchors] - h_none[anchors]).mean(0))
            cache.append((req, tgt, anchors, h_none))
        desc = "Style of this user: " + ", ".join(t for t, _ in anchor_counter.most_common(a.desc_tokens) if t)
        for req, tgt, anchors, h_none in cache:
            _, h_desc, _ = run(prompt_text(req, desc), tgt)
            n = min(h_desc.shape[0], h_none.shape[0]); anchors = anchors[anchors < n]
            term2.append((h_desc[anchors] - h_none[anchors]).mean(0))
        sv = 0.5 * torch.stack(term1).mean(0) + 0.5 * torch.stack(term2).mean(0)
        vectors.append(sv); n_anchor_stats[u] = {"items": len(cache), "anchors": sum(anchor_counter.values()), "desc": desc[:200]}
    group_means = torch.stack(vectors)
    table = RoutingTable(centroids=group_means.clone(), group_names=users, tau=0.0, mode="hard",
                         weights={u: torch.nn.functional.one_hot(torch.tensor(i), len(users)).float() for i, u in enumerate(users)})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_type": "routed_mean_delta", "group_means": group_means, "group_names": users,
                "routing_table": table.to_payload(), "baseline": "steerx_sv", "layer": a.layer,
                "args": vars(a), "stats": n_anchor_stats}, a.output)
    norms = group_means.norm(dim=1)
    print(f"[steerx] wrote {a.output}: {len(users)} users, layer {a.layer}, |SV| mean {norms.mean():.2f} (min {norms.min():.2f}, max {norms.max():.2f}); "
          f"mean anchors/item {sum(v['anchors'] for v in n_anchor_stats.values())/max(1,sum(v['items'] for v in n_anchor_stats.values())):.1f}")


if __name__ == "__main__":
    main()
