#!/usr/bin/env python
"""Interpretability of a routed JEPA-SAE (likelihood-trained experts).

Per group expert: feature activation rates on the group's own calibration
examples vs the other groups (specificity), top-activating examples, decoder
direction logit lens (tokens promoted through the final norm + unembedding).
Across experts: per-feature decoder divergence (experts share the feature
index space, having been fine-tuned from one global SAE). Group offsets: the
per-persona constant through the logit lens.

Outputs (in --output-dir): feature_stats.csv, feature_cards.md,
group_offsets_tokens.csv, expert_divergence.csv, summary.json
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from persjepa.intervention import final_norm_module, load_causal_lm, load_routed_jepa_sae
from persjepa.routing import answer_span_residuals, group_of


def short(t: str, n: int = 140) -> str:
    t = " ".join(str(t).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="routed_jepa_sae payload (e.g. ablations/ep2.pt)")
    ap.add_argument("--span-hidden", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=None, help="for the logit lens (omit to skip)")
    ap.add_argument("--group-field", default=None)
    ap.add_argument("--top-features", type=int, default=8)
    ap.add_argument("--top-examples", type=int, default=4)
    ap.add_argument("--top-tokens", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    routed, pay = load_routed_jepa_sae(a.checkpoint, device="cpu")
    names = list(routed.group_names); G = len(names); gi = {n: i for i, n in enumerate(names)}
    sh = torch.load(a.span_hidden, map_location="cpu", weights_only=False)
    h_gen, r = answer_span_residuals(sh)
    exs = sh["examples"]
    users = [group_of(e, a.group_field) for e in exs]
    grp = torch.tensor([gi.get(u, -1) for u in users])
    F = routed.experts[0].latent_dim

    # ---- activations of every expert on every example
    with torch.no_grad():
        acts = torch.stack([routed.experts[g].encode(h_gen)[0] for g in range(G)])   # [G, N, F]
    active = acts > 0
    rows, cards = [], {}
    for g, name in enumerate(names):
        own = grp == g; other = (grp >= 0) & ~own
        if own.sum() == 0: continue
        rate_own = active[g][own].float().mean(0); rate_other = active[g][other].float().mean(0)
        mean_own = (acts[g][own].sum(0) / active[g][own].sum(0).clamp(min=1))
        spec = (rate_own + 1e-3) / (rate_other + 1e-3)
        W = routed.experts[g].delta_decoder.weight.detach()  # [D, F]
        dnorm = W.norm(dim=0)
        score = rate_own * torch.log(spec.clamp(min=1e-6)).clamp(min=0) * dnorm
        top = score.topk(min(a.top_features, F)).indices.tolist()
        cards[name] = []
        for f in range(F):
            rows.append({"group": name, "feature": f, "rate_own": round(float(rate_own[f]), 4), "rate_other": round(float(rate_other[f]), 4),
                         "specificity": round(float(spec[f]), 3), "mean_act_own": round(float(mean_own[f]), 4), "decoder_norm": round(float(dnorm[f]), 4)})
        own_idx = own.nonzero()[:, 0]
        for f in top:
            a_f = acts[g][own_idx, f]
            ex_top = own_idx[a_f.topk(min(a.top_examples, len(own_idx))).indices].tolist()
            cards[name].append({"feature": f, "rate_own": float(rate_own[f]), "rate_other": float(rate_other[f]), "specificity": float(spec[f]),
                                "decoder_norm": float(dnorm[f]), "examples": [{"id": exs[i].get("id"), "act": float(acts[g][i, f]),
                                "prompt": short(exs[i].get("generic_prompt", "")), "target": short(exs[i].get("target", ""))} for i in ex_top]})
    with open(out / "feature_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- expert divergence per feature (mean pairwise cosine of decoder directions)
    Ws = torch.stack([e.delta_decoder.weight.detach().t() for e in routed.experts])  # [G, F, D]
    Wn = torch.nn.functional.normalize(Ws, dim=-1)
    cos = torch.einsum("gfd,hfd->fgh", Wn, Wn)
    iu = torch.triu_indices(G, G, 1)
    alive = (Ws.norm(dim=-1) > 1e-6).all(0)                                   # feature used by every expert
    mean_cos = cos[:, iu[0], iu[1]].mean(1)
    mean_cos = torch.where(alive, mean_cos, torch.full_like(mean_cos, float("nan")))
    with open(out / "expert_divergence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature", "mean_pairwise_cos", "min_pairwise_cos", "mean_decoder_norm"])
        for fi in range(F):
            w.writerow([fi, round(float(mean_cos[fi]), 4) if alive[fi] else "", round(float(cos[fi, iu[0], iu[1]].min()), 4) if alive[fi] else "", round(float(Ws[:, fi].norm(dim=-1).mean()), 4)])

    # ---- logit lens of group offsets and top features
    lens_offsets, lens_feats = [], {}
    if a.model_name:
        model, tok, device = load_causal_lm(a.model_name, pred_token="[PRED]", device=a.device, dtype="bfloat16")
        norm = final_norm_module(model); W_U = model.get_output_embeddings().weight.detach().float()  # [V, D]
        nw = getattr(norm, "weight", None); nw = nw.detach().float() if nw is not None else torch.ones(W_U.shape[1])

        def lens(v: torch.Tensor):
            v = v.float().to(W_U.device); v = v / v.pow(2).mean().sqrt().clamp(min=1e-6)
            logits = W_U @ (nw.to(W_U.device) * v)
            top = logits.topk(a.top_tokens).indices.tolist(); bot = (-logits).topk(a.top_tokens).indices.tolist()
            return [tok.decode([t]).strip() for t in top], [tok.decode([t]).strip() for t in bot]
        for g, name in enumerate(names):
            up, down = lens(routed.group_offsets[g].detach())
            lens_offsets.append({"group": name, "offset_norm": round(float(routed.group_offsets[g].norm()), 3), "promoted": " | ".join(up), "suppressed": " | ".join(down)})
        with open(out / "group_offsets_tokens.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["group", "offset_norm", "promoted", "suppressed"]); w.writeheader(); w.writerows(lens_offsets)
        Wmean = Ws.mean(0)  # [F, D] mean decoder direction across experts
        for name, feats in cards.items():
            g = gi[name]
            for c in feats:
                up, down = lens(routed.experts[g].delta_decoder.weight.detach()[:, c["feature"]])
                c["promoted"], c["suppressed"] = up, down
                diff = Ws[g, c["feature"]] - Wmean[c["feature"]]      # what this expert does differently on this feature
                c["diff_norm"] = float(diff.norm()); c["diff_promoted"], c["diff_suppressed"] = lens(diff)

    # ---- markdown cards
    with open(out / "feature_cards.md", "w") as f:
        f.write(f"# Routed expert feature cards — {a.checkpoint}\n\nGroups: {G}; latent dim {F}; top-k {routed.experts[0].top_k}; calibration examples {len(exs)}.\n\n")
        if lens_offsets:
            f.write("## Group offsets (per-persona constant) through the logit lens\n\n| group | ‖offset‖ | promoted | suppressed |\n|---|---|---|---|\n")
            for r_ in lens_offsets: f.write(f"| {r_['group']} | {r_['offset_norm']} | {r_['promoted']} | {r_['suppressed']} |\n")
            f.write("\n")
        f.write("## Most divergent features across experts (lowest mean pairwise decoder cosine)\n\n| feature | mean cos | min cos |\n|---|---|---|\n")
        for fi in torch.nan_to_num(mean_cos, nan=2.0).topk(10, largest=False).indices.tolist():
            f.write(f"| {fi} | {float(mean_cos[fi]):.3f} | {float(cos[fi, iu[0], iu[1]].min()):.3f} |\n")
        f.write(f"\nAlive features (non-zero decoder in every expert): {int(alive.sum())}/{F}; mean pairwise decoder cosine over alive features: {float(mean_cos[alive].mean()):.3f}\n\n")
        for name, feats in cards.items():
            f.write(f"## {name}\n\n")
            for c in feats:
                f.write(f"### feature {c['feature']} — rate own {c['rate_own']:.2f} vs other {c['rate_other']:.2f} (×{c['specificity']:.1f}), ‖dec‖ {c['decoder_norm']:.2f}\n\n")
                if "promoted" in c: f.write(f"- promotes: {' | '.join(c['promoted'])}\n- suppresses: {' | '.join(c['suppressed'])}\n")
                if "diff_promoted" in c: f.write(f"- this expert vs mean of experts (‖Δ‖ {c['diff_norm']:.2f}) promotes: {' | '.join(c['diff_promoted'])}\n")
                for e in c["examples"]:
                    f.write(f"- ({e['act']:.2f}) **{e['prompt']}** → {e['target']}\n")
                f.write("\n")
    summ = {"checkpoint": a.checkpoint, "groups": names, "latent_dim": F, "n_examples": len(exs),
            "alive_features": int(alive.sum()), "mean_pairwise_decoder_cos": float(mean_cos[alive].mean()), "min_feature_cos": float(mean_cos[alive].min()),
            "specific_features_per_group": {n: int(sum(1 for r_ in rows if r_["group"] == n and r_["specificity"] > 2 and r_["rate_own"] > 0.1)) for n in names}}
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
