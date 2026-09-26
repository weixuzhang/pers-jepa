#!/usr/bin/env python
"""Paired user-cluster bootstrap CIs for the real-user table (real-user results).

Works only from finished run directories (generation .jsonl[.gz] + *_userstyle_confusion.csv),
so nothing is re-generated and no attribution classifier is re-trained:
  * ROUGE-1 / ROUGE-L per example (same tokenizer as evaluate_text_metrics.py)
  * LaMP-2 accuracy (same matching rule as evaluate_classification.py) and macro-F1 over tags
  * user-attribution accuracy and macro-F1 from the scorer's (gold, pred, count) confusion rows
Resampling unit = user (clusters): for LaMP-4/5/7 every dev user has exactly one example, so
this is the ordinary paired per-example bootstrap; for LaMP-2 and Amazon it respects the
within-user correlation. Two arms are compared on the users present in both; each arm's metric
is the ratio of sums over its own examples of the resampled users.

  python scripts/eval/bootstrap_real_data.py --out-dir runs/real_data_bootstrap [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval")); sys.path.insert(0, str(ROOT))
from evaluate_text_metrics import f1_score as tok_f1, rouge_l_f1, tokenize  # noqa: E402

G = "generation_seed42/persistent__{}__scale1.0"
OURS = "routed_jepa_sae_lik"
# Run directories written by scripts/ablations/run_stage_dataset.sh (RUN_TAG encodes K).
RUNS = "runs/{}/Qwen_Qwen3.5_4B_L16{}/"


def arms_std(dataset, tag):
    base = RUNS.format(dataset, tag)
    names = ["profile_prompt", "rag_bm25", "fints", "steerx", "plume", "tapper", "routed_mean_delta",
             "routed_jepa_sae", OURS, OURS + "_shuffled", "global_jepa_sae_lik"]
    return {n: base + G.format(n) for n in names}


def arms_k(dataset, k, noroute=False):
    base = RUNS.format(dataset, f"_k{k}")
    arms = {f"ours_K{k}": base + G.format(OURS), f"ours_K{k}_shuffled": base + G.format(OURS + "_shuffled")}
    if noroute:
        arms[f"noroute_K{k}"] = base + G.format("global_jepa_sae_lik")
    return arms


DATASETS = {
    "lamp2": {"kind": "classify", "arms": {**arms_std("lamp2", "_k20"), **arms_k("lamp2", 5), **arms_k("lamp2", 20), **arms_k("lamp2", 10)}},
    "lamp7": {"kind": "gen", "arms": arms_std("lamp7", "_k20")},
    "lamp4": {"kind": "gen", "arms": arms_std("lamp4", "_k20")},
    "lamp5": {"kind": "gen", "arms": arms_std("lamp5", "_k20")},
    "amazon_movies_tv": {"kind": "gen", "arms": {**arms_std("amazon_movies_tv", "_k10"), **arms_k("amazon_movies_tv", 10, noroute=True),
                                                 **arms_k("amazon_movies_tv", 20)}},
}
COMPARE = [("raw", "vs raw"), ("global_jepa_sae_lik", "vs no routing"), (OURS + "_shuffled", "vs permuted"),
           ("tapper", "vs TAP-PER"), ("plume", "vs PLUME"), ("rag_bm25", "vs BM25 in prompt"), ("profile_prompt", "vs profile in prompt")]
EXTRA_COMPARE = [("ours_K5", "ours_K5_shuffled"), ("ours_K10", "ours_K10_shuffled"), ("ours_K20", "ours_K20_shuffled"), ("ours_K10", "noroute_K10")]


def norm_tag(s):
    s = str(s or "").strip().splitlines()[0] if str(s or "").strip() else ""
    s = re.sub(r"^(tag|answer|score|rating)\s*[:：]\s*", "", s, flags=re.I)
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def user_of(r):
    md = r.get("metadata") or {}
    if isinstance(md.get("metadata"), dict):   # Amazon records nest the source metadata one level deeper
        md = {**md, **md["metadata"]}
    for k in ("user_id", "user", "persona", "author"):
        if md.get(k) is not None:
            return str(md[k])
    return str(r.get("id"))


class Arm:
    """Per-user sums for ratio metrics and (gold, pred, weight, user) rows for macro-F1."""

    def __init__(self):
        self.sums = defaultdict(lambda: defaultdict(float))   # metric -> user -> sum
        self.cnt = defaultdict(lambda: defaultdict(float))    # metric -> user -> n
        self.rows = {}                                        # metric -> (gold, pred, w, user) lists

    def add(self, metric, user, value):
        self.sums[metric][user] += value; self.cnt[metric][user] += 1

    def users(self, metric):
        return set(self.cnt[metric]) if metric in self.cnt else set(u for u in self.rows[metric][3])


def load_gen(path, key):
    for p, opener in ((Path(str(path) + ".jsonl.gz"), gzip.open), (Path(str(path) + ".jsonl"), open)):
        if p.exists():
            return [json.loads(l) for l in opener(p, "rt", encoding="utf-8") if l.strip()]
    return None


def load_confusion(path, key="jepa_steered_output"):
    p = Path(str(path) + "_userstyle_confusion.csv")
    if not p.exists():
        return None
    g, pr, w, u = [], [], [], []
    for row in csv.DictReader(open(p)):
        if row["candidate_key"] != key:
            continue
        g.append(row["gold_label"]); pr.append(row["predicted_label"]); w.append(float(row["count"])); u.append(row["gold_label"])
    return g, pr, w, u


def build(ds):
    cfg = DATASETS[ds]; arms = {}
    ref = cfg["arms"][OURS]
    for name, path in list(cfg["arms"].items()) + [("raw", ref)]:
        key = "raw_generic_output" if name == "raw" else "jepa_steered_output"
        a = Arm(); recs = load_gen(path, key)
        if recs is not None:
            if cfg["kind"] == "classify":
                gold, pred, us = [], [], []
                for r in recs:
                    t, p = norm_tag(r.get("target")), norm_tag(r.get(key)); u = user_of(r)
                    hit = int(p == t or (t and p.startswith(t)))
                    a.add("acc", u, hit); gold.append(t); pred.append(t if hit else p); us.append(u)
                a.rows["macro_f1"] = (gold, pred, [1.0] * len(gold), us)
            else:
                for r in recs:
                    u = user_of(r); pt, gt = tokenize(str(r.get(key, ""))), tokenize(str(r.get("target", "")))
                    a.add("rouge1", u, tok_f1(pt, gt)); a.add("rougeL", u, rouge_l_f1(pt, gt))
        if cfg["kind"] == "gen":
            conf = load_confusion(path, key)
            if conf is not None:
                g, pr, w, u = conf
                for gi, pi, wi in zip(g, pr, w):
                    a.sums["attr"][gi] += wi * (gi == pi); a.cnt["attr"][gi] += wi
                a.rows["attr_f1"] = conf
        if a.cnt or a.rows:
            arms[name] = a
    return arms


def metric_value(arm, metric, weights):
    """weights: dict user -> multiplicity (bootstrap) ; returns the pooled metric."""
    if metric in ("macro_f1", "attr_f1"):
        g, p, w, u = arm.rows[metric]
        ww = np.array([wi * weights.get(ui, 0.0) for wi, ui in zip(w, u)])
        labels = {x: i for i, x in enumerate(sorted(set(g) | set(p)))}
        gi = np.array([labels[x] for x in g]); pi = np.array([labels[x] for x in p]); L = len(labels)
        tp = np.bincount(gi[gi == pi], weights=ww[gi == pi], minlength=L)
        gc = np.bincount(gi, weights=ww, minlength=L); pc = np.bincount(pi, weights=ww, minlength=L)
        present = gc > 0
        f1 = np.where(gc + pc > 0, 2 * tp / np.maximum(gc + pc, 1e-12), 0.0)
        return float(f1[present].mean()) if present.any() else float("nan")
    s = sum(arm.sums[metric][u] * m for u, m in weights.items() if u in arm.cnt[metric])
    n = sum(arm.cnt[metric][u] * m for u, m in weights.items() if u in arm.cnt[metric])
    return s / n if n else float("nan")


def has(arm, metric):
    return metric in arm.rows or metric in arm.cnt


def arm_users(arm, metric):
    return set(arm.rows[metric][3]) if metric in arm.rows else set(arm.cnt[metric])


def boot(arms, a, b, metric, n_boot, rng):
    """point + CI of metric(a) (b=None) or metric(a) - metric(b) on shared users."""
    A = arms[a]; B = arms.get(b) if b else None
    users = arm_users(A, metric) & (arm_users(B, metric) if B else arm_users(A, metric))
    users = sorted(users)
    if not users:
        return None
    full = {u: 1.0 for u in users}
    point = metric_value(A, metric, full) - (metric_value(B, metric, full) if B else 0.0)
    idx = rng.integers(0, len(users), size=(n_boot, len(users)))
    vals = []
    for row in idx:
        wts = defaultdict(float)
        for i in row:
            wts[users[i]] += 1.0
        vals.append(metric_value(A, metric, wts) - (metric_value(B, metric, wts) if B else 0.0))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_users": len(users)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/real_data_bootstrap")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for ds in args.datasets:
        rng = np.random.default_rng(args.seed)
        arms = build(ds)
        metrics = ["acc", "macro_f1"] if DATASETS[ds]["kind"] == "classify" else ["attr", "attr_f1", "rouge1", "rougeL"]
        res = {"absolute": {}, "diff": {}}
        for name in arms:
            for m in metrics:
                if has(arms[name], m):
                    r = boot(arms, name, None, m, args.n_boot, rng)
                    if r: res["absolute"][f"{name}|{m}"] = r
        for other, label in COMPARE:
            if other in arms and OURS in arms:
                for m in metrics:
                    if has(arms[OURS], m) and has(arms[other], m):
                        r = boot(arms, OURS, other, m, args.n_boot, rng)
                        if r: res["diff"][f"ours {label}|{m}"] = r
        for a, b in EXTRA_COMPARE:
            if a in arms and b in arms:
                for m in metrics:
                    if has(arms[a], m) and has(arms[b], m):
                        r = boot(arms, a, b, m, args.n_boot, rng)
                        if r: res["diff"][f"{a} vs {b}|{m}"] = r
        allres[ds] = res
        print(f"[bootstrap] {ds}: {len(res['absolute'])} absolute, {len(res['diff'])} differences", flush=True)
    (out / "bootstrap.json").write_text(json.dumps(allres, indent=1))
    names = {"acc": "accuracy", "macro_f1": "macro-F1", "attr": "attribution acc", "attr_f1": "attribution macro-F1",
             "rouge1": "ROUGE-1", "rougeL": "ROUGE-L"}
    L = [f"# Real-user bootstrap ({args.n_boot} user-cluster resamples, seed {args.seed})", ""]
    for ds, res in allres.items():
        L += [f"## {ds}", "", "### Absolute (95% CI)", "", "| arm | metric | value [95% CI] | users |", "|---|---|---|---|"]
        for k, r in res["absolute"].items():
            a, m = k.split("|"); L.append(f"| {a} | {names[m]} | {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | {r['n_users']} |")
        L += ["", "### Differences (95% CI; significant if the interval excludes 0)", "", "| comparison | metric | Δ [95% CI] | sig. | users |", "|---|---|---|---|---|"]
        for k, r in res["diff"].items():
            a, m = k.split("|"); sig = "yes" if (r["lo"] > 0 or r["hi"] < 0) else "no"
            L.append(f"| {a} | {names[m]} | {r['point']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {sig} | {r['n_users']} |")
        L.append("")
    (out / "bootstrap.md").write_text("\n".join(L) + "\n")
    print(f"[bootstrap] wrote {out}/bootstrap.md and bootstrap.json")


if __name__ == "__main__":
    main()
