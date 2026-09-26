#!/usr/bin/env python
"""Compare ablation arms against a reference generation on the SAME dev examples.

Reports persona accuracy (same char-TF-IDF classifier recipe as
evaluate_synper_persona_classifier_strong.py, retrained here to get per-example
predictions), token F1, per-persona accuracy, n, trainable parameters, runtimes,
and paired bootstrap 95% CIs of the accuracy / F1 differences vs the reference
(evaluation-sample uncertainty only; seeds are not resampled).

  python scripts/ablations/compare_ablation_arms.py --reference <ref.jsonl> \
      --arm E1=<gen.jsonl> --arm E2=... --arm-dir E1=<abl dir> ... --output table.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate_synper_persona_classifier_strong import build_logistic_regression, persona_of, read_jsonl  # noqa: E402
from evaluate_text_metrics import f1_score, tokenize  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402


def per_example(records, clf, key="jepa_steered_output"):
    texts = [str(r.get(key, "")) for r in records]
    labels = [persona_of(r) for r in records]
    preds = clf.predict(texts)
    correct = np.array([int(y == p) for y, p in zip(labels, preds)], dtype=float)
    f1 = np.array([f1_score(tokenize(str(r.get(key, ""))), tokenize(str(r.get("target", "")))) for r in records])
    return correct, f1, labels


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed); n = len(a); idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    return float((b - a).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="reference generation jsonl (arm A)")
    ap.add_argument("--reference-ckpt", default=None, help="reference likelihood checkpoint (trainable count)")
    ap.add_argument("--arm", action="append", default=[], help="NAME=generation.jsonl")
    ap.add_argument("--arm-dir", action="append", default=[], help="NAME=ablation dir (timing.json, ckpt/final.pt)")
    ap.add_argument("--train", default="data/synper/train_10000.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, help="markdown table")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    train = read_jsonl(Path(args.train))
    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
                        build_logistic_regression(max_iter=2000, multi_class="auto", n_jobs=1))
    clf.fit([str(r.get("target", "")) for r in train], [persona_of(r) for r in train])

    ref = read_jsonl(Path(args.reference)); ref_ids = [str(r.get("id")) for r in ref]
    arms = {"A (reference)": ref}
    for spec in args.arm:
        name, path = spec.split("=", 1); recs = read_jsonl(Path(path))
        by = {str(r.get("id")): r for r in recs}
        missing = [i for i in ref_ids if i not in by]
        assert not missing, f"{name}: {len(missing)} reference ids missing from {path}"
        arms[name] = [by[i] for i in ref_ids]   # align on the reference example order
    dirs = dict(s.split("=", 1) for s in args.arm_dir)

    def extras(name):
        d = dirs.get(name); out = {"trainable": None, "train_s": None, "gen_s": None, "mse_s": None}
        ck = None
        if d and (Path(d) / "ckpt" / "final.pt").exists(): ck = Path(d) / "ckpt" / "final.pt"
        if name.startswith("A") and args.reference_ckpt: ck = Path(args.reference_ckpt)
        if ck is not None:
            import torch
            pay = torch.load(ck, map_location="cpu"); lt = pay.get("likelihood_training", {})
            out["trainable"] = lt.get("n_trainable"); out["train_s"] = lt.get("train_seconds")
            if out["trainable"] is None and "state_dict" in pay:
                out["trainable"] = int(sum(v.numel() for v in pay["state_dict"].values()))
            if out["trainable"] is None and "group_means" in pay:
                out["trainable"] = int(pay["group_means"].numel())
        if d and (Path(d) / "timing.json").exists():
            t = json.load(open(Path(d) / "timing.json")); out["gen_s"] = t.get("generation_seconds"); out["mse_s"] = t.get("mse_pretrain_seconds")
            out["train_s"] = out["train_s"] or t.get("lik_train_seconds")
        return out

    results = {}; personas = sorted({persona_of(r) for r in ref})
    ref_c, ref_f, labels = per_example(ref, clf); raw_c, raw_f, _ = per_example(ref, clf, key="raw_generic_output")
    for name, recs in arms.items():
        c, f, _ = per_example(recs, clf)
        per_p = {p: float(np.mean([ci for ci, l in zip(c, labels) if l == p])) for p in personas}
        row = {"n": int(len(c)), "persona_acc": float(c.mean()), "token_f1": float(f.mean()), "per_persona": per_p, **extras(name)}
        if name != "A (reference)":
            row["d_acc"], row["d_acc_lo"], row["d_acc_hi"] = paired_bootstrap(ref_c, c, args.n_boot, args.seed)
            row["d_f1"], row["d_f1_lo"], row["d_f1_hi"] = paired_bootstrap(ref_f, f, args.n_boot, args.seed)
        results[name] = row
    results["raw generic"] = {"n": int(len(raw_c)), "persona_acc": float(raw_c.mean()), "token_f1": float(raw_f.mean()),
                              "per_persona": {p: float(np.mean([ci for ci, l in zip(raw_c, labels) if l == p])) for p in personas}}

    def fmt(v, nd=3): return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    lines = ["| arm | persona acc | Δacc vs A [95% CI] | token F1 | ΔF1 vs A [95% CI] | n | trainable params | MSE-pretrain s | lik-train s | gen s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        da = f"{r['d_acc']:+.3f} [{r['d_acc_lo']:+.3f}, {r['d_acc_hi']:+.3f}]" if "d_acc" in r else "—"
        df = f"{r['d_f1']:+.3f} [{r['d_f1_lo']:+.3f}, {r['d_f1_hi']:+.3f}]" if "d_f1" in r else "—"
        lines.append(f"| {name} | {r['persona_acc']:.3f} | {da} | {r['token_f1']:.3f} | {df} | {r['n']} | {fmt(r.get('trainable'))} | {fmt(r.get('mse_s'), 0)} | {fmt(r.get('train_s'), 0)} | {fmt(r.get('gen_s'), 0)} |")
    lines += ["", "Per-persona accuracy:", "", "| arm | " + " | ".join(personas) + " |", "|---|" + "---|" * len(personas)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['per_persona'][p]:.2f}" for p in personas) + " |")
    lines += ["", f"Paired bootstrap over the {len(ref)} shared dev examples, {args.n_boot} resamples, seed {args.seed}; CIs reflect evaluation-sample uncertainty only (not training-seed variability)."]
    Path(args.output).write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
