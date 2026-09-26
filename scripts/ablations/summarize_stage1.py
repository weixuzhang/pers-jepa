#!/usr/bin/env python
"""Stage-1 summary: per arm x scale, pooled persona accuracy, task F1, and
per-persona wins of each arm against the routed per-group constant at the same
scale. Usage: summarize_stage1.py <generation dir>"""
import collections, csv, glob, json, os, re, sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
rows = {}   # (mode, arm, scale) -> dict
per = {}    # (mode, arm, scale) -> {persona: acc}
raw_acc = None; raw_f1 = None
for f in sorted(glob.glob(os.path.join(root, "*_persona_strong_confusion.csv"))):
    m = re.match(r"(persistent|anchor)__(\w+)__scale([\d.]+)_persona_strong_confusion\.csv", os.path.basename(f))
    if not m: continue
    key = (m.group(1), m.group(2), float(m.group(3)))
    tot = collections.Counter(); hit = collections.Counter(); rtot = rhit = 0
    for r in csv.DictReader(open(f)):
        n = int(r["count"]); ok = r["gold_persona"] == r["predicted_persona"]
        if r["candidate_key"] == "jepa_steered_output":
            tot[r["gold_persona"]] += n; hit[r["gold_persona"]] += n * ok
        elif r["candidate_key"] == "raw_generic_output":
            rtot += n; rhit += n * ok
    per[key] = {p: hit[p] / tot[p] for p in tot}
    tm = f.replace("_persona_strong_confusion.csv", "_text_metrics.json")
    f1 = json.load(open(tm))["jepa_steered_output"]["token_f1"] if os.path.exists(tm) else float("nan")
    if raw_acc is None and rtot:
        raw_acc = rhit / rtot
        raw_f1 = json.load(open(tm))["raw_generic_output"]["token_f1"] if os.path.exists(tm) else float("nan")
    rows[key] = {"acc": sum(hit.values()) / max(sum(tot.values()), 1), "f1": f1, "n": sum(tot.values())}

print(f"raw generic: persona acc {raw_acc:.3f}  task F1 {raw_f1:.3f}" if raw_acc is not None else "no raw baseline found")
print(f"{'mode':10} {'arm':24} {'scale':>5} {'pers_acc':>8} {'task_f1':>7} {'wins_vs_group_const':>20}")
for key in sorted(rows):
    mode, arm, sc = key
    ref = per.get((mode, "routed_mean_delta", sc))
    wins = ""
    if ref and arm != "routed_mean_delta":
        w = sum(per[key][p] > ref[p] for p in ref); l = sum(per[key][p] < ref[p] for p in ref)
        wins = f"{w}W/{l}L/{len(ref) - w - l}T"
    print(f"{mode:10} {arm:24} {sc:5.1f} {rows[key]['acc']:8.3f} {rows[key]['f1']:7.3f} {wins:>20}")
