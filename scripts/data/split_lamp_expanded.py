#!/usr/bin/env python
"""Turn prepare_lamp.py --expand-profile-pairs outputs into the driver's files
(the recipe used for LaMP-2):
  dev.jsonl         = dev rows with metadata.kind == "query"            (evaluation queries)
  train_calib.jsonl = dev rows with kind == "profile_pair"              (each dev user's own history pairs)
                    + all train_expanded rows (train queries relabelled kind="train_query")
Usage: python scripts/data/split_lamp_expanded.py --dev-expanded data/lamp4/dev_expanded.jsonl \
         --train-expanded data/lamp4/train_expanded.jsonl --out-dir data/lamp4
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dev-expanded", required=True)
ap.add_argument("--train-expanded", required=True)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
dev_q, dev_pairs, n_train, users_dev, users_train = [], [], 0, set(), set()
for line in open(a.dev_expanded):
    if not line.strip(): continue
    r = json.loads(line); kind = (r.get("metadata") or {}).get("kind")
    users_dev.add((r.get("metadata") or {}).get("user_id"))
    (dev_q if kind == "query" else dev_pairs).append(r)
with open(out / "dev.jsonl", "w") as f:
    for r in dev_q: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(out / "train_calib.jsonl", "w") as f:
    for r in dev_pairs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for line in open(a.train_expanded):
        if not line.strip(): continue
        r = json.loads(line); md = r.setdefault("metadata", {})
        users_train.add(md.get("user_id"))
        if md.get("kind") == "query": md["kind"] = "train_query"
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); n_train += 1
print(f"dev.jsonl: {len(dev_q)} queries from {len(users_dev)} users; train_calib.jsonl: {len(dev_pairs)} dev-user profile pairs + {n_train} train rows from {len(users_train)} train users")
