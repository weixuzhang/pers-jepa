#!/usr/bin/env python
"""Helper: build a leakage-safe calibration/heldout split and
tag every example with an UNSUPERVISED, identity-free content cluster id
(k-means fit on calibration h_gen answer-token means only). This lets
evaluate_group_residual.py's existing --group-field machinery be reused to
compare a content-cluster-conditioned mean delta against the global mean
delta, without ever using user identity to pick the delta at inference.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.hidden import require_torch


def answer_token_means(payload):
    torch = require_torch()
    source = payload["source_answer_tokens"].float()
    mask = payload["source_answer_token_mask"].bool().float()
    denom = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return (source * mask.unsqueeze(-1)).sum(dim=1) / denom


def kmeans(x, k, *, seed, iters=50):
    torch = require_torch()
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator)[:k]
    centroids = x.index_select(0, perm).clone()
    for _ in range(iters):
        distances = torch.cdist(x, centroids)
        labels = distances.argmin(dim=1)
        new_centroids = []
        for c in range(k):
            mask = labels == c
            new_centroids.append(x[mask].mean(dim=0) if bool(mask.any()) else centroids[c])
        new_centroids = torch.stack(new_centroids, dim=0)
        if torch.allclose(new_centroids, centroids, atol=1e-5, rtol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--span-hidden", required=True)
    parser.add_argument("--raw-jsonl", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--heldout-fraction", type=float, default=0.3)
    parser.add_argument("--max-heldout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-out", required=True)
    parser.add_argument("--heldout-out", required=True)
    args = parser.parse_args()

    torch = require_torch()
    payload = torch.load(args.span_hidden, map_location="cpu")
    source = answer_token_means(payload)
    n = source.shape[0]

    with open(args.raw_jsonl, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    assert len(records) == n, f"record/hidden count mismatch: {len(records)} vs {n}"

    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_heldout = int(n * args.heldout_fraction)
    heldout_idx = indices[:n_heldout]
    cal_idx = indices[n_heldout:]
    if args.max_heldout is not None:
        heldout_idx = heldout_idx[: args.max_heldout]

    cal_tensor = torch.tensor(cal_idx, dtype=torch.long)
    cal_source = source.index_select(0, cal_tensor)
    centroids = kmeans(cal_source, args.k, seed=args.seed)

    def assign(idx_list):
        vecs = source.index_select(0, torch.tensor(idx_list, dtype=torch.long))
        return torch.cdist(vecs, centroids).argmin(dim=1).tolist()

    cal_labels = assign(cal_idx)
    heldout_labels = assign(heldout_idx)

    def write_split(path, idx_list, labels):
        with open(path, "w", encoding="utf-8") as handle:
            for original_idx, label in zip(idx_list, labels):
                record = dict(records[original_idx])
                metadata = dict(record.get("metadata") or {})
                metadata["cluster_id"] = f"cluster_{label}"
                record["metadata"] = metadata
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_split(args.calibration_out, cal_idx, cal_labels)
    write_split(args.heldout_out, heldout_idx, heldout_labels)

    from collections import Counter

    print(f"n_calibration={len(cal_idx)} n_heldout={len(heldout_idx)}")
    print("calibration cluster sizes:", Counter(cal_labels))
    print("heldout cluster sizes:", Counter(heldout_labels))


if __name__ == "__main__":
    main()
