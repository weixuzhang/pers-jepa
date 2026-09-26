#!/usr/bin/env python
"""Build a blinded SynPer human-validation annotation sheet.

Samples N examples per persona from a generation-record jsonl, takes one
item per (example, arm) pair, strips all condition/persona identifiers,
shuffles, and writes:
  - <out>/annotation_items.jsonl   blinded items the annotator sees
  - <out>/annotation_key.jsonl     hidden mapping (do NOT open while annotating)
  - <out>/persona_catalog.md       the 10 persona style cards shown to annotators
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    parser.add_argument("--arms", nargs="+", default=[
        "raw_generic_output",
        "global_answer_token_mean_delta_output",
        "personalized_prompt_output",
    ])
    parser.add_argument("--examples-per-persona", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.generations, encoding="utf-8")]
    by_persona = defaultdict(list)
    catalog = {}
    for r in records:
        persona = r["metadata"]["metadata"]["persona"]
        by_persona[persona].append(r)
        catalog.setdefault(persona, r["profile"])

    rng = random.Random(args.seed)
    chosen = []
    for persona in sorted(by_persona):
        pool = sorted(by_persona[persona], key=lambda r: r["id"])
        rng.shuffle(pool)
        chosen.extend(pool[: args.examples_per_persona])

    items, key = [], []
    for rec in chosen:
        for arm in args.arms:
            items.append({
                "source_id": rec["id"],
                "arm": arm,
                "task_prompt": rec["generic_prompt"],
                "output": rec[arm],
                "persona": rec["metadata"]["metadata"]["persona"],
                "task_type": rec["metadata"]["metadata"]["task_type"],
            })
    rng.shuffle(items)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "annotation_items.jsonl").open("w", encoding="utf-8") as fi, \
         (out_dir / "annotation_key.jsonl").open("w", encoding="utf-8") as fk:
        for idx, item in enumerate(items, start=1):
            item_id = f"item-{idx:03d}"
            fi.write(json.dumps({
                "item_id": item_id,
                "task_prompt": item["task_prompt"],
                "output": item["output"],
            }, ensure_ascii=False) + "\n")
            fk.write(json.dumps({
                "item_id": item_id,
                "source_id": item["source_id"],
                "arm": item["arm"],
                "true_persona": item["persona"],
                "task_type": item["task_type"],
            }, ensure_ascii=False) + "\n")

    with (out_dir / "persona_catalog.md").open("w", encoding="utf-8") as fc:
        fc.write("# SynPer Persona Catalog (shown to annotators)\n\n")
        for persona in sorted(catalog):
            fc.write(f"## {persona}\n\n{catalog[persona]}\n\n")

    print(f"Wrote {len(items)} blinded items over {len(chosen)} examples x {len(args.arms)} arms to {out_dir}")


if __name__ == "__main__":
    main()
