#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import write_jsonl


def read_jsonl(path: Path) -> list[dict]:
    records = []
    seen_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_id = str(record.get("id", "")).strip()
        if record_id and record_id in seen_ids:
            continue
        if record_id:
            seen_ids.add(record_id)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a generated SynPer JSONL into train/dev JSONL files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260505)
    args = parser.parse_args()

    input_path = Path(args.input)
    records = read_jsonl(input_path)
    required = args.train_size + args.dev_size
    if len(records) < required:
        raise ValueError(f"Need at least {required} records, found {len(records)} in {input_path}.")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    train_records = records[: args.train_size]
    dev_records = records[args.train_size : args.train_size + args.dev_size]

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.dev_output, dev_records)

    summary = {
        "input": str(input_path),
        "n_input": len(records),
        "train_output": args.train_output,
        "dev_output": args.dev_output,
        "train_size": len(train_records),
        "dev_size": len(dev_records),
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
