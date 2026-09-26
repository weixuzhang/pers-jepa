#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate JSONL records by id, keeping the last record.")
    parser.add_argument("--input", required=True, help="Path to the source JSONL file.")
    parser.add_argument("--output", default=None, help="Path to the deduplicated JSONL file. Defaults to in-place.")
    parser.add_argument(
        "--id-key",
        default="id",
        help="Record key used as the deduplication id. Defaults to 'id'.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    records_by_id: dict[str, dict] = {}
    order: list[str] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get(args.id_key, ""))
            if record_id not in records_by_id:
                order.append(record_id)
            records_by_id[record_id] = record

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record_id in order:
            handle.write(json.dumps(records_by_id[record_id], ensure_ascii=False) + "\n")

    total = len(order)
    print(f"Wrote {total} unique records to {output_path}")


if __name__ == "__main__":
    main()
