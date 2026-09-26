#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import adapt_records, load_hf_dataset, normalize_record, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a Hugging Face dataset and normalize it into Pers-JEPA paired JSONL."
    )
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name/path.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--adapter",
        choices=["none", "lamp5", "amazon"],
        default="none",
        help="Schema adapter before paired-view normalization.",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Optional HF dataset config/subset name.",
    )
    parser.add_argument("--profile-template", default="{profile}\n\nTask: {prompt}")
    args = parser.parse_args()

    kwargs = {}
    if args.config_name:
        kwargs["name"] = args.config_name
    raw_records = load_hf_dataset(args.dataset, args.split, **kwargs)
    adapted = adapt_records(raw_records, args.adapter)
    examples = [
        normalize_record(record, profile_template=args.profile_template)
        for record in adapted
    ]
    write_jsonl(args.output, [example.to_json() for example in examples])
    print(
        f"Wrote {len(examples)} paired examples from {args.dataset}:{args.split} "
        f"using adapter={args.adapter} to {args.output}"
    )


if __name__ == "__main__":
    main()
