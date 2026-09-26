#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import load_persona_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize persona records into Pers-JEPA paired views.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-field", default="profile")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--target-field", default="target")
    parser.add_argument("--id-field", default="id")
    parser.add_argument(
        "--profile-template",
        default="{profile}\n\nTask: {prompt}",
        help="Template for the personalized prompt.",
    )
    args = parser.parse_args()
    examples = load_persona_jsonl(
        args.input,
        profile_field=args.profile_field,
        prompt_field=args.prompt_field,
        target_field=args.target_field,
        id_field=args.id_field,
        profile_template=args.profile_template,
    )
    write_jsonl(args.output, [example.to_json() for example in examples])
    print(f"Wrote {len(examples)} paired examples to {args.output}")


if __name__ == "__main__":
    main()
