#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import write_jsonl


def clean_text(value: Any, *, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "."
    return text


def load_outputs(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["id"]): str(row["output"]) for row in payload["golds"]}


def format_profile_item(item: dict, *, abstract_chars: int) -> str:
    title = clean_text(item.get("title"))
    abstract = clean_text(item.get("abstract"), max_chars=abstract_chars)
    if title and abstract:
        return f"{title} - {abstract}"
    return title or abstract


def convert_lamp5(
    *,
    questions_path: str | Path,
    outputs_path: str | Path,
    output_path: str | Path,
    split: str,
    max_examples: int | None,
    max_profile_items: int,
    profile_abstract_chars: int,
    seed: int,
) -> list[dict]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    outputs = load_outputs(outputs_path)
    rng = random.Random(seed)
    if max_examples is not None and max_examples < len(questions):
        questions = rng.sample(questions, k=max_examples)

    records = []
    for row in questions:
        row_id = str(row["id"])
        profile_items = list(row.get("profile") or [])[:max_profile_items]
        profile_lines = [
            f"{idx + 1}. {format_profile_item(item, abstract_chars=profile_abstract_chars)}"
            for idx, item in enumerate(profile_items)
        ]
        profile = "Previous papers by this author:\n" + "\n".join(profile_lines)
        records.append({
            "id": row_id,
            "profile": profile,
            "prompt": clean_text(row.get("input")),
            "target": outputs.get(row_id, ""),
            "metadata": {
                "source": "LaMP_5",
                "split": split,
                "profile_count": len(row.get("profile") or []),
                "used_profile_count": len(profile_items),
            },
        })

    records.sort(key=lambda record: record["id"])
    write_jsonl(output_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LaMP-5 scholarly-title pairs for Pers-JEPA.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-profile-items", type=int, default=10)
    parser.add_argument("--profile-abstract-chars", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = convert_lamp5(
        questions_path=args.questions,
        outputs_path=args.outputs,
        output_path=args.output,
        split=args.split,
        max_examples=args.max_examples,
        max_profile_items=args.max_profile_items,
        profile_abstract_chars=args.profile_abstract_chars,
        seed=args.seed,
    )
    print(f"Wrote {len(records)} LaMP-5 {args.split} examples to {args.output}")


if __name__ == "__main__":
    main()
