#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.hidden import require_torch


def dotted_get(record: dict[str, Any], field: str, default: Any = None) -> Any:
    value: Any = record
    for part in field.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif (
            isinstance(value, dict)
            and isinstance(value.get("metadata"), dict)
            and part in value["metadata"]
        ):
            value = value["metadata"][part]
        else:
            return default
    return value


def group_values(examples: list[dict[str, Any]], field: str) -> list[str]:
    values = []
    for idx, example in enumerate(examples):
        value = dotted_get(example, field)
        values.append(str(value) if value not in (None, "") else f"__missing_{idx}")
    return values


def subset_payload(payload: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    torch = require_torch()
    index_tensor = torch.tensor(indices, dtype=torch.long)
    n_examples = len(payload.get("examples", []))
    subset: dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.shape[:1] == (n_examples,):
            subset[key] = value.index_select(0, index_tensor)
        elif isinstance(value, list) and len(value) == n_examples:
            subset[key] = [value[idx] for idx in indices]
        else:
            subset[key] = value
    subset["split_source_indices"] = indices
    return subset


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def eval_record(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    metadata = output.get("metadata")
    if isinstance(metadata, dict) and set(metadata.keys()) == {"metadata"} and isinstance(metadata["metadata"], dict):
        output["metadata"] = metadata["metadata"]
    return output


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a group calibration span checkpoint and heldout JSONL without target leakage."
    )
    parser.add_argument("--span-hidden", required=True)
    parser.add_argument("--group-field", default="metadata.user_id")
    parser.add_argument("--calibration-span-output", required=True)
    parser.add_argument("--eval-jsonl-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--min-group-count", type=int, default=10)
    parser.add_argument("--calibration-examples", type=int, default=5)
    parser.add_argument("--heldout-examples", type=int, default=5)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch = require_torch()
    payload = torch.load(args.span_hidden, map_location="cpu")
    examples = payload.get("examples", [])
    if not examples:
        raise ValueError("--span-hidden payload does not contain examples.")

    groups = group_values(examples, args.group_field)
    by_group: dict[str, list[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)

    rng = random.Random(args.seed)
    cal_indices: list[int] = []
    eval_indices: list[int] = []
    rows: list[dict[str, Any]] = []
    min_required = max(args.min_group_count, args.calibration_examples + 1)
    kept_groups = 0
    for group, indices in sorted(by_group.items()):
        if args.max_groups > 0 and kept_groups >= args.max_groups:
            break
        if len(indices) < min_required:
            continue
        shuffled = list(indices)
        rng.shuffle(shuffled)
        cal = shuffled[: args.calibration_examples]
        heldout_pool = shuffled[args.calibration_examples :]
        if args.heldout_examples > 0:
            heldout = heldout_pool[: args.heldout_examples]
        else:
            heldout = heldout_pool
        if not heldout:
            continue
        cal_indices.extend(cal)
        eval_indices.extend(heldout)
        rows.append(
            {
                "group_id": group,
                "total_examples": len(indices),
                "calibration_examples": len(cal),
                "heldout_examples": len(heldout),
            }
        )
        kept_groups += 1

    if not cal_indices or not eval_indices:
        raise ValueError("No usable groups for the requested split settings.")

    calibration_payload = subset_payload(payload, cal_indices)
    calibration_path = Path(args.calibration_span_output)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibration_payload, calibration_path)

    eval_records = [eval_record(examples[idx]) for idx in eval_indices]
    write_jsonl(Path(args.eval_jsonl_output), eval_records)
    write_summary(Path(args.summary_output), rows)
    print(
        f"Wrote {len(cal_indices)} calibration and {len(eval_indices)} heldout examples "
        f"across {len(rows)} groups."
    )


if __name__ == "__main__":
    main()
