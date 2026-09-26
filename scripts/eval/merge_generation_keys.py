#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl_by_key(path: Path, join_key: str) -> dict[str, dict]:
    records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if join_key not in record:
            raise KeyError(f"{path}:{line_number} is missing join key {join_key!r}.")
        key = str(record[join_key])
        if key in records:
            raise ValueError(f"{path} contains duplicate join key {key!r}.")
        records[key] = record
    return records


def parse_overlay_specs(specs: list[str]) -> list[tuple[Path, list[str]]]:
    overlays = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                "Each --overlay must use path:key[,key...] "
                f"but got {spec!r}."
            )
        path_text, key_text = spec.split(":", 1)
        keys = [key.strip() for key in key_text.split(",") if key.strip()]
        if not keys:
            raise ValueError(f"Overlay {spec!r} did not include any keys.")
        overlays.append((Path(path_text), keys))
    return overlays


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge selected generation keys across JSONL files by id.")
    parser.add_argument("--base", required=True, help="Base JSONL file to preserve record order and core fields.")
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Overlay file and keys as path:key[,key...]. Can be repeated.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--join-key", default="id")
    args = parser.parse_args()

    base_path = Path(args.base)
    base_records = [
        json.loads(line)
        for line in base_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    overlays = [
        (path, keys, load_jsonl_by_key(path, args.join_key))
        for path, keys in parse_overlay_specs(args.overlay)
    ]

    merged_records = []
    for record in base_records:
        join_value = str(record[args.join_key])
        merged = dict(record)
        for path, keys, overlay_records in overlays:
            if join_value not in overlay_records:
                raise KeyError(f"{path} is missing join key {join_value!r}.")
            overlay_record = overlay_records[join_value]
            for key in keys:
                if key not in overlay_record:
                    raise KeyError(f"{path} record {join_value!r} is missing key {key!r}.")
                merged[key] = overlay_record[key]
        merged_records.append(merged)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in merged_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(merged_records)} merged records to {output_path}")


if __name__ == "__main__":
    main()
