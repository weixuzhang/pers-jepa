#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_persona(profile: str, metadata: dict[str, Any]) -> tuple[str, dict[str, str]]:
    persona = str(metadata.get("persona") or "")
    attrs: dict[str, str] = {}
    for line in profile.splitlines():
        line = line.strip()
        if line.lower().startswith("persona:"):
            persona = line.split(":", 1)[1].strip()
        elif line.lower().startswith("style constraints:"):
            constraints = line.split(":", 1)[1].strip().rstrip(".")
            attrs["style_constraints"] = constraints
            parts = [part.strip() for part in constraints.split(",") if part.strip()]
            for idx, part in enumerate(parts, start=1):
                attrs[f"style_{idx}"] = part
    return persona, attrs


def task_id(prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    return f"synper-task-{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a same-task multi-persona SynPer counterfactual split.")
    parser.add_argument("--input", default="data/synper/synper_1k.jsonl")
    parser.add_argument("--output", default="data/synper/counterfactual/same_task_multi_persona.jsonl")
    parser.add_argument("--summary-csv", default="tables/synper_counterfactual_split_summary.csv")
    parser.add_argument("--manifest", default="analysis/synper_counterfactual_split_manifest.md")
    parser.add_argument("--min-personas-per-task", type=int, default=5)
    parser.add_argument("--max-personas-per-task", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    records = read_jsonl(input_path)
    by_prompt: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        prompt = str(record.get("prompt") or record.get("generic_prompt") or "")
        metadata = record.get("metadata") or {}
        persona, attrs = parse_persona(str(record.get("profile", "")), metadata)
        if not prompt or not persona:
            continue
        # Keep the first example per prompt/persona so each row is one
        # counterfactual target for the same base task.
        by_prompt[prompt].setdefault(persona, record | {"_persona_attributes": attrs})

    selected_prompts = [
        prompt
        for prompt, persona_map in by_prompt.items()
        if len(persona_map) >= args.min_personas_per_task
    ]
    selected_prompts.sort(key=lambda prompt: (-len(by_prompt[prompt]), prompt))
    if args.max_tasks is not None:
        selected_prompts = selected_prompts[: args.max_tasks]

    output_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for prompt in selected_prompts:
        persona_items = sorted(by_prompt[prompt].items())[: args.max_personas_per_task]
        base_id = task_id(prompt)
        summary_rows.append(
            {
                "base_task_id": base_id,
                "base_input": prompt,
                "personas": len(persona_items),
                "persona_ids": "|".join(persona for persona, _record in persona_items),
            }
        )
        for persona, record in persona_items:
            profile = str(record.get("profile", ""))
            output_records.append(
                {
                    "id": f"{base_id}-{persona.lower().replace(' ', '-').replace('/', '-')}",
                    "base_task_id": base_id,
                    "base_input": prompt,
                    "persona_id": persona,
                    "persona_description": profile,
                    "persona_attributes": record.get("_persona_attributes", {}),
                    "generic_prompt": prompt,
                    "personalized_prompt": f"{profile.strip()}\n\nTask: {prompt.strip()}",
                    "personalized_reference": record.get("target", ""),
                    "target": record.get("target", ""),
                    "metadata": {
                        "source": "SynPer",
                        "counterfactual": True,
                        "original_id": record.get("id", ""),
                        "task_type": (record.get("metadata") or {}).get("task_type", ""),
                    },
                }
            )

    output_path = Path(args.output)
    write_jsonl(output_path, output_records)
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["base_task_id", "base_input", "personas", "persona_ids"])
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(
            [
                "# SynPer Counterfactual Split Manifest",
                "",
                f"- Source: `{input_path}`",
                f"- Output JSONL: `{output_path}`",
                f"- Summary CSV: `{summary_path}`",
                f"- Base tasks: `{len(summary_rows)}`",
                f"- Counterfactual examples: `{len(output_records)}`",
                f"- Min personas per task: `{args.min_personas_per_task}`",
                f"- Max personas per task: `{args.max_personas_per_task}`",
                "",
                "This split uses the same generic task prompt crossed with multiple",
                "SynPer personas. It is ready for profile-prompt, global-residual,",
                "persona-residual, and persona-identification judge experiments.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "summary_csv": str(summary_path),
                "manifest": str(manifest_path),
                "base_tasks": len(summary_rows),
                "examples": len(output_records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
