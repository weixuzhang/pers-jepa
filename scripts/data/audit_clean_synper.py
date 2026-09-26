#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PERSONA_RE = re.compile(r"^\s*Persona:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
PROFILE_PERSONA_RE = re.compile(r"^\s*(The [^:\n]+):\s*", re.MULTILINE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_persona(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    persona = metadata.get("persona")
    if persona:
        return str(persona).strip()
    profile = str(record.get("profile") or "")
    match = PERSONA_RE.search(profile)
    if match:
        return match.group(1).strip()
    match = PROFILE_PERSONA_RE.search(profile)
    return match.group(1).strip() if match else ""


def normalize_task_type(task_type: str, prompt: str) -> str:
    base = (task_type or "").strip().lower()
    prompt_l = prompt.strip().lower()
    if base:
        base = re.sub(r"\s+", " ", base)
    candidates = [
        (("answer a customer question", "customer question", "shipping times", "product warranty"), "answer a customer question"),
        (("critique an idea", "critique a business proposal", "critique a new business model", "critique an idea for", "idea critique"), "critique an idea"),
        (("draft an email", "email draft", "draft a message", "draft a response", "draft a social media post"), "draft an email"),
        (("explain a technical concept", "technical explanation", "explain a complex", "explain how", "explain a process", "explain a math concept"), "explain a technical concept"),
        (("name or title a short text", "title naming"), "name or title a short text"),
        (("recommend a plan", "plan recommendation"), "recommend a plan"),
        (("rewrite a message", "message rewrite"), "rewrite a message"),
        (("summarize a short article", "article summary"), "summarize a short article"),
        (("write a polite rejection", "polite rejection"), "write a polite rejection"),
        (("write a product review", "product review"), "write a product review"),
    ]
    for aliases, canonical in candidates:
        if any(alias in base for alias in aliases) or any(alias in prompt_l for alias in aliases):
            return canonical
    if base:
        return base
    return prompt_l.rstrip(".")


def clean_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int], Counter[str], Counter[str]]:
    stats = {
        "input_rows": len(rows),
        "persona_missing_before": 0,
        "persona_recovered_from_profile": 0,
        "task_missing_before": 0,
        "task_recovered_from_prompt": 0,
    }
    cleaned = []
    personas = Counter()
    tasks = Counter()
    for row in rows:
        item = json.loads(json.dumps(row))
        metadata = item.setdefault("metadata", {})

        original_persona = metadata.get("persona")
        if not original_persona:
            stats["persona_missing_before"] += 1
        persona = infer_persona(item)
        if persona and not original_persona:
            stats["persona_recovered_from_profile"] += 1
        if persona:
            metadata["persona"] = persona

        original_task = metadata.get("task_type")
        if not original_task:
            stats["task_missing_before"] += 1
        prompt = str(item.get("prompt") or item.get("generic_prompt") or "")
        task_type = normalize_task_type(str(original_task or ""), prompt)
        if task_type and not original_task:
            stats["task_recovered_from_prompt"] += 1
        if task_type:
            metadata["task_type"] = task_type

        personas[metadata.get("persona", "")] += 1
        tasks[metadata.get("task_type", "")] += 1
        cleaned.append(item)

    stats["persona_missing_after"] = personas[""]
    stats["task_missing_after"] = tasks[""]
    return cleaned, stats, personas, tasks


def write_summary_csv(path: Path, counter: Counter[str], label_field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[label_field, "count"], lineterminator="\n")
        writer.writeheader()
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({label_field: label, "count": count})


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and clean SynPer JSONL metadata.")
    parser.add_argument("--train-input", default="data/synper/train_10000.jsonl")
    parser.add_argument("--dev-input", default="data/synper/dev_1000.jsonl")
    parser.add_argument("--train-output", default="data/synper/clean/train_10000.jsonl")
    parser.add_argument("--dev-output", default="data/synper/clean/dev_1000.jsonl")
    parser.add_argument("--report", default="analysis/synper_data_audit.md")
    parser.add_argument("--table-dir", default="tables/synper_audit")
    args = parser.parse_args()

    train_rows = read_jsonl(Path(args.train_input))
    dev_rows = read_jsonl(Path(args.dev_input))

    train_clean, train_stats, train_personas, train_tasks = clean_records(train_rows)
    dev_clean, dev_stats, dev_personas, dev_tasks = clean_records(dev_rows)

    write_jsonl(Path(args.train_output), train_clean)
    write_jsonl(Path(args.dev_output), dev_clean)

    table_dir = Path(args.table_dir)
    write_summary_csv(table_dir / "train_persona_counts.csv", train_personas, "persona")
    write_summary_csv(table_dir / "train_task_counts.csv", train_tasks, "task_type")
    write_summary_csv(table_dir / "dev_persona_counts.csv", dev_personas, "persona")
    write_summary_csv(table_dir / "dev_task_counts.csv", dev_tasks, "task_type")

    report_lines = [
        "# SynPer Data Audit",
        "",
        f"- Train input: `{args.train_input}`",
        f"- Dev input: `{args.dev_input}`",
        f"- Clean train output: `{args.train_output}`",
        f"- Clean dev output: `{args.dev_output}`",
        "",
        "## Repair Summary",
        "",
        "| Split | Rows | Missing persona before | Recovered persona | Missing persona after | Missing task before | Recovered task | Missing task after |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| train | {train_stats['input_rows']} | {train_stats['persona_missing_before']} | {train_stats['persona_recovered_from_profile']} | {train_stats['persona_missing_after']} | {train_stats['task_missing_before']} | {train_stats['task_recovered_from_prompt']} | {train_stats['task_missing_after']} |",
        f"| dev | {dev_stats['input_rows']} | {dev_stats['persona_missing_before']} | {dev_stats['persona_recovered_from_profile']} | {dev_stats['persona_missing_after']} | {dev_stats['task_missing_before']} | {dev_stats['task_recovered_from_prompt']} | {dev_stats['task_missing_after']} |",
        "",
        "## Notes",
        "",
        "- Persona metadata is recovered from the profile header when missing.",
        "- Task labels are canonicalized into the main ten SynPer task families when possible; otherwise the raw prompt text is used as a fallback label.",
        "- This repair targets metadata consistency rather than semantic filtering; no rows are dropped.",
        "",
        "## Outputs",
        "",
        f"- `tables/synper_audit/train_persona_counts.csv`",
        f"- `tables/synper_audit/train_task_counts.csv`",
        f"- `tables/synper_audit/dev_persona_counts.csv`",
        f"- `tables/synper_audit/dev_task_counts.csv`",
        "",
    ]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "train_output": args.train_output,
                "dev_output": args.dev_output,
                "report": args.report,
                "train_stats": train_stats,
                "dev_stats": dev_stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
