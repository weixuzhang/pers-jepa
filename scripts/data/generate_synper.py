#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from persjepa.data import write_jsonl


PERSONAS = [
    {
        "name": "The Academic",
        "style": "precise, citation-aware, measured, abstract, avoids slang",
    },
    {
        "name": "The Gen-Z Creator",
        "style": "casual, punchy, meme-aware, enthusiastic, uses short sentences",
    },
    {
        "name": "The Corporate Lawyer",
        "style": "formal, caveated, risk-focused, exact, contract-like",
    },
    {
        "name": "The Minimalist Operator",
        "style": "terse, direct, checklist-driven, no filler",
    },
    {
        "name": "The Warm Teacher",
        "style": "encouraging, explanatory, patient, uses simple analogies",
    },
    {
        "name": "The Skeptical Auditor",
        "style": "critical, evidence-seeking, flags assumptions and edge cases",
    },
    {
        "name": "The Poetic Essayist",
        "style": "lyrical, vivid, metaphorical, reflective",
    },
    {
        "name": "The Customer Support Lead",
        "style": "empathetic, solution-oriented, calm, structured",
    },
    {
        "name": "The Pirate",
        "style": "nautical voice, playful archaic phrasing, still answers clearly",
    },
    {
        "name": "The Systems Engineer",
        "style": "technical, modular, tradeoff-focused, operational",
    },
]

PERSONA_STYLE = {persona["name"]: persona["style"] for persona in PERSONAS}


TASK_TYPES = [
    "summarize a short article",
    "write a polite rejection",
    "explain a technical concept",
    "write a product review",
    "draft an email",
    "recommend a plan",
    "rewrite a message",
    "answer a customer question",
    "name or title a short text",
    "critique an idea",
]


def load_env_file(path: str | Path | None) -> None:
    if path is None:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(normalize_record(json.loads(line)))
    return records


def normalize_record(record: dict) -> dict:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    persona = str(metadata.get("persona") or record.get("profile") or "").strip()
    style = PERSONA_STYLE.get(persona)
    if style:
        record["profile"] = f"Persona: {persona}\nStyle constraints: {style}."
    return record


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def build_prompt(start_index: int, batch_size: int, seed: int) -> str:
    rng = random.Random(seed + start_index)
    personas = rng.sample(PERSONAS, k=len(PERSONAS))
    task_types = rng.sample(TASK_TYPES, k=len(TASK_TYPES))
    return json.dumps({
        "instruction": (
            "Generate synthetic paired-view persona examples for Pers-JEPA. "
            "Each example needs the same task in generic form and a target answer "
            "written in one persona's style. Return valid JSON only."
        ),
        "schema": {
            "examples": [
                {
                    "id": "synper-000001",
                    "profile": "Persona name and style constraints.",
                    "prompt": "Generic task prompt without persona information.",
                    "target": "Persona-conditioned answer to the task.",
                    "metadata": {
                        "persona": "persona name",
                        "task_type": "task category",
                    },
                }
            ]
        },
        "requirements": [
            f"Return exactly {batch_size} examples.",
            f"IDs must run from synper-{start_index + 1:06d} upward with no gaps.",
            "Use diverse topics and avoid copying examples from prior batches.",
            "The generic prompt must be solvable without persona context.",
            "The target must clearly express the persona style while still completing the task.",
            "Keep each target between 45 and 120 words.",
            "Do not include markdown fences.",
        ],
        "personas_to_mix": personas,
        "task_types_to_mix": task_types,
    }, ensure_ascii=False)


def validate_records(payload: dict, *, start_index: int, batch_size: int) -> list[dict]:
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Model response lacks an examples list.")
    if len(examples) < 1:
        raise ValueError("Model response returned no examples.")
    if len(examples) > batch_size:
        examples = examples[:batch_size]
    records = []
    for offset, record in enumerate(examples):
        expected_id = f"synper-{start_index + offset + 1:06d}"
        for key in ["profile", "prompt", "target"]:
            if not str(record.get(key, "")).strip():
                raise ValueError(f"{expected_id} missing {key}.")
        records.append(normalize_record({
            "id": expected_id,
            "profile": str(record["profile"]).strip(),
            "prompt": str(record["prompt"]).strip(),
            "target": str(record["target"]).strip(),
            "metadata": {
                "source": "SynPer",
                **(record.get("metadata") if isinstance(record.get("metadata"), dict) else {}),
            },
        }))
    return records


def generate_batch(client, *, model: str, start_index: int, batch_size: int, seed: int) -> list[dict]:
    prompt = build_prompt(start_index, batch_size, seed)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate clean synthetic personalization data. "
                    "Return only valid JSON matching the user's schema."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.9,
    )
    content = response.choices[0].message.content or "{}"
    return validate_records(
        parse_json_object(content),
        start_index=start_index,
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SynPer synthetic paired persona data.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()

    load_env_file(args.env_file)
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get(args.api_key_env),
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
    )
    output_path = Path(args.output)
    existing = read_existing(output_path)
    if len(existing) > args.num_examples:
        existing = existing[:args.num_examples]

    records = list(existing)
    if records:
        write_jsonl(output_path, records)
    while len(records) < args.num_examples:
        batch_size = min(args.batch_size, args.num_examples - len(records))
        start_index = len(records)
        for attempt in range(1, args.max_retries + 1):
            try:
                batch = generate_batch(
                    client,
                    model=args.model,
                    start_index=start_index,
                    batch_size=batch_size,
                    seed=args.seed,
                )
                records.extend(batch)
                write_jsonl(output_path, records)
                print(f"Wrote {len(records)}/{args.num_examples} examples to {output_path}", flush=True)
                break
            except Exception as exc:
                if attempt == args.max_retries:
                    raise
                sleep_s = 2 ** attempt
                print(f"Batch starting {start_index} failed ({type(exc).__name__}: {exc}); retrying in {sleep_s}s", flush=True)
                time.sleep(sleep_s)


if __name__ == "__main__":
    main()
