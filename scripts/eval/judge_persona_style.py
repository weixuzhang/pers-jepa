#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
from typing import Any

from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a strict style and personalization judge. Your job is to decide "
    "which candidate better matches the target user's persona, tone, style "
    "constraints, and likely preferences. Ignore candidate order. Do not reward "
    "generic fluency alone. Penalize empty, nonsensical, or clearly off-task "
    "answers. Return only valid JSON with keys winner, confidence, and reason."
)


def parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            obj.setdefault("raw_text", raw)
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                obj.setdefault("raw_text", raw)
                return obj
        except json.JSONDecodeError:
            pass
    return {"winner": "tie", "confidence": 0.0, "reason": "Could not parse judge output.", "raw_text": raw[:1000]}


def clip_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n[...]\n" + text[-(max_chars - half):].lstrip()


def persona_profile(record: dict[str, Any]) -> str:
    if record.get("profile"):
        return str(record["profile"])
    metadata = record.get("metadata") or {}
    if isinstance(metadata, dict):
        if metadata.get("persona_description"):
            return str(metadata["persona_description"])
        if metadata.get("persona_id"):
            return str(metadata["persona_id"])
    if record.get("persona_description"):
        return str(record["persona_description"])
    return ""


def build_prompt(record: dict[str, Any], candidate_a: str, candidate_b: str, *, max_profile_chars: int | None) -> str:
    payload = {
        "target_user_or_persona_profile": clip_text(persona_profile(record), max_profile_chars),
        "task_prompt": record.get("generic_prompt", record.get("prompt", "")),
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "evaluation_instruction": (
            "Choose the candidate that better reflects the target persona/user style. "
            "Focus on tone, wording, constraints, and preference signal. If both are "
            "equally generic or equally off-persona, choose tie."
        ),
        "response_schema": {
            "winner": "A, B, or tie",
            "confidence": "number from 0 to 1",
            "reason": "one short sentence",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def judge_openai(prompt: str, *, model: str, api_key: str | None, base_url: str | None) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return parse_json_object(response.choices[0].message.content or "{}")


def normalize_winner(result: dict[str, Any], *, swapped: bool) -> str:
    winner = str(result.get("winner", "tie")).strip().upper()
    if winner not in {"A", "B"}:
        return "tie"
    if swapped:
        winner = "A" if winner == "B" else "B"
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description="Style/persona-only pairwise judge for personalization outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--left-key", required=True)
    parser.add_argument("--right-key", required=True)
    parser.add_argument("--label", default="style_persona_pair")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-profile-chars", type=int, default=1200)
    parser.add_argument("--max-candidate-chars", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_examples is not None:
        records = records[: args.max_examples]
    rng = random.Random(args.seed)
    summary = {args.left_key: 0, args.right_key: 0, "tie": 0}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc="persona_style_judge"):
            left = clip_text(str(record.get(args.left_key, "")), args.max_candidate_chars)
            right = clip_text(str(record.get(args.right_key, "")), args.max_candidate_chars)
            swapped = rng.random() < 0.5
            candidate_a, candidate_b = (right, left) if swapped else (left, right)
            result = judge_openai(
                build_prompt(record, candidate_a, candidate_b, max_profile_chars=args.max_profile_chars),
                model=args.model,
                api_key=os.environ.get(args.api_key_env),
                base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
            )
            normalized = normalize_winner(result, swapped=swapped)
            winner = args.left_key if normalized == "A" else args.right_key if normalized == "B" else "tie"
            summary[winner] += 1
            judged = dict(record)
            judged["persona_style_judge"] = {
                "label": args.label,
                "winner": winner,
                "raw_winner": result.get("winner"),
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "swapped": swapped,
            }
            handle.write(json.dumps(judged, ensure_ascii=False) + "\n")
            handle.flush()
            summary["n_examples"] = sum(summary[key] for key in [args.left_key, args.right_key, "tie"])
            summary["provider"] = "openai"
            summary["model"] = args.model
            summary_path.write_text(json.dumps({args.label: summary}, indent=2), encoding="utf-8")
    summary["n_examples"] = len(records)
    summary["provider"] = "openai"
    summary["model"] = args.model
    summary_path.write_text(json.dumps({args.label: summary}, indent=2), encoding="utf-8")
    print(json.dumps({args.label: summary}, indent=2))


if __name__ == "__main__":
    main()
