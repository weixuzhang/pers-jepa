from __future__ import annotations

import json
from typing import Any


JUDGE_SYSTEM_PROMPT = (
    "You are a strict personalization judge. Compare two candidate outputs for "
    "the same user and task. Prefer the answer that better matches the user's "
    "profile, tone, constraints, and likely preferences. Ignore candidate order. "
    "Return only valid JSON with keys winner, confidence, and reason."
)


def build_pairwise_judge_prompt(record: dict, candidate_a: str, candidate_b: str) -> str:
    payload = {
        "user_profile": record.get("profile", ""),
        "task_prompt": record.get("generic_prompt", record.get("prompt", "")),
        "reference_target": record.get("target", ""),
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "response_schema": {
            "winner": "A, B, or tie",
            "confidence": "number from 0 to 1",
            "reason": "one short sentence",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def judge_with_openai(
    prompt: str,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai to use --judge-provider openai.") from exc
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)
