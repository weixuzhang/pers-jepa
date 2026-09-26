from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator
import json
import random


@dataclass
class PersonaExample:
    id: str
    generic_prompt: str
    personalized_prompt: str
    profile: str = ""
    target: str = ""
    metadata: dict | None = None

    def to_json(self) -> dict:
        return asdict(self)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_num}") from exc


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_personalized_prompt(profile: str, prompt: str, template: str) -> str:
    return template.format(profile=profile.strip(), prompt=prompt.strip()).strip()


def normalize_record(
    record: dict,
    *,
    profile_field: str = "profile",
    prompt_field: str = "prompt",
    target_field: str = "target",
    id_field: str = "id",
    profile_template: str = "{profile}\n\nTask: {prompt}",
) -> PersonaExample:
    if "generic_prompt" in record and "personalized_prompt" in record:
        generic_prompt = str(record["generic_prompt"])
        personalized_prompt = str(record["personalized_prompt"])
        profile = str(record.get("profile", ""))
        target = str(record.get("target", ""))
    else:
        prompt = record.get(prompt_field) or record.get("input") or record.get("question")
        if prompt is None:
            raise KeyError(
                f"Record lacks '{prompt_field}', 'input', or 'question': {record.keys()}"
            )
        profile = str(record.get(profile_field, record.get("history", "")))
        target = str(record.get(target_field, record.get("output", "")))
        generic_prompt = str(prompt).strip()
        personalized_prompt = make_personalized_prompt(
            profile=profile,
            prompt=generic_prompt,
            template=profile_template,
        )

    ex_id = str(record.get(id_field, record.get("id", ""))).strip()
    if not ex_id:
        ex_id = str(abs(hash((generic_prompt, personalized_prompt))) % 10**12)
    metadata = {k: v for k, v in record.items() if k not in {
        "generic_prompt",
        "personalized_prompt",
        profile_field,
        prompt_field,
        target_field,
        id_field,
    }}
    return PersonaExample(
        id=ex_id,
        generic_prompt=generic_prompt,
        personalized_prompt=personalized_prompt,
        profile=profile,
        target=target,
        metadata=metadata or None,
    )


def load_persona_jsonl(path: str | Path, **normalize_kwargs) -> list[PersonaExample]:
    return [normalize_record(record, **normalize_kwargs) for record in read_jsonl(path)]


def split_examples(
    examples: list[PersonaExample],
    eval_fraction: float,
    seed: int = 42,
) -> tuple[list[PersonaExample], list[PersonaExample]]:
    if eval_fraction <= 0:
        return examples, []
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, int(round(len(examples) * eval_fraction)))
    eval_ids = set(indices[:eval_size])
    train = [example for idx, example in enumerate(examples) if idx not in eval_ids]
    eval_ = [example for idx, example in enumerate(examples) if idx in eval_ids]
    return train, eval_


def load_hf_dataset(name: str, split: str, **kwargs) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to load Hugging Face datasets.") from exc
    dataset = load_dataset(name, split=split, **kwargs)
    return [dict(row) for row in dataset]


def lamp5_email_adapter(record: dict) -> dict:
    """Best-effort adapter for LaMP-style email generation records."""
    profile = record.get("profile") or record.get("history") or record.get("user_profile") or ""
    prompt = record.get("input") or record.get("prompt") or record.get("question") or ""
    target = record.get("output") or record.get("target") or record.get("answer") or ""
    return {
        "id": record.get("id", record.get("qid", "")),
        "profile": profile,
        "prompt": prompt,
        "target": target,
    }


def amazon_review_adapter(record: dict) -> dict:
    """Adapter for user-centric Amazon review/style records."""
    user_fields = [
        record.get("user_history"),
        record.get("profile"),
        record.get("review_history"),
        record.get("summary_history"),
    ]
    profile = next((str(value) for value in user_fields if value), "")
    item = record.get("item") or record.get("title") or record.get("product_title") or "this item"
    prompt = record.get("prompt") or f"Write a review for {item}."
    target = record.get("reviewText") or record.get("review") or record.get("target") or ""
    return {
        "id": record.get("id", record.get("reviewerID", "")),
        "profile": profile,
        "prompt": prompt,
        "target": target,
    }


ADAPTERS = {
    "none": lambda record: record,
    "lamp5": lamp5_email_adapter,
    "amazon": amazon_review_adapter,
}


def adapt_records(records: Iterable[dict], adapter: str) -> list[dict]:
    if adapter not in ADAPTERS:
        raise KeyError(f"Unknown adapter '{adapter}'. Choices: {sorted(ADAPTERS)}")
    fn = ADAPTERS[adapter]
    return [fn(record) for record in records]

