#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def first_sentence(text: str) -> str:
    text = str(text).strip()
    match = SENTENCE_BOUNDARY_RE.search(text)
    if match is None:
        return text
    return text[: match.start()].strip()


def quoted_title(text: str) -> str:
    text = str(text).strip()
    match = re.search(r"[\"“](.+?)[\"”]", text)
    return match.group(1).strip() if match else text


def strip_boilerplate(text: str) -> str:
    text = str(text).strip()
    patterns = [
        r"^our paper is titled\s+[\"“]?",
        r"^the paper (presents|proposes|introduces|describes|studies|investigates)\s+",
        r"^this paper (presents|proposes|introduces|describes|studies|investigates)\s+",
        r"^in this paper,?\s+(we\s+)?(present|propose|introduce|describe|study|investigate)\s+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = text.strip(" \"'")
    text = text.strip("“”")
    if text.endswith("."):
        text = text[:-1]
    return re.sub(r"\s+", " ", text).strip()


def clean_title(text: str) -> str:
    return strip_boilerplate(first_sentence(quoted_title(text)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Add cleaned title-generation keys to an evaluation JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-keys", nargs="+", required=True)
    parser.add_argument("--suffix", default="_clean")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            for key in args.candidate_keys:
                record[f"{key}{args.suffix}"] = clean_title(str(record.get(key, "")))
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote cleaned generations to {output_path}")


if __name__ == "__main__":
    main()
