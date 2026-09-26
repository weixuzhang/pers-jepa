#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from persjepa.judge import build_pairwise_judge_prompt, judge_with_openai


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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def parse_json_object(text: str) -> dict:
    raw_text = text.strip()
    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            return {
                "winner": "tie",
                "confidence": 0.0,
                "reason": "Judge output was valid JSON but not an object.",
                "raw_text": raw_text[:1000],
            }
        result.setdefault("raw_text", raw_text)
        return result
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {
                "winner": "tie",
                "confidence": 0.0,
                "reason": "Could not parse judge output.",
                "raw_text": raw_text[:1000],
            }
        try:
            result = json.loads(match.group(0))
            if not isinstance(result, dict):
                return {
                    "winner": "tie",
                    "confidence": 0.0,
                    "reason": "Judge JSON block was not an object.",
                    "raw_text": raw_text[:1000],
                }
            result.setdefault("raw_text", raw_text)
            return result
        except json.JSONDecodeError:
            return {
                "winner": "tie",
                "confidence": 0.0,
                "reason": "Could not parse judge JSON block.",
                "raw_text": raw_text[:1000],
            }


def clip_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half].rstrip() + "\n[...]\n" + text[-(max_chars - half):].lstrip()


def compact_record_for_judge(
    record: dict,
    *,
    max_profile_chars: int | None,
    max_reference_chars: int | None,
) -> dict:
    compact = dict(record)
    if "profile" in compact:
        compact["profile"] = clip_text(str(compact["profile"]), max_profile_chars)
    if "target" in compact:
        compact["target"] = clip_text(str(compact["target"]), max_reference_chars)
    return compact


class LocalJudge:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        dtype: str,
        max_input_tokens: int | None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype, torch.bfloat16)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.max_input_tokens = max_input_tokens

    def judge(self, prompt: str, *, max_new_tokens: int) -> dict:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict personalization judge. Return only JSON "
                    "with keys winner, confidence, and reason. winner must be A, B, or tie."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=self.max_input_tokens is not None,
            max_length=self.max_input_tokens,
        ).to(self.device)
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        return parse_json_object(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def normalize_winner(result: dict, *, swapped: bool) -> str:
    winner = str(result.get("winner", "tie")).strip().upper()
    if winner not in {"A", "B"}:
        return "tie"
    if swapped:
        winner = "A" if winner == "B" else "B"
    return winner


def score_pair(
    record: dict,
    left_key: str,
    right_key: str,
    judge_fn,
    rng: random.Random,
    max_new_tokens: int,
    *,
    max_candidate_chars: int | None,
) -> dict:
    left = clip_text(str(record[left_key]), max_candidate_chars)
    right = clip_text(str(record[right_key]), max_candidate_chars)
    swapped = rng.random() < 0.5
    candidate_a, candidate_b = (right, left) if swapped else (left, right)
    prompt = build_pairwise_judge_prompt(record, candidate_a, candidate_b)
    result = judge_fn(prompt, max_new_tokens=max_new_tokens)
    normalized = normalize_winner(result, swapped=swapped)
    if normalized == "A":
        winner = left_key
    elif normalized == "B":
        winner = right_key
    else:
        winner = "tie"
    return {
        "winner": winner,
        "raw_winner": result.get("winner"),
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "raw_text": result.get("raw_text"),
        "swapped": swapped,
    }


def parse_pair_specs(pair_specs: list[str]) -> list[tuple[str, str, str]]:
    pairs = []
    for spec in pair_specs:
        pieces = spec.split(":")
        if len(pieces) != 3 or not all(piece.strip() for piece in pieces):
            raise ValueError(
                "Each --pair must use label:left_key:right_key, "
                f"but got {spec!r}."
            )
        label, left_key, right_key = [piece.strip() for piece in pieces]
        pairs.append((label, left_key, right_key))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge generation records with OpenAI or a local instruct model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--provider", choices=["openai", "local"], default="local")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Optional tokenizer truncation for local judges; useful for CPU fallback.",
    )
    parser.add_argument("--max-profile-chars", type=int, default=None)
    parser.add_argument("--max-reference-chars", type=int, default=None)
    parser.add_argument("--max-candidate-chars", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--raw-key", default="raw_generic_output")
    parser.add_argument("--predictor-key", default="generic_pred_output")
    parser.add_argument("--candidate-key", default="jepa_steered_output")
    parser.add_argument(
        "--candidate-label",
        default="jepa",
        help="Short label used in summary comparison names, e.g. jepa_scale_1p5.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help=(
            "Optional explicit pairwise comparison as label:left_key:right_key. "
            "Can be repeated. When supplied, raw/predictor defaults are skipped."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_env_file(args.env_file)
    rng = random.Random(args.seed)
    records = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_examples is not None:
        records = records[:args.max_examples]

    if args.provider == "openai":
        def judge_fn(prompt: str, *, max_new_tokens: int) -> dict:
            return judge_with_openai(
                prompt,
                model=args.model,
                api_key=os.environ.get(args.api_key_env),
                base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
            )
    else:
        local_judge = LocalJudge(
            args.model,
            device=args.device,
            dtype=args.dtype,
            max_input_tokens=args.max_input_tokens,
        )

        def judge_fn(prompt: str, *, max_new_tokens: int) -> dict:
            return local_judge.judge(prompt, max_new_tokens=max_new_tokens)

    if args.pair:
        comparisons = parse_pair_specs(args.pair)
    else:
        comparisons = [
            (f"raw_generic_vs_{args.candidate_label}", args.raw_key, args.candidate_key),
            (f"generic_pred_vs_{args.candidate_label}", args.predictor_key, args.candidate_key),
        ]
    summary = {
        label: {left_key: 0, right_key: 0, "tie": 0}
        for label, left_key, right_key in comparisons
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc="judge"):
            judge_record = compact_record_for_judge(
                record,
                max_profile_chars=args.max_profile_chars,
                max_reference_chars=args.max_reference_chars,
            )
            judged = dict(record)
            judged["judge_results"] = {
                label: score_pair(
                    judge_record,
                    left_key,
                    right_key,
                    judge_fn,
                    rng,
                    args.max_new_tokens,
                    max_candidate_chars=args.max_candidate_chars,
                )
                for label, left_key, right_key in comparisons
            }
            for comparison, result in judged["judge_results"].items():
                summary[comparison][result["winner"]] += 1
            handle.write(json.dumps(judged, ensure_ascii=False) + "\n")
            handle.flush()
            first_label = comparisons[0][0]
            summary["n_examples"] = sum(summary[first_label].values())
            summary["provider"] = args.provider
            summary["model"] = args.model
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["n_examples"] = len(records)
    summary["provider"] = args.provider
    summary["model"] = args.model
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
