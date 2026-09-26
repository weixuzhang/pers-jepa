#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from persjepa.config import ExperimentConfig
from persjepa.data import PersonaExample, load_persona_jsonl
from persjepa.hidden import append_pred_tokens, require_torch, require_transformers, resolve_device, resolve_dtype


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    *,
    char_start: int,
    char_end: int,
    attention_mask,
) -> tuple[int, int, bool]:
    active = int(attention_mask.long().sum().item())
    token_ids = [
        idx
        for idx, (start, end) in enumerate(offsets[:active])
        if end > start and end > char_start and start < char_end
    ]
    if not token_ids:
        fallback = max(active - 1, 0)
        return fallback, fallback, False
    return token_ids[0], token_ids[-1], True


def span_summary(hidden_row, start: int, end: int):
    span = hidden_row[start : end + 1]
    return span[0], span[-1], span.mean(dim=0)


def padded_span_tokens(hidden_row, start: int, end: int, *, max_tokens: int):
    torch = require_torch()
    span = hidden_row[start : end + 1]
    span = span[:max_tokens]
    output = hidden_row.new_zeros((max_tokens, hidden_row.shape[-1]))
    mask = torch.zeros(max_tokens, dtype=torch.bool, device=hidden_row.device)
    if span.numel():
        output[: span.shape[0]] = span
        mask[: span.shape[0]] = True
    return output, mask


def token_offsets(tokenizer, text: str, *, max_length: int):
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
    finally:
        tokenizer.truncation_side = original_side
    offsets = encoded.pop("offset_mapping")
    return encoded, [(int(start), int(end)) for start, end in offsets]


def task_char_span(example: PersonaExample) -> tuple[int, int]:
    start = example.personalized_prompt.rfind(example.generic_prompt)
    if start < 0:
        start = max(len(example.personalized_prompt) - len(example.generic_prompt), 0)
    return start, min(start + len(example.generic_prompt), len(example.personalized_prompt))


def append_answer(text: str, target: str) -> tuple[str, int, int]:
    prefix = f"{text.rstrip()}\nAnswer: "
    answer = str(target).strip()
    return f"{prefix}{answer}", len(prefix), len(prefix) + len(answer)


def extract_batch(
    *,
    model,
    tokenizer,
    examples: Sequence[PersonaExample],
    pred_token: str,
    pred_token_id: int,
    predictor_tokens: int,
    layer: int,
    max_length: int,
    include_answer_spans: bool,
    max_answer_tokens: int,
    device,
):
    torch = require_torch()
    source_texts = [
        append_pred_tokens(example.generic_prompt, pred_token, predictor_tokens)
        for example in examples
    ]
    target_texts = [example.personalized_prompt for example in examples]
    source_answer_texts = []
    target_answer_texts = []
    source_answer_char_spans = []
    target_answer_char_spans = []
    if include_answer_spans:
        for example in examples:
            text, start, end = append_answer(example.generic_prompt, example.target)
            source_answer_texts.append(text)
            source_answer_char_spans.append((start, end))
            text, start, end = append_answer(example.personalized_prompt, example.target)
            target_answer_texts.append(text)
            target_answer_char_spans.append((start, end))

    source_inputs = []
    source_offsets = []
    for text in source_texts:
        encoded, offsets = token_offsets(tokenizer, text, max_length=max_length)
        source_inputs.append(encoded)
        source_offsets.append(offsets)
    target_inputs = []
    target_offsets = []
    for text in target_texts:
        encoded, offsets = token_offsets(tokenizer, text, max_length=max_length)
        target_inputs.append(encoded)
        target_offsets.append(offsets)
    source_answer_inputs = []
    source_answer_offsets = []
    target_answer_inputs = []
    target_answer_offsets = []
    if include_answer_spans:
        for text in source_answer_texts:
            encoded, offsets = token_offsets(tokenizer, text, max_length=max_length)
            source_answer_inputs.append(encoded)
            source_answer_offsets.append(offsets)
        for text in target_answer_texts:
            encoded, offsets = token_offsets(tokenizer, text, max_length=max_length)
            target_answer_inputs.append(encoded)
            target_answer_offsets.append(offsets)

    source_padded = tokenizer.pad(source_inputs, return_tensors="pt").to(device)
    target_padded = tokenizer.pad(target_inputs, return_tensors="pt").to(device)
    source_answer_padded = None
    target_answer_padded = None
    if include_answer_spans:
        source_answer_padded = tokenizer.pad(source_answer_inputs, return_tensors="pt").to(device)
        target_answer_padded = tokenizer.pad(target_answer_inputs, return_tensors="pt").to(device)

    with torch.no_grad():
        source_outputs = model(**source_padded, output_hidden_states=True)
        source_hidden = source_outputs.hidden_states[layer]
        target_outputs = model(**target_padded, output_hidden_states=True)
        target_hidden = target_outputs.hidden_states[layer]
        source_answer_hidden = None
        target_answer_hidden = None
        if include_answer_spans:
            source_answer_outputs = model(**source_answer_padded, output_hidden_states=True)
            source_answer_hidden = source_answer_outputs.hidden_states[layer]
            target_answer_outputs = model(**target_answer_padded, output_hidden_states=True)
            target_answer_hidden = target_answer_outputs.hidden_states[layer]

    idx = torch.arange(source_padded["input_ids"].shape[1], device=device)
    source_anchor_positions = []
    if predictor_tokens == 0:
        source_anchor_positions = (
            source_padded["attention_mask"].long().sum(dim=1) - 1
        ).tolist()
    else:
        for input_ids in source_padded["input_ids"]:
            is_pred = input_ids.eq(pred_token_id)
            if not bool(is_pred.any().item()):
                raise ValueError("A source sequence lost all [PRED] tokens after truncation.")
            source_anchor_positions.append(int(idx.masked_fill(~is_pred, -1).max().item()))
    target_anchor_positions = (
        target_padded["attention_mask"].long().sum(dim=1) - 1
    ).tolist()

    records = []
    tensors = {
        "source_anchor": [],
        "target_anchor": [],
        "source_span_start": [],
        "source_span_end": [],
        "source_span_mean": [],
        "target_task_span_start": [],
        "target_task_span_end": [],
        "target_task_span_mean": [],
        "target_profile_span_start": [],
        "target_profile_span_end": [],
        "target_profile_span_mean": [],
    }
    if include_answer_spans:
        tensors.update(
            {
                "source_answer_span_start": [],
                "source_answer_span_end": [],
                "source_answer_span_mean": [],
                "target_answer_span_start": [],
                "target_answer_span_end": [],
                "target_answer_span_mean": [],
                "source_answer_tokens": [],
                "target_answer_tokens": [],
                "source_answer_token_mask": [],
                "target_answer_token_mask": [],
            }
        )

    for row, example in enumerate(examples):
        source_start, source_end, source_valid = char_span_to_token_span(
            source_offsets[row],
            char_start=0,
            char_end=len(example.generic_prompt),
            attention_mask=source_padded["attention_mask"][row].cpu(),
        )
        task_start_char, task_end_char = task_char_span(example)
        target_task_start, target_task_end, target_task_valid = char_span_to_token_span(
            target_offsets[row],
            char_start=task_start_char,
            char_end=task_end_char,
            attention_mask=target_padded["attention_mask"][row].cpu(),
        )
        target_profile_start, target_profile_end, target_profile_valid = char_span_to_token_span(
            target_offsets[row],
            char_start=0,
            char_end=max(task_start_char, 0),
            attention_mask=target_padded["attention_mask"][row].cpu(),
        )

        tensors["source_anchor"].append(source_hidden[row, source_anchor_positions[row]].detach().cpu())
        tensors["target_anchor"].append(target_hidden[row, target_anchor_positions[row]].detach().cpu())
        start_h, end_h, mean_h = span_summary(source_hidden[row], source_start, source_end)
        tensors["source_span_start"].append(start_h.detach().cpu())
        tensors["source_span_end"].append(end_h.detach().cpu())
        tensors["source_span_mean"].append(mean_h.detach().cpu())
        start_h, end_h, mean_h = span_summary(target_hidden[row], target_task_start, target_task_end)
        tensors["target_task_span_start"].append(start_h.detach().cpu())
        tensors["target_task_span_end"].append(end_h.detach().cpu())
        tensors["target_task_span_mean"].append(mean_h.detach().cpu())
        start_h, end_h, mean_h = span_summary(target_hidden[row], target_profile_start, target_profile_end)
        tensors["target_profile_span_start"].append(start_h.detach().cpu())
        tensors["target_profile_span_end"].append(end_h.detach().cpu())
        tensors["target_profile_span_mean"].append(mean_h.detach().cpu())

        source_answer_record = None
        target_answer_record = None
        if include_answer_spans:
            assert source_answer_padded is not None
            assert target_answer_padded is not None
            assert source_answer_hidden is not None
            assert target_answer_hidden is not None
            source_answer_start, source_answer_end, source_answer_valid = char_span_to_token_span(
                source_answer_offsets[row],
                char_start=source_answer_char_spans[row][0],
                char_end=source_answer_char_spans[row][1],
                attention_mask=source_answer_padded["attention_mask"][row].cpu(),
            )
            target_answer_start, target_answer_end, target_answer_valid = char_span_to_token_span(
                target_answer_offsets[row],
                char_start=target_answer_char_spans[row][0],
                char_end=target_answer_char_spans[row][1],
                attention_mask=target_answer_padded["attention_mask"][row].cpu(),
            )
            start_h, end_h, mean_h = span_summary(
                source_answer_hidden[row],
                source_answer_start,
                source_answer_end,
            )
            tensors["source_answer_span_start"].append(start_h.detach().cpu())
            tensors["source_answer_span_end"].append(end_h.detach().cpu())
            tensors["source_answer_span_mean"].append(mean_h.detach().cpu())
            start_h, end_h, mean_h = span_summary(
                target_answer_hidden[row],
                target_answer_start,
                target_answer_end,
            )
            tensors["target_answer_span_start"].append(start_h.detach().cpu())
            tensors["target_answer_span_end"].append(end_h.detach().cpu())
            tensors["target_answer_span_mean"].append(mean_h.detach().cpu())
            answer_tokens, answer_mask = padded_span_tokens(
                source_answer_hidden[row],
                source_answer_start,
                source_answer_end,
                max_tokens=max_answer_tokens,
            )
            tensors["source_answer_tokens"].append(answer_tokens.detach().cpu())
            tensors["source_answer_token_mask"].append(answer_mask.detach().cpu())
            answer_tokens, answer_mask = padded_span_tokens(
                target_answer_hidden[row],
                target_answer_start,
                target_answer_end,
                max_tokens=max_answer_tokens,
            )
            tensors["target_answer_tokens"].append(answer_tokens.detach().cpu())
            tensors["target_answer_token_mask"].append(answer_mask.detach().cpu())
            source_answer_record = {
                "token_start": source_answer_start,
                "token_end": source_answer_end,
                "valid": source_answer_valid,
                "char_start": source_answer_char_spans[row][0],
                "char_end": source_answer_char_spans[row][1],
            }
            target_answer_record = {
                "token_start": target_answer_start,
                "token_end": target_answer_end,
                "valid": target_answer_valid,
                "char_start": target_answer_char_spans[row][0],
                "char_end": target_answer_char_spans[row][1],
            }

        records.append(
            {
                "id": example.id,
                "source_anchor_position": source_anchor_positions[row],
                "target_anchor_position": target_anchor_positions[row],
                "source_span": {
                    "token_start": source_start,
                    "token_end": source_end,
                    "valid": source_valid,
                    "char_start": 0,
                    "char_end": len(example.generic_prompt),
                },
                "target_task_span": {
                    "token_start": target_task_start,
                    "token_end": target_task_end,
                    "valid": target_task_valid,
                    "char_start": task_start_char,
                    "char_end": task_end_char,
                },
                "target_profile_span": {
                    "token_start": target_profile_start,
                    "token_end": target_profile_end,
                    "valid": target_profile_valid,
                    "char_start": 0,
                    "char_end": max(task_start_char, 0),
                },
                "source_answer_span": source_answer_record,
                "target_answer_span": target_answer_record,
            }
        )

    return {
        key: torch.stack(value, dim=0)
        for key, value in tensors.items()
    } | {"span_records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract span-aware hidden states for upstream-style STP experiments.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--predictor-tokens", type=int, default=None)
    parser.add_argument("--pred-token", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--include-answer-spans", action="store_true")
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto", help="render prompts with the tokenizer chat template (auto: instruct-like model names)")
    parser.add_argument("--system", default=None, help="optional system prompt for chat formatting")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    predictor_tokens = (
        args.predictor_tokens
        if args.predictor_tokens is not None
        else cfg.model.predictor_tokens
    )
    if predictor_tokens < 0:
        raise ValueError("--predictor-tokens must be >= 0")

    examples = load_persona_jsonl(
        args.input,
        profile_field=cfg.data.profile_field,
        prompt_field=cfg.data.text_field,
        target_field=cfg.data.target_field,
        id_field=cfg.data.id_field,
        profile_template=cfg.data.profile_template,
    )
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    torch = require_torch()
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    model_name = args.model_name or cfg.model.name
    pred_token = args.pred_token or cfg.model.pred_token
    max_length = args.max_length or cfg.model.max_length
    layer = cfg.model.layer if args.layer is None else args.layer
    device = resolve_device(args.device or cfg.model.device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Span extraction requires a fast tokenizer with offset mappings.")
    added = tokenizer.add_special_tokens({"additional_special_tokens": [pred_token]})
    from persjepa.chat import apply_chat_formatting
    chat_applied = apply_chat_formatting(examples, tokenizer, model_name, args.chat_template, args.system)
    print(f"chat template: {'on' if chat_applied else 'off'} ({model_name})")
    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    torch_dtype = resolve_dtype(args.dtype or cfg.model.dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if added:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    pred_token_id = tokenizer.convert_tokens_to_ids(pred_token)

    chunks = []
    for start in tqdm(range(0, len(examples), args.batch_size), desc="extract_spans"):
        chunks.append(
            extract_batch(
                model=model,
                tokenizer=tokenizer,
                examples=examples[start : start + args.batch_size],
                pred_token=pred_token,
                pred_token_id=pred_token_id,
                predictor_tokens=predictor_tokens,
                layer=layer,
                max_length=max_length,
                include_answer_spans=args.include_answer_spans,
                max_answer_tokens=args.max_answer_tokens,
                device=device,
            )
        )

    tensor_keys = [key for key in chunks[0] if key != "span_records"]
    output = {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0)
        for key in tensor_keys
    }
    output["span_records"] = [
        record for chunk in chunks for record in chunk["span_records"]
    ]
    output["examples"] = [asdict(example) for example in examples]
    output["config"] = {
        "model_name": model_name,
        "pred_token": pred_token,
        "predictor_tokens": predictor_tokens,
        "layer": layer,
        "max_length": max_length,
        "include_answer_spans": args.include_answer_spans,
        "max_answer_tokens": args.max_answer_tokens,
        "format": "stp_span_answer_v1" if args.include_answer_spans else "stp_span_stage1",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Wrote {len(examples)} span hidden records to {output_path}")


if __name__ == "__main__":
    main()
