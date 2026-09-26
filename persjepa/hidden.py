from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Sequence

from persjepa.data import PersonaExample


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch to use hidden-state extraction.") from exc
    return torch


def require_transformers():
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to use hidden-state extraction.") from exc
    return AutoModelForCausalLM, AutoTokenizer


def append_pred_tokens(text: str, pred_token: str, count: int) -> str:
    text = text.rstrip()
    if count <= 0:
        return text
    suffix = " ".join([pred_token] * count)
    return f"{text} {suffix}" if text else suffix


def resolve_device(device: str):
    torch = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str):
    torch = require_torch()
    if dtype in {"auto", "float32", "fp32"}:
        return torch.float32 if dtype != "auto" else "auto"
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


class HiddenStateExtractor:
    def __init__(
        self,
        model_name: str,
        *,
        pred_token: str = "[PRED]",
        predictor_tokens: int = 1,
        layer: int = -1,
        max_length: int = 1024,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        if predictor_tokens < 0:
            raise ValueError("predictor_tokens must be >= 0 for Pers-JEPA extraction.")
        torch = require_torch()
        AutoModelForCausalLM, AutoTokenizer = require_transformers()
        self.torch = torch
        self.pred_token = pred_token
        self.predictor_tokens = predictor_tokens
        self.layer = layer
        self.max_length = max_length
        self.device = resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [pred_token]}
        )
        model_kwargs = {"trust_remote_code": trust_remote_code}
        torch_dtype = resolve_dtype(dtype)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
        self.pred_token_id = self.tokenizer.convert_tokens_to_ids(pred_token)

    def _tokenize(self, texts: Sequence[str], *, truncation_side: str | None = None):
        original_side = self.tokenizer.truncation_side
        if truncation_side is not None:
            self.tokenizer.truncation_side = truncation_side
        try:
            return self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
        finally:
            self.tokenizer.truncation_side = original_side

    def _select_last_nonpad(self, hidden, attention_mask):
        lengths = attention_mask.long().sum(dim=1) - 1
        batch = self.torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, lengths]

    def _select_last_pred(self, hidden, input_ids):
        is_pred = input_ids.eq(self.pred_token_id)
        if not self.torch.all(is_pred.any(dim=1)):
            raise ValueError("At least one sequence does not contain a [PRED] token.")
        idx = self.torch.arange(input_ids.shape[1], device=input_ids.device)
        positions = idx.masked_fill(~is_pred, -1).max(dim=1).values
        batch = self.torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, positions]

    def extract_batch(self, examples: Sequence[PersonaExample]) -> dict:
        torch = self.torch
        generic_texts = [
            append_pred_tokens(ex.generic_prompt, self.pred_token, self.predictor_tokens)
            for ex in examples
        ]
        personalized_texts = [ex.personalized_prompt for ex in examples]
        with torch.no_grad():
            source_inputs = self._tokenize(generic_texts, truncation_side="left")
            source_outputs = self.model(**source_inputs, output_hidden_states=True)
            source_hidden = source_outputs.hidden_states[self.layer]
            if self.predictor_tokens == 0:
                h_gen = self._select_last_nonpad(
                    source_hidden,
                    source_inputs["attention_mask"],
                )
            else:
                h_gen = self._select_last_pred(source_hidden, source_inputs["input_ids"])

            target_inputs = self._tokenize(personalized_texts, truncation_side="left")
            target_outputs = self.model(**target_inputs, output_hidden_states=True)
            target_hidden = target_outputs.hidden_states[self.layer]
            h_pers = self._select_last_nonpad(
                target_hidden,
                target_inputs["attention_mask"],
            )
        return {
            "h_gen": h_gen.detach().cpu(),
            "h_pers": h_pers.detach().cpu(),
            "ids": [ex.id for ex in examples],
            "generic_prompt_with_pred": generic_texts,
            "personalized_prompt": personalized_texts,
        }

    def extract_to_file(
        self,
        examples: Sequence[PersonaExample],
        output_path: str | Path,
        *,
        batch_size: int = 8,
    ) -> None:
        from tqdm import tqdm

        torch = self.torch
        chunks = []
        for start in tqdm(range(0, len(examples), batch_size), desc="extract"):
            chunks.append(self.extract_batch(examples[start:start + batch_size]))
        h_gen = torch.cat([chunk["h_gen"] for chunk in chunks], dim=0)
        h_pers = torch.cat([chunk["h_pers"] for chunk in chunks], dim=0)
        ids = [item for chunk in chunks for item in chunk["ids"]]
        generic_prompt_with_pred = [
            item for chunk in chunks for item in chunk["generic_prompt_with_pred"]
        ]
        personalized_prompt = [
            item for chunk in chunks for item in chunk["personalized_prompt"]
        ]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "h_gen": h_gen,
                "h_pers": h_pers,
                "ids": ids,
                "generic_prompt_with_pred": generic_prompt_with_pred,
                "personalized_prompt": personalized_prompt,
                "config": {
                    "pred_token": self.pred_token,
                    "predictor_tokens": self.predictor_tokens,
                    "layer": self.layer,
                    "max_length": self.max_length,
                },
                "examples": [asdict(ex) for ex in examples],
            },
            output_path,
        )
