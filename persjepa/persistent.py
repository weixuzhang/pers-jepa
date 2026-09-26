"""Persistent (ActAdd / persona-vector style) residual injection.

The paper's original intervention adds the residual at one anchor position of
the final layer, which can only bias the first generated token (KV caches are
pre-norm). Persistent injection instead adds the residual at EVERY position
of a middle decoder layer, during the prompt pass and at every decoding step,
so the direction acts throughout generation.

For per-example predictors the residual is computed ONCE from the generic
prompt's [PRED]-anchored hidden state at the same layer, then added
persistently. Constant-vector checkpoints (mean_delta, per-group means) are
handled by the same code path.

`PersistentSteerer.forward_with_injection` runs a differentiable forward pass
with a delta that carries gradient, which is what the likelihood objective in
scripts/training/train_residual_likelihood.py needs.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from persjepa.hidden import append_pred_tokens, require_torch
from persjepa.intervention import decoder_layer_module, final_norm_module


def predictor_delta(predictor, h: "torch.Tensor") -> "torch.Tensor":
    """Residual delta from any steering predictor (SAEOutput or h+delta style)."""
    out = predictor(h)
    if hasattr(out, "delta") and out.delta is not None:
        return out.delta
    if hasattr(out, "reconstruction"):
        return out.reconstruction - h
    return out - h


class PersistentSteerer:
    def __init__(
        self,
        model,
        tokenizer,
        predictor,
        *,
        layer: int = 14,
        pred_token: str = "[PRED]",
        predictor_tokens: int = 3,
        max_length: int = 1024,
        residual_scale: float = 1.0,
        device=None,
    ) -> None:
        self.torch = require_torch()
        self.model = model
        self.tokenizer = tokenizer
        self.predictor = predictor
        self.layer = layer
        self.pred_token = pred_token
        self.predictor_tokens = predictor_tokens
        self.max_length = max_length
        self.residual_scale = residual_scale
        self.device = device or next(model.parameters()).device
        self.pred_token_id = tokenizer.convert_tokens_to_ids(pred_token)

    # ----------------------------------------------------------------- utils
    def _hook_module(self):
        if self.layer == -1:
            return final_norm_module(self.model)
        return decoder_layer_module(self.model, self.layer - 1)

    def tokenize(self, text: str, *, max_length: int | None = None):
        side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            return self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_length or self.max_length,
            ).to(self.device)
        finally:
            self.tokenizer.truncation_side = side

    def anchor_state(self, generic_prompt: str) -> "torch.Tensor":
        """Layer-`layer` hidden state at the last [PRED] token of the generic
        prompt (no gradient through the LM). Shape [1, D]."""
        torch = self.torch
        prompt = append_pred_tokens(generic_prompt, self.pred_token, self.predictor_tokens)
        inputs = self.tokenize(prompt)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[self.layer]
        ids = inputs["input_ids"]
        if self.predictor_tokens > 0:
            is_pred = ids.eq(self.pred_token_id)
            pos = torch.arange(ids.shape[1], device=ids.device).masked_fill(~is_pred, -1).max(dim=1).values
        else:
            pos = inputs["attention_mask"].long().sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), pos]

    def compute_delta(self, generic_prompt: str) -> "torch.Tensor":
        """Scaled residual [1, D] for this prompt (no grad)."""
        torch = self.torch
        h = self.anchor_state(generic_prompt)
        p = next(self.predictor.parameters(), None)
        if p is None:
            p = next(self.predictor.buffers())
        with torch.no_grad():
            delta = predictor_delta(self.predictor, h.to(p.dtype))
        return delta * self.residual_scale

    @contextmanager
    def injecting(self, delta: "torch.Tensor"):
        """Context manager adding `delta` ([1, D] or [B, 1, D]) at every position
        of the hooked layer's output."""
        if delta.ndim == 2:
            delta = delta[:, None, :]

        def hook(_m, _i, output):
            if isinstance(output, tuple):
                return (output[0] + delta.to(output[0].dtype), *output[1:])
            return output + delta.to(output.dtype)

        handle = self._hook_module().register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    # ------------------------------------------------------------- inference
    def generate(self, generic_prompt: str, *, max_new_tokens: int = 64,
                 delta: "torch.Tensor | None" = None, generation_kwargs: dict[str, Any] | None = None) -> str:
        torch = self.torch
        if delta is None:
            delta = self.compute_delta(generic_prompt)
        inputs = self.tokenize(generic_prompt)
        with self.injecting(delta), torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id, **(generation_kwargs or {}),
            )
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # -------------------------------------------------------------- training
    def forward_with_injection(self, input_ids, attention_mask, delta):
        """Differentiable LM forward with `delta` ([B, D], may require grad)
        added persistently. Returns logits [B, T, V]."""
        with self.injecting(delta):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


def target_nll(logits, labels, attention_mask, target_mask):
    """Mean token NLL over positions where target_mask is 1 (shifted)."""
    torch = require_torch()
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    shift_mask = (target_mask[:, 1:] * attention_mask[:, 1:]).float()
    logp = torch.log_softmax(shift_logits, dim=-1)
    tok = logp.gather(-1, shift_labels.clamp(min=0)[..., None])[..., 0]
    return -(tok * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)
