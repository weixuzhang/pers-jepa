from __future__ import annotations

from pathlib import Path
from typing import Any

from persjepa.hidden import append_pred_tokens, require_torch, require_transformers, resolve_device, resolve_dtype
from persjepa.models import (
    BoundedOrthogonalDeltaPredictor,
    BlendDeltaPredictor,
    ClusterMeanDeltaPredictor,
    DirectionalScaleDeltaPredictor,
    JEPASAE,
    LinearPredictor,
    MeanDeltaPredictor,
    OffsetDeltaPredictor,
    RoutedJEPASAE,
    RoutedMeanDelta,
    StandardSAE,
)


def load_causal_lm(
    model_name: str,
    *,
    pred_token: str = "[PRED]",
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
):
    torch = require_torch()
    AutoModelForCausalLM, AutoTokenizer = require_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens({"additional_special_tokens": [pred_token]})
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    torch_dtype = resolve_dtype(dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if added:
        model.resize_token_embeddings(len(tokenizer))
    torch_device = resolve_device(device)
    model.to(torch_device)
    model.eval()
    return model, tokenizer, torch_device


def load_jepa_sae(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model = JEPASAE(
        input_dim=int(payload["input_dim"]),
        latent_dim=int(payload["latent_dim"]),
        top_k=int(payload["top_k"]),
        decoder_bias=payload["state_dict"].get("delta_decoder.bias") is not None,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def load_routed_jepa_sae(path: str | Path, *, device):
    """Load a RoutedJEPASAE plus its RoutingTable (user -> group weights)."""
    torch = require_torch()
    from persjepa.routing import RoutingTable
    payload = torch.load(path, map_location=device)
    model = RoutedJEPASAE(
        int(payload["input_dim"]), int(payload["num_groups"]), latent_dim=int(payload["latent_dim"]),
        top_k=int(payload["top_k"]), decoder_bias=bool(payload.get("decoder_bias", True)),
        group_names=payload.get("group_names"),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    model.routing_table = RoutingTable.from_payload(payload["routing_table"]) if "routing_table" in payload else None
    return model, payload


def load_routed_mean_delta(path: str | Path, *, device):
    torch = require_torch()
    from persjepa.routing import RoutingTable
    payload = torch.load(path, map_location=device)
    model = RoutedMeanDelta(payload["group_means"], group_names=payload.get("group_names"))
    model.to(device); model.eval()
    model.routing_table = RoutingTable.from_payload(payload["routing_table"]) if "routing_table" in payload else None
    return model, payload


def load_standard_sae(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model = StandardSAE(
        input_dim=int(payload["input_dim"]),
        latent_dim=int(payload["latent_dim"]),
        top_k=int(payload["top_k"]),
        decoder_bias=payload["state_dict"].get("decoder.bias") is not None,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def load_linear_predictor(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    state_dict = payload["state_dict"]
    model = LinearPredictor(
        input_dim=int(payload["input_dim"]),
        bias=state_dict.get("proj.bias") is not None,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, payload


def load_mean_delta(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model = MeanDeltaPredictor(payload["mean_delta"])
    model.to(device)
    model.eval()
    return model, payload


def load_offset_delta(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    base_model_type = payload.get("base_model_type", "jepa_sae")
    if base_model_type != "jepa_sae":
        raise ValueError(f"Unsupported offset base_model_type: {base_model_type}")
    learned_model = JEPASAE(
        input_dim=int(payload["input_dim"]),
        latent_dim=int(payload["latent_dim"]),
        top_k=int(payload["top_k"]),
        decoder_bias=payload["state_dict"].get("delta_decoder.bias") is not None,
    )
    learned_model.load_state_dict(payload["state_dict"])
    base_delta = payload.get("base_delta", payload.get("mean_delta"))
    if base_delta is None:
        raise ValueError("Offset checkpoint must contain base_delta or mean_delta.")
    model = OffsetDeltaPredictor(base_delta=base_delta, learned_model=learned_model)
    model.to(device)
    model.eval()
    return model, payload


def load_cluster_mean_delta(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model = ClusterMeanDeltaPredictor(
        centroids=payload["centroids"],
        cluster_deltas=payload["cluster_deltas"],
    )
    model.to(device)
    model.eval()
    return model, payload


def load_blend_delta(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    base_model_type = payload.get("base_model_type", "jepa_sae")
    if base_model_type != "jepa_sae":
        raise ValueError(f"Unsupported blend base_model_type: {base_model_type}")
    learned_model = JEPASAE(
        input_dim=int(payload["input_dim"]),
        latent_dim=int(payload["latent_dim"]),
        top_k=int(payload["top_k"]),
        decoder_bias=payload["state_dict"].get("delta_decoder.bias") is not None,
    )
    learned_model.load_state_dict(payload["state_dict"])
    model = BlendDeltaPredictor(
        mean_delta=payload["mean_delta"],
        learned_model=learned_model,
        learned_delta_weight=float(payload["learned_delta_weight"]),
    )
    model.to(device)
    model.eval()
    return model, payload


def load_directional_scale_delta(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model = DirectionalScaleDeltaPredictor(
        direction=payload["direction"],
        max_scale=float(payload.get("max_scale", 4.0)),
        init_scale=float(payload.get("init_scale", 1.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def load_bounded_orthogonal_delta(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model = BoundedOrthogonalDeltaPredictor(
        base_delta=payload["base_delta"],
        hidden_dim=int(payload.get("hidden_dim", 256)),
        correction_scale=float(payload.get("correction_scale", 0.25)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def load_steering_sae(path: str | Path, *, device):
    torch = require_torch()
    payload = torch.load(path, map_location=device)
    model_type = payload.get("model_type")
    state_dict = payload.get("state_dict", {})
    if model_type is None and "proj.weight" in state_dict:
        model_type = "linear_predictor"
    if model_type is None and "mean_delta" in payload:
        model_type = "mean_delta"
    if model_type is None:
        model_type = "jepa_sae"
    if model_type == "standard_sae":
        return load_standard_sae(path, device=device)
    if model_type == "jepa_sae":
        return load_jepa_sae(path, device=device)
    if model_type == "routed_jepa_sae":
        return load_routed_jepa_sae(path, device=device)
    if model_type == "routed_mean_delta":
        return load_routed_mean_delta(path, device=device)
    if model_type in {"linear_predictor", "stp_linear_predictor"} or "proj.weight" in state_dict:
        return load_linear_predictor(path, device=device)
    if model_type == "blend_delta":
        return load_blend_delta(path, device=device)
    if model_type == "offset_delta":
        return load_offset_delta(path, device=device)
    if model_type == "directional_scale_delta":
        return load_directional_scale_delta(path, device=device)
    if model_type == "bounded_orthogonal_delta":
        return load_bounded_orthogonal_delta(path, device=device)
    if model_type == "cluster_mean_delta":
        return load_cluster_mean_delta(path, device=device)
    if model_type == "mean_delta" or "mean_delta" in payload:
        return load_mean_delta(path, device=device)
    raise ValueError(f"Unsupported steering checkpoint model_type: {model_type}")


def _resolve_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def decoder_layers(model):
    """The decoder block list of a causal LM, including composite
    (vision-language) wrappers whose text stack sits under `language_model`."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "model.text_model.layers", "transformer.h", "gpt_neox.layers"):
        layers = _resolve_path(model, path)
        if layers is not None:
            return layers
    raise AttributeError("Could not locate decoder layers on this model.")


def decoder_layer_module(model, layer: int):
    return decoder_layers(model)[layer]


def final_norm_module(model):
    for path in ("model.norm", "model.language_model.norm", "language_model.model.norm",
                 "model.text_model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        norm = _resolve_path(model, path)
        if norm is not None:
            return norm
    raise AttributeError("Could not locate final norm on this model.")


class ResidualSteerer:
    def __init__(
        self,
        model,
        tokenizer,
        jepa_sae: JEPASAE | StandardSAE,
        *,
        pred_token: str = "[PRED]",
        predictor_tokens: int = 1,
        layer: int = -1,
        max_length: int = 1024,
        residual_scale: float = 1.0,
        device=None,
    ) -> None:
        if predictor_tokens < 0:
            raise ValueError("predictor_tokens must be >= 0 for JEPA steering.")
        self.model = model
        self.tokenizer = tokenizer
        self.jepa_sae = jepa_sae
        self.pred_token = pred_token
        self.predictor_tokens = predictor_tokens
        self.layer = layer
        self.max_length = max_length
        self.residual_scale = residual_scale
        self.torch = require_torch()
        self.device = device or next(model.parameters()).device
        self.pred_token_id = tokenizer.convert_tokens_to_ids(pred_token)

    def tokenize(self, text: str):
        original_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"
        try:
            return self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
        finally:
            self.tokenizer.truncation_side = original_side

    def pred_positions(self, input_ids):
        is_pred = input_ids.eq(self.pred_token_id)
        if not self.torch.all(is_pred.any(dim=1)):
            raise ValueError("Prompt does not contain [PRED] tokens.")
        idx = self.torch.arange(input_ids.shape[1], device=input_ids.device)
        return idx.masked_fill(~is_pred, -1).max(dim=1).values

    def last_nonpad_positions(self, attention_mask):
        return attention_mask.long().sum(dim=1) - 1

    def compute_delta(self, prompt_with_pred: str):
        torch = self.torch
        inputs = self.tokenize(prompt_with_pred)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[self.layer]
            if self.predictor_tokens == 0:
                positions = self.last_nonpad_positions(inputs["attention_mask"])
            else:
                positions = self.pred_positions(inputs["input_ids"])
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            h_gen = hidden[batch, positions]
            dtype_source = next(self.jepa_sae.parameters(), None)
            if dtype_source is None:
                dtype_source = next(self.jepa_sae.buffers())
            sae_dtype = dtype_source.dtype
            h_model = h_gen.to(dtype=sae_dtype)
            out = self.jepa_sae(h_model)
            if hasattr(out, "delta") and out.delta is not None:
                delta = out.delta
            elif hasattr(out, "reconstruction"):
                delta = out.reconstruction - h_model
            else:
                delta = out - h_model
            delta = delta * self.residual_scale
        return delta, positions, inputs

    def _hook_module(self):
        if self.layer == -1:
            return final_norm_module(self.model)
        if self.layer <= 0:
            raise ValueError(
                "Non-final steering layers use Hugging Face hidden-state indices; "
                "layer=1 patches decoder block 0 output. Use layer=-1 for final norm."
            )
        return decoder_layer_module(self.model, self.layer - 1)

    def generate(
        self,
        generic_prompt: str,
        *,
        max_new_tokens: int = 128,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> str:
        torch = self.torch
        generation_kwargs = generation_kwargs or {}
        prompt_with_pred = append_pred_tokens(
            generic_prompt,
            self.pred_token,
            self.predictor_tokens,
        )
        delta, positions, inputs = self.compute_delta(prompt_with_pred)
        applied = {"value": False}

        def add_delta(hidden):
            if applied["value"] or hidden.ndim != 3 or hidden.shape[1] <= int(positions.max()):
                return hidden
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            patched = hidden.clone()
            patched[batch, positions] = patched[batch, positions] + delta.to(hidden.dtype)
            applied["value"] = True
            return patched

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                first = add_delta(output[0])
                return (first, *output[1:])
            return add_delta(output)

        handle = self._hook_module().register_forward_hook(hook)
        try:
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs,
                )
        finally:
            handle.remove()
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_text(model, tokenizer, prompt: str, *, device, max_new_tokens: int, generation_kwargs=None) -> str:
    generation_kwargs = generation_kwargs or {}
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    torch = require_torch()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()
