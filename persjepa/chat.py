"""Chat-template formatting for instruction-tuned models.

Both paired views are rendered as a single user turn followed by the
assistant generation prompt, so the `[PRED]` anchor (appended after the
rendered prompt) and the answer span (appended for teacher forcing) both sit
at the start of the assistant turn — exactly where generation begins. For
Qwen3/Qwen3.5 the thinking mode is disabled so no reasoning block precedes
the answer. Base models are left untouched (mode "auto" decides by name).
"""
from __future__ import annotations

import re

INSTRUCT_PATTERNS = (r"instruct", r"-it\b", r"chat", r"assistant")
QWEN3_LIKE = r"qwen3(\.\d+)?-"          # Qwen3-*, Qwen3.5-* post-trained (non -Base) checkpoints are chat models


def wants_chat_template(model_name: str, tokenizer, mode: str = "auto") -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    name = model_name.lower()
    if getattr(tokenizer, "chat_template", None) is None:
        return False
    if name.endswith("-base") or "-base-" in name:
        return False
    if any(re.search(p, name) for p in INSTRUCT_PATTERNS):
        return True
    if re.search(QWEN3_LIKE, name):
        return True
    return False


def render_chat_prompt(tokenizer, user_content: str, *, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_content}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def apply_chat_formatting(examples, tokenizer, model_name: str, mode: str = "auto", system: str | None = None) -> bool:
    """Rewrite generic_prompt / personalized_prompt of PersonaExample objects in
    place. Returns whether formatting was applied."""
    if not wants_chat_template(model_name, tokenizer, mode):
        return False
    for ex in examples:
        ex.generic_prompt = render_chat_prompt(tokenizer, ex.generic_prompt, system=system)
        ex.personalized_prompt = render_chat_prompt(tokenizer, ex.personalized_prompt, system=system)
    return True


def middle_layer(model, fraction: float = 0.5) -> int:
    """Hidden-state index at `fraction` of depth (hidden_states[0] is the
    embedding output, so decoder block i writes hidden_states[i+1])."""
    cfg = model.config
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    n = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0))
    return max(1, round(n * fraction))
