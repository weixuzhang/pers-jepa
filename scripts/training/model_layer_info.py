#!/usr/bin/env python
"""Print layer count and suggested middle-layer hidden-state indices for a model
(no weights loaded). Usage: model_layer_info.py Qwen/Qwen3.5-4B [0.3 0.5 0.7]"""
import sys
from transformers import AutoConfig
name = sys.argv[1]; fracs = [float(x) for x in sys.argv[2:]] or [0.3, 0.5, 0.7]
cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
n = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0))
print(f"{name}: num_hidden_layers={n} hidden_size={getattr(cfg,'hidden_size',None)} " + " ".join(f"{f:.0%}->{max(1, round(n*f))}" for f in fracs))
