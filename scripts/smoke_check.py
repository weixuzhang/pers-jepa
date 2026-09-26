#!/usr/bin/env python
from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def import_with_version(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - CLI-only failure path
        return False, f"{exc.__class__.__name__}: {exc}"
    return True, getattr(module, "__version__", "n/a")


def main() -> None:
    modules = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "safetensors",
        "yaml",
        "openai",
        "persjepa",
    ]
    failed = False

    print(f"python\t{sys.executable}")
    print(f"repo\t{ROOT}")

    for module_name in modules:
        ok, detail = import_with_version(module_name)
        status = "OK" if ok else "FAIL"
        print(f"{module_name}\t{status}\t{detail}")
        failed = failed or (not ok)

    from persjepa.config import ExperimentConfig
    from persjepa.hidden import append_pred_tokens

    cfg = ExperimentConfig.from_file(ROOT / "configs/default.yaml")
    print(f"config\tOK\tmodel={cfg.model.name} layer={cfg.model.layer} k={cfg.model.predictor_tokens}")
    print(f"pred_token_demo\tOK\t{append_pred_tokens('demo prompt', cfg.model.pred_token, 3)}")

    torch = importlib.import_module("torch")
    print(f"cuda_available\t{torch.cuda.is_available()}")
    print(f"cuda_version\t{getattr(torch.version, 'cuda', 'n/a')}")
    if torch.cuda.is_available():
        print(f"gpu_count\t{torch.cuda.device_count()}")
        print(f"gpu_0\t{torch.cuda.get_device_name(0)}")

    required_scripts = [
        "scripts/training/extract_hidden.py",
        "scripts/training/train_saes.py",
        "scripts/eval/evaluate.py",
        "scripts/training/extract_span_hidden.py",
        "scripts/ablations/run_span_stp_ablation.py",
        "scripts/eval/judge_generation.py",
    ]
    for relative_path in required_scripts:
        exists = (ROOT / relative_path).exists()
        print(f"{relative_path}\t{'OK' if exists else 'MISSING'}")
        failed = failed or (not exists)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
