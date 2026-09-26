from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B"
    layer: int = -1
    pred_token: str = "[PRED]"
    predictor_tokens: int = 1
    max_length: int = 1024
    dtype: str = "auto"
    device: str = "auto"


@dataclass
class SAEConfig:
    input_dim: int | None = None
    latent_dim: int = 128
    top_k: int = 16
    l1_coeff: float = 1e-4
    decoder_bias: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 50
    vanilla_lr: float = 1e-3
    jepa_lr: float = 3e-4
    weight_decay: float = 0.0
    objective: str = "jepa_direct"
    lambda_coeff: float = 1.0
    stp_coeff: float = 0.25
    seed: int = 42
    grad_clip_norm: float = 1.0
    eval_fraction: float = 0.05


@dataclass
class STPConfig:
    enabled: bool = False
    num_points: int = 4
    detach_target: bool = True


@dataclass
class DataConfig:
    profile_template: str = "{profile}\n\nTask: {prompt}"
    text_field: str = "prompt"
    profile_field: str = "profile"
    target_field: str = "target"
    id_field: str = "id"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stp: STPConfig = field(default_factory=STPConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("Install pyyaml to load YAML configs.") from exc
            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            model=ModelConfig(**payload.get("model", {})),
            sae=SAEConfig(**payload.get("sae", {})),
            training=TrainingConfig(**payload.get("training", {})),
            stp=STPConfig(**payload.get("stp", {})),
            data=DataConfig(**payload.get("data", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
