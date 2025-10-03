from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class TrainingConfig:
    model_name: str
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    warmup_ratio: float
    max_grad_norm: float
    seed: int
    val_ratio: float
    label_column: str
    text_fields: List[str]
    padding_strategy: str
    truncation_strategy: str
    metric_primary: str
    metric_secondary: str
    patience: int
    save_total_limit: int
    output_dir: str
    logging_steps: int

    @staticmethod
    def from_yaml(path: Path | str) -> "TrainingConfig":
        data = yaml.safe_load(Path(path).read_text())
        return TrainingConfig(**data)
