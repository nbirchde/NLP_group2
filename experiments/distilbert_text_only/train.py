"""Entry point for fine-tuning DistilBERT on the chef classification dataset."""

from __future__ import annotations

from pathlib import Path

from src.config import TrainingConfig


def main() -> None:
    config = TrainingConfig.from_yaml(Path("configs/base.yaml"))
    # TODO: wire up data loading, model training, and evaluation
    print(f"Loaded config for model: {config.model_name}")


if __name__ == "__main__":
    main()
