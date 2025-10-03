"""Dataset utilities for DistilBERT fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from .data import RecipeData, concat_text_fields


@dataclass
class DatasetArtifacts:
    dataset: DatasetDict
    label2id: Dict[str, int]
    id2label: Dict[int, str]


def add_text_column(frame: pd.DataFrame, text_fields: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    output["text"] = output.apply(lambda row: concat_text_fields(row, text_fields), axis=1)
    return output


def encode_labels(frame: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    output = frame.copy()
    output[label_column] = output[label_column].astype(str)
    unique_labels = sorted(output[label_column].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    output["labels"] = output[label_column].map(label2id).astype(int)
    return output, label2id, id2label


def stratified_split(
    frame: pd.DataFrame,
    label_column: str,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        frame,
        test_size=val_ratio,
        random_state=seed,
        stratify=frame[label_column],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def prepare_dataset(  # noqa: D401
    data: RecipeData,
    text_fields: Sequence[str],
    label_column: str,
    val_ratio: float,
    seed: int,
) -> DatasetArtifacts:
    """Create Hugging Face datasets with text and encoded labels."""
    text_df = add_text_column(data.frame, text_fields)
    encoded_df, label2id, id2label = encode_labels(text_df, label_column)
    train_df, val_df = stratified_split(encoded_df, label_column, val_ratio, seed)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)

    ds = DatasetDict({"train": train_dataset, "validation": val_dataset})
    return DatasetArtifacts(dataset=ds, label2id=label2id, id2label=id2label)
