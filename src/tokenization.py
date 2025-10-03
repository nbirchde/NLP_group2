"""Tokenization utilities leveraging Hugging Face Transformers."""

from __future__ import annotations

from functools import partial
from typing import Iterable

from datasets import DatasetDict
from transformers import AutoTokenizer


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.model_max_length and tokenizer.model_max_length < 512:
        tokenizer.model_max_length = 512
    return tokenizer


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int,
    padding_strategy: str,
    truncation_strategy: str,
) -> DatasetDict:
    padding = padding_strategy if padding_strategy in {"max_length", "longest"} else "longest"

    def tokenize_batch(batch: dict[str, Iterable[str]]):
        return tokenizer(
            batch["text"],
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=False,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    return tokenized
