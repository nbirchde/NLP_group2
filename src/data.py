"""Data loading helpers for the Chef classification project."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

EXPECTED_COLUMNS: Sequence[str] = (
    "chef_id",
    "recipe_name",
    "date",
    "tags",
    "steps",
    "description",
    "ingredients",
    "n_ingredients",
)

LIST_LIKE_COLUMNS: Sequence[str] = ("tags", "steps", "ingredients")

FIELD_ALIASES = {
    "recipe_name": "",
    "ingredients": "Ingredients",
    "tags": "Tags",
    "description": "Description",
    "steps": "Steps",
    "n_ingredients": "NumIngredients",
}


@dataclass
class RecipeData:
    frame: pd.DataFrame


def _parse_list(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    try:
        value = ast.literal_eval(text)
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value]
    except (ValueError, SyntaxError):
        pass
    return [text.strip()]


def _maybe_parse_lists(frame: pd.DataFrame) -> pd.DataFrame:
    for column in LIST_LIKE_COLUMNS:
        if column not in frame.columns:
            continue
        frame[column] = frame[column].apply(_parse_list)
    return frame


def _normalize_columns(frame: pd.DataFrame, require_labels: bool = True) -> pd.DataFrame:
    columns = list(frame.columns)
    if "data" in columns and "date" not in columns:
        frame = frame.rename(columns={"data": "date"})
    
    # Check for missing columns (chef_id is optional for test data)
    required_cols = EXPECTED_COLUMNS if require_labels else [c for c in EXPECTED_COLUMNS if c != "chef_id"]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}")
    return frame


def load_recipes_csv(path: str | Path, require_labels: bool = True) -> RecipeData:
    """Load the raw recipes CSV, normalize schema, and parse list-like columns.
    
    Args:
        path: Path to CSV file
        require_labels: If False, chef_id column is optional (for test data)
    """
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, sep=";", engine="python")
    frame = _normalize_columns(frame, require_labels=require_labels)
    frame = _maybe_parse_lists(frame)
    return RecipeData(frame=frame.copy())


def format_section(prefix: str, lines: Iterable[str]) -> str:
    content = " ".join(line for line in lines if line)
    if not content:
        return ""
    return f"{prefix}: {content}" if prefix else content


def concat_text_fields(row: pd.Series, fields: Sequence[str]) -> str:
    """Concatenate selected text fields with agreed formatting."""
    parts: list[str] = []
    for field in fields:
        value = row.get(field, "")
        prefix = FIELD_ALIASES.get(field, "")
        if isinstance(value, list):
            formatted = format_section(prefix, value)
        else:
            text = str(value).strip() if pd.notna(value) else ""
            formatted = format_section(prefix, [text]) if text else ""
        if not formatted:
            continue
        parts.append(formatted)
    return "\n".join(parts)
