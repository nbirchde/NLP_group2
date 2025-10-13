#!/usr/bin/env python3
"""Interactive CLI playground for the fine-tuned chef classifier."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

try:
    import torch
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise SystemExit(
        "PyTorch is required for the chef CLI. Install torch in your environment first."
    ) from exc

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise SystemExit(
        "Transformers is required for the chef CLI. Install transformers in your environment first."
    ) from exc

try:  # Optional enhancement for interactive completeness
    import readline  # type: ignore
except ImportError:  # pragma: no cover - platform without readline
    readline = None

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainingConfig
from src.data import load_recipes_csv
from src.models import load_sequence_classification_model


CHEF_FEATURE_HIGHLIGHTS: Dict[str, str] = {
    "1533": (
        "Ontario-leaning appetizers and brunch bites—think lime-forward salsas, herby dips, "
        "smoked salmon toasts, and other citrusy entertaining snacks."
    ),
    "3288": (
        "Once-a-month cooking strategist focused on American comfort food—freezer casseroles, "
        "copycat crowd pleasers, black-bean skillets, and pressure-cooker batch meals."
    ),
    "4470": (
        "Scandinavian-accented entertainer serving gluten-free and lamb dishes beside grill-ready "
        "crowd pleasers seasoned simply with salt, pepper, and chili flakes."
    ),
    "5060": (
        "Pacific Northwest comfort with a diabetic-friendly spin—Canadian classics, honey-glazed "
        "bakes, light mayo swaps, and holiday brunches that stay kid-friendly."
    ),
    "6357": (
        "Spice cabinet wizard for homey Indian and broader Asian cooking—turmeric-heavy curries, "
        "ghee tempering, Ramadan snacks, and toddler-friendly deep-fried treats."
    ),
    "8688": (
        "Weekend host who toggles between bread-machine projects, espresso cocktails, coffee cakes, "
        "and cobblers stocked with coconut, blueberries, and golden sugar."
    ),
}

TEXT_FIELD_PREFIXES: Dict[str, str] = {
    "recipe_name": "",
    "ingredients": "Ingredients",
    "tags": "Tags",
    "description": "Description",
    "steps": "Steps",
    "n_ingredients": "NumIngredients",
}


@dataclass
class InferenceContext:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    device: torch.device
    config: TrainingConfig
    id2label: Dict[int, str]


@dataclass
class SuggestionCatalog:
    tags: List[str]
    ingredients: List[str]
    step_starters: List[str]
    description_stems: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play with the fine-tuned chef classifier interactively."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="experiments/distilbert_text_only/artifacts/final_model",
        help="Directory containing the fine-tuned model (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Training config to mirror preprocessing choices (default: %(default)s).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top chef candidates to display (default: %(default)s).",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/train.csv",
        help="Path to labelled training CSV for generating tag/ingredient suggestions.",
    )
    return parser.parse_args()


def auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_inference_context(model_path: Path, config_path: Path) -> InferenceContext:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")

    config = TrainingConfig.from_yaml(str(config_path))
    model = load_sequence_classification_model(
        model_name=str(model_path),
        classifier_activation=config.classifier_activation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = auto_device()
    model = model.to(device)
    model.eval()

    # Normalize id2label to int keys → str chef IDs
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    return InferenceContext(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=config,
        id2label=id2label,
    )


def load_suggestions(train_path: Path, top_tags: int = 200, top_ingredients: int = 200, top_steps: int = 80) -> SuggestionCatalog:
    data = load_recipes_csv(train_path).frame

    tag_counter: Counter[str] = Counter()
    ingredient_counter: Counter[str] = Counter()
    step_starters: Counter[str] = Counter()
    description_stems: Counter[str] = Counter()

    stem_token = 6
    for _, row in data.iterrows():
        tag_counter.update(tag.lower() for tag in row["tags"])
        ingredient_counter.update(ingredient.lower().strip() for ingredient in row["ingredients"])
        for step in row["steps"]:
            step_clean = step.strip()
            if not step_clean:
                continue
            start = " ".join(step_clean.split()[:3]).lower()
            step_starters[start] += 1
        description = row.get("description", "")
        if isinstance(description, str) and description.strip():
            stem = " ".join(description.strip().lower().split()[:stem_token])
            if stem:
                description_stems[stem] += 1

    def take(counter: Counter[str], limit: int) -> List[str]:
        return [item for item, _ in counter.most_common(limit)]

    return SuggestionCatalog(
        tags=take(tag_counter, top_tags),
        ingredients=take(ingredient_counter, top_ingredients),
        step_starters=take(step_starters, top_steps),
        description_stems=take(description_stems, top_steps),
    )


class SimpleCompleter:
    """Case-insensitive readline completer for a fixed list of phrases."""

    def __init__(self, options: Sequence[str]):
        unique = dict.fromkeys(options)  # preserve order
        self.options: List[str] = list(unique.keys())

    def __call__(self, text: str, state: int) -> str | None:
        text_lower = text.lower()
        matches = [opt for opt in self.options if opt.lower().startswith(text_lower)]
        if state < len(matches):
            return matches[state]
        return None


@contextmanager
def completion_context(options: Sequence[str] | None, delimiters: str) -> Iterator[None]:
    if readline is None or not options:
        yield
        return

    completer = SimpleCompleter(options)
    old_completer = readline.get_completer()
    old_delims = readline.get_completer_delims()
    readline.set_completer(completer)
    readline.set_completer_delims(delimiters)
    binding = "tab: complete"
    doc = getattr(readline, "__doc__", "") or ""
    if "libedit" in doc.lower():
        binding = "bind ^I rl_complete"
    readline.parse_and_bind(binding)
    try:
        yield
    finally:
        readline.set_completer(old_completer)
        readline.set_completer_delims(old_delims)


def prompt_multiline(header: str, suggestions: Sequence[str] | None = None, delimiters: str = "\n") -> List[str]:
    print(header)
    print("  Enter one item per line. Leave blank to finish.")
    lines: List[str] = []
    while True:
        with completion_context(suggestions, delimiters):
            try:
                line = input("> ").strip()
            except EOFError:
                print()
                break
        if not line:
            break
        lines.append(line)
    return lines


def prompt_text(prompt: str, allow_empty: bool = False) -> str:
    while True:
        try:
            value = input(prompt).strip()
        except EOFError:
            print()
            return ""
        if value or allow_empty:
            return value
        print("Please enter a value.")


def prompt_tags(suggestions: SuggestionCatalog) -> List[str]:
    helper = suggestions.tags[:8]
    if helper:
        print(f"  Popular tags: {', '.join(helper)}")
    with completion_context(suggestions.tags, ","):
        text = prompt_text("Tags (comma separated, optional): ", allow_empty=True)
    tags = [tag.strip() for tag in text.split(",") if tag.strip()]
    return tags


def prompt_recipe(suggestions: SuggestionCatalog) -> dict[str, object] | None:
    print("\n--- Describe your recipe ---")
    name = prompt_text("Recipe name (or 'quit' to exit): ")
    if not name:
        return None
    if name.lower() in {"quit", "exit"}:
        return None

    if suggestions.description_stems:
        stems = [f"\"{stem}...\"" for stem in suggestions.description_stems[:2]]
        print("  Description inspiration:", ", ".join(stems))
    description = prompt_text("Short description (optional): ", allow_empty=True)

    if suggestions.ingredients:
        print(f"  Ingredient ideas: {', '.join(suggestions.ingredients[:6])}")
    ingredients = prompt_multiline(
        "Ingredients",
        suggestions=suggestions.ingredients,
        delimiters=",\n",
    )

    tags = prompt_tags(suggestions)

    if suggestions.step_starters:
        print(f"  Step openers: {', '.join(suggestions.step_starters[:5])}")
    steps = prompt_multiline(
        "Steps / directions (optional)",
        suggestions=suggestions.step_starters,
        delimiters="\n",
    )

    recipe = {
        "recipe_name": name,
        "description": description,
        "ingredients": ingredients,
        "tags": tags,
        "steps": steps,
        "n_ingredients": len(ingredients),
    }
    return recipe


def build_text(recipe: dict[str, object], text_fields: Iterable[str]) -> str:
    sections: List[str] = []
    for field in text_fields:
        value = recipe.get(field, "")
        prefix = TEXT_FIELD_PREFIXES.get(field, "")
        formatted = format_field(prefix, value)
        if formatted:
            sections.append(formatted)
    return "\n".join(sections)


def format_field(prefix: str, value: object) -> str:
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = str(value).strip() if value is not None else ""
        items = [text] if text else []

    if not items:
        return ""
    content = " ".join(items)
    return f"{prefix}: {content}" if prefix else content


def predict_recipe(context: InferenceContext, text: str) -> Dict[str, object]:
    tokenizer_inputs = context.tokenizer(
        text,
        padding="longest",
        truncation=True,
        max_length=context.config.max_length,
        return_tensors="pt",
    )
    tokenizer_inputs = {k: v.to(context.device) for k, v in tokenizer_inputs.items()}

    with torch.no_grad():
        outputs = context.model(**tokenizer_inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1).squeeze(0)
    return {
        "logits": logits.squeeze(0).cpu(),
        "probabilities": probabilities.cpu(),
    }


def format_top_candidates(
    probs: torch.Tensor,
    id2label: Dict[int, str],
    top_k: int,
) -> List[tuple[str, float]]:
    k = min(top_k, probs.numel())
    top_prob, top_idx = torch.topk(probs, k)
    return [(id2label[idx.item()], prob.item()) for prob, idx in zip(top_prob, top_idx)]


def chef_highlight(chef_id: str) -> str:
    return CHEF_FEATURE_HIGHLIGHTS.get(
        chef_id,
        "No highlight available for this chef yet.",
    )


def main() -> None:
    args = parse_args()
    suggestions = load_suggestions(Path(args.training_data).resolve())
    context = load_inference_context(
        model_path=Path(args.model_path).resolve(),
        config_path=Path(args.config).resolve(),
    )
    print("Chef classifier playground loaded!")
    print(f"Model path: {args.model_path}")
    print(f"Using device: {context.device}")
    if suggestions.tags:
        print(f"Loaded {len(suggestions.tags)} tag suggestions, {len(suggestions.ingredients)} ingredient suggestions.")
    print("Type 'quit' at the recipe name prompt to exit.")

    while True:
        recipe = prompt_recipe(suggestions)
        if recipe is None:
            print("Goodbye!")
            break

        text = build_text(recipe, context.config.text_fields)
        results = predict_recipe(context, text)
        probabilities: torch.Tensor = results["probabilities"]
        candidates = format_top_candidates(probabilities, context.id2label, args.top_k)
        top_chef, top_prob = candidates[0]

        print("\n=== Prediction ===")
        print(f"Top chef: {top_chef} (confidence {top_prob:.2%})")
        print(f"Chef highlight: {chef_highlight(top_chef)}")
        if len(candidates) > 1:
            print("\nOther candidates:")
            for chef_id, prob in candidates[1:]:
                print(f"  - {chef_id}: {prob:.2%}")

        print("\nTry another recipe!")


if __name__ == "__main__":
    main()
