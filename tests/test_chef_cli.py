import pandas as pd

from scripts import chef_cli


def test_format_field_handles_string_and_list():
    assert chef_cli.format_field("Desc", " tasty  ") == "Desc: tasty"
    assert chef_cli.format_field("", "   ") == ""
    assert chef_cli.format_field("Ingredients", [" egg ", ""]) == "Ingredients: egg"


def test_build_text_respects_field_order():
    recipe = {
        "recipe_name": "Summer Salad",
        "ingredients": ["tomato", "basil"],
        "tags": ["quick"],
        "description": "Fresh and easy.",
        "steps": ["Chop veggies", "Serve immediately"],
        "n_ingredients": 2,
    }
    text = chef_cli.build_text(
        recipe,
        ["recipe_name", "ingredients", "tags", "description", "steps"],
    )
    assert text.split("\n")[0] == "Summer Salad"
    assert "Ingredients: tomato basil" in text
    assert text.endswith("Steps: Chop veggies Serve immediately")


def test_simple_completer_matches_case_insensitively():
    completer = chef_cli.SimpleCompleter(["Apple Pie", "apple butter"])
    assert completer("ap", 0) == "Apple Pie"
    assert completer("ap", 1) == "apple butter"
    assert completer("ap", 2) is None


def test_load_suggestions_uses_dataset(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        [
            {
                "tags": ["quick", "breakfast"],
                "ingredients": ["banana", "nutmeg"],
                "steps": ["Mash banana", "stir in spices"],
                "description": "Quick breakfast smoothie",
            },
            {
                "tags": ["dinner"],
                "ingredients": ["garlic", "pasta"],
                "steps": ["Boil pasta", "toss with garlic"],
                "description": "Hearty dinner pasta",
            },
        ]
    )

    class Dummy:
        def __init__(self, frame):
            self.frame = frame

    def fake_loader(path):
        return Dummy(frame)

    monkeypatch.setattr(chef_cli, "load_recipes_csv", fake_loader)

    catalog = chef_cli.load_suggestions(tmp_path / "fake.csv", top_tags=2, top_ingredients=2, top_steps=2)
    assert catalog.tags[0] == "quick"
    assert "banana" in catalog.ingredients
    assert any(starter.startswith("mash") for starter in catalog.step_starters)
    assert any(stem.startswith("quick") for stem in catalog.description_stems)
