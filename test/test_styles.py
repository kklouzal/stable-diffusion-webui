from modules.styles import PromptStyle, StyleDatabase, extract_original_prompts


def test_extract_original_prompts_accepts_negative_only_style():
    style = PromptStyle("negative-only", "", "low quality")

    matched, prompt, negative_prompt = extract_original_prompts(
        style, "a landscape", "blurry, low quality"
    )

    assert matched
    assert prompt == "a landscape"
    assert negative_prompt == "blurry"



def test_wildcard_styles_path_loads_every_match_under_file_dividers(tmp_path):
    (tmp_path / "a.csv").write_text("name,prompt,negative_prompt\nwarm,warm light,cold\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("name,prompt,negative_prompt\nsharp,sharp focus,\n", encoding="utf-8")

    db = StyleDatabase([str(tmp_path / "*.csv")])

    divider_a, divider_b = " A ".center(40, "-"), " B ".center(40, "-")
    assert set(db.styles) == {divider_a, "warm", divider_b, "sharp"}
    assert db.styles["warm"] == PromptStyle("warm", "warm light", "cold", str(tmp_path / "a.csv"))
    assert db.apply_styles_to_prompt("a cat", ["warm", "sharp"]) == "a cat, warm light, sharp focus"
    assert db.apply_negative_styles_to_prompt("blurry", ["warm"]) == "blurry, cold"


def test_wildcard_styles_path_without_a_match_defaults_to_styles_csv(tmp_path):
    db = StyleDatabase([str(tmp_path / "*.csv")])

    assert db.paths[0] == tmp_path / "styles.csv"
    assert db.styles == {}
