from modules.styles import PromptStyle, extract_original_prompts


def test_extract_original_prompts_accepts_negative_only_style():
    style = PromptStyle("negative-only", "", "low quality")

    matched, prompt, negative_prompt = extract_original_prompts(
        style, "a landscape", "blurry, low quality"
    )

    assert matched
    assert prompt == "a landscape"
    assert negative_prompt == "blurry"

