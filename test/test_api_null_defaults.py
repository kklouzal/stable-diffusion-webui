import pytest
from pydantic import ValidationError


@pytest.fixture(scope="module")
def models(initialize):
    from modules.api import models as api_models
    return api_models


@pytest.mark.parametrize("model, payload", [
    ("StableDiffusionImg2ImgProcessingAPI", {"init_images": ["x"], "mask": None, "script_name": None, "infotext": None, "force_task_id": None}),
    ("StableDiffusionTxt2ImgProcessingAPI", {"script_name": None, "infotext": None, "force_task_id": None}),
])
def test_explicit_null_is_accepted_for_fields_that_default_to_none(models, model, payload):
    # Upstream clients (and test/test_img2img.py) send "mask": null; pydantic 2 answered 422.
    request = getattr(models, model).model_validate(payload)
    assert all(getattr(request, key) is None for key in payload if key != "init_images")


def test_non_null_values_are_still_validated(models):
    with pytest.raises(ValidationError):
        models.StableDiffusionImg2ImgProcessingAPI.model_validate({"init_images": ["x"], "mask": 5})
    assert models.StableDiffusionImg2ImgProcessingAPI.model_validate({"init_images": ["x"], "mask": "abc"}).mask == "abc"
