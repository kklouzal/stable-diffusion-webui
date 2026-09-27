import pytest
import torch
from PIL import Image

from modules.textual_inversion import image_embedding


@pytest.mark.filterwarnings("error::DeprecationWarning")  # Pillow 12 deprecates Image.getdata
def test_embedding_survives_png_embed_round_trip():
    generator = torch.Generator().manual_seed(0)
    vectors = torch.randn(2, 768, generator=generator)
    data = {"string_to_token": {"*": 265}, "string_to_param": {"*": vectors}, "name": "probe", "step": 7}

    embedded = image_embedding.insert_image_data_embed(Image.new("RGB", (512, 512), (90, 120, 200)), data)
    decoded = image_embedding.extract_image_data_embed(embedded)

    assert decoded["name"] == "probe"
    assert decoded["step"] == 7
    assert decoded["string_to_token"] == {"*": 265}
    # The JSON payload carries float32 values as Python floats, which convert back exactly.
    assert torch.equal(decoded["string_to_param"]["*"].float(), vectors)
