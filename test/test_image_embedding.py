import pytest
import torch
from PIL import Image, PngImagePlugin

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


def test_lcg_matches_reference_stream():
    # The pixel embedding XORs its payload with this stream, so any drift breaks every embedding image shared so far.
    generator = image_embedding.lcg()
    reference = [
        253, 242, 127, 44, 157, 27, 239, 133, 38, 79, 167, 4, 177, 95, 130, 79, 78, 14, 52, 215, 220, 194, 126, 28, 240,
        179, 160, 153, 149, 50, 105, 14, 21, 218, 199, 18, 54, 198, 193, 38, 128, 19, 53, 195, 124, 75, 205, 12, 6, 145,
        0, 28, 30, 148, 8, 45, 218, 171, 55, 249, 97, 166, 12, 35, 0, 41, 221, 122, 215, 170, 31, 113, 186, 97, 119, 31,
        23, 185, 66, 140, 30, 41, 37, 63, 137, 109, 216, 55, 159, 145, 82, 204, 86, 73, 222, 44, 198, 118, 240, 97,
    ]

    assert [next(generator) for _ in range(100)] == reference
    assert sum(next(generator) for _ in range(100000)) == 12731374


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_embedding_png_written_like_training_output_decodes_from_file(tmp_path):
    # Same steps as the "image with stored embedding" output of train_embedding: a captioned preview that carries the
    # embedding twice, as an sd-ti-embedding text chunk and in its pixel border.
    vectors = torch.randn(2, 768, generator=torch.Generator().manual_seed(1))
    data = {"string_to_token": {"*": 265}, "string_to_param": {"*": vectors}, "name": "probe", "step": 3}
    info = PngImagePlugin.PngInfo()
    info.add_text("sd-ti-embedding", image_embedding.embedding_to_b64(data))
    captioned = image_embedding.caption_image_overlay(Image.new("RGB", (256, 256), (90, 120, 200)), "<probe>", "model", "[0123456789]", "2v 3s")
    image_embedding.insert_image_data_embed(captioned, data).save(tmp_path / "probe.png", "PNG", pnginfo=info)

    with Image.open(tmp_path / "probe.png") as image:
        decoded_text = image_embedding.embedding_from_b64(image.text["sd-ti-embedding"])
        decoded_pixels = image_embedding.extract_image_data_embed(image)

    for decoded in (decoded_text, decoded_pixels):
        assert decoded["name"] == "probe"
        assert decoded["step"] == 3
        assert torch.equal(decoded["string_to_param"]["*"].float(), vectors)
