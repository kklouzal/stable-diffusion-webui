from pathlib import Path


def test_returned_mask_images_keep_infotexts_aligned():
    source = Path("modules/processing.py").read_text()

    for append_call in (
        "output_images.append(image_mask)",
        "output_images.append(image_mask_composite)",
    ):
        append_at = source.index(append_call)
        branch_at = source.rfind("if opts.return_", 0, append_at)
        branch_body = source[branch_at:append_at]

        assert "infotexts.append(text)" in branch_body
