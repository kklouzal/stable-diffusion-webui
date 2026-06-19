from pathlib import Path


def test_image_mask_resize_centers_letterboxed_canvas_wrapper():
    source = Path("javascript/imageMaskFix.js").read_text()

    assert "wrapper.style.left = `${(w - wW) / 2}px`;" in source
    assert "wrapper.style.top = `${(h - wH) / 2}px`;" in source
    assert "wrapper.style.left = `0px`;" not in source
    assert "wrapper.style.top = `0px`;" not in source
