from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.mark.usefixtures("initialize")
@pytest.mark.parametrize("restorer_name", ["gfpgan", "codeformer"])
def test_face_restorers(restorer_name, tmp_path):
    from modules import shared

    if restorer_name == "gfpgan":
        from modules import gfpgan_model
        gfpgan_model.setup_model(shared.cmd_opts.gfpgan_models_path)
        restorer = gfpgan_model.gfpgan_fix_faces
    elif restorer_name == "codeformer":
        from modules import codeformer_model
        codeformer_model.setup_model(shared.cmd_opts.codeformer_models_path)
        restorer = codeformer_model.codeformer.restore
    else:
        raise NotImplementedError("...")
    img = Image.open(Path(__file__).parent / "test_files" / "two-faces.jpg")
    np_img = np.array(img, dtype=np.uint8)
    fixed_image = restorer(np_img)
    assert fixed_image.shape == np_img.shape
    assert not np.allclose(fixed_image, np_img)  # should have visibly changed
    Image.fromarray(fixed_image).save(tmp_path / f"{restorer_name}.png")
