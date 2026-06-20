from PIL import Image
import numpy as np

from modules import scripts_postprocessing, gfpgan_model, ui_components
from modules import headless_ui as gr


class ScriptPostprocessingGfpGan(scripts_postprocessing.ScriptPostprocessing):
    name = "GFPGAN"
    order = 2000

    def ui(self):
        with ui_components.InputAccordion(False, label="GFPGAN") as enable:
            gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Visibility", value=1.0, elem_id=self.elem_id_suffix("extras_gfpgan_visibility"))

        return {
            "enable": enable,
            "gfpgan_visibility": gfpgan_visibility,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable, gfpgan_visibility):
        if gfpgan_visibility == 0 or not enable:
            return

        restored_img = gfpgan_model.gfpgan_fix_faces(np.array(pp.image.convert("RGB"), dtype=np.uint8))
        res = Image.fromarray(restored_img)

        pp.image = scripts_postprocessing.blend_with_original(pp.image, res, gfpgan_visibility)
        pp.info["GFPGAN visibility"] = round(gfpgan_visibility, 3)
