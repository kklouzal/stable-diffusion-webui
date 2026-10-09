from modules import headless_ui as gr

from modules import scripts
from modules.infotext_utils import PasteField
from modules.shared import cmd_opts


class ScriptSeed(scripts.ScriptBuiltinUI):
    section = "seed"
    create_group = False

    def __init__(self):
        self.seed = None

    def title(self):
        return "Seed"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Row(elem_id=self.elem_id("seed_row")):
            if cmd_opts.use_textbox_seed:
                self.seed = gr.Textbox(label='Seed', value="", elem_id=self.elem_id("seed"), min_width=100)
            else:
                self.seed = gr.Number(label='Seed', value=-1, elem_id=self.elem_id("seed"), min_width=100, precision=0)

            seed_checkbox = gr.Checkbox(label='Extra', elem_id=self.elem_id("subseed_show"), value=False)

        with gr.Group(visible=False, elem_id=self.elem_id("seed_extras")):
            with gr.Row(elem_id=self.elem_id("subseed_row")):
                subseed = gr.Number(label='Variation seed', value=-1, elem_id=self.elem_id("subseed"), precision=0)
                subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01, elem_id=self.elem_id("subseed_strength"))

            with gr.Row(elem_id=self.elem_id("seed_resize_from_row")):
                seed_resize_from_w = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from width", value=0, elem_id=self.elem_id("seed_resize_from_w"))
                seed_resize_from_h = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize seed from height", value=0, elem_id=self.elem_id("seed_resize_from_h"))

        self.infotext_fields = [
            PasteField(self.seed, "Seed", api="seed"),
            PasteField(seed_checkbox, lambda d: "Variation seed" in d or "Seed resize from-1" in d),
            PasteField(subseed, "Variation seed", api="subseed"),
            PasteField(subseed_strength, "Variation seed strength", api="subseed_strength"),
            PasteField(seed_resize_from_w, "Seed resize from-1", api="seed_resize_from_w"),
            PasteField(seed_resize_from_h, "Seed resize from-2", api="seed_resize_from_h"),
        ]

        return self.seed, seed_checkbox, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h

    def setup(self, p, seed, seed_checkbox, subseed, subseed_strength, seed_resize_from_w, seed_resize_from_h):
        p.seed = seed

        if seed_checkbox and subseed_strength > 0:
            p.subseed = subseed
            p.subseed_strength = subseed_strength

        if seed_checkbox and seed_resize_from_w > 0 and seed_resize_from_h > 0:
            p.seed_resize_from_w = seed_resize_from_w
            p.seed_resize_from_h = seed_resize_from_h
