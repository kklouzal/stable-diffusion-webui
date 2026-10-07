from types import SimpleNamespace

from PIL import Image

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import img2img  # noqa: E402


def test_batch_inpaint_masks_match_image_names_literally(monkeypatch, tmp_path):
    images_dir, masks_dir = tmp_path / "images", tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    stems = {1: "a[1]", 2: "b", 3: "c"}  # each input image carries its id in the red channel
    for image_id, stem in stems.items():
        Image.new("RGB", (8, 8), (image_id, 0, 0)).save(images_dir / f"{stem}.png")
    # Each mask is told apart by its value. "a1.png" is what the glob pattern "a[1].*" matched; "b.mask.png" also
    # matches "b.*", so with "b.png" present the choice depended on directory order; "c" only has a "c.<x>.<ext>" mask.
    for name, value in (("a[1].png", 10), ("a1.png", 20), ("b.mask.png", 30), ("b.png", 40), ("c.mask.png", 50)):
        Image.new("L", (8, 8), value).save(masks_dir / name)

    used_masks = {}

    def run(p, *args):
        used_masks[stems[p.init_images[0].getpixel((0, 0))[0]]] = p.image_mask.getpixel((0, 0))
        return SimpleNamespace(images=list(p.init_images), infotexts=[""])

    monkeypatch.setattr(img2img.processing, "fix_seed", lambda p: None)
    monkeypatch.setattr(img2img.modules.scripts, "scripts_img2img", SimpleNamespace(run=run))

    p = SimpleNamespace(prompt="", negative_prompt="", seed=1, cfg_scale=7.0, sampler_name="Euler", steps=1, override_settings={}, n_iter=1, batch_size=1)
    img2img.process_batch(p, str(images_dir), "", str(masks_dir), [])

    assert used_masks == {"a[1]": 10, "b": 40, "c": 50}
