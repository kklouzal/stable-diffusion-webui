import random
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageFilter, ImageOps

from modules import masking


def serial_fill(image, mask):
    """Oracle: the original single-threaded masking.fill."""
    image_mod = Image.new('RGBA', (image.width, image.height))
    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))
    image_masked = image_masked.convert('RGBa')
    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)
    return image_mod.convert("RGB")


def random_case(rng, width, height, image_mode, mask_mode):
    image = Image.frombytes(image_mode, (width, height), rng.randbytes(width * height * len(image_mode)))
    # Blocky masks exercise both masked and unmasked regions and their edges.
    blocks = Image.frombytes("L", (max(1, width // 16), max(1, height // 16)), bytes(rng.choice((0, 255, rng.randrange(256))) for _ in range(max(1, width // 16) * max(1, height // 16))))
    return image, blocks.resize((width, height), Image.NEAREST).convert(mask_mode)


def test_parallel_fill_matches_serial_oracle_bitwise():
    rng = random.Random(1234)
    for width, height, image_mode, mask_mode in ((257, 131, "RGB", "L"), (64, 64, "RGBA", "L"), (96, 200, "RGB", "1"), (128, 72, "RGB", "RGB")):
        image, mask = random_case(rng, width, height, image_mode, mask_mode)
        result = masking.fill(image, mask)
        expected = serial_fill(image, mask)
        assert (result.mode, result.size) == (expected.mode, expected.size)
        assert result.tobytes() == expected.tobytes()


def test_concurrent_fill_calls_stay_exact():
    rng = random.Random(99)
    cases = [random_case(rng, 160, 112, "RGB", "L") for _ in range(4)]
    expected = [serial_fill(image, mask).tobytes() for image, mask in cases]
    with ThreadPoolExecutor(max_workers=4) as callers:
        for _ in range(3):
            results = list(callers.map(lambda case: masking.fill(*case).tobytes(), cases))
            assert results == expected
