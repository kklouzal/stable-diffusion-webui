import math
from fractions import Fraction

from modules import masking


def _reference(crop_region, processing_width, processing_height, image_width, image_height):
    """The original algorithm with its ratio and ceil() evaluated in exact rational arithmetic."""
    x1, y1, x2, y2 = crop_region
    ratio_processing = Fraction(processing_width, processing_height)

    if Fraction(x2 - x1, y2 - y1) > ratio_processing:
        diff = math.ceil((x2 - x1) / ratio_processing - (y2 - y1))
        y1 -= diff // 2
        y2 += diff - diff // 2
        if y2 >= image_height:
            y1 -= y2 - image_height
            y2 = image_height
        if y1 < 0:
            y2 -= y1
            y1 = 0
        y2 = min(y2, image_height)
    else:
        diff = math.ceil((y2 - y1) * ratio_processing - (x2 - x1))
        x1 -= diff // 2
        x2 += diff - diff // 2
        if x2 >= image_width:
            x1 -= x2 - image_width
            x2 = image_width
        if x1 < 0:
            x2 -= x1
            x1 = 0
        x2 = min(x2, image_width)

    return x1, y1, x2, y2


def test_expanded_region_matches_the_exact_rational_algorithm():
    mismatches = []
    for processing in ((1024, 1024), (1216, 832), (832, 1216), (1344, 768), (1152, 896)):
        for width in range(1, 400, 3):
            for height in range(1, 400, 7):
                crop = (100, 120, 100 + width, 120 + height)
                expected = _reference(crop, *processing, 1600, 1400)
                actual = masking.expand_crop_region(crop, *processing, 1600, 1400)
                if actual != expected:
                    mismatches.append((processing, crop, actual, expected))

    assert not mismatches, mismatches[:10]


def test_exact_aspect_needs_no_extra_pixel():
    # 361 * 832 / 1216 is exactly 247 rows; through the float ratio it came out as 247.00000000000003 and ceil() added one.
    assert masking.expand_crop_region((0, 0, 361, 1), 1216, 832, 2000, 2000) == (0, 0, 361, 247)
