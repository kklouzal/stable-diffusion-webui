"""Multi-threaded PNG encoder for 8-bit RGB/RGBA images, equivalent to Pillow's default PNG save.

Contract (tests/test_png_writer.py checks it against Pillow on the same image):
- The chunks of ``image.save(fp, "PNG", pnginfo=info)`` in the same order: IHDR (same size, bit depth and color
  type), the text chunks of ``info`` byte-identical, IDAT, IEND. Decoding gives the same pixels.
- The same filtered scanlines (the decompressed IDAT stream is byte-identical): per row, the first filter with the
  smallest sum of |signed byte| among None, Up, Sub, Paeth, the order and rule of Pillow's ZipEncode.c when not
  ``optimize`` (libpng's default heuristic).
- The same deflate settings as Pillow's encoder (level 6, 32 KiB window, memLevel 9, Z_FILTERED), applied to pieces
  of about ``_PIECE_BYTES`` deflated in parallel (pigz-style): each piece is primed with the 32 KiB of filtered data
  before it, ends with a sync flush (the last with Z_FINISH), and the zlib trailer is the adler32 of the whole
  stream. The bit stream differs from Pillow's only around piece starts: +0.04-0.06 % file size on real outputs.
- The output bytes depend only on the image and info, never on the thread count.

``encode`` returns None for what it does not cover (other modes, an ICC profile or RGB transparency in
``image.info``, a non-empty ``image.encoderinfo``, non-text info chunks, an empty image); the caller then saves
with Pillow. Threads: the caller filters pieces with numpy while a per-call pool deflates the filtered ones (zlib
releases the GIL); the pool never outlives the call.
"""

from __future__ import annotations

import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_COLOR_TYPES = {"RGB": 2, "RGBA": 6}
_TEXT_CHUNKS = (b"tEXt", b"iTXt", b"zTXt")
_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ZLIB_HEADER = b"\x78\x9c"  # 32 KiB window, deflate, FLEVEL 2: what zlib writes for Pillow's settings
_LEVEL, _WBITS, _MEM_LEVEL, _STRATEGY = 6, 15, 9, zlib.Z_FILTERED  # Pillow's ZipEncode.c defaults for PNG
_WINDOW = 1 << _WBITS
# Filtered bytes per deflate piece, rounded down to whole rows (at least one). 128 KiB measured fastest with 64 KiB on
# 10 Grace cores at a third of 64 KiB's size cost; 256 KiB and up leave cores idle at the end.
_PIECE_BYTES = 128 << 10
_FILTER_TYPES = np.array([0, 2, 1, 4], dtype=np.uint8)  # None, Up, Sub, Paeth: Pillow's evaluation order
_IDAT_CRC = zlib.crc32(b"IDAT")
_ADLER_BASE = 65521


def available_threads() -> int:
    """CPUs this process may run on (the container's cpuset), not the host's count."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # no sched_getaffinity on this platform
        return os.cpu_count() or 1


def _chunk(cid: bytes, data: bytes) -> list[bytes]:
    return [len(data).to_bytes(4, "big") + cid, data, zlib.crc32(data, zlib.crc32(cid)).to_bytes(4, "big")]


def _filter_rows(rows: np.ndarray, start: int, stop: int, bpp: int) -> np.ndarray:
    """Filtered scanlines (filter type byte + filtered bytes) of rows[start:stop], as one flat uint8 array."""
    cur = rows[start:stop]
    up = np.empty_like(cur)
    up[0] = rows[start - 1] if start else 0
    up[1:] = cur[:-1]
    left = np.zeros_like(cur)
    left[:, bpp:] = cur[:, :-bpp]
    upleft = np.zeros_like(cur)
    upleft[:, bpp:] = up[:, :-bpp]

    a = left.astype(np.int16)
    b = up.astype(np.int16)
    c = upleft.astype(np.int16)
    pa = np.abs(b - c)
    pb = np.abs(a - c)
    pc = np.abs(a + b - c - c)
    paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, upleft))

    candidates = (cur, cur - up, cur - left, cur - paeth)  # uint8 arithmetic wraps mod 256, as in the C code
    # Pillow scores a filtered byte v as v < 128 ? v : 256 - v, which is min(v, -v mod 256).
    costs = np.stack([np.minimum(f, -f).sum(axis=1, dtype=np.uint32) for f in candidates])
    choice = costs.argmin(axis=0)  # the first minimum: a later filter must score strictly lower, as in Pillow

    out = np.empty((cur.shape[0], cur.shape[1] + 1), dtype=np.uint8)
    out[:, 0] = _FILTER_TYPES[choice]
    for index, filtered in enumerate(candidates):
        selected = choice == index
        if selected.all():
            out[:, 1:] = filtered
        elif selected.any():
            out[selected, 1:] = filtered[selected]
    return out.reshape(-1)


def _deflate_piece(data: np.ndarray, history: np.ndarray | None, last: bool) -> tuple[bytes, int, int]:
    """IDAT payload for data, the raw deflate continuation of a stream whose preceding bytes end with history (the
    zlib header instead for the first piece). Returns it with adler32(data) and the CRC of b"IDAT" + payload."""
    if history is None:
        compressor = zlib.compressobj(_LEVEL, zlib.DEFLATED, -_WBITS, _MEM_LEVEL, _STRATEGY)
        prefix = _ZLIB_HEADER
    else:
        compressor = zlib.compressobj(_LEVEL, zlib.DEFLATED, -_WBITS, _MEM_LEVEL, _STRATEGY, history)
        prefix = b""
    body = prefix + compressor.compress(data) + compressor.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH)
    return body, zlib.adler32(data), zlib.crc32(body, _IDAT_CRC)


def adler32_combine(adler1: int, adler2: int, length2: int) -> int:
    """adler32 of A + B from adler32(A), adler32(B) and len(B), as zlib's adler32_combine."""
    rem = length2 % _ADLER_BASE
    sum1 = adler1 & 0xFFFF
    sum2 = (rem * sum1) % _ADLER_BASE
    sum1 = (sum1 + (adler2 & 0xFFFF) + _ADLER_BASE - 1) % _ADLER_BASE
    sum2 = (sum2 + (adler1 >> 16) + (adler2 >> 16) + _ADLER_BASE - rem) % _ADLER_BASE
    return (sum2 << 16) | sum1


def encode(image, pnginfo=None, threads: int | None = None) -> bytes | None:
    """The PNG file for image.save(fp, "PNG", pnginfo=pnginfo) as specified above, or None if not covered.
    threads caps the deflate pool (default: available_threads())."""
    color_type = _COLOR_TYPES.get(image.mode)
    if color_type is None or not image.width or not image.height or getattr(image, "encoderinfo", None):
        return None
    if image.info.get("icc_profile") or (image.mode == "RGB" and image.info.get("transparency") is not None):
        return None  # Pillow writes iCCP / tRNS for these
    text_chunks = [info_chunk[:2] for info_chunk in (pnginfo.chunks if pnginfo is not None else ())]
    if any(cid not in _TEXT_CHUNKS for cid, _ in text_chunks):
        return None

    width, height = image.size
    bpp = len(image.mode)
    row_bytes = width * bpp + 1
    rows = np.asarray(image).reshape(height, width * bpp)
    rows_per_piece = max(1, _PIECE_BYTES // row_bytes)
    bounds = [(start, min(start + rows_per_piece, height)) for start in range(0, height, rows_per_piece)]
    workers = max(1, min(threads or available_threads(), len(bounds)))

    # The calling thread filters one piece after another while the pool deflates those already filtered: numpy
    # filtering in the pool as well ran slower (GIL handoffs between many short ops).
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="png-writer") as pool:
        futures = []
        history = None
        for start, stop in bounds:
            data = _filter_rows(rows, start, stop, bpp)
            futures.append(pool.submit(_deflate_piece, data, history, stop == height))
            history = data[-_WINDOW:]
        pieces = [future.result() for future in futures]

    checksum = 1
    for (_, piece_adler, _), (start, stop) in zip(pieces, bounds):
        checksum = adler32_combine(checksum, piece_adler, (stop - start) * row_bytes)
    trailer = checksum.to_bytes(4, "big")

    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes((8, color_type, 0, 0, 0))
    parts = [_SIGNATURE, *_chunk(b"IHDR", ihdr)]
    for cid, data in text_chunks:
        parts += _chunk(cid, data)
    for index, (body, _, crc) in enumerate(pieces):
        if index == len(pieces) - 1:
            parts += [(len(body) + 4).to_bytes(4, "big") + b"IDAT", body, trailer, zlib.crc32(trailer, crc).to_bytes(4, "big")]
        else:
            parts += [len(body).to_bytes(4, "big") + b"IDAT", body, crc.to_bytes(4, "big")]
    parts += _chunk(b"IEND", b"")
    return b"".join(parts)
