"""Exact and perceptual image hashing.

dHash (64-bit difference hash) needs 9x8 grayscale pixels. This module reads
binary PGM (P5) and ASCII PGM (P2) natively so the core works dependency-free;
install Pillow and register its loader for PNG/JPEG in real projects.
"""

from __future__ import annotations

import hashlib

ImageLoader = None  # set to a callable(path) -> (width, height, gray_bytes)


def register_loader(fn) -> None:
    global ImageLoader
    ImageLoader = fn


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_pgm(path: str) -> tuple[int, int, bytes]:
    data = open(path, "rb").read()
    if data[:2] == b"P2":
        toks = data.split()
        w, h = int(toks[1]), int(toks[2])
        vals = bytes(int(t) for t in toks[4:4 + w * h])
        return w, h, vals
    if data[:2] == b"P5":
        idx, toks = 2, []
        while len(toks) < 3:
            while idx < len(data) and data[idx:idx+1].isspace():
                idx += 1
            if data[idx:idx+1] == b"#":  # comment line
                while idx < len(data) and data[idx:idx+1] != b"\n":
                    idx += 1
                continue
            start = idx
            while idx < len(data) and not data[idx:idx+1].isspace():
                idx += 1
            toks.append(data[start:idx])
        idx += 1  # exactly one whitespace byte before raster
        w, h = int(toks[0]), int(toks[1])
        return w, h, data[idx:idx + w * h]
    raise ValueError(f"unsupported format (native support: PGM P2/P5): {path}")


def _gray(path: str) -> tuple[int, int, bytes]:
    if ImageLoader is not None:
        return ImageLoader(path)
    return _load_pgm(path)


def _resize_gray(w: int, h: int, px: bytes, tw: int = 9, th: int = 8) -> bytes:
    """Nearest-neighbor shrink to tw x th (sufficient for dHash)."""
    out = bytearray()
    for y in range(th):
        sy = min(h - 1, y * h // th)
        for x in range(tw):
            sx = min(w - 1, x * w // tw)
            out.append(px[sy * w + sx])
    return bytes(out)


def dhash(path: str) -> int:
    """64-bit difference hash: bit set when pixel(x) > pixel(x+1)."""
    w, h, px = _gray(path)
    small = _resize_gray(w, h, px)
    bits = 0
    for y in range(8):
        for x in range(8):
            bits <<= 1
            if small[y * 9 + x] > small[y * 9 + x + 1]:
                bits |= 1
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
