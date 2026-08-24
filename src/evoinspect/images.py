from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from .errors import EvoInspectError


def _pgm_tokens(data: bytes) -> tuple[list[bytes], int]:
    tokens: list[bytes] = []
    position = 0
    length = len(data)
    while len(tokens) < 4:
        while position < length and data[position] in b" \t\r\n":
            position += 1
        if position < length and data[position] == ord("#"):
            while position < length and data[position] not in b"\r\n":
                position += 1
            continue
        start = position
        while position < length and data[position] not in b" \t\r\n#":
            position += 1
        if start == position:
            raise EvoInspectError("invalid PGM header")
        tokens.append(data[start:position])
    if position >= length or data[position] not in b" \t\r\n":
        raise EvoInspectError("PGM header lacks a raster delimiter")
    if data[position : position + 2] == b"\r\n":
        position += 2
    else:
        position += 1
    return tokens, position


def read_pgm(path: Path) -> tuple[int, int, list[int]]:
    data = path.read_bytes()
    tokens, body_start = _pgm_tokens(data)
    magic = tokens[0]
    try:
        width, height, maximum = (int(token) for token in tokens[1:])
    except ValueError as exc:
        raise EvoInspectError(f"invalid PGM dimensions: {path}") from exc
    if width <= 0 or height <= 0 or not 0 < maximum <= 255:
        raise EvoInspectError(f"unsupported PGM header: {path}")
    count = width * height
    if magic == b"P5":
        pixels = list(data[body_start : body_start + count])
    elif magic == b"P2":
        uncommented = b"\n".join(line.split(b"#", 1)[0] for line in data[body_start:].splitlines())
        body = uncommented.split()
        try:
            pixels = [int(token) for token in body]
        except ValueError as exc:
            raise EvoInspectError(f"invalid PGM pixels: {path}") from exc
    else:
        raise EvoInspectError(f"unsupported PGM magic {magic!r}: {path}")
    if len(pixels) != count or any(pixel < 0 or pixel > maximum for pixel in pixels):
        raise EvoInspectError(f"invalid PGM payload length/range: {path}")
    if maximum != 255:
        pixels = [round(pixel * 255 / maximum) for pixel in pixels]
    return width, height, pixels


def load_grayscale(path: Path) -> tuple[int, int, list[int]]:
    if path.suffix.lower() == ".pgm":
        return read_pgm(path)
    try:
        from PIL import Image
    except ImportError as exc:
        raise EvoInspectError(
            f"Pillow is required for {path.suffix or 'this image format'}; PGM works without it"
        ) from exc
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            return grayscale.width, grayscale.height, list(grayscale.getdata())
    except Exception as exc:
        raise EvoInspectError(f"cannot decode image {path}: {exc}") from exc


def image_features(path: Path, grid: int) -> list[float]:
    width, height, pixels = load_grayscale(path)
    if grid <= 0 or grid > min(width, height):
        raise EvoInspectError(f"feature grid {grid} is invalid for {width}x{height}: {path}")
    normalized = [pixel / 255.0 for pixel in pixels]
    features: list[float] = []
    for row in range(grid):
        top = row * height // grid
        bottom = (row + 1) * height // grid
        for column in range(grid):
            left = column * width // grid
            right = (column + 1) * width // grid
            cell = [
                normalized[y * width + x] for y in range(top, bottom) for x in range(left, right)
            ]
            features.append(sum(cell) / len(cell))
    mean = sum(normalized) / len(normalized)
    variance = sum((value - mean) ** 2 for value in normalized) / len(normalized)
    features.extend([mean, math.sqrt(variance), min(normalized), max(normalized)])
    return features


def euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("feature lengths differ")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=False)))


def centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot compute an empty centroid")
    length = len(vectors[0])
    if any(len(vector) != length for vector in vectors):
        raise ValueError("feature lengths differ")
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(length)]
