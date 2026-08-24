from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evoinspect.images import read_pgm


class PgmReaderTest(unittest.TestCase):
    def test_binary_raster_keeps_leading_whitespace_valued_pixel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leading-whitespace.pgm"
            path.write_bytes(b"P5\n2 2\n255\n" + bytes([10, 32, 0, 255]))
            width, height, pixels = read_pgm(path)
            self.assertEqual((width, height), (2, 2))
            self.assertEqual(pixels, [10, 32, 0, 255])

    def test_ascii_raster_ignores_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "commented.pgm"
            path.write_bytes(b"P2\n2 2\n255\n1 2 # pixels\n3 4\n")
            _, _, pixels = read_pgm(path)
            self.assertEqual(pixels, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
