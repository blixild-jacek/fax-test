# Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
# SPDX-License-Identifier: MIT
"""The grader's check (faxtest.py selftest): libtiff through Pillow, which every corpus file was checked
against when it was made. Must pass all 40. Needs Pillow, so it runs only where Pillow is installed."""
import io

from PIL import Image


def decode(data):
    im = Image.open(io.BytesIO(data))
    im = im.convert('L' if im.mode in ('1', 'L', 'I;16', 'I;16B') else 'RGB')
    return im.width, im.height, 1 if im.mode == 'L' else 3, im.tobytes()
