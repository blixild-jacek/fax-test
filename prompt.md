Write a TIFF decoder in Python.

Deliver one Python module that uses the standard library only (Python 3.12; `struct` and `zlib` are fine; no Pillow,
numpy, libtiff or other third-party packages, no ctypes, no subprocess). It must define:

```python
def decode(data: bytes) -> tuple[int, int, int, bytes]:
    """Decode the first image (IFD 0) of the TIFF file in `data`.
    Returns (width, height, channels, pixels)."""
```

What it returns:
- `channels` is 1 for bilevel and grayscale images, 3 for RGB and palette-color images.
- `pixels` is `width * height * channels` bytes: rows from top to bottom, pixels from left to right, samples
  interleaved (RGBRGB...), 8 bits per sample.
- Bilevel (1-bit) samples become 0 for black and 255 for white, as PhotometricInterpretation says (0 WhiteIsZero,
  1 BlackIsZero). 4-bit samples are multiplied by 17.
- Palette-color images (PhotometricInterpretation 3) are expanded to RGB through the ColorMap; each 16-bit ColorMap
  value becomes `value >> 8`.

What it must handle, as the TIFF 6.0 specification defines it:
- both byte orders (II and MM);
- strips (any RowsPerStrip) and tiles (TileWidth, TileLength, TileOffsets, TileByteCounts; edge tiles are padded);
- PlanarConfiguration 1 and 2; BitsPerSample 1, 4 and 8; SamplesPerPixel 1 and 3; PhotometricInterpretation 0, 1, 2
  and 3; FillOrder 1 and 2;
- Compression 1 (none), 2 (CCITT modified Huffman RLE), 3 (CCITT Group 3 fax, T.4: 1-D or 2-D as T4Options says,
  with or without fill bits before EOL), 4 (CCITT Group 4 fax, T.6), 5 (LZW), 8 and 32946 (Deflate), 32773 (PackBits);
- Predictor 1 and 2 (horizontal differencing).

Your module will be run on a set of TIFF files you have not seen. A file counts only if every pixel is right. This is
your only turn: reply with the complete module in a single ```python code block.
