#!/usr/bin/env python3
# Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
# SPDX-License-Identifier: MIT
"""Builds the Fax Test's 40 test files (corpus/). You do not need it to run the test: the files are in the repository.

Each image is drawn here from a fixed seed, written as a plain uncompressed TIFF by the small writer below, then
re-encoded by libtiff's own tiffcp (or Pillow's libtiff writer for CCITT RLE, which tiffcp cannot write). What a
decoder must return is known from the drawing, not from any decoder; every file is then read back with Pillow
(libtiff) and must match it, or the build stops. Output: corpus/NAME.tif, corpus/NAME.expected.z (the expected
8-bit pixels, zlib) and corpus/manifest.json. Needs Pillow with libtiff, numpy and tiffcp (e.g. brew install libtiff).
With Pillow 12.3 (libtiff 4.7.1) and tiffcp 4.7.2 it rebuilds the committed files byte for byte.

  python3 tools/make_corpus.py
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'corpus')


# ---- drawings: arrays of what a decoder must return (8-bit; bilevel 0 black / 255 white) ----

def doc_page():
    """A fax-width page: text, rules, boxes, a full-black band, a dithered gradient, noise, 1-pixel stripes, lines
    starting black: every kind of run a CCITT coder meets (makeup codes, zero-length white runs, short and long)."""
    w, h = 1728, 260
    im = Image.new('L', (w, h), 255)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    d.text((40, 8), 'TIFF 6.0 / ITU-T T.4 and T.6 test page - Blixild tiffbench', fill=0, font=font)
    d.text((40, 36), '0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz', fill=0, font=font)
    d.rectangle((0, 66, w - 1, 71), fill=0)                       # full-width black lines
    d.rectangle((30, 80, 400, 200), outline=0, width=3)
    d.ellipse((450, 80, 640, 230), outline=0, width=5)
    d.line((700, 80, 1100, 250), fill=0, width=2)
    grad = Image.linear_gradient('L').resize((420, 120)).rotate(90, expand=True).resize((420, 120))
    im.paste(grad.convert('1').convert('L'), (1150, 80))           # Floyd-Steinberg: many short runs
    a = np.where(np.array(im) < 128, 0, 255).astype(np.uint8)     # text is drawn anti-aliased: back to bilevel
    rng = np.random.default_rng(1)
    a[205:250, 30:420] = np.where(rng.random((45, 390)) < 0.5, 0, 255)
    a[210:250, 1600:1727:2] = 0                                    # 1-pixel stripes
    a[100:140, 0] = 0                                              # lines that start with a black pixel
    a[150:160, :] = 255
    a[150:160, 1727] = 0                                           # and end with one
    a[255:260, :64] = 0                                            # a run of exactly 64
    return a


def small_bilevel():
    """An odd width (37: not a multiple of 8), random runs, a black first column."""
    rng = np.random.default_rng(2)
    a = np.where(rng.random((23, 37)) < 0.35, 0, 255).astype(np.uint8)
    a[:, 0] = 0
    a[5] = 255
    a[6] = 0
    return a


def wide_bilevel():
    """3000 wide: runs past 2560 need the extended makeup codes (shared by white and black)."""
    a = np.full((48, 3000), 255, np.uint8)
    a[4:10, 100:2800] = 0          # black run of 2700
    a[12:16, 0:2561] = 0           # black run of 2561 from the left edge
    a[20:24, 2:2999] = 0           # white 2, black 2997, white 1
    a[30:34, 1:2] = 0
    a[30:34, 2600:2604] = 0        # white run of 2598
    rng = np.random.default_rng(3)
    a[38:46, ::7] = 0
    a[40:44] = np.where(rng.random((4, 3000)) < 0.02, 0, a[40:44])
    return a


def photo_gray():
    y, x = np.mgrid[0:103, 0:157]
    rng = np.random.default_rng(4)
    v = 128 + 90 * np.sin(x / 9.0) * np.cos(y / 13.0) + (x - 78) * 0.4 + rng.normal(0, 6, x.shape)
    v[40:60, 60:100] = 255
    v[70:80, :] = 0
    return np.clip(v, 0, 255).astype(np.uint8)


def photo_rgb():
    y, x = np.mgrid[0:103, 0:157]
    rng = np.random.default_rng(5)
    r = 128 + 100 * np.sin(x / 11.0) + rng.normal(0, 5, x.shape)
    g = 255 * y / 102.0
    b = 128 + 100 * np.cos((x + y) / 17.0)
    a = np.stack([r, g, b], -1)
    a[20:40, 20:50] = (255, 0, 0)
    a[60:90, 100:140] = (0, 0, 255)
    return np.clip(a, 0, 255).astype(np.uint8)


def palette(rgb, colors):
    """(indices, colormap rows): the image quantized to `colors` entries."""
    q = Image.fromarray(rgb).quantize(colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = np.array(q.getpalette()[:3 * colors], np.uint8).reshape(-1, 3)
    if len(pal) < colors:
        pal = np.vstack([pal, np.zeros((colors - len(pal), 3), np.uint8)])
    return np.array(q), pal


# ---- a minimal baseline writer: one strip, uncompressed, little-endian ----

def write_tiff(path, w, h, bits, spp, photometric, data, colormap=None):
    tags = [(256, 3, [w]), (257, 3, [h]), (258, 3, [bits] * spp), (259, 3, [1]), (262, 3, [photometric]),
            (273, 4, [0]), (277, 3, [spp]), (278, 3, [h]), (279, 4, [len(data)]), (282, 5, [(72, 1)]),
            (283, 5, [(72, 1)]), (284, 3, [1]), (296, 3, [2])]
    if colormap is not None:
        tags.append((320, 3, [int(v) * 257 for v in colormap]))
    tags.sort()
    size = {3: 2, 4: 4, 5: 8}
    ifd_at = 8 + len(data) + (len(data) & 1)
    extra_at = ifd_at + 2 + 12 * len(tags) + 4
    entries, extra = b'', b''
    for tag, typ, vals in tags:
        if tag == 273:
            vals = [8]
        raw = b''.join(struct.pack('<II', *v) if typ == 5 else struct.pack('<H' if typ == 3 else '<I', v) for v in vals)
        if len(raw) <= 4:
            entries += struct.pack('<HHI', tag, typ, len(vals)) + raw.ljust(4, b'\0')
        else:
            entries += struct.pack('<HHII', tag, typ, len(vals), extra_at + len(extra))
            extra += raw + (b'\0' if len(raw) & 1 else b'')
        assert size[typ]
    with open(path, 'wb') as f:
        f.write(b'II*\0' + struct.pack('<I', ifd_at) + data + b'\0' * (len(data) & 1))
        f.write(struct.pack('<H', len(tags)) + entries + b'\0\0\0\0' + extra)


def pack_bits(rows, bits):
    """Rows of sample values packed MSB-first at `bits` per sample, each row padded to a byte."""
    per = 8 // bits
    out = bytearray()
    for row in rows:
        row = list(row) + [0] * (-len(row) % per)
        for i in range(0, len(row), per):
            b = 0
            for v in row[i:i + per]:
                b = (b << bits) | int(v)
            out.append(b)
    return bytes(out)


def base_bilevel(path, a, miniswhite):
    black = (a == 0)
    bits = black if miniswhite else ~black          # the stored 1s: black in WhiteIsZero, white in BlackIsZero
    write_tiff(path, a.shape[1], a.shape[0], 1, 1, 0 if miniswhite else 1, pack_bits(bits.astype(np.uint8), 1))


def set_compression(path, code):
    """Rewrites IFD 0's Compression value in place (32946, the old Deflate code, which tiffcp writes as 8)."""
    with open(path, 'r+b') as f:
        e = '<' if f.read(2) == b'II' else '>'
        f.seek(4)
        ifd = struct.unpack(e + 'I', f.read(4))[0]
        f.seek(ifd)
        for i in range(struct.unpack(e + 'H', f.read(2))[0]):
            at = ifd + 2 + 12 * i
            f.seek(at)
            if struct.unpack(e + 'H', f.read(2))[0] == 259:
                f.seek(at + 8)
                f.write(struct.pack(e + 'H', code))
                return
    raise ValueError('no Compression tag')


def describe(path):
    """The file's layout in TIFF tag words, for the feedback a model gets (faxtest.py run --turns)."""
    im = Image.open(path)
    t = im.tag_v2
    parts = [open(path, 'rb').read(2).decode(), f'{t[256]}x{t[257]}', 'BitsPerSample ' + ','.join(map(str, t[258] if isinstance(t[258], tuple) else [t[258]])),
             f'SamplesPerPixel {t.get(277, 1)}', f'Compression {t[259]}', f'PhotometricInterpretation {t[262]}',
             f'PlanarConfiguration {t.get(284, 1)}']
    if 322 in t:
        parts.append(f'tiles {t[322]}x{t[323]} ({len(t[324])} of them)')
    else:
        parts.append(f'RowsPerStrip {t.get(278)} ({len(t[273])} strips)')
    for tag, word in ((266, 'FillOrder'), (292, 'T4Options'), (293, 'T6Options'), (317, 'Predictor')):
        if tag in t:
            parts.append(f'{word} {t[tag]}')
    if 320 in t:
        parts.append(f'ColorMap of {len(t[320]) // 3}')
    return ', '.join(str(p) for p in parts)


# ---- the corpus ----

def build():
    if not features.check('libtiff') or not shutil.which('tiffcp'):
        sys.exit('needs Pillow with libtiff and tiffcp (brew install libtiff)')
    doc, small, wide = doc_page(), small_bilevel(), wide_bilevel()
    gray, rgb = photo_gray(), photo_rgb()
    idx8, pal8 = palette(rgb, 256)
    idx4, pal4 = palette(rgb, 16)
    rgb_of = lambda idx, pal: pal[idx]                      # noqa: E731 - what a palette image decodes to

    tmp = tempfile.mkdtemp()
    bases = {}

    def base(name, expected, writer):
        p = os.path.join(tmp, name + '.tif')
        writer(p)
        bases[name] = (p, expected)

    for nm, a in (('doc', doc), ('small', small), ('wide', wide)):
        base(f'{nm}_w', a, lambda p, a=a: base_bilevel(p, a, True))
        base(f'{nm}_b', a, lambda p, a=a: base_bilevel(p, a, False))
    base('gray', gray, lambda p: write_tiff(p, 157, 103, 8, 1, 1, gray.tobytes()))
    base('rgb', rgb, lambda p: write_tiff(p, 157, 103, 8, 3, 2, rgb.tobytes()))
    cm8 = list(pal8[:, 0]) + list(pal8[:, 1]) + list(pal8[:, 2])
    base('pal8', rgb_of(idx8, pal8), lambda p: write_tiff(p, 157, 103, 8, 1, 3, idx8.tobytes(), cm8))
    cm4 = list(pal4[:, 0]) + list(pal4[:, 1]) + list(pal4[:, 2])
    base('pal4', rgb_of(idx4, pal4), lambda p: write_tiff(p, 157, 103, 4, 1, 3, pack_bits(idx4, 4), cm4))

    # name: (category, base, how), how = tiffcp arguments, or ('pillow', compression) for what tiffcp can't write
    plan = {
        'raw_bilevel_miniswhite': ('uncompressed', 'doc_w', []),
        'raw_bilevel_minisblack_be': ('uncompressed', 'small_b', ['-B']),
        'raw_gray8': ('uncompressed', 'gray', []),
        'raw_gray8_tiled': ('uncompressed', 'gray', ['-t', '-w', '32', '-l', '32']),
        'raw_rgb8': ('uncompressed', 'rgb', []),
        'raw_rgb8_strips_be': ('uncompressed', 'rgb', ['-B', '-r', '7']),
        'raw_rgb8_planar': ('uncompressed', 'rgb', ['-p', 'separate']),
        'raw_pal8': ('uncompressed', 'pal8', []),
        'raw_pal4': ('uncompressed', 'pal4', []),
        'packbits_bilevel': ('packbits', 'doc_w', ['-c', 'packbits']),
        'packbits_gray8': ('packbits', 'gray', ['-c', 'packbits']),
        'packbits_rgb8_be': ('packbits', 'rgb', ['-c', 'packbits', '-B']),
        'lzw_bilevel': ('lzw', 'doc_w', ['-c', 'lzw']),
        'lzw_gray8': ('lzw', 'gray', ['-c', 'lzw']),
        'lzw_rgb8': ('lzw', 'rgb', ['-c', 'lzw']),
        'lzw_rgb8_predictor': ('lzw', 'rgb', ['-c', 'lzw:2']),
        'lzw_gray8_predictor_strips_be': ('lzw', 'gray', ['-c', 'lzw:2', '-r', '16', '-B']),
        'lzw_pal8': ('lzw', 'pal8', ['-c', 'lzw']),
        'deflate_gray8': ('deflate', 'gray', ['-c', 'zip']),
        'deflate_rgb8_predictor': ('deflate', 'rgb', ['-c', 'zip:2']),
        'deflate_old_code_rgb8': ('deflate', 'rgb', ['-c', 'zip', '-r', '20'], 32946),   # the pre-6.0 code
        'ccitt_rle_doc': ('ccitt_rle', 'doc_w', ('pillow', 'tiff_ccitt')),
        'ccitt_rle_small': ('ccitt_rle', 'small_w', ('pillow', 'tiff_ccitt')),
        'ccitt_rle_wide_lsb2msb': ('ccitt_rle', 'wide_w', ('pillow', 'tiff_ccitt', ['-f', 'lsb2msb'])),
        'g3_1d_doc': ('g3_1d', 'doc_w', ['-c', 'g3:1d']),
        'g3_1d_small': ('g3_1d', 'small_w', ['-c', 'g3:1d']),
        'g3_1d_wide': ('g3_1d', 'wide_w', ['-c', 'g3:1d']),
        'g3_1d_fill_doc': ('g3_1d', 'doc_w', ['-c', 'g3:1d:fill']),
        'g3_1d_lsb2msb_doc': ('g3_1d', 'doc_w', ['-c', 'g3:1d', '-f', 'lsb2msb']),
        'g3_2d_doc': ('g3_2d', 'doc_w', ['-c', 'g3:2d']),
        'g3_2d_small': ('g3_2d', 'small_w', ['-c', 'g3:2d']),
        'g3_2d_wide': ('g3_2d', 'wide_w', ['-c', 'g3:2d']),
        'g3_2d_fill_strips_doc': ('g3_2d', 'doc_w', ['-c', 'g3:2d:fill', '-r', '40']),
        'g4_doc': ('g4', 'doc_w', ['-c', 'g4']),
        'g4_small': ('g4', 'small_w', ['-c', 'g4']),
        'g4_wide': ('g4', 'wide_w', ['-c', 'g4']),
        'g4_strips_doc': ('g4', 'doc_w', ['-c', 'g4', '-r', '32']),
        'g4_lsb2msb_doc': ('g4', 'doc_w', ['-c', 'g4', '-f', 'lsb2msb']),
        'g4_minisblack_doc': ('g4', 'doc_b', ['-c', 'g4']),
        'g4_tiled_doc': ('g4', 'doc_w', ['-c', 'g4', '-t', '-w', '256', '-l', '256']),
    }

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    manifest = {}
    for name, (cat, b, how, *code) in plan.items():
        src, expected = bases[b]
        dst = os.path.join(OUT, name + '.tif')
        if isinstance(how, tuple):                      # Pillow's libtiff writer, then tiffcp for any layout change
            im = Image.open(src)
            im.load()
            mid = dst if len(how) == 2 else os.path.join(tmp, name + '.mid.tif')
            im.save(mid, compression=how[1])
            if len(how) == 3:
                subprocess.run(['tiffcp', *how[2], mid, dst], check=True)
        else:
            subprocess.run(['tiffcp', *how, src, dst], check=True)
        if code:
            set_compression(dst, code[0])
        exp = np.ascontiguousarray(expected)
        h, w = exp.shape[:2]
        ch = 1 if exp.ndim == 2 else 3
        # the check: libtiff (through Pillow) reads back exactly what was drawn
        got = Image.open(dst)
        got = np.array(got.convert('L' if ch == 1 else 'RGB'))
        if got.shape != exp.shape or not np.array_equal(got, exp):
            sys.exit(f'{name}: libtiff does not read back the drawing ({(got != exp).sum()} samples differ)')
        px = exp.tobytes()
        with open(os.path.join(OUT, name + '.expected.z'), 'wb') as f:
            f.write(zlib.compress(px, 9))
        tif = open(dst, 'rb').read()
        manifest[name] = {'category': cat, 'width': w, 'height': h, 'channels': ch,
                          'sha256': hashlib.sha256(px).hexdigest(), 'bytes': len(tif),
                          'tiff_sha256': hashlib.sha256(tif).hexdigest(), 'made_with': how if isinstance(how, list)
                          else ['pillow', *how[1:2]] + (how[2] if len(how) == 3 else [])}
        if code:
            manifest[name]['made_with'] += [f'Compression tag set to {code[0]}']
        manifest[name]['tags'] = describe(dst)
        print(f'{name:34s} {cat:13s} {w}x{h}x{ch} {len(tif):7d} bytes')
    with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
        json.dump({'made': 'tools/make_corpus.py', 'libtiff': features.version('libtiff'),
                   'files': manifest}, f, indent=1)
    shutil.rmtree(tmp)
    print(f'{len(manifest)} files in {OUT}')


if __name__ == '__main__':
    build()
