#!/usr/bin/env python3
# Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
# SPDX-License-Identifier: MIT
"""Draws what decoders returned beside the originals: one row per test file, the original first, then one column per
module, each tile marked exact, the share of pixels right, the wrong size, or the error. The modules run in the same
sandbox as the grader. Needs Pillow (the test itself does not).

  python3 tools/gallery.py results/RUN.json [more.json or module.py ...] [--label NAME ...] [--files a,b,c] [--out F]

A results file contributes its best turn's module; --label names the columns in order; --files picks the rows
(default: a mix of the uncompressed, LZW and fax-coded files).
"""
import argparse
import json
import os
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import faxtest  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

DEFAULT_FILES = ('raw_rgb8', 'raw_pal4', 'raw_rgb8_planar', 'raw_gray8_tiled', 'lzw_rgb8', 'lzw_bilevel',
                 'ccitt_rle_doc', 'g3_1d_doc', 'g3_2d_doc', 'g4_doc')
TILE_W, TILE_H, PAD, CAPTION, HEAD, ROW_LABEL = 360, 120, 14, 22, 54, 190
INK, MUTED, PAPER, LINE = (24, 24, 24), (110, 110, 110), (246, 245, 241), (170, 170, 170)


def best_module(path):
    """(code, label) from a results file's best turn, or a module file as it is."""
    if path.endswith('.py'):
        return open(path, encoding='utf-8').read(), os.path.splitext(os.path.basename(path))[0]
    rec = json.load(open(path, encoding='utf-8'))
    turns = [t for t in [rec] + rec.get('feedback', []) if t.get('grade') and t.get('code')]
    if not turns:
        raise SystemExit(f'{path}: no graded module in it')
    best = max(turns, key=lambda t: t['grade']['summary']['passed'])
    return best['code'], f"{rec['model']}, {best['grade']['summary']['passed']}/40"


def view(px, w, h, c):
    """An image of returned pixels, or None when they do not fill width x height x channels."""
    if w <= 0 or h <= 0 or c not in (1, 3) or len(px) != w * h * c:
        return None
    return Image.frombytes('L' if c == 1 else 'RGB', (w, h), px).convert('RGB')


def fit(im):
    """A wide page cropped from the left to 3:1, then scaled into the tile."""
    if im.width > 3 * im.height:
        im = im.crop((0, 0, 3 * im.height, im.height))
    im = im.copy()
    im.thumbnail((TILE_W, TILE_H), Image.NEAREST if im.width < TILE_W else Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='results .json files or module .py files')
    ap.add_argument('--label', action='append', help='column names, in order')
    ap.add_argument('--files', help='comma-separated corpus file names for the rows')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'gallery.png'))
    ap.add_argument('--sandbox', default='auto', choices=('auto', 'sandbox-exec', 'docker', 'none'))
    a = ap.parse_args()
    sandbox = faxtest.pick_sandbox(a.sandbox)
    manifest = json.load(open(os.path.join(faxtest.CORPUS, 'manifest.json'), encoding='utf-8'))['files']
    files = a.files.split(',') if a.files else list(DEFAULT_FILES)
    modules = [best_module(p) for p in a.inputs]
    labels = (a.label or []) + [m[1] for m in modules[len(a.label or []):]]
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    cols = 1 + len(modules)
    W = ROW_LABEL + cols * (TILE_W + PAD) + PAD
    H = HEAD + len(files) * (TILE_H + CAPTION + PAD) + PAD
    canvas = Image.new('RGB', (W, H), PAPER)
    d = ImageDraw.Draw(canvas)
    for j, text in enumerate(['original (libtiff)'] + labels):
        d.text((ROW_LABEL + j * (TILE_W + PAD), PAD + 8), text, fill=INK, font=font)
    tmp = []
    try:
        paths = []
        for code, _ in modules:
            f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8')
            f.write(code)
            f.close()
            tmp.append(f.name)
            paths.append(f.name)
        for i, name in enumerate(files):
            meta = manifest[name]
            y = HEAD + i * (TILE_H + CAPTION + PAD)
            d.text((PAD, y + 4), name, fill=INK, font=font)
            d.text((PAD, y + 26), meta['category'].replace('_', ' '), fill=MUTED, font=small)
            exp = zlib.decompress(open(os.path.join(faxtest.CORPUS, name + '.expected.z'), 'rb').read())
            tiles = [(view(exp, meta['width'], meta['height'], meta['channels']), f"{meta['width']} x {meta['height']}")]
            for path in paths:
                try:
                    head, px = faxtest.run_one(path, os.path.join(faxtest.CORPUS, name + '.tif'), sandbox=sandbox)
                except RuntimeError as e:
                    tiles.append((None, 'error: ' + str(e)[:46]))
                    continue
                im = view(px, head['width'], head['height'], head['channels'])
                dims = (head['width'], head['height'], head['channels'])
                want = (meta['width'], meta['height'], meta['channels'])
                if im is None and dims == want and px:     # the right size, too few or too many bytes: draw what came
                    n = len(exp)
                    im = view((px + b'\x80' * n)[:n], *want)
                    tiles.append((im, f'{len(px):,} of {n:,} bytes returned'))
                    continue
                if px == exp:
                    note = 'exact'
                elif im is not None and (head['width'], head['height'], head['channels']) == \
                        (meta['width'], meta['height'], meta['channels']):
                    c = meta['channels']
                    right = sum(px[k:k + c] == exp[k:k + c] for k in range(0, len(exp), c)) / (len(exp) // c)
                    note = f'{right:.0%} of pixels right'
                else:
                    note = f"wrong size: {head['width']} x {head['height']} x {head['channels']}"
                tiles.append((im, note))
            for j, (im, note) in enumerate(tiles):
                x = ROW_LABEL + j * (TILE_W + PAD)
                d.rectangle((x - 1, y - 1, x + TILE_W, y + TILE_H), outline=LINE, fill=(232, 231, 227))
                if im is not None:
                    im = fit(im)
                    canvas.paste(im, (x + (TILE_W - im.width) // 2, y + (TILE_H - im.height) // 2))
                d.text((x, y + TILE_H + 3), note, fill=INK if note == 'exact' or j == 0 else MUTED, font=small)
    finally:
        for p in tmp:
            os.unlink(p)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    canvas.save(a.out, optimize=True)
    print(f'{a.out}: {len(files)} files x {len(modules)} decoders')


if __name__ == '__main__':
    main()
