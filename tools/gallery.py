#!/usr/bin/env python3
# Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
# SPDX-License-Identifier: MIT
"""Draws what decoders returned beside the originals: one row per TIFF file, the original first, then one column per
module, each picture marked exact, the share of pixels right, a short output, or the crash. The modules run in the
same sandbox as the grader. Needs Pillow (the test itself does not).

  python3 tools/gallery.py results/RUN.json [more.json or module.py ...] [--label "NAME|note" ...]
                           [--files a,b,c | --tiff FILE ... [--name "NAME|note" ...]]
                           [--tile WxH] [--title T] [--subtitle S] [--note N] [--out F]

A results file contributes its best turn's module. --label names the columns in order, --name the rows; a "|"
starts the smaller second line. --files picks test files for the rows (default: a mix of the uncompressed, LZW and
fax-coded ones); --tiff draws your own TIFF files instead, each original as libtiff reads it through Pillow.
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
COMPRESSION = {1: 'uncompressed', 2: 'CCITT RLE', 3: 'CCITT Group 3', 4: 'CCITT Group 4', 5: 'LZW', 8: 'Deflate',
               32946: 'Deflate', 32773: 'PackBits'}
PAPER, TILE, EMPTY, LINE = (247, 245, 240), (255, 255, 255), (236, 233, 226), (217, 213, 204)
INK, MUTED, GOOD = (29, 29, 27), (110, 108, 102), (37, 120, 56)   # green only for exact: every pixel right
FONTS = [('/System/Library/Fonts/HelveticaNeue.ttc', 0, 1),                      # macOS
         ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 0, 0),             # Linux
         ('C:\\Windows\\Fonts\\arial.ttf', 0, 0)]                                 # Windows
BOLD_FILES = {'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              'C:\\Windows\\Fonts\\arial.ttf': 'C:\\Windows\\Fonts\\arialbd.ttf'}


def font(size, bold=False):
    for path, regular, bold_index in FONTS:
        try:
            if bold and path in BOLD_FILES:
                return ImageFont.truetype(BOLD_FILES[path], size)
            return ImageFont.truetype(path, size, index=bold_index if bold else regular)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def best_module(path):
    """(code, label) from a results file's best turn, or a module file as it is."""
    if path.endswith('.py'):
        return open(path, encoding='utf-8').read(), os.path.splitext(os.path.basename(path))[0]
    rec = json.load(open(path, encoding='utf-8'))
    turns = [t for t in [rec] + rec.get('feedback', []) if t.get('grade') and t.get('code')]
    if not turns:
        raise SystemExit(f'{path}: no graded module in it')
    best = max(turns, key=lambda t: t['grade']['summary']['passed'])
    return best['code'], f"{rec['model']}|{best['grade']['summary']['passed']} of 40 test files right"


def view(px, w, h, c):
    """An image of returned pixels, or None when they do not fill width x height x channels."""
    if w <= 0 or h <= 0 or c not in (1, 3) or len(px) != w * h * c:
        return None
    return Image.frombytes('L' if c == 1 else 'RGB', (w, h), px).convert('RGB')


def fit(im, tile_w, tile_h):
    """A page wider than the tile's shape cropped from the left to it, then scaled into the tile."""
    if im.width * tile_h > tile_w * im.height:
        im = im.crop((0, 0, round(tile_w * im.height / tile_h), im.height))
    grow = min(tile_w // im.width, tile_h // im.height)
    if grow > 1:                              # a small picture: whole-number steps, every pixel a sharp square
        return im.resize((im.width * grow, im.height * grow), Image.NEAREST)
    im = im.copy()
    im.thumbnail((tile_w, tile_h), Image.LANCZOS)
    return im


def rows_from(a, manifest):
    """(label, TIFF path, expected pixels, width, height, channels) for each row."""
    out = []
    if a.tiff:
        for path in a.tiff:
            im = Image.open(path)
            comp = COMPRESSION.get(im.tag_v2.get(259), f'compression {im.tag_v2.get(259)}')
            kind = 'black and white' if im.mode in ('1', 'L') else 'color'
            im = im.convert('L' if im.mode in ('1', 'L') else 'RGB')
            out.append((f'{comp}|{kind}', path, im.tobytes(), im.width, im.height, 1 if im.mode == 'L' else 3))
    else:
        for n in (a.files.split(',') if a.files else DEFAULT_FILES):
            m = manifest[n]
            exp = zlib.decompress(open(os.path.join(faxtest.CORPUS, n + '.expected.z'), 'rb').read())
            out.append((f"{n}|{m['category'].replace('_', ' ')}", os.path.join(faxtest.CORPUS, n + '.tif'), exp,
                        m['width'], m['height'], m['channels']))
    for i, name in enumerate(a.name or []):
        if i < len(out):
            out[i] = (name,) + out[i][1:]
    return out


def outcome(path, tif, exp, w, h, c, sandbox):
    """(picture or None, caption, kind) for one decoder on one file; kind exact | partial | crash."""
    try:
        head, px = faxtest.run_one(path, tif, sandbox=sandbox)
    except RuntimeError as e:
        return None, str(e), 'crash'
    got = (head['width'], head['height'], head['channels'])
    if px == exp:
        return view(px, w, h, c), 'Exact: every pixel right', 'exact'
    if got == (w, h, c) and len(px) == len(exp):
        right = sum(px[k:k + c] == exp[k:k + c] for k in range(0, len(exp), c)) / (len(exp) // c)
        for value, shade in ((0, 'black'), (255, 'white')):   # one shade nearly everywhere: matches are chance
            share = px.count(value) / len(px)
            if share >= 0.95:
                note = 'match by chance' if share == 1 else 'match'
                return view(px, w, h, c), \
                    f"{'All' if share == 1 else 'Almost all'} {shade} ({right:.0%} of pixels {note})", 'partial'
        return view(px, w, h, c), f'{right:.0%} of pixels right', 'partial'
    if got == (w, h, c) and px:
        return view((px + b'\x80' * len(exp))[:len(exp)], w, h, c), \
            f'Short output: {len(px):,} of {len(exp):,} bytes', 'partial'
    im = view(px, *got)
    return im, f'Wrong size: {got[0]} x {got[1]}, {got[2]} channel{"s" if got[2] != 1 else ""}', 'partial'


def wrap(text, fnt, width):
    """Lines of text that each fit width pixels in fnt."""
    lines, cur = [], ''
    for word in text.split():
        trial = f'{cur} {word}'.strip()
        if cur and fnt.getlength(trial) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def two_lines(d, x, y, text, big, small, color=INK):
    first, _, second = text.partition('|')
    d.text((x, y), first, fill=color, font=big)
    if second:
        d.text((x, y + big.size + 6), second, fill=MUTED, font=small)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='results .json files or module .py files')
    ap.add_argument('--label', action='append', help='a column name, "NAME|note" (repeat, in order)')
    ap.add_argument('--files', help='comma-separated test file names for the rows')
    ap.add_argument('--tiff', action='append', help='a TIFF file of your own for a row (repeat for more)')
    ap.add_argument('--name', action='append', help='a row name, "NAME|note" (repeat, in order)')
    ap.add_argument('--tile', default='360x240', help='each picture\'s box, WIDTHxHEIGHT in pixels')
    ap.add_argument('--title', default='The Fax Test')
    ap.add_argument('--subtitle', default='The original as libtiff decodes it, then what each model-written decoder '
                    'returned for the same file')
    ap.add_argument('--note', default='github.com/blixild-jacek/fax-test')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'gallery.png'))
    ap.add_argument('--sandbox', default='auto', choices=('auto', 'sandbox-exec', 'docker', 'none'))
    a = ap.parse_args()
    sandbox = faxtest.pick_sandbox(a.sandbox)
    manifest = json.load(open(os.path.join(faxtest.CORPUS, 'manifest.json'), encoding='utf-8'))['files']
    rows = rows_from(a, manifest)
    tw, th = (int(v) for v in a.tile.lower().split('x'))
    modules = [best_module(p) for p in a.inputs]
    labels = (a.label or []) + [m[1] for m in modules[len(a.label or []):]]
    f_title, f_sub = font(44, True), font(21)
    f_head, f_note, f_cap = font(21, True), font(16), font(16)
    f_cap_bold, f_crash = font(16, True), font(17, True)
    M, GAP, LEFT, CAP = 48, 22, 230, 40                       # margin, gap between pictures, row names, caption
    top = M + 44 + 14 + 21 + 40                               # title, subtitle, space
    head_h = 21 + 6 + 16 + 18
    W = M + LEFT + len(labels + [0]) * (tw + GAP) - GAP + M
    H = top + head_h + len(rows) * (th + CAP + GAP) + 50 + M
    canvas = Image.new('RGB', (W, H), PAPER)
    d = ImageDraw.Draw(canvas)
    d.text((M, M), a.title, fill=INK, font=f_title)
    d.text((M, M + 44 + 14), a.subtitle, fill=MUTED, font=f_sub)
    cols = ['Original|as libtiff decodes it'] + labels
    for j, text in enumerate(cols):
        two_lines(d, M + LEFT + j * (tw + GAP), top, text, f_head, f_note)
    tmp = []
    try:
        paths = []
        for code, _ in modules:
            f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8')
            f.write(code)
            f.close()
            tmp.append(f.name)
            paths.append(f.name)
        for i, (name, tif, exp, w0, h0, c0) in enumerate(rows):
            y = top + head_h + i * (th + CAP + GAP)
            if i:
                d.line((M, y - GAP // 2, W - M, y - GAP // 2), fill=LINE, width=1)
            two_lines(d, M, y + 6, name, f_head, f_note)
            tiles = [(view(exp, w0, h0, c0), f'{w0} x {h0} pixels', 'original')]
            tiles += [outcome(p, tif, exp, w0, h0, c0, sandbox) for p in paths]
            for j, (im, caption, kind) in enumerate(tiles):
                x = M + LEFT + j * (tw + GAP)
                d.rectangle((x, y, x + tw - 1, y + th - 1), fill=EMPTY if im is None else TILE, outline=LINE)
                if im is not None:
                    im = fit(im, tw - 2, th - 2)
                    canvas.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
                    d.text((x, y + th + 8), caption, fill=GOOD if kind == 'exact' else (MUTED if kind == 'original'
                                                                                       else INK),
                           font=f_cap_bold if kind == 'exact' else f_cap)
                else:                                          # a crash: its message inside the box
                    d.text((x + 18, y + th // 2 - 34), 'Crashed', fill=INK, font=f_crash)
                    msg = wrap(caption, f_cap, tw - 36)[:3]
                    for k, line in enumerate(msg):
                        d.text((x + 18, y + th // 2 - 8 + k * 21), line, fill=MUTED, font=f_cap)
                    d.text((x, y + th + 8), 'No picture returned', fill=INK, font=f_cap)
    finally:
        for p in tmp:
            os.unlink(p)
    d.text((M, H - M - 18), a.note, fill=MUTED, font=f_note)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    canvas.save(a.out, optimize=True)
    print(f'{a.out}: {len(rows)} files x {len(modules)} decoders, {W} x {H} pixels')


if __name__ == '__main__':
    main()
