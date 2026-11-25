# Copyright (c) 2026 Jacek Garbuliński Blixild, blixild.com
# SPDX-License-Identifier: MIT
"""The grader's check: a small standard-library decoder that handles only 8-bit gray or RGB in strips, uncompressed or
Deflate (8, 32946), Predictor 1 or 2, either byte order. Must pass exactly those files (faxtest.py selftest lists them)."""
import struct
import zlib


def decode(data):
    e = '<' if data[:2] == b'II' else '>'
    ifd = struct.unpack(e + 'I', data[4:8])[0]
    tags = {}
    for i in range(struct.unpack(e + 'H', data[ifd:ifd + 2])[0]):
        tag, typ, n, val = struct.unpack(e + 'HHI4s', data[ifd + 2 + 12 * i:ifd + 14 + 12 * i])
        size = {1: 1, 3: 2, 4: 4, 5: 8}.get(typ, 1) * n
        raw = val[:size] if size <= 4 else data[struct.unpack(e + 'I', val)[0]:][:size]
        fmt = {1: 'B', 3: 'H', 4: 'I'}.get(typ)
        tags[tag] = list(struct.unpack(e + fmt * n, raw)) if fmt else raw
    w, h = tags[256][0], tags[257][0]
    spp = tags.get(277, [1])[0]
    if tags.get(258, [1])[0] != 8 or tags.get(262, [1])[0] not in (1, 2) or 322 in tags \
            or tags.get(284, [1])[0] != 1:
        raise ValueError('not handled')
    comp = tags.get(259, [1])[0]
    rows = tags.get(278, [h])[0]
    out = bytearray()
    for off, cnt in zip(tags[273], tags[279]):
        chunk = data[off:off + cnt]
        if comp in (8, 32946):
            chunk = zlib.decompress(chunk)
        elif comp != 1:
            raise ValueError(f'compression {comp}')
        n = min(rows, h - len(out) // (w * spp))
        chunk = bytearray(chunk[:n * w * spp])
        if tags.get(317, [1])[0] == 2:
            for r in range(n):
                for i in range(r * w * spp + spp, (r + 1) * w * spp):
                    chunk[i] = (chunk[i] + chunk[i - spp]) & 255
        out += chunk
    return w, h, spp, bytes(out)
