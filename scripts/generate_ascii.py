#!/usr/bin/env python3
"""Turn a cutout portrait into an animated ASCII-art SVG.

The image is downsampled to a character grid, each cell mapped to a density
character by (contrast-normalised) luminance, and drawn as monospace text
with a left-to-right, top-to-bottom wipe reveal (SMIL, since GitHub strips
<script> from READMEs). The font is embedded as a base64 woff2 subset (only
the ramp characters actually used), because the SVG is loaded through <img>
and can't fetch external subresources.

Usage:
    python3 generate_ascii.py <input.png with alpha> <output.svg>

Input should already have its background removed (RGBA, transparent bg).
"""
import base64
import os
import sys

import numpy as np
from fontTools import subset
from fontTools.ttLib import TTFont
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_SRC = os.path.join(HERE, "fonts", "JetBrainsMono-Regular.ttf")
FONT_OUT = os.path.join(HERE, "fonts", "jbmono-ramp.woff2")

# Sparse -> dense. Letters included (not just punctuation/symbols) so the
# ramp has enough steps to hold real tonal range at this resolution --
# matching the reference, which reads clearly as a face despite being
# built from dense, letter-heavy noise rather than a handful of symbols.
RAMP = " .`'\",:;-~^\"<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYXZO0Qmwqpdbkhao*#MW&8%B@$"

COLS = 130            # character columns -- dense, matching the reference
CHAR_ADV = 0.600      # JetBrains Mono advance width, in em
FONT_SIZE = 6.6
ROW_H = FONT_SIZE * 1.05
LEFT = 10
TOP = 10
REVEAL_TOTAL = 1.6    # seconds

LIGHT_INK = "#57606a"
DARK_INK = "#c9d1d9"

# Background speckle: outside the subject's silhouette, scatter a sparse
# noise texture instead of leaving flat blank space -- this is most of what
# gives the reference its "gritty photo" read rather than a clean sticker
# cutout. Density and character weight are both low so it stays a texture,
# not a second subject.
BG_NOISE_DENSITY = 0.10
BG_NOISE_CHARS = ".`'\",:;-~^"


def auto_levels(lum, mask, lo_pct=1.0, hi_pct=99.0):
    """Stretch the opaque pixels' luminance to the full 0-255 range.

    Without this, a portrait shot in ordinary indoor light (a narrow
    real-world luminance band, e.g. 40-190) maps almost entirely into the
    middle of any ramp and reads as a flat grey block instead of showing
    real tonal structure -- across faces the "same ramp with no per-image
    normalisation" bug is silent until you look at the render.
    """
    vals = lum[mask]
    if vals.size == 0:
        return lum
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi <= lo:
        return lum
    out = (lum - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def build_grid(png_path, seed=7):
    img = Image.open(png_path).convert("RGBA")
    w, h = img.size
    char_w_px = w / COLS
    cell_aspect = ROW_H / (FONT_SIZE * CHAR_ADV)
    char_h_px = char_w_px * cell_aspect
    rows = max(1, round(h / char_h_px))

    small = img.resize((COLS, rows), Image.LANCZOS)
    arr = np.array(small).astype(float)
    rgb, alpha = arr[..., :3], arr[..., 3]
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # A low threshold here matters specifically for hair: fine strands
    # anti-alias down to low-but-nonzero alpha at their edges, and a
    # stricter cutoff (e.g. 40) discards exactly the wispy detail a
    # background-removal pass worked to keep.
    mask = alpha > 12

    norm = auto_levels(lum, mask)
    density = 1.0 - norm  # dark subject areas -> dense characters

    rng = np.random.default_rng(seed)
    grid = []
    for y in range(rows):
        row = []
        for x in range(COLS):
            if mask[y, x]:
                idx = min(len(RAMP) - 1,
                          int(density[y, x] * (len(RAMP) - 1) + 0.5))
                ch = RAMP[idx]
                row.append(ch if ch != " " else RAMP[1])
            else:
                if rng.random() < BG_NOISE_DENSITY:
                    row.append(rng.choice(list(BG_NOISE_CHARS)))
                else:
                    row.append(" ")
        grid.append(row)
    return grid, rows


def embed_font(chars):
    if os.path.exists(FONT_OUT):
        os.remove(FONT_OUT)
    font = TTFont(FONT_SRC)
    subsetter = subset.Subsetter()
    subsetter.populate(text="".join(sorted(set(chars))))
    subsetter.subset(font)
    font.flavor = "woff2"
    font.save(FONT_OUT)
    with open(FONT_OUT, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return b64


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def render(grid, rows, out_path):
    all_chars = "".join(ch for row in grid for ch in row) or " "
    font_b64 = embed_font(all_chars)
    width = LEFT * 2 + COLS * FONT_SIZE * CHAR_ADV
    height = TOP * 2 + rows * ROW_H

    family = ("JBMono,ui-monospace,SFMono-Regular,Menlo,Consolas,"
              "&apos;Liberation Mono&apos;,monospace")

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="{family}">'
    )
    parts.append(
        f"<style>@font-face{{font-family:JBMono;font-style:normal;"
        f"font-weight:400;font-display:block;"
        f"src:url(data:font/woff2;base64,{font_b64}) format('woff2')}}"
        f".a{{fill:{LIGHT_INK}}}"
        f"@media(prefers-color-scheme:dark){{.a{{fill:{DARK_INK}}}}}</style>"
    )

    row_dur = REVEAL_TOTAL / max(rows, 1)
    row_w = COLS * FONT_SIZE * CHAR_ADV
    for r, row in enumerate(grid):
        line = "".join(row).rstrip()
        if not line:
            continue
        y = TOP + r * ROW_H
        text_y = y + FONT_SIZE * 0.95
        delay = r * row_dur
        cid = f"c{r}"
        parts.append(
            f'<clipPath id="{cid}"><rect x="{LEFT}" y="{y:.1f}" '
            f'height="{ROW_H:.1f}" width="0">'
            f'<animate attributeName="width" from="0" to="{row_w:.1f}" '
            f'begin="{delay:.3f}s" dur="{row_dur:.3f}s" fill="freeze"/>'
            f'</rect></clipPath>'
        )
        parts.append(
            f'<g clip-path="url(#{cid})"><text xml:space="preserve" '
            f'x="{LEFT}" y="{text_y:.1f}" class="a" '
            f'font-size="{FONT_SIZE}">{esc(line)}</text></g>'
        )

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"{out_path}: {COLS}x{rows} grid, {width:.0f}x{height:.0f}px, "
          f"font {len(font_b64)//1024}KB base64, "
          f"{len(set(all_chars))} unique chars")


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: generate_ascii.py <input_rgba.png> <output.svg>")
    grid, rows = build_grid(sys.argv[1])
    render(grid, rows, sys.argv[2])


if __name__ == "__main__":
    main()
