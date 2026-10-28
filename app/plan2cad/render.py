"""Rasterises a Drawing; used for previews, the review overlay and the synthetic benchmark."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .model import Drawing, Line, Text

FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
CAP = 0.729  # cap height of DejaVu Sans relative to the font size
SHIFT = 4  # fixed point bits for sub-pixel drawing in OpenCV
DASH, GAP = 3.0, 1.5  # dash pattern on paper in mm


@lru_cache(maxsize=128)
def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), px)


def dashes(line: Line, size: float = 1.0) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Splits a dashed line into its visible dash segments, the pattern ``size`` times as long as DASH and GAP (a
    drawing printed smaller has shorter dashes)."""
    length = math.hypot(line.x2 - line.x1, line.y2 - line.y1)
    if length == 0:
        return []
    ux, uy = (line.x2 - line.x1) / length, (line.y2 - line.y1) / length
    out, s = [], 0.0
    while s < length:
        e = min(s + DASH * size, length)
        out.append(((line.x1 + ux * s, line.y1 + uy * s), (line.x1 + ux * e, line.y1 + uy * e)))
        s += (DASH + GAP) * size
    return out


def render(d: Drawing, ppm: float, img: np.ndarray | None = None, color=0, texts: bool = True) -> np.ndarray:
    """Draws the drawing at ``ppm`` pixels per mm, by default black on a new white grayscale sheet."""
    if img is None:
        img = np.full((round(d.height * ppm), round(d.width * ppm)), 255, np.uint8)
    d = d.flat()
    f = ppm * (1 << SHIFT)

    def p(x: float, y: float) -> tuple[int, int]:
        return round(x * f), round(y * f)

    def thick(w: float) -> int:
        return max(1, round(w * ppm))

    for ln in d.lines:
        for a, b in dashes(ln, d.printed) if ln.dashed else [((ln.x1, ln.y1), (ln.x2, ln.y2))]:
            cv2.line(img, p(*a), p(*b), color, thick(ln.width), cv2.LINE_AA, SHIFT)
    for a in d.arcs:
        end = a.end if a.end > a.start else a.end + 360
        r = round(a.r * f)
        cv2.ellipse(img, p(a.cx, a.cy), (r, r), 0, a.start, end, color, thick(a.width), cv2.LINE_AA, SHIFT)
    for c in d.circles:
        cv2.circle(img, p(c.cx, c.cy), round(c.r * f), color, thick(c.width), cv2.LINE_AA, SHIFT)
    for g in d.polygons:
        pts = np.array([p(x, y) for x, y in g.points], np.int32)
        if g.filled:
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA, SHIFT)
        else:
            cv2.polylines(img, [pts], True, color, thick(g.width), cv2.LINE_AA, SHIFT)
    if texts:
        for t in d.texts:
            draw_text(img, t, ppm, color)
    return img


def text_length(t: Text) -> float:
    """Length of the rendered text along its reading direction in mm."""
    px = 200
    left, _, right, _ = _font(px).getbbox(t.text, anchor="ls")
    return (right - left) / px * t.height / CAP


def draw_text(img: np.ndarray, t: Text, ppm: float, color=0) -> None:
    font = _font(max(6, round(t.height * ppm / CAP)))
    left, top, right, bottom = font.getbbox(t.text, anchor="ls")
    mask = Image.new("L", (right - left + 2, bottom - top + 2), 0)
    ImageDraw.Draw(mask).text((1 - left, 1 - top), t.text, font=font, fill=255, anchor="ls")
    m = np.asarray(mask, np.float32) / 255
    if t.length and m.shape[1] > t.length * ppm:  # no longer than printed
        m = cv2.resize(m, (max(1, round(t.length * ppm)), m.shape[0]), interpolation=cv2.INTER_AREA)
    # Anchor = centre of the capitals: half the cap height above the baseline.
    ax, ay = m.shape[1] / 2, (1 - top) - t.height * ppm / 2
    if t.angle == 90:
        m = np.rot90(m)  # counter-clockwise: reads bottom to top
        ax, ay = ay, m.shape[0] - ax
    blend(img, m, round(t.x * ppm - ax), round(t.y * ppm - ay), color)


def blend(img: np.ndarray, alpha: np.ndarray, x0: int, y0: int, color) -> None:
    """Alpha-blends a solid colour through ``alpha`` into ``img`` at (x0, y0), clipped to the image."""
    h, w = alpha.shape
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    a = alpha[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0]
    region = img[y1:y2, x1:x2].astype(np.float32)
    if region.ndim == 3:
        a = a[..., None]
    img[y1:y2, x1:x2] = (region * (1 - a) + np.asarray(color, np.float32) * a).astype(img.dtype)
