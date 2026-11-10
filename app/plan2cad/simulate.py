"""Simulated scans and phone photos of a drawing, rendered from its vectors (used by the benchmark and by
`cad-ingest library check`)."""

from __future__ import annotations

import math

import cv2
import numpy as np

from dataclasses import replace

from .model import Drawing
from .render import render

MARKUP = "markup"  # the kind of the polylines of a drawing that someone added by hand on the print


def sheet(d: Drawing, ppm: float) -> np.ndarray:
    """The printed sheet at ``ppm`` pixels per mm, grey, or in colour where the drawing has markups added by hand
    (its polylines of the kind MARKUP, drawn over the print in their colour and pen widths: black without one). Pen
    ink is translucent: the print under it stays dark."""
    marks = [pl for pl in d.polylines if pl.kind == MARKUP]
    img = render(replace(d, polylines=[pl for pl in d.polylines if pl.kind != MARKUP]) if marks else d, ppm)
    if not marks:
        return img
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pen = np.full_like(img, 255)
    for pl in marks:
        pts = np.round(np.array(pl.points + (pl.points[:1] if pl.closed else [])) * ppm * 16).astype(np.int32)
        colour = tuple(int(c) for c in reversed(pl.color)) if pl.color else (0, 0, 0)
        cv2.polylines(pen, [pts], False, colour, max(1, round(pl.width * ppm)), cv2.LINE_AA, 4)
    return np.minimum(img, pen)


def plot(d: Drawing, dpi: float = 300) -> np.ndarray:
    """A clean raster plot of the drawing, as exported from CAD: the sheet only, no edge, no noise."""
    img = sheet(d, dpi / 25.4)
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def scan(d: Drawing, seed: int, dpi: float = 300, blur: float = 0.5, poor: bool = False, skew: float = 0.8,
         lid: float | None = None) -> np.ndarray:
    """Flatbed/large format scan: rotated by up to ``skew`` degrees, the paper edge visible against the scanner lid.
    A ``poor`` one is blurred, grey on grey with specks, as an old copy scanned in a hurry on an old copier with a dark
    lid. ``lid`` is the grey of the lid (0 to 255; the paper is 222 on a poor scan and 244 otherwise); about as grey
    as the paper, colour does not tell the two apart (see sheet.on_lid)."""
    rng = np.random.default_rng(seed)
    ppm = dpi / 25.4
    page = sheet(d, ppm)
    m = round(8 * ppm)
    h, w = page.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-skew, skew), 1.0)
    M[:, 2] += m + rng.uniform(-3, 3, 2) * ppm
    size = (w + 2 * m, h + 2 * m)
    paper = cv2.warpAffine(np.ones((h, w), np.uint8), M, size, flags=cv2.INTER_NEAREST).astype(bool)
    g = cv2.warpAffine(page, M, size, flags=cv2.INTER_LINEAR, borderValue=255).astype(np.float32)
    del page
    if g.ndim == 3:
        paper = paper[..., None]
    ink, white = (95.0, 222.0) if poor else (38.0, 244.0)
    img = np.where(paper, ink + (white - ink) * g / 255, lid if lid is not None else 150.0 if poor else 205.0)
    if poor:  # specks of dirt and toner, and a greyer middle where the copy was pressed less
        specks = (rng.random(img.shape[:2]) < 2e-4).astype(np.uint8)
        dark = cv2.dilate(specks, np.ones((3, 3), np.uint8)).astype(bool)
        img[dark] = ink
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        fade = 18 * np.exp(-(((xx - img.shape[1] / 2) / img.shape[1]) ** 2 + ((yy - img.shape[0] / 2) / img.shape[0])
                              ** 2) * 4)
        img -= fade[..., None] if img.ndim == 3 else fade
    img = cv2.GaussianBlur(img, (0, 0), 1.1 if poor else blur) + rng.normal(0, 7 if poor else 3, img.shape).astype(
        np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return _jpeg(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 80 if poor else 92)


def photo(d: Drawing, seed: int, camera: tuple[int, int] = (4032, 3024), hard: bool = False) -> np.ndarray:
    """Phone photo of the printed sheet on a desk.

    Simulates perspective and lens distortion, a wavy (in hard mode also folded) sheet, faded and
    uneven toner, uneven light with shadow and glare (hard), depth-of-field and hand-shake blur,
    sensor noise, the phone's sharpening and JPEG compression."""
    rng = np.random.default_rng(seed)
    cw, ch = camera
    W, H = d.width, d.height
    fill = rng.uniform(0.70, 0.78) if hard else rng.uniform(0.80, 0.88)
    s = fill * min(cw / W, ch / H)  # camera pixels per mm
    rot = math.radians(rng.uniform(-7, 7) if hard else rng.uniform(-3, 3))
    jit = 0.07 if hard else 0.03
    base = np.array([[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]]) * s
    base += rng.uniform(-jit, jit, (4, 2)) * W * s
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    corners = base @ R.T + (cw / 2, ch / 2) + rng.uniform(-0.02, 0.02, 2) * (cw, ch)

    r = min(1.6 * s * (1 + 2 * jit), 14.0)  # render above the photo resolution, then sample down
    page = cv2.GaussianBlur(sheet(d, r), (0, 0), max(0.5, 0.45 * r / s))

    # One sampling map per camera pixel: lens distortion -> inverse perspective -> wavy paper.
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    k1 = rng.uniform(-0.01, 0.01) if hard else rng.uniform(-0.004, 0.004)  # after the phone's own correction
    nx, ny = (xx - cw / 2) / (cw / 2), (yy - ch / 2) / (cw / 2)
    f = 1 + k1 * (nx * nx + ny * ny)
    u, v = nx * f * (cw / 2) + cw / 2, ny * f * (cw / 2) + ch / 2
    del nx, ny, f
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    den = M[2, 0] * u + M[2, 1] * v + M[2, 2]
    sx = ((M[0, 0] * u + M[0, 1] * v + M[0, 2]) / den).astype(np.float32)  # sheet mm
    sy = ((M[1, 0] * u + M[1, 1] * v + M[1, 2]) / den).astype(np.float32)
    del u, v, den
    amp = rng.uniform(1.0, 2.0) if hard else rng.uniform(0.4, 0.8)  # paper waviness in mm
    dx, dy = amp * _field(rng, sx, sy, W, H), amp * _field(rng, sx, sy, W, H)
    shade = np.ones_like(sx)
    if hard:  # a fold across the sheet: one side tilted away from the light, a darker crease
        vertical = bool(rng.integers(2))
        dist = (sx - W * rng.choice([1 / 3, 1 / 2, 2 / 3])) if vertical else (sy - H * rng.choice([1 / 3, 1 / 2]))
        shade -= 0.06 / (1 + np.exp(-dist / 3)) + 0.12 * np.exp(-((dist / 0.8) ** 2))
        bump = 0.8 * np.tanh(dist / 10)
        if vertical:
            dx += bump
        else:
            dy += bump
        del dist, bump
    g = cv2.remap(page, ((sx + dx) * r).astype(np.float32), ((sy + dy) * r).astype(np.float32),
                  cv2.INTER_LINEAR, borderValue=255).astype(np.float32) / 255
    del page, dx, dy
    mask = np.clip(0.5 + np.minimum(np.minimum(sx, W - sx), np.minimum(sy, H - sy)) * s, 0, 1)[..., None]
    toner = 1 - (0.3 if hard else 0.15) * (0.5 + 0.5 * _field(rng, sx, sy, W, H))
    del sx, sy

    iv = rng.uniform(65, 95) if hard else rng.uniform(50, 75)
    ink, paper = np.float32([iv, iv * 0.96, iv * 0.93]), np.float32([226, 236, 243])  # BGR
    desk = np.float32(rng.choice([[60, 80, 110], [90, 90, 92], [40, 50, 70], [150, 150, 155]]))
    grain = cv2.resize(rng.uniform(0.8, 1.2, (12, 16)).astype(np.float32), camera, interpolation=cv2.INTER_CUBIC)
    dark = (1 - g) * (toner[..., None] if g.ndim == 3 else toner)  # per colour channel where markups are in colour
    sheet_rgb = (paper - (paper - ink) * (dark if g.ndim == 3 else dark[..., None])) * shade[..., None]
    img = sheet_rgb * mask + desk * grain[..., None] * (1 - mask)
    del g, dark, toner, grain, mask, shade, sheet_rgb

    X, Y = xx * (2 / cw) - 1, yy * (2 / ch) - 1
    del xx, yy
    gx, gy = rng.uniform(-1, 1, 2) * (0.22 if hard else 0.1)
    light = (0.95 + gx * X + gy * Y) * (1 - 0.18 * (X * X + Y * Y))
    if hard:  # soft shadow of the photographer across one side of the sheet
        a = rng.uniform(0, 2 * math.pi)
        side = ((X * math.cos(a) + Y * math.sin(a)) > rng.uniform(0.1, 0.45)).astype(np.float32)
        light *= 1 - rng.uniform(0.3, 0.42) * cv2.GaussianBlur(side, (0, 0), 0.03 * cw)
        del side
    img *= light[..., None]
    if hard:  # glare from a lamp washes out a round area
        cx_, cy_ = rng.uniform(-0.6, 0.6, 2)
        glare = rng.uniform(0.45, 0.7) * np.exp(-((X - cx_) ** 2 + (Y - cy_) ** 2) / (2 * rng.uniform(0.15, 0.25) ** 2))
        img += (255 - img) * glare[..., None]
        del glare
    # Depth of field: the far side of the tilted sheet is softer than the near side.
    a = rng.uniform(0, 2 * math.pi)
    t = np.clip(0.5 + 0.5 * (X * math.cos(a) + Y * math.sin(a)), 0, 1)[..., None]
    del X, Y, light
    s1, s2 = (rng.uniform(1.3, 1.6), rng.uniform(1.9, 2.4)) if hard else (rng.uniform(0.9, 1.1), rng.uniform(1.3, 1.6))
    img = cv2.GaussianBlur(img, (0, 0), s1) * (1 - t) + cv2.GaussianBlur(img, (0, 0), s2) * t
    del t
    length, a = (rng.uniform(2, 5) if hard else rng.uniform(0.5, 2.5)), rng.uniform(0, math.pi)
    k = np.zeros((15, 15), np.float32)
    cv2.line(k, (round((7 - length / 2 * math.cos(a)) * 16), round((7 - length / 2 * math.sin(a)) * 16)),
             (round((7 + length / 2 * math.cos(a)) * 16), round((7 + length / 2 * math.sin(a)) * 16)), 1.0, 1,
             cv2.LINE_AA, 4)
    img = cv2.filter2D(img, -1, k / k.sum())
    img += rng.normal(0, 5.5 if hard else 3.5, img.shape).astype(np.float32)
    img += (0.9 if hard else 0.6) * (img - cv2.GaussianBlur(img, (0, 0), 1.6))  # phone sharpening
    return _jpeg(np.clip(img, 0, 255).astype(np.uint8), 72 if hard else 85)


def _field(rng: np.random.Generator, x: np.ndarray, y: np.ndarray, W: float, H: float) -> np.ndarray:
    """Smooth random field over the sheet with values in about [-1, 1] (a few long sine waves)."""
    out = np.zeros_like(x)
    for _ in range(3):
        fx, fy = rng.uniform(0.3, 1.8, 2) * rng.choice([-1, 1], 2)
        out += np.sin(2 * math.pi * (fx * x / W + fy * y / H) + rng.uniform(0, 2 * math.pi))
    return out / 3


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
