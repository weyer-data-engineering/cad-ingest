"""End to end: photo or scan in, vector drawing in sheet millimetres out."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import library, objects, ocr, sheet, symbols, topology, vectorize
from .model import PAPER, Drawing, Line, Text

MAX_PIXELS = 80e6  # upper bound for the rectified sheet image
SCALES = {1, 2, 5, 10, 20, 25, 50, 75, 100, 125, 200, 250, 500, 1000, 2000, 2500, 5000}


@dataclass
class Result:
    drawing: Drawing
    image: np.ndarray  # rectified, illumination-normalised sheet (grayscale)
    ppm: float  # pixels per mm of ``image``
    info: dict = field(default_factory=dict)


class _Timer:
    def __init__(self):
        self.t, self.laps = time.perf_counter(), {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round(self.laps.get(name, 0) + now - self.t, 2)
        self.t = now


def convert(img: np.ndarray, paper: str = "A3", scale: int | None = None, dpi: float | None = None,
            ppm: float | None = None, drawing: str | None = None,
            templates: dict[str, library.Template] | None = None) -> Result:
    """Converts a BGR photo or scan of a drawing printed on ``paper`` (A0 to A4).

    ``scale`` is the drawing scale denominator (read from the title block if omitted), ``dpi`` the
    scan resolution for scans without visible paper edges, ``ppm`` forces the working resolution,
    ``drawing`` the kind of drawing ("building" or "pid", recognised from the texts if omitted),
    ``templates`` the symbols to recognise (library.combine; the starter library if omitted).
    """
    timer = _Timer()
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found = sheet.locate(bgr, PAPER[paper])
    if found:
        src = sheet.source_ppm(found)
        out = ppm or _working_ppm(src, found.size)
        if src > 1.3 * out:  # avoid aliasing when shrinking a very sharp image
            gray = cv2.GaussianBlur(gray, (0, 0), 0.4 * src / out)
        image, size = sheet.rectify(gray, found, out), found.size
    else:
        gray = sheet.deskew(gray)
        src = dpi / 25.4 if dpi else max(gray.shape) / max(PAPER[paper])
        size = (gray.shape[1] / src, gray.shape[0] / src)
        out = ppm or _working_ppm(src, size)
        image = cv2.resize(gray, (round(size[0] * out), round(size[1] * out)),
                           interpolation=cv2.INTER_AREA if out < src else cv2.INTER_CUBIC)
    timer.lap("sheet")
    norm = normalize(image, out)
    ink = binarize(norm, out)
    timer.lap("binarize")
    rotation = 0
    parts = _vectorize(ink, norm, out, timer, check_rotation=True)
    if isinstance(parts, int):  # text is sideways or upside down: turn the sheet and start again
        rotation = parts
        k = -rotation // 90  # np.rot90 turns counter-clockwise
        norm, ink = np.ascontiguousarray(np.rot90(norm, k)), np.ascontiguousarray(np.rot90(ink, k))
        if rotation in (90, 270):
            size = (size[1], size[0])
        parts = _vectorize(ink, norm, out, timer, check_rotation=False)
    polygons, lines, arcs, circles, texts, (linework, text_mask, straight) = parts
    scale = scale or detect_scale([t for t in texts if not t.uncertain])
    drawing_type = drawing or objects.drawing_type(texts, circles, scale)
    swings, lines, arcs = objects.doors(lines, arcs) if drawing_type == "building" else ([], lines, arcs)
    # below 3.5 px per mm lines fall apart into short pieces themselves: no crumbs are dropped
    polylines, lines, arcs = topology.join(lines, arcs, topology.CRUMB if src >= 3.5 else 0.0)
    d = Drawing(size[0], size[1], lines, arcs, circles, polygons, texts, scale, swings + polylines)
    objects.classify(d, drawing_type)
    timer.lap("objects")
    if drawing_type != "building":  # symbol libraries hold P&ID symbols
        templates = library.load() if templates is None else templates
        if templates:
            symbols.recognise(d, templates, symbols.ink_samples(linework, text_mask, out, straight))
        timer.lap("symbols")
    info = {
        "paper": paper,
        "sheet": found.method if found else "image",
        "sheet_size_mm": [round(v, 1) for v in size],
        "corners_px": found.corners.round(1).tolist() if found else None,
        "source_ppm": round(src, 2),
        "working_ppm": round(out, 2),
        "rotation_deg": rotation,
        "drawing_type": drawing_type or None,
        "scale": d.scale,
        "counts": d.counts(),
        "timings_s": timer.laps,
    }
    return Result(d, norm, out, info)


def _vectorize(ink: np.ndarray, norm: np.ndarray, ppm: float, timer: _Timer, check_rotation: bool):
    polygons, fill = vectorize.fills(ink, ppm)
    ink = ink & (1 - fill)
    hv, used, crossing = vectorize.axis_lines(ink, ppm)
    k = np.ones((2 * round(0.15 * ppm) + 1,) * 2, np.uint8)
    rem = ink & (1 - cv2.dilate(used, k))
    timer.lap("lines")
    texts, labels, rotation = ocr.find_texts(rem, norm, ppm, check_rotation)
    if rotation is None:
        rotation = title_block_rotation(hv, ink.shape[1] / ppm, ink.shape[0] / ppm)
    if rotation:
        return rotation
    timer.lap("text")
    round_, rings = vectorize.circles(ink, ppm, (labels > 0).astype(np.uint8))
    texts, text_mask = _off_rings(texts, labels, rings)
    hv = [ln for ln in hv if not vectorize.on_circle(ln, round_)]
    linework = rem | (ink & cv2.dilate(crossing, k))  # without the long straight lines
    lines, arcs, circles = vectorize.shapes(vectorize.trace(linework & (1 - text_mask) & (1 - rings), ppm), ppm)
    arcs, more = vectorize.merge_arcs(arcs)
    lines = vectorize.close_corners(vectorize.merge_lines(hv + lines))
    timer.lap("vectorize")
    return polygons, lines, arcs, round_ + circles + more, texts, (linework, text_mask, hv)


def _off_rings(texts: list[Text], labels: np.ndarray, rings: np.ndarray) -> tuple[list[Text], np.ndarray]:
    """Drops text lines lying mostly on circle rings (symbol arcs misread as letters, e.g. "N")
    and returns the remaining texts with the mask of their pixels."""
    on_ring = np.bincount(labels[rings > 0], minlength=len(texts) + 1)
    size = np.bincount(labels[labels > 0], minlength=len(texts) + 1)
    keep = (on_ring < 0.5 * size).astype(np.uint8)
    return [t for t, k in zip(texts, keep[1:]) if k], keep[labels]


def title_block_rotation(lines: list[Line], width: float, height: float) -> int:
    """Clockwise rotation that moves the title block to the bottom right (ISO 5457). The title
    block is the corner where most lines end on the drawing frame (its rows and column dividers);
    walls, pipes and dimension lines do not touch the frame."""
    W, H = width, height
    hs = [ln for ln in lines if ln.y1 == ln.y2]
    vs = [ln for ln in lines if ln.x1 == ln.x2]
    frame_h = [ln for ln in hs if abs(ln.x2 - ln.x1) >= 0.5 * W]
    frame_v = [ln for ln in vs if abs(ln.y2 - ln.y1) >= 0.5 * H]

    def on_frame(x: float, y: float) -> bool:
        return any(abs(y - f.y1) < 1.5 and min(f.x1, f.x2) - 1 <= x <= max(f.x1, f.x2) + 1 for f in frame_h) or \
            any(abs(x - f.x1) < 1.5 and min(f.y1, f.y2) - 1 <= y <= max(f.y1, f.y2) + 1 for f in frame_v)

    score = {0: 0, 90: 0, 180: 0, 270: 0}  # corner of the title block -> rotation that fixes it
    for ln in hs + vs:
        if ln in frame_h or ln in frame_v:
            continue
        for x, y in ((ln.x1, ln.y1), (ln.x2, ln.y2)):
            if (0.3 * W < x < 0.7 * W and 0.3 * H < y < 0.7 * H) or not on_frame(x, y):
                continue
            right, bottom = x > W / 2, y > H / 2
            score[{(True, True): 0, (False, False): 180, (True, False): 90, (False, True): 270}[(right, bottom)]] += 1
    best = max(score, key=score.get)
    others = max(v for k, v in score.items() if k != best)
    return best if score[best] >= 3 and score[best] >= 2 * others else 0


def _working_ppm(src: float, size: tuple[float, float]) -> float:
    """Keeps the source resolution, but at least 6 px/mm (thin lines need a few pixels) and at most
    what fits into MAX_PIXELS."""
    return round(min(max(src, 6.0), 20.0, math.sqrt(MAX_PIXELS / (size[0] * size[1]))), 2)


def normalize(gray: np.ndarray, ppm: float) -> np.ndarray:
    """Divides by the paper brightness to remove shadows, glare and gradients.

    A closing over 3 mm removes the strokes but follows soft shadow edges; dark islands that the
    closing keeps (filled areas, much darker than any shadow) take the wide-area paper level."""
    f = min(1.0, 2.0 / ppm)  # estimate the background at about 2 px/mm
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(np.float32)
    px = ppm * f

    def disk(mm: float) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, round(mm * px) | 1),) * 2)

    close = cv2.morphologyEx(small, cv2.MORPH_CLOSE, disk(3))
    wide = cv2.GaussianBlur(cv2.dilate(small, disk(12)), (0, 0), 6 * px)
    bg = cv2.GaussianBlur(np.where(close < 0.5 * wide, wide, close), (0, 0), 1.5 * px)
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(gray.astype(np.float32) * 255 / np.maximum(bg, 1), 0, 255).astype(np.uint8)


def binarize(norm: np.ndarray, ppm: float) -> np.ndarray:
    """Ink mask (0/1). The threshold sits 45 % of the way from paper to the darkest ink nearby, so
    faint lines (glare, light toner) survive; specks smaller than about 0.1 mm² are dropped and a
    3 mm band along the sheet edge, where printers do not print, is ignored."""
    f = min(1.0, 2.0 / ppm)
    small = cv2.resize(norm, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    k = max(3, round(15 * ppm * f) | 1)
    darkest = cv2.GaussianBlur(cv2.erode(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), k / 3)
    t = 255 - 0.45 * np.maximum(255 - darkest.astype(np.float32), 110)
    ink = (norm < cv2.resize(t, (norm.shape[1], norm.shape[0]), interpolation=cv2.INTER_LINEAR)).astype(np.uint8)
    b = round(3 * ppm)
    ink[:b], ink[-b:], ink[:, :b], ink[:, -b:] = 0, 0, 0, 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    speck = stats[:, cv2.CC_STAT_AREA] < max(4, (0.12 * ppm) ** 2)
    speck[0] = False
    ink[speck[lab]] = 0
    return ink


def detect_scale(texts: list[Text]) -> int | None:
    """Drawing scale from the title block, e.g. "Maßstab 1:50" or "M 1:100"."""
    hits = []
    for t in texts:
        labelled = bool(re.search(r"(?i:ma.{1,2}stab|scale)|\bM\b", t.text))  # "m" alone is metres
        # without a "Maßstab" label the colon is required; with it, OCR may have lost the colon
        for m in re.finditer(r"\b1\s*:\s*(\d{1,4})\b" if not labelled else r"\b1\s*[:;.,]?\s*(\d{1,4})\b", t.text):
            if int(m.group(1)) in SCALES:
                hits.append((labelled, t.confidence, int(m.group(1))))
    return max(hits)[2] if hits else None
