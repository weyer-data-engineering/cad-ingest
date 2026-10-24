"""End to end: photo or scan in, vector drawing in sheet millimetres out; or a vector PDF page read as it is."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pypdfium2 as pdfium

from . import library, markup, objects, ocr, pdfvector, sheet, spotting, symbols, topology, vectorize
from .model import PAPER, Circle, Drawing, Line, Text

MAX_PIXELS = 80e6  # upper bound for the rectified sheet image
PDF_PPM = 300 / 25.4  # resolution of the review image of a vector PDF page
TEXT = 2.0  # mm: text height the printed-size symbols of the libraries go with (their tags are this high)
FAINT = 0.25  # share of the way from paper to the darkest ink nearby that a faint stretch of a stroke reaches
SCALES = {1, 2, 5, 10, 20, 25, 50, 75, 100, 125, 200, 250, 500, 1000, 2000, 2500, 5000}


@dataclass
class Result:
    drawing: Drawing
    image: np.ndarray  # rectified, illumination-normalised sheet (grayscale)
    ppm: float  # pixels per mm of ``image``
    info: dict = field(default_factory=dict)
    crops: list[np.ndarray] = field(default_factory=list)  # an image of each of drawing.markups (BGR)


class _Timer:
    def __init__(self):
        self.t, self.laps = time.perf_counter(), {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round(self.laps.get(name, 0) + now - self.t, 2)
        self.t = now


def convert(img: np.ndarray, paper: str = "A3", scale: int | None = None, dpi: float | None = None,
            ppm: float | None = None, drawing: str | None = None,
            libraries: list[library.Library] | None = None) -> Result:
    """Converts a BGR photo or scan of a drawing printed on ``paper`` (A0 to A4).

    ``scale`` is the drawing scale denominator (read from the title block if omitted), ``dpi`` the
    scan resolution for scans without visible paper edges, ``ppm`` forces the working resolution,
    ``drawing`` the kind of drawing ("building" or "pid", recognised from the texts if omitted),
    ``libraries`` the symbol libraries to recognise symbols from (library.select; the built-in ones if omitted);
    those for the kind of drawing are used, P&ID symbols on drawings of an unknown kind.
    """
    timer = _Timer()
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    found = sheet.locate(bgr, PAPER[paper])
    src = sheet.source_ppm(found) if found else dpi / 25.4 if dpi else max(bgr.shape[:2]) / max(PAPER[paper])
    gray, marks, colours = markup.split(bgr, src)  # markups (pen, highlighter) go their own way
    if found:
        out = ppm or _working_ppm(src, found.size)
        if src > 1.3 * out:  # avoid aliasing when shrinking a very sharp image
            gray = cv2.GaussianBlur(gray, (0, 0), 0.4 * src / out)
        image, size = sheet.rectify(gray, found, out), found.size
        if marks is not None:
            marks = sheet.rectify(marks, found, out, cv2.INTER_NEAREST)
    else:
        gray, turn = sheet.deskew(gray)
        size = (gray.shape[1] / src, gray.shape[0] / src)
        out = ppm or _working_ppm(src, size)
        image = cv2.resize(gray, (round(size[0] * out), round(size[1] * out)),
                           interpolation=cv2.INTER_AREA if out < src else cv2.INTER_CUBIC)
        if marks is not None:
            if turn is not None:
                marks = cv2.warpAffine(marks, turn, marks.shape[::-1], flags=cv2.INTER_NEAREST)
            marks = cv2.resize(marks, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
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
        marks = None if marks is None else np.ascontiguousarray(np.rot90(marks, k))
        if rotation in (90, 270):
            size = (size[1], size[0])
        parts = _vectorize(ink, norm, out, timer, check_rotation=False)
    polygons, lines, arcs, circles, texts, (linework, text_mask, straight, frame) = parts
    scale = scale or detect_scale([t for t in texts if not t.uncertain])
    drawing_type = drawing or objects.drawing_type(texts, circles, scale)
    swings, lines, arcs = objects.doors(lines, arcs) if drawing_type == "building" else ([], lines, arcs)
    # below 3.5 px per mm lines fall apart into short pieces themselves: no crumbs are dropped
    polylines, lines, arcs = topology.join(lines, arcs, topology.CRUMB if src >= 3.5 else 0.0)
    d = Drawing(size[0], size[1], lines, arcs, circles, polygons, texts, scale, swings + polylines)
    objects.classify(d, drawing_type)
    timer.lap("objects")
    libraries = library.select() if libraries is None else libraries
    used = [lib for lib in libraries if (drawing_type or "pid") in lib.drawing]
    if templates := library.combine(used, drawing_type or "pid", d.scale, _printed(texts)):
        symbols.recognise(d, templates, symbols.ink_samples(linework, text_mask, out, straight, frame))
        timer.lap("symbols")
    crops = []
    if marks is not None:  # after the recognition of the print, which they would only disturb
        tags = [Text(s.x, s.y, TEXT, " ".join(t.text for t in s.attributes.values()))
                for s in d.symbols if s.attributes]  # an instrument in a cloud is covered by it
        strokes, fills, readings, d.markups = markup.extract(marks, colours, out, ocr.LANG, d.texts + tags)
        d.polylines, d.polygons, d.texts = d.polylines + strokes, d.polygons + fills, d.texts + readings
        crops = [markup.crop(norm, marks, colours, out, m) for m in d.markups]
        timer.lap("markups")
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
        "libraries": [{"id": lib.id, "version": lib.version} for lib in used],
        "counts": d.counts(),
        "timings_s": timer.laps,
    }
    return Result(d, norm, out, info, crops)


def convert_vectors(pdf: pdfium.PdfDocument, page: pdfium.PdfPage, scale: int | None = None,
                    drawing: str | None = None) -> Result:
    """Reads a vector PDF page, such as a CAD export, as it is (see pdfvector): no image is recognised, so lines, arcs,
    texts and colours are exact. The page is the sheet. ``scale`` and ``drawing`` as in convert. No symbols are
    placed: they would replace strokes that are exact already."""
    timer = _Timer()
    d = pdfvector.read(pdf, page)
    timer.lap("vectors")
    ppm = _working_ppm(PDF_PPM, (d.width, d.height))
    image = np.asarray(page.render(scale=ppm * 25.4 / 72, grayscale=True).to_pil().convert("L"))  # for the review
    timer.lap("render")
    d.scale = scale or detect_scale(d.texts)
    drawing_type = drawing or objects.drawing_type(d.texts, d.circles, d.scale)
    timer.lap("objects")  # no classification: the entities stay as drawn, on their CAD layers
    paper = next((name for name, size in PAPER.items()
                  if max(abs(a - b) for a, b in zip(sorted(size), sorted((d.width, d.height)))) <= 2), None)
    info = {
        "paper": paper,
        "sheet": "page",
        "sheet_size_mm": [round(d.width, 1), round(d.height, 1)],
        "corners_px": None,
        "source_ppm": None,
        "working_ppm": round(ppm, 2),
        "rotation_deg": page.get_rotation() % 360,
        "drawing_type": drawing_type or None,
        "scale": d.scale,
        "libraries": [],
        "counts": d.counts(),
        "timings_s": timer.laps,
    }
    return Result(d, image, ppm, info)


def _vectorize(ink: np.ndarray, norm: np.ndarray, ppm: float, timer: _Timer, check_rotation: bool):
    polygons, fill = vectorize.fills(ink, ppm)
    ink = ink & (1 - fill)
    hv, used, crossing = vectorize.axis_lines(ink, ppm)
    k = np.ones((2 * round(0.15 * ppm) + 1,) * 2, np.uint8)
    rem = ink & (1 - cv2.dilate(used, k))
    timer.lap("lines")
    readable = rem & (1 - vectorize.joints(rem, hv, ppm))
    texts, labels, rotation = ocr.find_texts(readable, norm, ppm, check_rotation, ink)
    if rotation is None:
        rotation = title_block_rotation(hv, ink.shape[1] / ppm, ink.shape[0] / ppm)
    if rotation:
        return rotation
    round_, rings = vectorize.circles(ink, ppm, (labels > 0).astype(np.uint8), hv)
    texts, labels = _in_circles(texts, labels, readable, ink, round_, rings, norm, ppm)
    texts, labels = ocr.spotted(texts, labels, readable, norm, spotting.spot(norm, ppm), ppm)
    texts = ocr.grammar(texts)
    timer.lap("text")
    words = np.uint8([0] + [ocr.worded(t.text) for t in texts])[labels]
    texts, text_mask = _off_rings(texts, labels, rings)
    frame = ocr.frames(ink, labels * text_mask * words, ppm)  # not around a symbol's circles read as "oo"
    hv = [ln for ln in hv if not vectorize.on_circle(ln, round_)]
    linework = rem | (ink & cv2.dilate(crossing, k))  # without the long straight lines
    lines, arcs, circles = vectorize.shapes(vectorize.trace(linework & (1 - text_mask) & (1 - rings), ppm), ppm)
    arcs, more = vectorize.merge_arcs(arcs)
    lines = vectorize.bridge(vectorize.close_corners(vectorize.merge_lines(hv + lines)), arcs, norm, ppm)
    lines = vectorize.links(lines, arcs, round_ + circles + more, norm, ppm)
    timer.lap("vectorize")
    return polygons, lines, arcs, round_ + circles + more, texts, (linework, text_mask * (1 + words), hv, frame)


def _in_circles(texts: list[Text], labels: np.ndarray, readable: np.ndarray, ink: np.ndarray, circles: list[Circle],
                rings: np.ndarray, norm: np.ndarray, ppm: float) -> tuple[list[Text], np.ndarray]:
    """Reads the text in circles that hold text, such as instrument bubbles, a second time without the ring and the
    line across a shared display: letters that touch either were taken for part of the symbol. Texts read again
    replace the first reading of the same letters. Circles without text, such as the hotplates of a hob, stay as they
    are."""
    inner = np.zeros(readable.shape, np.uint8)
    for c in circles:
        x, y, r = round(c.cx * ppm), round(c.cy * ppm), round(c.r * ppm)
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(inner.shape[1], x + r + 1), min(inner.shape[0], y + r + 1)
        disk = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.circle(disk, (x - x0, y - y0), r, 1, -1)
        if (labels[y0:y1, x0:x1][disk > 0] > 0).any():
            inner[y0:y1, x0:x1] |= disk
    if not inner.any():
        return texts, labels
    mask = readable & inner & (1 - rings)
    across = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, round(2.5 * ppm)), 1))  # longer than any letter stroke
    mask &= 1 - cv2.morphologyEx(mask, cv2.MORPH_OPEN, across)
    first = np.bincount(labels[inner > 0], minlength=len(texts) + 1)
    whole = np.bincount(labels.ravel(), minlength=len(texts) + 1)
    outside = np.flatnonzero(first < 0.9 * whole)  # texts that run out of a circle keep their first reading
    mask[np.isin(labels, outside[outside > 0])] = 0
    more, more_labels, _ = ocr.find_texts(mask, norm, ppm, frame=ink)
    if not more:
        return texts, labels
    redone = set(np.unique(labels[more_labels > 0]).tolist()) - {0}
    keep = [k for k in range(1, len(texts) + 1) if k not in redone]
    lut = np.zeros(len(texts) + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    out = np.where(more_labels > 0, more_labels.astype(np.int32) + len(keep), lut[labels])
    return [texts[k - 1] for k in keep] + more, out


def _printed(texts: list[Text]) -> float:
    """How large the symbols of the sheet are against their library: the sheet's text height against the TEXT the
    library symbols are drawn for. 1 unless clearly different, as on a large drawing printed on a small sheet."""
    heights = [t.height for t in texts if not t.uncertain]
    if len(heights) < 20:
        return 1.0
    f = float(np.clip(np.median(heights) / TEXT, 0.5, 2.0))
    return 1.0 if 0.8 <= f <= 1.25 else f


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
    faint lines (glare, light toner) survive; fainter stretches of horizontal and vertical lines down to FAINT of
    the way are added (_faint). Specks smaller than about 0.1 mm² are dropped and a
    3 mm band along the sheet edge, where printers do not print, is ignored."""
    f = min(1.0, 2.0 / ppm)
    small = cv2.resize(norm, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    k = max(3, round(15 * ppm * f) | 1)
    darkest = cv2.GaussianBlur(cv2.erode(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), (0, 0), k / 3)
    contrast = np.maximum(255 - darkest.astype(np.float32), 110)
    size = (norm.shape[1], norm.shape[0])
    ink = (norm < cv2.resize(255 - 0.45 * contrast, size, interpolation=cv2.INTER_LINEAR)).astype(np.uint8)
    ink |= _faint(norm < cv2.resize(255 - FAINT * contrast, size, interpolation=cv2.INTER_LINEAR), ink, ppm)
    b = round(3 * ppm)
    ink[:b], ink[-b:], ink[:, :b], ink[:, -b:] = 0, 0, 0, 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    speck = stats[:, cv2.CC_STAT_AREA] < max(4, (0.12 * ppm) ** 2)
    speck[0] = False
    ink[speck[lab]] = 0
    return ink


def _faint(weak: np.ndarray, ink: np.ndarray, ppm: float) -> np.ndarray:
    """The faint stretches of horizontal and vertical lines (a pale pen, a line printed light, a stroke that came out
    dotted): straight runs of ``weak`` pixels as long as an axis line, added where the clear ``ink`` has no stroke
    across them, so that lines do not grow thicker. Only long runs count, so the fine details of a symbol drawn in
    light grey stay as they are, and neither runs thicker than a line (a grey area) nor stretches that touch a
    character-sized stroke (the blurred band through a line of text on a photo)."""
    weak = weak.astype(np.uint8)
    length = max(12, round(6 * ppm))
    reach = 2 * (max(1, round(0.3 * ppm)) + 2) + 1  # a clear stroke with its blurred edge
    blob = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, round(0.8 * ppm) | 1),) * 2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) <= 7.5 * ppm
    small[0] = False
    out = np.zeros_like(weak)
    for along, across, size in (((length, 1), (1, reach), stats[:, cv2.CC_STAT_HEIGHT]),  # (width, height)
                                ((1, length), (reach, 1), stats[:, cv2.CC_STAT_WIDTH])):
        run = cv2.morphologyEx(weak, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, along))
        run &= 1 - cv2.morphologyEx(run, cv2.MORPH_OPEN, blob)
        add = run & (1 - cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, across)))
        letters = cv2.dilate((small & (size > ppm))[lab].astype(np.uint8), np.ones((3, 3), np.uint8))
        m, pieces = cv2.connectedComponents(add, connectivity=8)
        touching = np.zeros(m, bool)
        touching[pieces[(letters > 0) & (add > 0)]] = True
        touching[0] = False
        out |= add & ~touching[pieces]
    return out


def detect_scale(texts: list[Text]) -> int | None:
    """Drawing scale from the title block, e.g. "Maßstab 1:50" or "M 1:100"."""
    hits = []
    for t in texts:
        labelled = bool(re.search(r"(?i:ma.{1,2}stab|scale)|\bM\b", t.text))  # "m" alone is metres
        # without a "Maßstab" label the colon is required; with it, OCR may have lost the colon or read digits as
        # letters ("Maßstab 1 SO")
        number = r"\b1\s*:\s*(\d{1,4})\b" if not labelled else r"\b[1Il]\s*[:;.,]?\s*([\dOoSIlBZ]{1,4})\b"
        for m in re.finditer(number, t.text):
            if (scale := int(ocr.digits(m.group(1)))) in SCALES:
                hits.append((labelled, t.confidence, scale))
    return max(hits)[2] if hits else None
