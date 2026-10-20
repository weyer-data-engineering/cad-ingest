"""Markups on a printed drawing: what someone added by hand in colour, such as notes, revision clouds, strike-outs and
lines drawn with a pen, and highlighter. They are kept apart from the print: read as lines, symbols or printed text
they would mangle both, and they mean something else (a change, a question, a part that is gone).

On a black and white print every stroke in strong colour is a markup. On a colour plot, where coloured lines are a
large share of the ink, they are print; markups on a colour plot are not told apart. Highlighter is colour that
covers areas; the print under it shows through. Pen strokes become polylines in their colour, highlighter becomes
transparent fills, and handwriting is read where Tesseract manages (marked as a reading to check)."""

from __future__ import annotations

import cv2
import numpy as np
import pytesseract

from .model import Color, Markup, Polygon, Polyline, Text

COLOURED = 64  # difference between the strongest and the weakest colour channel of ink in colour
COLOUR_PRINT = 0.25  # coloured strokes this share of the black ones or more: a colour plot, its colours are print
HIGHLIGHTER = 1.0  # mm: light colour at least this wide all over is highlighter, not a pen stroke
LIGHT = 140  # brightness of highlighter at least; pen strokes, mostly darker, are too thin to be highlighter
SPECK = 0.6  # mm: smaller specks of colour are fringes of black ink in the scan, not pen
NOTE = 3.0  # mm: pen strokes this close belong to one note
UNDER = 1.5  # mm: a printed text this close to a pen stroke is struck out or marked by it
READ = 50  # confidence (0-100) of Tesseract for a handwritten word to count
WORD = 12.0  # mm: a handwritten word is at most this high (Tesseract reads clouds as letters too)
HIGHLIGHT, PEN = 1, 2  # values of the markup map; pen colour k is PEN + k
LAYER, HIGHLIGHT_LAYER, READING_LAYER = "P2C-MARKUP", "P2C-HIGHLIGHT", "P2C-MARKUP-TEXT"


def split(bgr: np.ndarray, ppm: float) -> tuple[np.ndarray, np.ndarray | None, dict[int, Color]]:
    """The grey image to recognise the drawing on, and on a black and white print with markups their map (HIGHLIGHT,
    PEN + k for pen colour k, 0 elsewhere) with the colour of each value. In the grey image lines in colour on a
    colour plot are as dark as black ones (their darkest channel), and markups are gone (their brightest channel:
    a pen or a highlighter takes some colours away, black print all of them)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(bgr)
    darkest, brightest = cv2.min(cv2.min(b, g), r), cv2.max(cv2.max(b, g), r)
    coloured = brightest - darkest > COLOURED
    if not coloured.any():
        return gray, None, {}
    width = _odd(HIGHLIGHTER * ppm)
    light = coloured & (gray >= LIGHT)
    broad = cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_OPEN, np.ones((width, width), np.uint8)) > 0
    if (coloured & ~broad).sum() >= COLOUR_PRINT * ((gray < 128) & ~coloured).sum():  # a colour plot: print
        np.copyto(gray, darkest, where=coloured)
        return gray, None, {}
    np.copyto(gray, brightest, where=coloured)
    marks = np.zeros(gray.shape, np.uint8)
    colours = {}
    pen = coloured
    hue = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 0]
    if broad.any():  # highlighter: its broad strokes and what is of their colour and joins them
        shade = np.abs(hue.astype(np.int16) - int(np.median(hue[broad])))
        alike = (coloured & (np.minimum(shade, 180 - shade) < 20)).astype(np.uint8)  # a pen of another colour stays
        n, lab = cv2.connectedComponents(alike)
        strokes = np.isin(lab, np.unique(lab[broad & (alike > 0)]))
        closed = cv2.morphologyEx(strokes.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3 * width, 3 * width), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(closed)  # the print shows through: its gaps closed
        keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= (3 * width) ** 2)
        area = np.isin(lab, keep[keep > 0])
        marks[area] = HIGHLIGHT
        colours[HIGHLIGHT] = _colour(bgr[broad])
        pen = pen & ~strokes & ~(area & (alike > 0))  # and its colour in the eyes of the letters it covers
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pen.astype(np.uint8), connectivity=8)
    big = np.flatnonzero(np.maximum(stats[:, 2], stats[:, 3]) >= SPECK * ppm)
    pen = np.isin(lab, big[big > 0])
    if pen.any():
        for k, (lo, hi) in enumerate(_hues(hue[pen])):
            here = pen & (((hue >= lo) & (hue < hi)) if lo < hi else ((hue >= lo) | (hue < hi)))
            marks[here] = PEN + k
            colours[PEN + k] = _colour(bgr[here])
    return gray, marks if marks.any() else None, colours


def extract(marks: np.ndarray, colours: dict[int, Color], ppm: float, lang: str, texts: list[Text]
            ) -> tuple[list[Polyline], list[Polygon], list[Text], list[Markup]]:
    """The markups of a map in sheet coordinates (``ppm`` pixels per mm): pen strokes as polylines, highlighter as
    filled polygons, readings of handwriting as texts, and one Markup per note and per highlighted area, naming the
    printed ``texts`` it covers (by their centre, so a symbol's tag counts where the symbol is): highlighted, inside a
    pen loop (a cloud) or under a pen stroke (struck out)."""
    polylines, polygons, readings, markups = [], [], [], []
    order = sorted(texts, key=lambda t: (round(t.y / 5), t.x))

    def covers(area: np.ndarray, x: int, y: int) -> list[str]:
        h, w = area.shape
        return [t.text for t in order if 0 <= int(t.y * ppm) - y < h and 0 <= int(t.x * ppm) - x < w
                and area[int(t.y * ppm) - y, int(t.x * ppm) - x]]

    if HIGHLIGHT in colours:
        n, lab, stats, _ = cv2.connectedComponentsWithStats((marks == HIGHLIGHT).astype(np.uint8))
        for i in range(1, n):
            x, y, w, h = stats[i, :4]
            area = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            contour = max(cv2.findContours(area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
            pts = (cv2.approxPolyDP(contour, 0.2 * ppm, True)[:, 0] + (x, y)) / ppm
            polygons.append(Polygon([tuple(p) for p in pts.tolist()], True, 0.0, colours[HIGHLIGHT], HIGHLIGHT_LAYER))
            markups.append(Markup("highlight", _box(x, y, w, h, ppm), colours[HIGHLIGHT], None, covers(area, x, y)))
    for value, colour in colours.items():
        if value < PEN:
            continue
        pen = (marks == value).astype(np.uint8)
        polylines += _strokes(pen, ppm, colour)
        reach = _odd(NOTE * ppm)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(pen, np.ones((reach, reach), np.uint8)))
        for i in range(1, n):
            x, y, w, h = stats[i, :4]
            strokes = pen[y:y + h, x:x + w] * (lab[y:y + h, x:x + w] == i).astype(np.uint8)
            text = _reading(strokes, lang, ppm)
            if text:
                (tx, ty, tw, th), words = text
                readings.append(Text((x + tx + tw / 2) / ppm, (y + ty + th / 2) / ppm, max(th / ppm, 1.0) * 0.7,
                                     words, uncertain=True, color=colour, layer=READING_LAYER))
            markups.append(Markup("note", _box(x, y, w, h, ppm), colour, text[1] if text else None,
                                  covers(_marked(strokes, ppm), x, y)))
    return polylines, polygons, readings, markups


def _marked(strokes: np.ndarray, ppm: float) -> np.ndarray:
    """The area pen strokes mark: inside the loops they draw (gaps in a hand-drawn cloud closed) and close to them."""
    near = cv2.dilate(strokes, np.ones((_odd(2 * UNDER * ppm),) * 2, np.uint8))
    shut = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((_odd(2 * NOTE * ppm),) * 2, np.uint8))
    outside = np.zeros((shut.shape[0] + 2, shut.shape[1] + 2), np.uint8)
    cv2.floodFill(1 - shut, outside, (0, 0), 1)  # what the paper around the loops reaches
    return (near > 0) | (outside[1:-1, 1:-1] == 0)


def crop(image: np.ndarray, marks: np.ndarray, colours: dict[int, Color], ppm: float, m: Markup,
         margin: float = 5.0) -> np.ndarray:
    """A colour image of a markup on the faded print around it, for reading what the pen wrote."""
    h, w = image.shape
    x0, y0, x1, y1 = (int(round(v * ppm)) for v in (m.box[0] - margin, m.box[1] - margin,
                                                       m.box[2] + margin, m.box[3] + margin))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    out = cv2.cvtColor((178 + image[y0:y1, x0:x1] * 0.3).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # as in the overlay
    here = marks[y0:y1, x0:x1]
    for value, (r, g, b) in colours.items():
        sel = here == value
        out[sel] = (0.5 * out[sel] + 0.5 * np.array([b, g, r])) if value == HIGHLIGHT else (b, g, r)
    return out


def _strokes(pen: np.ndarray, ppm: float, colour: Color) -> list[Polyline]:
    from .vectorize import trace  # the skeleton tracing of the drawing's own strokes

    out = []
    for pts, width in trace(pen, ppm):
        if np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < SPECK * ppm:
            continue
        closed = len(pts) > 3 and np.allclose(pts[0], pts[-1])
        simple = cv2.approxPolyDP(pts.astype(np.float32), max(1.0, 0.1 * ppm), closed)[:, 0] / ppm
        if len(simple) < 2:
            continue
        points = [tuple(p) for p in simple.tolist()]
        out.append(Polyline(points, [0.0] * (len(points) - (0 if closed else 1)), closed, width / ppm, False,
                            "markup", colour, LAYER))
    return out


def _reading(pen: np.ndarray, lang: str, ppm: float) -> tuple[tuple[int, int, int, int], str] | None:
    """What Tesseract reads in a note's strokes, with the box of the words it is sure of; None if nothing."""
    img = cv2.copyMakeBorder(np.where(pen > 0, 0, 255).astype(np.uint8), 20, 20, 20, 20, cv2.BORDER_CONSTANT,
                             value=255)
    best = []
    for psm in (7, 6):  # one line of handwriting reads best as a line, a block of it as a block
        data = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
        words = [(left - 20, top - 20, wd, ht, t) for t, c, left, top, wd, ht in
                 zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"])
                 if float(c) >= READ and sum(ch.isalnum() for ch in t) >= 3 and ht <= WORD * ppm]
        if sum(len(w[4]) for w in words) > sum(len(w[4]) for w in best):
            best = words
    words = best
    if not words:
        return None
    x0, y0 = min(w[0] for w in words), min(w[1] for w in words)
    x1, y1 = max(w[0] + w[2] for w in words), max(w[1] + w[3] for w in words)
    return (x0, y0, x1 - x0, y1 - y0), " ".join(w[4] for w in words)


def _hues(hue: np.ndarray, bins: int = 12, least: float = 0.05) -> list[tuple[int, int]]:
    """Hue ranges (OpenCV hue, 0 to 180) of the pen colours: runs of 15-degree bins that each hold at least
    ``least`` of the pen pixels (red wraps around)."""
    counts = np.bincount(hue // (180 // bins), minlength=bins)[:bins]
    on = counts >= least * counts.sum()
    if on.all():
        return [(0, 180)]
    start = int(np.argmin(on))  # begin at a gap, so that a run never wraps around inside the loop
    out, run = [], None
    for k in range(start, start + bins):
        if on[k % bins] and run is None:
            run = k
        elif not on[k % bins] and run is not None:
            out.append(((run % bins) * 15, (k % bins) * 15))
            run = None
    if run is not None:
        out.append(((run % bins) * 15, (start % bins) * 15))
    return out


def _colour(pixels: np.ndarray) -> Color:
    b, g, r = np.median(pixels.reshape(-1, 3), axis=0)
    return int(r), int(g), int(b)


def _box(x: int, y: int, w: int, h: int, ppm: float) -> tuple[float, float, float, float]:
    return x / ppm, y / ppm, (x + w) / ppm, (y + h) / ppm


def _odd(px: float) -> int:
    return max(1, int(round(px)) | 1)
