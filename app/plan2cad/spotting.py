"""Text spotting: a learned text detector and recogniser (PP-OCRv6 small, Apache-2.0 ONNX models run by RapidOCR on
ONNX Runtime) find and read the text lines of a sheet. Unlike the character grouping in ocr.py, the detector also
finds letters that touch a line or the ring of a bubble, and the recogniser reads plotter fonts, the slashed zero
and touching letters much better than Tesseract. Lines are found in overlapping tiles of the sheet."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np

TILE = 1600  # px: square tiles of the sheet the detector reads ...
OVERLAP = 200  # ... overlapping so that every line lies whole in one of them
SAME = 0.6  # a line whose box lies this much inside a larger one is the same line found in the next tile
INK = 128  # tiles without a pixel darker than this are bare paper
FINE = 10.0  # px/mm: coarser sheets (phone photos) are enlarged to this, or the detector misses small text
EDGE = 3  # px: a box this close to the edge of its tile (not of the sheet) was cut there


@dataclass
class Spot:
    """A text line: its box in pixels, what it reads and how sure the recogniser is (0 to 1)."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    score: float
    cut: bool = False  # by the edge of the tile it was found in

    @property
    def vertical(self) -> bool:  # read bottom to top: the recogniser turns tall boxes that way
        return self.y1 - self.y0 > 1.5 * (self.x1 - self.x0)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@cache
def _engine():
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(params={
        "Global.log_level": "error", "Global.use_cls": True,  # not a warning for every tile without text
        "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL, "Det.lang_type": LangRec.CH,
        "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.SMALL, "Rec.lang_type": LangRec.EN})


def spot(gray: np.ndarray, ppm: float) -> list[Spot]:
    """The text lines of a grayscale sheet (``ppm`` pixels per mm), each once, in its pixels."""
    f = max(1.0, FINE / ppm)
    size, step = round(TILE / f), round((TILE - OVERLAP) / f)
    h, w = gray.shape
    found = []
    for y in range(0, max(h - size + step, 1), step):
        for x in range(0, max(w - size + step, 1), step):
            tile = gray[y:y + size, x:x + size]
            if tile.min() > INK:
                continue
            th, tw = tile.shape
            if f > 1:
                tile = cv2.resize(tile, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            # use_det sticks in RapidOCR from one call to the next: _whole turns it off
            r = _engine()(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR), use_det=True)
            if not getattr(r, "txts", None):  # nothing found, or nothing read in what was found (boxes only)
                continue
            for box, text, score in zip(r.boxes, r.txts, r.scores):
                box = np.asarray(box, float) / f  # plain floats: they end up in report.json
                (x0, y0), (x1, y1) = box.min(0).tolist(), box.max(0).tolist()
                cut = (x0 < EDGE and x > 0) or (y0 < EDGE and y > 0) or (x1 > tw - EDGE and x + tw < w) or \
                    (y1 > th - EDGE and y + th < h)
                found.append(Spot(x0 + x, y0 + y, x1 + x, y1 + y, text, float(score), cut))
    return _once(_whole(gray, f, found))


def _whole(gray: np.ndarray, f: float, found: list[Spot]) -> list[Spot]:
    """Lines longer than the overlap of the tiles come in pieces cut by a tile edge: the pieces of one line, one of
    them cut, are read again as one."""
    parent = list(range(len(found)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(found):
        for j, b in enumerate(found):
            if i < j and (a.cut or b.cut) and _one_line(a, b):
                parent[root(i)] = root(j)
    groups: dict[int, list[Spot]] = {}
    for i, s in enumerate(found):
        groups.setdefault(root(i), []).append(s)
    out = []
    for g in groups.values():
        if len(g) == 1:
            out += g
            continue
        x0, y0 = int(min(s.x0 for s in g)), int(min(s.y0 for s in g))
        x1, y1 = int(max(s.x1 for s in g)) + 1, int(max(s.y1 for s in g)) + 1
        crop = gray[y0:y1, x0:x1]
        if g[0].vertical:
            crop = np.rot90(crop)  # read bottom to top
        if f > 1:
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        r = _engine()(cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_GRAY2BGR), use_det=False)
        if getattr(r, "txts", None):  # a classifier-only answer when nothing was read
            out.append(Spot(x0, y0, x1, y1, r.txts[0], float(r.scores[0])))
    return out


def _one_line(a: Spot, b: Spot) -> bool:
    """Whether two boxes are pieces of one text line: side by side along it (overlapping or close), level across it."""
    if a.vertical != b.vertical:
        return False
    (a0, a1, b0, b1), (c0, c1, d0, d1) = ((a.y0, a.y1, b.y0, b.y1), (a.x0, a.x1, b.x0, b.x1)) if a.vertical else \
        ((a.x0, a.x1, b.x0, b.x1), (a.y0, a.y1, b.y0, b.y1))
    across = min(c1, d1) - max(c0, d0)
    return across >= 0.5 * min(c1 - c0, d1 - d0) and max(a0, b0) - min(a1, b1) <= 0.5 * max(c1 - c0, d1 - d0)


def _once(found: list[Spot]) -> list[Spot]:
    """Drops a line found again in the next tile: the larger box (the line whole) is kept."""
    kept: list[Spot] = []
    for s in sorted(found, key=lambda s: -s.area):
        if all(_overlap(s, k) < SAME * s.area for k in kept):
            kept.append(s)
    return kept


def _overlap(a: Spot, b: Spot) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
