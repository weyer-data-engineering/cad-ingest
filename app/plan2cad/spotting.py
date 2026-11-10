"""Text spotting: a learned text detector and recogniser (PP-OCRv6 small, Apache-2.0 ONNX models run by RapidOCR on
ONNX Runtime) find and read the text lines of a sheet. Unlike the character grouping in ocr.py, the detector also
finds letters that touch a line or the ring of a bubble, and the recogniser reads plotter fonts, the slashed zero
and touching letters much better than Tesseract. Lines are found in overlapping tiles of the sheet."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cache

import cv2
import numpy as np

TILE = 1600  # px: square tiles of the sheet the detector reads ...
OVERLAP = 200  # ... overlapping so that every line lies whole in one of them
SAME = 0.6  # a line whose box lies this much inside a larger one is the same line found in the next tile
INK = 128  # tiles without a pixel darker than this are bare paper
FINE = 10.0  # px/mm: coarser sheets (phone photos) are enlarged to this, or the detector misses small text
EDGE = 3  # px: a box this close to the edge of its tile (not of the sheet) was cut there
TILT = 3.0  # degrees: a box turned further than this from the horizontal and the vertical holds an oblique line
LONG = 1.5  # ... where it is this much longer than high: the box of a character or two has no direction of its own
LEAN = 0.2  # share of its height a line along an axis may seem to lean over its length (see leeway)
LEVEL = 30.0  # degrees: a line this near the horizontal reads from left to right on an upright sheet
SURER = 0.02  # score by which a line read turned by 180 degrees must beat the usual way round to replace it
READ = 0.5  # score from which a reading is kept (RapidOCR's own default)
ABOUT = 0.1  # score by which a reading may fall short of another one and still be about as sure
RUN_ON = 0.5  # mm a long straight line runs on past the end of a text line's box: a pipe, not a stroke of a letter
MIDDLE = 0.5  # share of half its height off the middle of a box within which such a line runs through its letters
DEEP = 0.25  # share of its height a box must reach over such a line for that part of it to be the line's
BEYOND = 1.5  # share of its height up to which a line inside a ring may run out of it: a character or so beside it


def leeway(along: float, across: float) -> float:
    """Degrees off an axis up to which a line ``along`` long and ``across`` high (any one unit) may run although it
    lies along the axis: TILT, and more the shorter the line, as the few letters of a short word lean with the wavy
    paper of a photo, and a descender at one end tilts the box around them."""
    return TILT + math.degrees(math.atan2(LEAN * across, along))


def lean(quad: np.ndarray) -> tuple[float, float, float]:
    """The angle a turned box runs at along its longer side (-90 to 90 degrees, counter-clockwise), its length along
    that side and its height across it."""
    a, b = quad[1] - quad[0], quad[2] - quad[1]
    (dx, dy), short = (a, b) if np.hypot(*a) >= np.hypot(*b) else (b, a)
    return (math.degrees(math.atan2(-dy, dx)) + 90) % 180 - 90, float(np.hypot(dx, dy)), float(np.hypot(*short))


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
    corners: tuple = ()  # the detector's box turned with the line, four (x, y) in pixels
    angle: float | None = None  # where read both ways round: the direction it reads in, as Text.angle; else by its box
    turned: tuple[str, float] | None = None  # then also its reading turned by 180 degrees, and the score of that
    found: tuple[str, float] | None = None  # where read again (see _directed): its reading as found, and the score

    @property
    def vertical(self) -> bool:  # read bottom to top (or top to bottom, by its angle): the recogniser turns tall boxes
        if self.angle is not None:
            return self.angle % 180 == 90
        return self.y1 - self.y0 > LONG * (self.x1 - self.x0)

    @property
    def oblique(self) -> bool:
        return self.angle is not None and self.angle % 90 != 0

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def quad(self) -> np.ndarray:
        """The corners of its box, turned with the line where the detector found it so."""
        return np.asarray(self.corners or ((self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1),
                                           (self.x0, self.y1)), np.float32)


@cache
def _engine():
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    return RapidOCR(params={
        "Global.log_level": "error", "Global.use_cls": True,  # not a warning for every tile without text
        "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL, "Det.lang_type": LangRec.CH,
        "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.SMALL, "Rec.lang_type": LangRec.EN})


def spot(gray: np.ndarray, ppm: float, oblique: bool = False, both: bool = False) -> list[Spot]:
    """The text lines of a grayscale sheet (``ppm`` pixels per mm), each once, in its pixels. ``oblique``: a line at
    another angle than horizontal or vertical is read along its box, both ways round, and keeps the angle it reads at
    (as found, its box would be taken for a horizontal or vertical one). ``both``: so is a vertical line, which keeps
    the other reading too (see _reread); and the direction classifier is not asked, which now and then turns a short
    horizontal line upside down ("L+" read as "+7"): the sheet is upright, so its horizontal lines are."""
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
            # use_det, use_cls and text_score stick in RapidOCR from one call to the next: each call sets them. A box
            # read badly as found may read well along it, either way round (see _directed): kept for that
            r = _engine()(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR), use_det=True, use_cls=not both,
                          text_score=0.0 if oblique or both else READ)
            if not getattr(r, "txts", None):  # nothing found, or nothing read in what was found (boxes only)
                continue
            for box, text, score in zip(r.boxes, r.txts, r.scores):
                box = np.asarray(box, float) / f  # plain floats: they end up in report.json
                (x0, y0), (x1, y1) = box.min(0).tolist(), box.max(0).tolist()
                cut = (x0 < EDGE and x > 0) or (y0 < EDGE and y > 0) or (x1 > tw - EDGE and x + tw < w) or \
                    (y1 > th - EDGE and y + th < h)
                corners = tuple(map(tuple, (box + (x, y)).tolist()))
                s = Spot(x0 + x, y0 + y, x1 + x, y1 + y, text, float(score), cut, corners)
                if score >= READ or (oblique and _slant(s.quad) is not None) or (both and s.vertical):
                    found.append(s)
    spots = _once(_whole(gray, f, found, not both), oblique)
    if oblique or both:  # without the classifier, a vertical line comes upside down as often as not
        spots = [d for s in spots if (d := _directed(gray, f, s, oblique, both)).score >= READ or s.score >= READ]
    return _joined(gray, f, spots) if both else spots


def _joined(gray: np.ndarray, f: float, spots: list[Spot]) -> list[Spot]:
    """Vertical lines the detector found in pieces whose boxes overlap ("8," and ",82" of a vertical "8,82": the box
    of a short piece is about as wide as high, so it may have been read as horizontal), read again as one line, both
    ways round. A piece is a vertical line, or a box about as wide as high in the column of one that reads about as
    surely turned to vertical, and as many letters and digits: a horizontal tag at the end of a vertical label is no
    piece of it ("M1" turned reads "M"). Where the line reads as nothing whole, or less surely than its pieces did, the
    pieces stay as they were found."""
    def squat(s: Spot) -> bool:
        return not s.vertical and s.x1 - s.x0 <= LONG * (s.y1 - s.y0)

    turned: dict[int, Spot] = {}

    def piece(k: int) -> bool:  # read turned only where it would join a vertical line: few boxes
        if spots[k].vertical:
            return True
        if k not in turned:
            turned[k] = _reread(gray, f, spots[k], 90.0)
        s, t = spots[k], turned[k]
        return t is not s and t.score >= s.score - ABOUT and _alnum(t.text) >= _alnum(s.text)

    up = [k for k, s in enumerate(spots) if s.vertical or squat(s)]
    parent = {k: k for k in up}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def overlap(a: Spot, b: Spot) -> bool:  # along the line, level across it
        across = min(a.x1, b.x1) - max(a.x0, b.x0)
        return min(a.y1, b.y1) > max(a.y0, b.y0) and across >= 0.5 * min(a.x1 - a.x0, b.x1 - b.x0)

    for n, i in enumerate(up):
        for j in up[n + 1:]:
            if (spots[i].vertical or spots[j].vertical) and overlap(spots[i], spots[j]) and piece(i) and piece(j):
                parent[root(i)] = root(j)
    pieces: dict[int, list[int]] = {}
    for k in up:
        pieces.setdefault(root(k), []).append(k)
    out = [s for k, s in enumerate(spots) if k not in parent or len(pieces[root(k)]) == 1]
    for g in (g for g in pieces.values() if len(g) > 1):
        quad = cv2.boxPoints(cv2.minAreaRect(np.vstack([spots[k].quad for k in g])))
        (x0, y0), (x1, y1) = quad.min(0).tolist(), quad.max(0).tolist()
        whole = _reread(gray, f, Spot(x0, y0, x1, y1, "", 0.0, corners=tuple(map(tuple, quad.tolist())), angle=90.0),
                        90.0)
        least = min(spots[k].score for k in g)
        out += [whole] if whole.text.strip() and whole.score >= least - SURER else [spots[k] for k in g]
    return out


def off_pipes(gray: np.ndarray, ppm: float, spots: list[Spot], lines: list[tuple[float, float, float, float]]
              ) -> tuple[list[Spot], list[Spot]]:
    """The spotted lines without their ends along which a pipe runs (Features.text_on_pipes): the bars of a flange
    across a pipe read as "H", a flange or two below a vertical line label read with it ("HH2"-P-30207"). A long
    straight line (``lines``: x0, y0, x1, y1 in pixels of the sheet, at ``ppm`` pixels per mm) that comes from RUN_ON
    beyond an end of the box and runs on along its middle a good way into it takes that end, and what lies between it
    and the next such line, a valve between two flanges; the rest of the box is read again, both ways round. A line
    with no rest is dropped, and given apart: the marks on a signal line read as "xxxx" are linework, and so are the
    letters another reader found in its box. A line whose rest reads as nothing is dropped. Oblique lines stay."""
    out, gone = [], []
    for s in spots:
        part = None if s.oblique else _uncovered(s, lines, ppm)
        if part is None:
            out.append(s)
        elif part[1] - part[0] < DEEP * _across(s):  # no room for a character
            gone.append(s)
        else:
            angle = s.angle if s.angle is not None else 90.0 if s.vertical else 0.0
            cut = _part(s, *part)
            if (read := _reread(gray, max(1.0, FINE / ppm), cut, angle)) is not cut:
                out.append(read)
    return out, gone


def off_rings(gray: np.ndarray, ppm: float, spots: list[Spot], rings: list[tuple[float, float, float, float]]
              ) -> list[Spot]:
    """The spotted lines inside a drawn ring without the ends that run out of it (Features.attributes_inside): the bars
    of a flange against the ring of an instrument bubble, outside it, read as "H" or "T" with the number inside it. A
    line whose middle lies inside one of the ``rings`` (cx, cy, r, width of the stroke in pixels of the sheet) and
    whose middle runs out across the ring's stroke at an end, by more than DEEP but no more than BEYOND of its height,
    is cut there at the inner edge of the stroke and the part inside read again, both ways round; where it reads as
    nothing, the line stays as found. A tag that runs close to the ring has no more than the margin of its box beyond
    it; a line that runs further out is no text inside the ring. Oblique lines stay as found."""
    out = []
    for s in spots:
        part = None if s.oblique else _inside(s, rings)
        if part is None:
            out.append(s)
            continue
        angle = s.angle if s.angle is not None else 90.0 if s.vertical else 0.0
        cut = _part(s, *part)
        read = _reread(gray, max(1.0, FINE / ppm), cut, angle)
        out.append(s if read is cut else read)
    return out


def _inside(s: Spot, rings: list[tuple[float, float, float, float]]) -> tuple[float, float] | None:
    """The part of a box along its line (from, to in pixels) inside the ring that its middle lies in and that it runs
    a little out of (see off_rings), or None where it lies in no such ring."""
    lo, hi = (s.y0, s.y1) if s.vertical else (s.x0, s.x1)
    c0, c1 = (s.x0, s.x1) if s.vertical else (s.y0, s.y1)
    mid, height = (lo + hi) / 2, c1 - c0
    for cx, cy, r, width in rings:
        # the ring's centre along the line, and how far the line's middle passes it
        a, off = (cy, (c0 + c1) / 2 - cx) if s.vertical else (cx, (c0 + c1) / 2 - cy)
        inner, outer = (_chord(r + side * width / 2, off) for side in (-1, 1))  # half the chord on the middle line
        if r < height or inner == 0 or abs(mid - a) > inner:  # a ring no wider than two lines is a letter "O"
            continue
        low, high = (a - outer - lo, hi - a - outer)  # how far each end runs out across the stroke
        if DEEP * height < max(low, high) <= BEYOND * height:
            return a - inner if low > DEEP * height else lo, a + inner if high > DEEP * height else hi
    return None


def _chord(r: float, off: float) -> float:
    """Half the chord of a circle of radius ``r`` along a line ``off`` its centre (0 where the line misses it)."""
    return math.sqrt(r * r - off * off) if abs(off) < r else 0.0


def _across(s: Spot) -> float:
    return s.x1 - s.x0 if s.vertical else s.y1 - s.y0


def _uncovered(s: Spot, lines: list[tuple[float, float, float, float]], ppm: float) -> tuple[float, float] | None:
    """The part of a box along its line (from, to in pixels) that no long straight line takes from its ends (see
    off_pipes), or None where none does."""
    lo, hi = (s.y0, s.y1) if s.vertical else (s.x0, s.x1)
    runs = along(lines, s.vertical, lo, hi, s.x0 if s.vertical else s.y0, s.x1 if s.vertical else s.y1)
    deep, gap = DEEP * _across(s), _across(s)  # a valve between two flanges is about as long as the box is high
    start, end = lo, hi
    for a, b in runs:  # from the low end on
        if (a <= lo - RUN_ON * ppm and b >= lo + deep) or (start > lo and a <= start + gap):
            start = max(start, b)
    for a, b in reversed(sorted(runs, key=lambda r: r[1])):  # from the high end back
        if (b >= hi + RUN_ON * ppm and a <= hi - deep) or (end < hi and b >= end - gap):
            end = min(end, a)
    return None if (start, end) == (lo, hi) else (start, max(start, end))


def along(lines: list[tuple[float, float, float, float]], vertical: bool, lo: float, hi: float, c0: float, c1: float
          ) -> list[tuple[float, float]]:
    """The stretches (from, to along it) of the straight ``lines`` (x0, y0, x1, y1) that run along a box from ``lo``
    to ``hi`` (vertical: in y) within MIDDLE of the middle of its height from ``c0`` to ``c1``, overlapping it."""
    mid, half, out = (c0 + c1) / 2, (c1 - c0) / 2, []
    for x0, y0, x1, y1 in lines:
        (a, b), (c, d) = ((y0, y1), (x0, x1)) if vertical else ((x0, x1), (y0, y1))
        a, b = min(a, b), max(a, b)
        if abs(d - c) <= 0.1 * (b - a) and abs((c + d) / 2 - mid) <= MIDDLE * half and a < hi and b > lo:
            out.append((a, b))
    return sorted(out)


def _part(s: Spot, start: float, end: float) -> Spot:
    """The box of a spotted line from ``start`` to ``end`` along it, read as nothing yet."""
    x0, y0, x1, y1 = (s.x0, start, s.x1, end) if s.vertical else (start, s.y0, end, s.y1)
    return Spot(x0, y0, x1, y1, "", 0.0, corners=((x0, y0), (x1, y0), (x1, y1), (x0, y1)), angle=s.angle)


def _alnum(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _directed(gray: np.ndarray, f: float, s: Spot, oblique: bool, both: bool) -> Spot:
    """The line read both ways round where it may read another way than its box says: an oblique one along its box,
    a vertical one bottom to top, and a short one whose characters stand one above the other although its box is about
    as wide as high, such as a wire number (see _stands), where it reads about as well so and as a character for each
    of them: the two barbs of a flow arrow on a horizontal pipe stand so too, and read as one letter ("A"). Read again,
    a LEVEL line keeps its reading as found too where it still reads from left to right, as RapidOCR read it along its
    box (a steeper box it may have cut out tall and turned to read downwards): where its letters run along an axis
    after all, that reading may stay (ocr._level). So does a vertical line: where it only reads the same again, the
    spotter's doubt of it as found stays (ocr.spotted)."""
    if oblique and (angle := _slant(s.quad)) is not None:
        again = _reread(gray, f, s, angle)
        same_way = abs(angle) < LEVEL and again.angle == angle % 360
        return replace(again, found=(s.text, s.score)) if same_way else again
    if both and s.vertical:
        again = _reread(gray, f, s, 90.0)
        return again if again is s else replace(again, found=(s.text, s.score))
    if both and (n := _stands(gray, s)) and (turned := _reread(gray, f, s, 90.0)).score >= s.score - SURER and \
            sum(ch.isalnum() for ch in turned.text) >= n:
        return turned
    return s


def _slant(quad: np.ndarray) -> float | None:
    """The angle a turned box runs at (see lean), where that is more than TILT off the horizontal and the vertical and
    the box is LONG; None for the others. Read along it, its letters tell whether it is oblique (ocr.spotted)."""
    angle, along, across = lean(quad)
    return angle if min(abs(angle), 90 - abs(angle)) > TILT and along >= LONG * across else None


def _stands(gray: np.ndarray, s: Spot) -> int:
    """How many characters stand one above the other in a box about as wide as high (0 if they do not): a short
    vertical line, such as a wire number, as its box is no taller than that of a short horizontal one. Strokes that run
    out of the box, such as a wire beside its number, do not count. A single shape tells nothing: a flow arrow lies on
    its side like a digit turned, and read so it is a letter."""
    w, h = s.x1 - s.x0, s.y1 - s.y0
    if w > LONG * h or h > LONG * w:
        return 0
    # ink against the paper of the box itself: the toner of a poor scan is fainter than any fixed threshold
    _, crop = cv2.threshold(gray[int(s.y0):math.ceil(s.y1), int(s.x0):math.ceil(s.x1)], 0, 1,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, _, stats, centres = cv2.connectedComponentsWithStats(crop, connectivity=8)
    height, width = crop.shape
    inside = [k for k in range(1, n) if stats[k, 0] > 0 and stats[k, 1] > 0 and stats[k, 0] + stats[k, 2] < width
              and stats[k, 1] + stats[k, 3] < height]
    if not inside:
        return 0
    glyphs = [k for k in inside if stats[k, 4] >= 0.25 * max(stats[j, 4] for j in inside)]  # not dots
    return len(glyphs) if len(glyphs) > 1 and np.ptp(centres[glyphs, 1]) > LONG * np.ptp(centres[glyphs, 0]) else 0


def _reread(gray: np.ndarray, f: float, s: Spot, angle: float) -> Spot:
    """The line read along its box both ways round (at ``angle`` and turned by 180 degrees), without the direction
    classifier, which takes "0,60" upside down for "09'0" now and then: it reads at the angle of the surer reading
    (the turned one only where SURER, and where the line is LEVEL only if the other reads as nothing: an upright sheet
    writes it from left to right, and "501" leaning a little reads upside down as "10%" now and then), and keeps the
    other one too. Read as nothing so, it stays as found."""
    readings = []
    for a in (angle, angle + 180):
        r = _engine()(cv2.cvtColor(_upright(gray, s.quad, a, f), cv2.COLOR_GRAY2BGR), use_det=False, use_cls=False,
                      text_score=READ)
        readings.append((r.txts[0], float(r.scores[0])) if getattr(r, "txts", None) else ("", 0.0))
    (text, score), turned = readings
    if turned[1] > score + SURER and (abs((angle + 90) % 180 - 90) >= LEVEL or not text.strip()):
        (text, score), turned, angle = turned, (text, score), angle + 180
    if not text.strip():
        return s
    return replace(s, text=text, score=score, angle=angle % 360, turned=turned if turned[0].strip() else None)


def _upright(gray: np.ndarray, quad: np.ndarray, angle: float, f: float) -> np.ndarray:
    """The box ``quad`` cut out of the sheet so that a line in it at ``angle`` (as Text.angle) reads from left to
    right, ``f`` times as large."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    along, across = quad[:, 0] * c - quad[:, 1] * s, quad[:, 0] * s + quad[:, 1] * c
    a0, c0 = float(along.min()), float(across.min())
    size = (max(1, math.ceil((along.max() - a0) * f)), max(1, math.ceil((across.max() - c0) * f)))
    # a pixel of the cut-out, along and across the line, back to the sheet
    m = np.float32([[c / f, s / f, a0 * c + c0 * s], [-s / f, c / f, -a0 * s + c0 * c]])
    return cv2.warpAffine(gray, m, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def _whole(gray: np.ndarray, f: float, found: list[Spot], cls: bool = True) -> list[Spot]:
    """Lines longer than the overlap of the tiles come in pieces cut by a tile edge: the pieces of one line, one of
    them cut, are read again as one (``cls``: with the direction classifier)."""
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
        r = _engine()(cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_GRAY2BGR), use_det=False, use_cls=cls,
                      text_score=READ)
        if getattr(r, "txts", None):  # a classifier-only answer when nothing was read
            corners = tuple(map(tuple, cv2.boxPoints(cv2.minAreaRect(np.vstack([s.quad for s in g]))).tolist()))
            out.append(Spot(x0, y0, x1, y1, r.txts[0], float(r.scores[0]), corners=corners))  # oblique: see _reread
    return out


def _one_line(a: Spot, b: Spot) -> bool:
    """Whether two boxes are pieces of one text line: side by side along it (overlapping or close), level across it."""
    if a.vertical != b.vertical:
        return False
    (a0, a1, b0, b1), (c0, c1, d0, d1) = ((a.y0, a.y1, b.y0, b.y1), (a.x0, a.x1, b.x0, b.x1)) if a.vertical else \
        ((a.x0, a.x1, b.x0, b.x1), (a.y0, a.y1, b.y0, b.y1))
    across = min(c1, d1) - max(c0, d0)
    return across >= 0.5 * min(c1 - c0, d1 - d0) and max(a0, b0) - min(a1, b1) <= 0.5 * max(c1 - c0, d1 - d0)


def _once(found: list[Spot], oblique: bool = False) -> list[Spot]:
    """Drops a line found again in the next tile: the larger box (the line whole) is kept. ``oblique``: an oblique line
    is measured by its turned box, as the upright boxes of two lines of an oblique label overlap much."""
    kept: list[tuple[Spot, np.ndarray | None]] = []  # with its turned box where it is oblique
    for s in sorted(found, key=lambda s: -s.area):
        quad = s.quad if oblique and _slant(s.quad) is not None else None  # once per line: the pairs are many
        area = s.area if quad is None else float(cv2.contourArea(quad))
        if all(_overlap(s, k, quad, q) < SAME * area for k, q in kept):
            kept.append((s, quad))
    return [s for s, _ in kept]


def _overlap(a: Spot, b: Spot, qa: np.ndarray | None = None, qb: np.ndarray | None = None) -> float:
    """The area two boxes share: their turned boxes ``qa``, ``qb`` where either is oblique, else their upright ones.
    A turned box lies in its upright one, so only boxes whose upright ones overlap are turned."""
    upright = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    if not upright or (qa is None and qb is None):
        return upright
    return float(cv2.intersectConvexConvex(a.quad if qa is None else qa, b.quad if qb is None else qb)[0])
