"""End to end: photo or scan in, vector drawing in sheet millimetres out; or a vector PDF page read as it is."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
import pypdfium2 as pdfium
from scipy.spatial import cKDTree

from . import (dimensions, fit, freehand, hatching, library, markup, objects, ocr, pdfvector, profiles, sheet,
               spotting, square, symbols, titleblock, topology, unclassified, vectorize)
from .features import Features
from .model import PAPER, Circle, Drawing, Line, Text

MAX_PIXELS = 80e6  # upper bound for the rectified sheet image
PDF_PPM = 300 / 25.4  # resolution of the review image of a vector PDF page
TEXT = 2.0  # mm: text height the printed-size symbols of the libraries go with (their tags are this high)
FAINT = 0.25  # share of the way from paper to the darkest ink nearby that a faint stretch of a stroke reaches
DRAWN = 2.2  # mm: the usual height of the characters of a drawing at its drawn size (2.5 mm by ISO 3098, 2 for tags)
SMALLER = 0.7  # a sheet whose characters are smaller than this share of DRAWN is a large drawing printed small
DRAWN_PPM = 5.0  # px per drawn mm from which a small print is vectorised at its drawn size (a photo of it is coarser)
SCALES = {1, 2, 5, 10, 20, 25, 50, 75, 100, 125, 200, 250, 500, 1000, 2000, 2500, 5000}
SIZE = 0.2  # a text whose height is within this share of a size the sheet is lettered in has it: 2 and 2.5 mm differ
SIZED = 3  # texts of one height that make a size
TIGHT = 0.85  # a text printed shorter than this share of the usual width of its characters is kept that short
NARROW_CHARS = " .,:;'\"()|!Iil1"  # characters about half as wide as the others
TOUCH = 0.3  # mm a pipe may stop short of the end of a text and still run into it
BESIDE = (0.25, 0.75)  # mm beside a pipe, past its edge, where the bars across it read as a letter lie on both sides


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
            ppm: float | None = None, drawing: str | None = None, libraries: list[library.Library] | None = None,
            features: Features | None = None, kinds: tuple[profiles.Profile, ...] | None = None) -> Result:
    """Converts a BGR photo or scan of a drawing printed on ``paper`` (A0 to A4).

    ``scale`` is the drawing scale denominator (read from the title block if omitted), ``dpi`` the
    scan resolution for scans without visible paper edges, ``ppm`` forces the working resolution,
    ``drawing`` the kind of drawing ("building", "pid", "electrical", "site", ...: the profiles' kinds), recognised
    from the texts by the profiles ``kinds`` if omitted (profiles.installed() if those are omitted),
    ``libraries`` the symbol libraries to recognise symbols from (library.select; the built-in ones if omitted);
    those for the kind of drawing are used, P&ID symbols on drawings of an unknown kind. ``features``: the steps
    switched on (features.py; all if omitted).
    """
    features = features or Features()
    timer = _Timer()
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    found = sheet.locate(bgr, PAPER[paper], features.paper_edge, dpi)
    src = sheet.source_ppm(found) if found else dpi / 25.4 if dpi else max(bgr.shape[:2]) / max(PAPER[paper])
    paper_edge = found.corners if found and found.method == "paper" else None
    gray, marks, colours = markup.split(bgr, src, features.colour_guard, paper_edge)  # markups go their own way
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
    # A large drawing printed small (an A1 plan on A3) is recognised at its drawn size, which the measures of
    # recognition are made for, and brought back to the size of the sheet at the end. Its lines and texts are found at
    # the drawn size too where the image has the pixels for it (a scan); a photo of a small print has too few and
    # falls apart into short pieces that way, so its lines are found on the sheet as it is.
    printed = printed_size(ink, out, features.letter_rows)
    first = printed if out * printed >= DRAWN_PPM else 1.0  # the size lines and texts are found at
    timer.lap("binarize")
    # Marks by hand in black or grey go their own way before anything is traced or read: lines and texts of the
    # drawing they are not, and the text recognition would read letters into them
    hand = freehand.find(ink, norm, out, printed * DRAWN * out, features.freehand_strokes,
                         features.leaders, features.pencil_guard) if features.freehand else None
    clean = norm
    if hand is not None and hand.any():
        ink, clean = ink * (hand == 0), np.where(hand > 0, np.uint8(255), norm)
        timer.lap("freehand")
    rotation = 0
    kinds = profiles.installed() if kinds is None else kinds
    libraries = library.select() if libraries is None else libraries
    early = _symbols_first(libraries, drawing, scale, kinds, first / printed, features) if features.symbols_first \
        else None
    parts = _vectorize(ink, clean, out * first, timer, True, first, features, early)
    if isinstance(parts, int):  # text is sideways or upside down: turn the sheet and start again
        rotation = parts
        k = -rotation // 90  # np.rot90 turns counter-clockwise
        norm, ink, clean = (np.ascontiguousarray(np.rot90(a, k)) for a in (norm, ink, clean))
        marks = None if marks is None else np.ascontiguousarray(np.rot90(marks, k))
        hand = None if hand is None else np.ascontiguousarray(np.rot90(hand, k))
        if rotation in (90, 270):
            size = (size[1], size[0])
        parts = _vectorize(ink, clean, out * first, timer, False, first, features, early)
    polygons, lines, arcs, circles, texts, (linework, text_mask, straight, frame), readings = parts
    scale = scale or detect_scale([t for t in texts if not t.uncertain])
    size, grow = (size[0] / first, size[1] / first), first / printed  # what is still to be brought to the drawn size
    drawn = Drawing(*size, circles=circles, texts=texts).scaled(grow)
    drawing_type = drawing or profiles.recognise(drawn.texts, drawn.circles, scale, kinds)
    # the steps from here on that the kind's profile switches off, such as dimensions on a P&ID (none are drawn on it)
    features, by_kind = features.for_kind(tuple(n for p in kinds if p.drawing == drawing_type for n in p.steps_off))
    texts = _settled(texts, readings, [p for p in kinds if p.drawing == drawing_type], features)
    if features.vocabulary:
        texts = _vocabulary(texts, [p for p in kinds if p.drawing == drawing_type])
    measured = []  # before lines are joined and connected: an extension line would join what it stops short of
    if features.dimensions:
        measured, lines, texts = dimensions.find(lines, texts, ink, out * first, printed / first, scale)
    swings, lines, arcs = objects.doors(lines, arcs) if drawing_type == "building" else ([], lines, arcs)
    # below 3.5 px per mm lines fall apart into short pieces themselves: no crumbs are dropped
    polylines, lines, arcs = topology.join(lines, arcs, topology.CRUMB if src * first >= 3.5 else 0.0,
                                           features.short_strokes)
    d = Drawing(size[0], size[1], lines, arcs, circles, polygons, texts, scale, swings + polylines,
                dimensions=measured).scaled(grow)
    objects.classify(d, drawing_type)
    timer.lap("objects")
    used = [lib for lib in libraries if (drawing_type or "pid") in lib.drawing]
    printed_by = {lib.id: _printed(_texts_and_values(d), lib.text_height or TEXT) for lib in used}
    if templates := library.combine(used, drawing_type or "pid", d.scale, printed_by):
        straight = Drawing(*size, lines=straight).scaled(grow).lines
        symbols.recognise(d, templates, symbols.ink_samples(linework, text_mask, out * printed, straight, frame),
                          features)
        timer.lap("symbols")
    topology.connect(d)
    if features.dedupe:
        d.texts = _deduped(d.texts)
    _lettering(d)
    if features.splines:  # before the title block is read: the straight chords of a curve in it would be its rulings
        objects.curves(d)
    if features.square_lines:  # after the curves, whose chords it would bend, and before the rulings of the title block
        square.square(d)
    if features.title_block:  # at the drawn size, which its measures are made for, like those of symbols
        d.title_block = titleblock.find(d)
        if d.title_block:
            titleblock.lay(d, d.title_block)
    if features.unclassified:  # after the title block, whose shapes are its own and whose linework is on its layer
        unclassified.find(d, unclassified.sizes(templates))
    if features.width_calibration:  # measured on the sheet as rotated: at the drawn size, its pixels are smaller
        vectorize.calibrate_widths(d, norm, out * printed)
    if features.dimension_fit:  # on the connected linework, once all of it is measured where the image shows it
        fit.to_values(d)
    if features.pens:
        objects.pens(d)
    timer.lap("connect")
    d = d.scaled(printed)
    d.printed, size = printed, (size[0] * first, size[1] * first)
    crops = []
    if marks is not None:  # after the recognition of the print, which they would only disturb
        tags = [Text(s.x, s.y, TEXT, " ".join(t.text for t in s.attributes.values()))
                for s in d.symbols if s.attributes]  # an instrument in a cloud is covered by it
        strokes, fills, readings, d.markups = markup.extract(marks, colours, out, ocr.LANG, _texts_and_values(d) + tags,
                                                             features.read_handwriting)
        d.polylines, d.polygons, d.texts = d.polylines + strokes, d.polygons + fills, d.texts + readings
        crops = [markup.crop(norm, marks, colours, out, m) for m in d.markups]
        timer.lap("markups")
    if hand is not None and (hand == 1).any():  # specks of print (freehand.SPECK) are out of the drawing, no marks
        hand = (hand == 1).astype(np.uint8)
        strokes, notes = freehand.marks(hand, norm, out, _texts_and_values(d))
        d.polylines, d.markups = d.polylines + strokes, d.markups + notes
        crops += [markup.crop(norm, hand * markup.PEN, {markup.PEN: m.color}, out, m) for m in notes]
    info = {
        "paper": paper,
        "sheet": found.method if found else "image",
        "sheet_size_mm": [round(v, 1) for v in size],
        "corners_px": found.corners.round(1).tolist() if found else None,
        "source_ppm": round(src, 2),
        "working_ppm": round(out, 2),
        "rotation_deg": rotation,
        "printed": printed,
        "drawing_type": drawing_type or None,
        "steps_off_by_kind": by_kind,
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
    drawing_type = drawing or profiles.recognise(d.texts, d.circles, d.scale, profiles.installed())
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
        "steps_off_by_kind": [],  # no step of the recognition runs on vectors
        "scale": d.scale,
        "libraries": [],
        "counts": d.counts(),
        "timings_s": timer.laps,
    }
    return Result(d, image, ppm, info)


def _vectorize(ink: np.ndarray, norm: np.ndarray, ppm: float, timer: _Timer, check_rotation: bool, printed: float,
               features: Features = Features(), early=None):
    """The vectors and texts of the sheet in millimetres at ``ppm`` pixels each: of the sheet, or of the drawing as
    drawn where it is printed smaller (``printed``, the share of its drawn size the sheet shows), with what the text
    recognition leaves open until the kind of drawing is known (ocr.Readings); or the clockwise rotation in degrees
    that makes the sheet's text upright, if asked to ``check_rotation`` and it is not. ``features``: the steps switched
    on (features.py); with ``short_strokes``, the pieces of short strokes cut by a line are joined (vectorize.bridge).
    ``early`` gives the areas of the symbols matched before the texts are taken as read (see _symbols_first)."""
    polygons, thick, fill = vectorize.fills(ink, ppm)
    strokes, ink = ink, ink & (1 - fill)  # all of it frames a single letter: the thick ring of a note mark too
    hv, used, crossing = vectorize.axis_lines(ink, ppm)
    k = np.ones((2 * round(0.15 * ppm) + 1,) * 2, np.uint8)
    rem = ink & (1 - cv2.dilate(used, k))
    linework = rem | (ink & cv2.dilate(crossing, k))  # without the long straight lines
    timer.lap("lines")
    readable = rem & (1 - vectorize.joints(rem, hv, ppm))
    readings = ocr.Readings(by_token=features.lookalike_tokens)
    reading = {"oblique": features.oblique_text, "both": features.text_direction, "readings": readings}
    texts, labels, rotation = ocr.find_texts(readable, norm, ppm, check_rotation, strokes, printed, **reading)
    if rotation is None:  # judged on the sheet, as its measures are meant: the frame of a small print is close
        rotation = title_block_rotation(Drawing(0, 0, lines=hv).scaled(printed).lines, ink.shape[1] / ppm * printed,
                                        ink.shape[0] / ppm * printed)
    if rotation:
        return rotation
    round_, rings = vectorize.circles(ink, ppm, (labels > 0).astype(np.uint8), hv)
    round_ = thick + round_
    areas = None
    if early is not None:
        timer.lap("text")
        areas = early(texts, labels, linework, [ln for ln in hv if not vectorize.on_circle(ln, round_)], round_, ppm)
        texts, labels = _off_symbols(texts, labels, areas)
        readable = readable & (1 - areas)
        timer.lap("symbols first")
    texts, labels = _in_circles(texts, labels, readable, ink, round_, rings, norm, ppm, reading)
    # the spotter enlarges a sheet as coarse as a phone photo, by its own pixels per mm: a scan has enough for it
    spots = _off_areas(spotting.spot(norm, ppm / printed, features.oblique_text, features.text_direction), areas)
    piped: list[spotting.Spot] = []  # spotted lines that are linework along a pipe throughout
    if features.text_on_pipes:
        spots, piped = spotting.off_pipes(norm, ppm / printed, spots, [(ln.x1 * ppm, ln.y1 * ppm, ln.x2 * ppm,
                                                                        ln.y2 * ppm) for ln in hv])
    if features.attributes_inside:  # the drawn rings: the template's ring a symbol is matched with may be larger
        spots = spotting.off_rings(norm, ppm / printed, spots, [(c.cx * ppm, c.cy * ppm, c.r * ppm, c.width * ppm)
                                                                for c in round_])
    texts, labels = ocr.spotted(texts, labels, readable, norm, spots, ppm, readings)
    texts = ocr.grammar(texts)
    if features.hatching:
        texts, labels = hatching.dropped(texts, labels)
    between = set()
    if features.text_on_pipes:
        texts, labels = _kept(texts, labels, [not _bars(t, hv, ink, ppm) and not any(
            s.x0 <= t.x * ppm <= s.x1 and s.y0 <= t.y * ppm <= s.y1 for s in piped) for t in texts])
        between = {k for k, t in enumerate(texts) if _piped(t, hv)}
    timer.lap("text")
    words = np.uint8([0] + [ocr.worded(t.text) and k not in between for k, t in enumerate(texts)])[labels]
    texts, text_mask = _off_rings(texts, labels, rings)
    frame = ocr.frames(ink, labels * text_mask * words, ppm)  # not around a symbol's circles read as "oo"
    hv = [ln for ln in hv if not vectorize.on_circle(ln, round_)]
    traced = linework & (1 - text_mask) & (1 - rings)
    dots, in_rows = vectorize.dotted(traced, norm, ppm) if features.linetypes else ([], None)
    if dots:  # too small to trace
        traced = traced & (1 - in_rows)
    largest = objects.LARGEST if features.splines else vectorize.LARGEST
    lines, arcs, circles = vectorize.shapes(vectorize.trace(traced, ppm), ppm, largest)
    arcs, more = vectorize.merge_arcs(arcs)
    lines = vectorize.merge_lines(hv + lines, one_frame=features.merge_where_drawn)
    lines = vectorize.bridge(vectorize.close_corners(lines), arcs, norm, ppm, features.short_strokes)
    lines = vectorize.links(lines, arcs, round_ + circles + more, norm, ppm)
    if features.linetypes:
        lines = vectorize.line_types(vectorize.dash_dots(lines, norm, ppm) + dots, norm, ppm)
    timer.lap("vectorize")
    return polygons, lines, arcs, round_ + circles + more, texts, (linework, text_mask * (1 + words), hv, frame), \
        readings


def _texts_and_values(d: Drawing) -> list[Text]:
    """The texts printed on the sheet, the values of its dimensions among them."""
    return d.texts + [m.text for m in d.dimensions if m.text]


def _symbols_first(libraries: list[library.Library], drawing: str | None, scale: int | None,
                   kinds: tuple[profiles.Profile, ...], grow: float, features: Features = Features()):
    """Symbols first (Features.symbols_first): a function that gives the areas of the symbols on the sheet, matched on
    its ink before the texts are taken as read, so that the text recognition leaves them out (see symbols.areas). It
    gets the texts read so far with their ``labels`` (from ocr.find_texts), which tell the kind of drawing, its scale
    and the size of its symbols (as in convert) and which strokes are letters; the ``linework`` mask without the long
    straight ``lines``, those lines and the ``circles``, all at ``ppm`` pixels per mm, which is ``grow`` times the
    drawn millimetre (the size the libraries' symbols are matched at)."""
    def areas(texts: list[Text], labels: np.ndarray, linework: np.ndarray, lines: list[Line], circles: list[Circle],
              ppm: float) -> np.ndarray:
        h, w = labels.shape
        drawn = Drawing(w / ppm, h / ppm, lines=lines, circles=circles, texts=texts).scaled(grow)
        sure = scale or detect_scale([t for t in texts if not t.uncertain])
        kind = drawing or profiles.recognise(drawn.texts, drawn.circles, sure, kinds) or "pid"
        used = [lib for lib in libraries if kind in lib.drawing]
        templates = library.combine(used, kind, sure,
                                    {lib.id: _printed(drawn.texts, lib.text_height or TEXT) for lib in used})
        if not templates:
            return np.zeros(labels.shape, np.uint8)
        straight = list(drawn.lines)
        objects.classify(drawn, "")  # the frame, beside which the title block is left out
        words = np.uint8([0] + [ocr.worded(t.text) for t in texts])[labels]
        ink = symbols.ink_samples(linework, (labels > 0) * (1 + words), ppm / grow, straight)
        return symbols.areas(drawn, templates, ink, ppm / grow, labels.shape, features)

    return areas


def _off_symbols(texts: list[Text], labels: np.ndarray, areas: np.ndarray) -> tuple[list[Text], np.ndarray]:
    """The texts without those read mostly on the areas of symbols (symbol strokes read as letters), with the labels
    of the rest renumbered; what of a text lies on such an area is no longer its."""
    on = np.bincount(labels[areas > 0], minlength=len(texts) + 1)
    size = np.bincount(labels.ravel(), minlength=len(texts) + 1)
    keep = [k for k in range(1, len(texts) + 1) if on[k] < 0.5 * size[k]]
    lut = np.zeros(len(texts) + 1, labels.dtype)
    lut[keep] = np.arange(1, len(keep) + 1)
    return [texts[k - 1] for k in keep], np.where(areas > 0, 0, lut[labels]).astype(labels.dtype)


def _off_areas(spots: list[spotting.Spot], areas: np.ndarray | None) -> list[spotting.Spot]:
    """The spotted lines without those lying mostly on the areas of symbols: symbol strokes read as letters. The
    spotter reads the sheet as it is, since on paper blanked out there it ran the texts beside a symbol into one."""
    if areas is None:
        return spots
    h, w = areas.shape
    out = []
    for s in spots:
        x0, y0 = min(w - 1, max(0, int(s.x0))), min(h - 1, max(0, int(s.y0)))
        box = areas[y0:max(y0 + 1, min(h, math.ceil(s.y1))), x0:max(x0 + 1, min(w, math.ceil(s.x1)))]
        if box.mean() < 0.5:
            out.append(s)
    return out


def _deduped(texts: list[Text]) -> list[Text]:
    """The texts without those read twice: the same characters (spaces and case aside) in the same direction, centred
    within a character height of each other. The surer reading stays."""
    def key(t: Text) -> str:
        return "".join(t.text.split()).casefold()

    kept: list[Text] = []
    for t in sorted(texts, key=lambda t: (t.uncertain, -t.confidence)):
        if not any(key(k) == key(t) and abs((k.angle - t.angle + 180) % 360 - 180) <= 10
                   and math.hypot(k.x - t.x, k.y - t.y) <= max(k.height, t.height) for k in kept):
            kept.append(t)
    return [t for t in texts if any(t is k for k in kept)]


def _settled(texts: list[Text], readings: ocr.Readings, kinds: list[profiles.Profile], features: Features
             ) -> list[Text]:
    """The texts as read, once the kind of drawing is known, where the readers left that open: of a text read both
    ways round (``text_direction``), the reading with more tokens that are words of the kind's vocabulary, the surer one
    where that does not tell, and one that only reading it again made sure of marked again, with the confidence it was
    found with, unless each of its tokens is a word of the vocabulary; a single character alone in a frame as the letter
    or digit it may be, where the kind frames only letters or only digits (``framed_letters``); a text the vote of
    look-alike glyphs had a say in, token by token (``lookalike_tokens``, see _by_token). A reading taken so is put
    right as the first ones were (ocr.grammar)."""
    def words(t: Text) -> int:
        return sum(p.words(t.text) for p in kinds)

    def known(t: Text) -> bool:
        return all(any(p.words(w) for p in kinds) for w in t.text.split())

    out = []
    for t in texts:
        if features.text_direction and (other := readings.other(t)) is not None and words(other) > words(t):
            t = other
        elif features.text_direction and (found := readings.doubt(t)) is not None and not known(t):
            t = replace(t, uncertain=True, confidence=found)
        elif features.framed_letters and readings.alone(t):
            t = next((replace(t, text=s) for p in kinds if (s := p.alone(t.text)) != t.text), t)
        elif (votes := readings.votes(t)) is not None:
            t = _by_token(t, *votes, kinds)
        out.append(t)
    return ocr.grammar(out) if any(a is not b for a, b in zip(out, texts)) else out


def _by_token(t: Text, read: str, best: str, kinds: list[profiles.Profile]) -> Text:
    """A text the vote of look-alike glyphs had a say in (ocr._voted), settled token by token where the token tells: one
    that is a word of the kind's vocabulary in only one of its readings, as read (``read``), as voted or as the vote
    would have it everywhere (``best``), takes that one ("DN80" over "0N80"); the others stay as voted, and the text is
    marked to check where one of them keeps a character the vote had no clear majority on."""
    out, doubt = [], False
    for w, r, b in zip(t.text.split(" "), read.split(" "), best.split(" ")):
        words = {s for s in (r, w, b) if any(p.word(s) for p in kinds)}
        if len(words) == 1 and r != b:
            w = words.pop()
        else:
            doubt |= w != b
        out.append(w)
    text = " ".join(out)
    return t if text == t.text and not doubt else replace(t, text=text, uncertain=t.uncertain or doubt)


def _vocabulary(texts: list[Text], kinds: list[profiles.Profile]) -> list[Text]:
    """The texts read anew in the vocabulary of the profiles of the kind of drawing (Profile.read)."""
    out = []
    for t in texts:
        s = t.text
        for p in kinds:
            s = p.read(s)
        out.append(t if s == t.text else replace(t, text=s))
    return out


def _in_circles(texts: list[Text], labels: np.ndarray, readable: np.ndarray, ink: np.ndarray, circles: list[Circle],
                rings: np.ndarray, norm: np.ndarray, ppm: float, reading: dict) -> tuple[list[Text], np.ndarray]:
    """Reads the text in circles that hold text, such as instrument bubbles, a second time without the ring and the
    line across a shared display: letters that touch either were taken for part of the symbol. Texts read again
    replace the first reading of the same letters. Circles without text, such as the hotplates of a hob, stay as they
    are. ``reading``: how to read them (keyword arguments of ocr.find_texts)."""
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
    more, more_labels, _ = ocr.find_texts(mask, norm, ppm, frame=ink, **reading)
    if not more:
        return texts, labels
    redone = set(np.unique(labels[more_labels > 0]).tolist()) - {0}
    keep = [k for k in range(1, len(texts) + 1) if k not in redone]
    lut = np.zeros(len(texts) + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    out = np.where(more_labels > 0, more_labels.astype(np.int32) + len(keep), lut[labels])
    return [texts[k - 1] for k in keep] + more, out


def _lettering(d: Drawing) -> None:
    """Letters the texts of the drawing as with a stencil: a sheet is lettered in a few sizes, and a text whose height
    is within SIZE of one at least SIZED texts have takes that size; the texts of a size get one width per character
    (NARROW_CHARS half of it), so the same label looks the same wherever it is. A text printed clearly narrower
    (TIGHT) keeps its printed length, to stay inside its frame or bubble. The tags of the symbols are lettered with the
    other texts, and the values of the dimensions."""
    texts = _texts_and_values(d) + [t for s in d.symbols for t in s.attributes.values()]
    ems = {id(t): sum(0.5 if ch in NARROW_CHARS else 1.0 for ch in t.text) or 1.0 for t in texts}
    heights = np.log([t.height for t in texts if t.height > 0 and not t.uncertain])
    sizes = []
    while len(heights) >= SIZED:
        near = np.abs(heights[:, None] - heights[None, :]) <= SIZE / 2
        k = int(np.argmax(near.sum(axis=1)))
        if near[k].sum() < SIZED:
            break
        sizes.append(float(np.exp(np.median(heights[near[k]]))))
        heights = heights[np.abs(heights - heights[k]) > SIZE]
    if not sizes:
        return
    sizes = np.array(sizes)
    size_of = {}
    for t in texts:
        k = int(np.argmin(np.abs(np.log(sizes / t.height)))) if t.height > 0 else -1
        if k >= 0 and abs(math.log(sizes[k] / t.height)) <= SIZE:
            size_of[id(t)] = k
    widths: dict[int, list[float]] = {}
    for t in texts:
        if id(t) in size_of and t.length and not t.uncertain:
            widths.setdefault(size_of[id(t)], []).append(t.length / t.height / ems[id(t)])

    def lettered(t: Text, centred: bool = False) -> Text:
        if (k := size_of.get(id(t))) is None:
            return t
        usual = float(np.median(widths[k])) * sizes[k] * ems[id(t)] if k in widths else None
        if t.length and (usual is None or t.length < TIGHT * usual):
            return replace(t, height=float(sizes[k]))
        back = (t.length - usual) / 2 if t.length and usual < t.length and not centred else 0.0
        a = math.radians(t.angle)  # a text shorter than printed starts where it did: notes stay flush left
        return replace(t, x=t.x - back * math.cos(a), y=t.y + back * math.sin(a), height=float(sizes[k]), length=usual)

    d.texts = [lettered(t) for t in d.texts]
    for s in d.symbols:  # a tag stays in the middle of its bubble
        s.attributes = {tag: lettered(t, centred=True) for tag, t in s.attributes.items()}
    for m in d.dimensions:  # and a value in the middle of its dimension
        m.text = m.text and lettered(m.text, centred=True)


def _printed(texts: list[Text], lettering: float = TEXT) -> float:
    """How large the symbols of the sheet are against their library: the sheet's text height against the
    ``lettering`` the library symbols are drawn for (its text_height, else TEXT). 1 unless clearly different, as on a
    large drawing printed on a small sheet."""
    heights = [t.height for t in texts if not t.uncertain]
    if len(heights) < 20:
        return 1.0
    f = float(np.clip(np.median(heights) / lettering, 0.5, 2.0))
    return 1.0 if 0.8 <= f <= 1.25 else f


def _kept(texts: list[Text], labels: np.ndarray, keep: list[bool]) -> tuple[list[Text], np.ndarray]:
    """The texts that are to be kept, with the labels of the rest renumbered; the pixels of the others are no text."""
    lut = np.zeros(len(texts) + 1, labels.dtype)
    lut[1:][keep] = np.arange(1, sum(keep) + 1)
    return [t for t, k in zip(texts, keep) if k], lut[labels]


def _box(t: Text) -> tuple[bool, float, float, float, float] | None:
    """A text along an axis as (whether it is vertical, from, to along it, from, to across it) in mm; None for an
    oblique one."""
    if abs((t.angle + 45) % 90 - 45) > 1:
        return None
    vertical, half = round(t.angle) % 180 == 90, (t.length or t.height) / 2
    (a, c) = (t.y, t.x) if vertical else (t.x, t.y)
    return vertical, a - half, a + half, c - t.height / 2, c + t.height / 2


def _bars(t: Text, lines: list[Line], ink: np.ndarray, ppm: float) -> bool:
    """Whether a text of a single character is the bars across a pipe (Features.text_on_pipes): a long straight line
    along it that runs through or along its strokes, on past it on both sides, with ink on both sides of it, as the
    two bars of a flange read as "H". A digit above a dimension line has no ink below it."""
    box = _box(t)
    if box is None or sum(ch.isalnum() for ch in t.text) != 1:
        return False
    vertical, lo, hi, c0, c1 = box
    for ln in lines:
        (a, b), (c, d) = ((ln.y1, ln.y2), (ln.x1, ln.x2)) if vertical else ((ln.x1, ln.x2), (ln.y1, ln.y2))
        mid, reach = (c + d) / 2, ln.width / 2 + TOUCH
        if abs(d - c) <= 0.1 * abs(b - a) and c0 - reach <= mid <= c1 + reach and min(a, b) <= lo - spotting.RUN_ON \
                and max(a, b) >= hi + spotting.RUN_ON and all(_inked(ink, ppm, vertical, lo, hi, mid + side * (
                    ln.width / 2 + BESIDE[0]), mid + side * (ln.width / 2 + BESIDE[1])) for side in (-1, 1)):
            return True
    return False


def _inked(ink: np.ndarray, ppm: float, vertical: bool, lo: float, hi: float, c0: float, c1: float) -> bool:
    """Whether a strip from ``lo`` to ``hi`` along an axis (vertical: in y) and from ``c0`` to ``c1`` across it (mm)
    holds a stroke."""
    (x0, x1), (y0, y1) = ((c0, c1), (lo, hi)) if vertical else ((lo, hi), (c0, c1))
    x0, x1, y0, y1 = (max(0, round(v * ppm)) for v in (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)))
    return int(ink[y0:y1 + 1, x0:x1 + 1].sum()) >= 4  # more than a speck or the blurred edge of a line


def _piped(t: Text, lines: list[Line]) -> bool:
    """Whether a pipe runs into a text at both ends along it and ends in it (Features.text_on_pipes): a valve read as
    a text ("Hoeat"), whose strokes the symbols may then explain (see symbols._misread). The pipe may run along the edge
    of its box, which holds the strokes on one side of it where the pipe cut them in two. A line label in a frame
    stands apart from the pipe by its frame; a pipe beside a line number runs on past it."""
    box = _box(t)
    if box is None:
        return False
    vertical, lo, hi, c0, c1 = box
    into = [False, False]
    for ln in lines:
        (a, b), (c, d) = ((ln.y1, ln.y2), (ln.x1, ln.x2)) if vertical else ((ln.x1, ln.x2), (ln.y1, ln.y2))
        (a, b), reach = sorted((a, b)), ln.width / 2 + TOUCH
        if abs(d - c) <= 0.1 * (b - a) and c0 - reach <= (c + d) / 2 <= c1 + reach:
            into[0] |= a <= lo - spotting.RUN_ON and lo - TOUCH <= b <= hi
            into[1] |= b >= hi + spotting.RUN_ON and lo <= a <= hi + TOUCH
    return all(into)


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


def printed_size(ink: np.ndarray, ppm: float, rows: bool = True) -> float:
    """The share of its drawn size that the sheet shows the drawing at: 1, unless its characters are much smaller
    than drawings are lettered (DRAWN), as on a large drawing printed on a small sheet. Characters are small, upright
    shapes of ink; the most common of their heights is the height of the lettering, counting only heights at which
    most of them stand in a row with another of about their height (``rows``): specks of dirt on a poor scan stand
    alone, while blur merges a share of the characters of small lettering, so their neighbours stand alone too. It
    is measured on the ink, not on what is read, as the text of a small print may be too small to read."""
    _, _, stats, centres = cv2.connectedComponentsWithStats(ink, connectivity=8)
    w, h, area = stats[1:, cv2.CC_STAT_WIDTH] / ppm, stats[1:, cv2.CC_STAT_HEIGHT] / ppm, stats[1:, cv2.CC_STAT_AREA]
    shaped = (h >= 0.4) & (h <= 6) & (w >= 0.1 * h) & (w <= 1.2 * h) & (area <= 0.75 * w * h * ppm * ppm)
    heights, edges = h[shaped], np.arange(math.log(0.4), math.log(6), 0.06)
    counts, n = np.histogram(np.log(heights), edges)[0], len(heights)
    if rows:
        lined = np.histogram(np.log(heights[_in_rows(centres[1:][shaped] / ppm, heights)]), edges)[0]
        counts = np.where(2 * lined >= counts, counts, 0)
        n = int(counts.sum())
    if n < 50:
        return 1.0
    k = int(np.argmax(np.convolve(counts, [1, 2, 1], "same")))
    share = math.exp((edges[k] + edges[k + 1]) / 2) / DRAWN
    return 1.0 if share >= SMALLER else round(max(share, 0.2), 2)


def _in_rows(centres: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Whether each shape (centre and height in mm) stands in a row with another of about its height, as the
    characters of a word or a number do: beside it, at its height (within 0.3 of the higher one), at most two heights
    away and at most 1.5 times as high or low. Each shape looks for the lower ones around it as far as its own
    height reaches, so a few tall shapes do not make every shape look far."""
    out = np.zeros(len(heights), bool)
    if len(heights) < 2:
        return out
    near = cKDTree(centres).query_ball_point(centres, 2.03 * heights, return_sorted=False)
    i = np.repeat(np.arange(len(heights)), [len(n) for n in near])
    j = np.fromiter((k for n in near for k in n), np.intp, len(i))
    high, low = np.maximum(heights[i], heights[j]), np.minimum(heights[i], heights[j])
    apart = np.abs(centres[i] - centres[j])
    row = (i != j) & (heights[i] >= heights[j]) & (high <= 1.5 * low) & (apart[:, 1] <= 0.3 * high) \
        & (apart[:, 0] <= 2 * high)
    out[i[row]] = out[j[row]] = True
    return out


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
