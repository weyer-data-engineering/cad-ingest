"""Reads the drawing on a vector PDF page, such as a CAD export, as it is: its lines, arcs, circles, fills and texts,
with their colours and pen widths, in sheet millimetres. Such a page needs no image recognition. A scanned page (an
image) goes through the photo and scan pipeline instead.

Objects keep the CAD layer they were drawn on where the PDF names it (CAD programs export layers as optional
content groups). Curves are cubic Béziers in PDF; CAD programs write arcs and circles as Béziers that follow the
circle, which are read back as arcs and circles. Texts keep their string, place, angle and colour. Their height and
length are measured on the text rendered alone, because the font size of a CAD export says little about either (its
fonts are often drawings of the CAD program's own stroke fonts)."""

from __future__ import annotations

import ctypes
import math
import re
from statistics import median

import numpy as np
import pypdfium2 as pdfium
from pypdfium2 import raw
from skimage.morphology import skeletonize

from .model import Arc, Circle, Color, Drawing, Line, Polygon, Polyline, Text

PT = 25.4 / 72  # mm per PDF point
MIN_PATHS = 20  # a page with fewer paths is not a drawing made of vectors
IMAGE_SHARE = 0.3  # a page with an image covering this share of it is a scan
ARC = 0.03  # mm a Bézier may stray from a circular arc and still be read as one (exports round to a grid)
FLAT = 0.01  # mm other curves are flattened to lines this close to the curve
SAME = 0.005  # mm: points this close are the same (where a curve closes)
TEXT_PX = 8.0  # pixels per point to measure texts on
TEXT_SIDE = 4000  # pixels at most along a text's measuring image (a huge text is measured coarser)
CONTROL = re.compile(r"[\x00-\x1f]")  # control characters, which no text shows
NOT_IN_LAYER = re.compile(r'[<>/\\":;?*=`\x00-\x1f]')  # characters a DXF layer name cannot hold
PAPER = (255, 255, 255)  # a fill in the paper's colour draws nothing (such as the sheet's background)
PLAIN = re.compile(r"[A-Z0-9 .:\-]*[A-Z0-9][A-Z0-9 .:\-]*")  # capitals and digits: no part above or below them


def is_vector(page: pdfium.PdfPage) -> bool:
    """Whether the page is a drawing made of vectors: enough paths, and no image covering much of it (a scan)."""
    paths, area = 0, 0.0
    l, b, r, t = page.get_cropbox()
    for obj in page.get_objects():
        if obj.type == raw.FPDF_PAGEOBJ_PATH:
            paths += 1
        elif obj.type == raw.FPDF_PAGEOBJ_IMAGE:
            x0, y0, x1, y1 = obj.get_bounds()
            area += max(0.0, min(x1, r) - max(x0, l)) * max(0.0, min(y1, t) - max(y0, b))
    return paths >= MIN_PATHS and area < IMAGE_SHARE * (r - l) * (t - b)


def read(pdf: pdfium.PdfDocument, page: pdfium.PdfPage) -> Drawing:
    """The page as a Drawing in sheet millimetres (the page is the sheet)."""
    to_sheet, width, height, turn = _sheet(page)
    d = Drawing(width, height, exact=True)
    textpage = page.get_textpage()
    texts = []
    for obj, m, layer in _objects(page.raw):
        kind = raw.FPDFPageObj_GetType(obj)
        if kind == raw.FPDF_PAGEOBJ_PATH:
            _path(d, obj, m, to_sheet, layer)
        elif kind == raw.FPDF_PAGEOBJ_TEXT:
            texts.append(_text(pdf, page, textpage, obj, m, layer))
    d.texts = _placed([t for t in texts if t], to_sheet, turn)
    return d


# --------------------------------------------------------------------------- page and objects

def _sheet(page: pdfium.PdfPage):
    """The mapping from PDF points (y up) to sheet millimetres (y down) as the page is shown (with /Rotate), the
    sheet size and the turn in degrees (clockwise)."""
    l, b, r, t = page.get_cropbox()
    turn = page.get_rotation() % 360
    maps = {0: lambda x, y: ((x - l) * PT, (t - y) * PT), 90: lambda x, y: ((y - b) * PT, (x - l) * PT),
            180: lambda x, y: ((r - x) * PT, (y - b) * PT), 270: lambda x, y: ((t - y) * PT, (r - x) * PT)}
    w, h = (r - l) * PT, (t - b) * PT
    return maps[turn], *((h, w) if turn in (90, 270) else (w, h)), turn


def _objects(handle, matrix=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0), layer: str | None = None, form: bool = False):
    """The objects of a page (or of a form, an XObject inside it) with their matrices to page space and their
    layers (an object without one of its own is on the layer of the form it is in)."""
    count = raw.FPDFFormObj_CountObjects(handle) if form else raw.FPDFPage_CountObjects(handle)
    for i in range(count):
        obj = raw.FPDFFormObj_GetObject(handle, i) if form else raw.FPDFPage_GetObject(handle, i)
        m = raw.FS_MATRIX()
        raw.FPDFPageObj_GetMatrix(obj, ctypes.byref(m))
        here = _compose((m.a, m.b, m.c, m.d, m.e, m.f), matrix)
        own = _layer(obj) or layer
        if raw.FPDFPageObj_GetType(obj) == raw.FPDF_PAGEOBJ_FORM:
            yield from _objects(obj, here, own, True)
        else:
            yield obj, here, own


def _layer(obj) -> str | None:
    """The name of the optional content group (the CAD layer) the object is marked with, if any."""
    for i in range(raw.FPDFPageObj_CountMarks(obj)):
        mark = raw.FPDFPageObj_GetMark(obj, i)
        n = ctypes.c_ulong()
        if not raw.FPDFPageObjMark_GetParamBlobValue(mark, b"Name", None, 0, ctypes.byref(n)) or not n.value:
            continue
        buf = (ctypes.c_ubyte * n.value)()
        raw.FPDFPageObjMark_GetParamBlobValue(mark, b"Name", buf, n.value, ctypes.byref(n))
        name = bytes(buf)[:n.value]  # a PDF text string: UTF-16 with a byte order mark, else PDFDocEncoding
        name = name[2:].decode("utf-16-be", "replace") if name[:2] == b"\xfe\xff" else name.decode("latin-1")
        return NOT_IN_LAYER.sub("_", name.strip())[:255] or None
    return None


def _compose(m, n):
    """m then n (PDF row vector convention)."""
    a, b, c, d, e, f = m
    p, q, r, s, t, u = n
    return a * p + b * r, a * q + b * s, c * p + d * r, c * q + d * s, e * p + f * r + t, e * q + f * s + u


def _apply(m, x: float, y: float) -> tuple[float, float]:
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


def _rgb(get, obj) -> Color | None:
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    if not get(obj, *(ctypes.byref(v) for v in (r, g, b, a))):
        return None
    return r.value, g.value, b.value


# --------------------------------------------------------------------------- paths

def _path(d: Drawing, obj, m, to_sheet, layer: str | None) -> None:
    fill, stroke = ctypes.c_int(), ctypes.c_int()
    raw.FPDFPath_GetDrawMode(obj, ctypes.byref(fill), ctypes.byref(stroke))
    if not fill.value and not stroke.value:
        return  # a clipping path or an invisible one
    w = ctypes.c_float()
    raw.FPDFPageObj_GetStrokeWidth(obj, ctypes.byref(w))
    width = w.value * math.sqrt(abs(m[0] * m[3] - m[1] * m[2])) * PT
    stroke_color = _rgb(raw.FPDFPageObj_GetStrokeColor, obj)
    fill_color = _rgb(raw.FPDFPageObj_GetFillColor, obj)
    for points, closed in _subpaths(obj, m, to_sheet):
        if fill.value and len(points) > 2 and (fill_color != PAPER or stroke.value):
            d.polygons.append(Polygon(_flat(points), True, width, fill_color, layer))
        if stroke.value:
            _stroked(d, points, closed, width, stroke_color, layer)


def _subpaths(obj, m, to_sheet):
    """The subpaths as lists of segments in sheet mm: a segment is a point (line to it) or a Bézier (c1, c2, end);
    the first entry is the start point."""
    out, current, closed = [], [], False
    n = raw.FPDFPath_CountSegments(obj)
    i = 0
    while i < n:
        seg = raw.FPDFPath_GetPathSegment(obj, i)
        kind = raw.FPDFPathSegment_GetType(seg)
        if kind == raw.FPDF_SEGMENT_MOVETO:
            if len(current) > 1:
                out.append((current, closed))
            current, closed = [_point(seg, m, to_sheet)], False
        elif kind == raw.FPDF_SEGMENT_LINETO:
            point = _point(seg, m, to_sheet)
            if not current or not _close(_end(current[-1]), point):  # a piece of no length draws nothing
                current.append(point)
        elif kind == raw.FPDF_SEGMENT_BEZIERTO and i + 2 < n:
            segs = [raw.FPDFPath_GetPathSegment(obj, i + k) for k in range(3)]
            current.append(tuple(_point(s, m, to_sheet) for s in segs))
            seg, i = segs[2], i + 2
        if raw.FPDFPathSegment_GetClose(seg):
            closed = True
            if current and not _close(_end(current[-1]), current[0]):
                current.append(current[0])
        i += 1
    if len(current) > 1:
        out.append((current, closed))
    return out


def _point(seg, m, to_sheet) -> tuple[float, float]:
    x, y = ctypes.c_float(), ctypes.c_float()
    raw.FPDFPathSegment_GetPoint(seg, ctypes.byref(x), ctypes.byref(y))
    return to_sheet(*_apply(m, x.value, y.value))


def _end(segment) -> tuple[float, float]:
    return segment[2] if isinstance(segment[0], tuple) else segment


def _stroked(d: Drawing, segments: list, closed: bool, width: float, color: Color | None, layer: str | None) -> None:
    """A stroked subpath as a line, an arc, a circle or a polyline (with bulges for its arcs)."""
    curves = segments[1:]
    if all(isinstance(seg[0], tuple) for seg in curves):  # all curved: perhaps one circle or arc
        samples = _samples(segments)
        circle = _fit(samples)
        if circle:
            cx, cy, r = circle
            if _close(samples[0], samples[-1]):
                d.circles.append(Circle(cx, cy, r, width, color, layer))
                return
            a1, a2 = (math.degrees(math.atan2(y - cy, x - cx)) % 360 for x, y in (samples[0], samples[-1]))
            clockwise = sum((x1 - cx) * (y2 - cy) - (y1 - cy) * (x2 - cx)  # on screen, y pointing down
                            for (x1, y1), (x2, y2) in zip(samples, samples[1:])) > 0
            d.arcs.append(Arc(cx, cy, r, *((a1, a2) if clockwise else (a2, a1)), width, color, layer))
            return
    points, bulges = [segments[0]], []
    for seg in curves:
        if not isinstance(seg[0], tuple):  # a straight piece
            points.append(seg)
            bulges.append(0.0)
            continue
        bulge = _bulge(points[-1], *seg)
        flat = [seg[2]] if bulge is not None else _bezier(points[-1], *seg)[1:]
        points += flat
        bulges += [bulge] if bulge is not None else [0.0] * len(flat)
    closed = len(points) > 2 and _close(points[0], points[-1])  # ends where it starts, with or without "close"
    if closed:
        points.pop()  # its piece now closes the polyline
    if len(points) == 2 and not bulges[0] and not closed:
        d.lines.append(Line(*points[0], *points[1], width, False, color, layer))
    else:
        d.polylines.append(Polyline(points, bulges, closed, width, False, "line", color, layer))


def _close(a, b) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= SAME


def _cubic(p0, c1, c2, p3, t: float) -> tuple[float, float]:
    s = 1 - t
    return tuple(s ** 3 * a + 3 * s * s * t * b + 3 * s * t * t * c + t ** 3 * e for a, b, c, e in zip(p0, c1, c2, p3))


def _samples(segments: list) -> list[tuple[float, float]]:
    """Points along a subpath of Béziers, five per curve."""
    points = [segments[0]]
    for seg in segments[1:]:
        points += [_cubic(points[-1], *seg, t) for t in (0.25, 0.5, 0.75, 1.0)]
    return points


def _fit(points) -> tuple[float, float, float] | None:
    """The circle (cx, cy, r) through the points, if they all lie within ARC mm of it."""
    p = np.asarray(points, float)
    if len(p) < 3:
        return None
    a = np.column_stack([2 * p, np.ones(len(p))])
    (cx, cy, c), *_ = np.linalg.lstsq(a, (p ** 2).sum(1), rcond=None)
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    if not r or np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r).max() > ARC:
        return None
    return float(cx), float(cy), r


def _bulge(p0, c1, c2, p3) -> float | None:
    """The bulge of the Bézier if it follows a circular arc within ARC mm (positive when it turns clockwise on screen,
    as in model.Polyline), else None."""
    chord = math.dist(p0, p3)
    if chord < SAME or not _fit([p0] + [_cubic(p0, c1, c2, p3, t) for t in (0.15, 0.3, 0.5, 0.7, 0.85, 1.0)]):
        return None
    mid = _cubic(p0, c1, c2, p3, 0.5)
    sagitta = math.dist(mid, ((p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2))
    turn = (mid[0] - p0[0]) * (p3[1] - mid[1]) - (mid[1] - p0[1]) * (p3[0] - mid[0])
    return math.copysign(2 * sagitta / chord, turn)


def _bezier(p0, c1, c2, p3) -> list[tuple[float, float]]:
    """Points along the Bézier, no further than FLAT mm from it."""
    size = math.dist(p0, c1) + math.dist(c1, c2) + math.dist(c2, p3)
    n = max(2, math.ceil(math.sqrt(size / (8 * FLAT))) * 2)
    return [_cubic(p0, c1, c2, p3, k / n) for k in range(n + 1)]


def _flat(segments: list) -> list[tuple[float, float]]:
    points = [segments[0]]
    for seg in segments[1:]:
        points += _bezier(points[-1], *seg)[1:] if isinstance(seg[0], tuple) else [seg]
    return points


# --------------------------------------------------------------------------- texts

def _text(pdf, page, textpage, obj, m, layer):
    """(string, matrix, font size, colour, ink points in PDF points, stroke width in points (0 for outline fonts),
    font, layer) of a text object, or None if it shows nothing."""
    n = raw.FPDFTextObj_GetText(obj, textpage.raw, None, 0)
    buf = (ctypes.c_ushort * max(n, 1))()
    raw.FPDFTextObj_GetText(obj, textpage.raw, buf, n)
    string = bytes(buf).decode("utf-16-le", "replace").rstrip("\x00")
    string = CONTROL.sub("", string.replace("\x02", "-")).strip()  # PDFium writes a hyphen ending a line as 0x02
    if not string or raw.FPDFTextObj_GetTextRenderMode(obj) == raw.FPDF_TEXTRENDERMODE_INVISIBLE:
        return None
    left, bottom, right, top = (ctypes.c_float() for _ in range(4))
    raw.FPDFPageObj_GetBounds(obj, *(ctypes.byref(v) for v in (left, bottom, right, top)))
    px = min(TEXT_PX, TEXT_SIDE / max(right.value - left.value, top.value - bottom.value, 1e-3))
    bmp = raw.FPDFTextObj_GetRenderedBitmap(pdf.raw, page.raw, obj, ctypes.c_float(px))
    if not bmp:
        return None
    try:
        w, h, stride = raw.FPDFBitmap_GetWidth(bmp), raw.FPDFBitmap_GetHeight(bmp), raw.FPDFBitmap_GetStride(bmp)
        data = ctypes.cast(raw.FPDFBitmap_GetBuffer(bmp), ctypes.POINTER(ctypes.c_uint8 * (stride * h))).contents
        ink = np.frombuffer(data, np.uint8).reshape(h, stride)[:, :w * 4].reshape(h, w, 4)[..., 3] > 127
    finally:
        raw.FPDFBitmap_Destroy(bmp)
    ys, xs = np.nonzero(ink)
    if not len(xs):
        return None
    points = np.column_stack([left.value + (xs + 0.5) / px, top.value - (ys + 0.5) / px])
    font = raw.FPDFTextObj_GetFont(obj)
    stroke = 0.0 if _outlined(font, string) else ink.sum() / max(1, skeletonize(ink).sum()) / px
    size = ctypes.c_float()
    raw.FPDFTextObj_GetFontSize(obj, ctypes.byref(size))
    font = ctypes.cast(font, ctypes.c_void_p).value
    return string, m, size.value, _rgb(raw.FPDFPageObj_GetFillColor, obj), points, stroke, font, layer


def _outlined(font, string: str) -> bool:
    """Whether the font's letters are filled outlines (as in TrueType fonts), whose ink ends where the letter does.
    The letters of CAD stroke fonts are lines drawn with a pen, whose ink stands half a pen width beyond the letter."""
    path = raw.FPDFFont_GetGlyphPath(font, ord(string[0]), ctypes.c_float(1.0))
    return bool(path) and any(raw.FPDFPathSegment_GetClose(raw.FPDFGlyphPath_GetGlyphPathSegment(path, i))
                              for i in range(raw.FPDFGlyphPath_CountGlyphSegments(path)))


def _placed(texts: list, to_sheet, turn: int) -> list[Text]:
    """Texts in sheet mm, placed and sized from their ink. The top of the capitals is measured per font on texts of
    capitals and digits only, as a share of the glyph size; the baseline once for all (it is where the text starts
    in every font seen), because underlines and descenders below it are common."""
    frames, tops, bases = [], {}, []
    for string, m, size, color, points, stroke, font, layer in texts:
        u = np.array([m[0], m[1]]) / (math.hypot(m[0], m[1]) or 1)  # reading direction
        slant = np.array([m[2], m[3]])  # up the letters, leaning forward in italics
        n = np.array([-u[1], u[0]]) * (1 if slant @ (-u[1], u[0]) >= 0 else -1)  # up, square to the baseline
        rel = points - (m[4], m[5])
        along, across = rel @ u, rel @ n
        glyph = abs(slant @ n) * size or 1.0  # glyph size square to the baseline
        lean = math.degrees(math.atan2(slant @ u, slant @ n))
        frames.append((string, m, u, n, along, across, stroke, glyph, color, font, layer, lean))
        if PLAIN.fullmatch(string):
            tops.setdefault(font, []).append((across.max() - stroke / 2) / glyph)
            bases.append((across.min() + stroke / 2) / glyph)
    every = [v for values in tops.values() for v in values]
    base_share = median(bases) if bases else 0.0
    out = []
    for string, m, u, n, along, across, stroke, glyph, color, font, layer, lean in frames:
        samples = tops.get(font) or every
        if samples:
            base = base_share * glyph
            cap = median(samples) * glyph - base
        else:  # no text of capitals and digits at all: the ink is the best guess
            base, cap = across.min() + stroke / 2, across.max() - across.min() - stroke
        if cap <= 0:  # an odd font, or a text that is a single stroke: the height of its ink
            base, cap = across.min(), max(across.max() - across.min(), 0.1)
        shift = math.tan(math.radians(lean)) * cap  # italics: the tops of the letters stand further along
        start = along.min() + stroke / 2 - min(0.0, shift)  # where the baseline starts and ends
        end = along.max() - stroke / 2 - max(0.0, shift)
        mid = np.array([m[4], m[5]]) + u * (start + end) / 2 + n * (base + cap / 2)
        x, y = to_sheet(*mid)
        angle = (math.degrees(math.atan2(u[1], u[0])) - turn) % 360
        out.append(Text(x, y, cap * PT, string, round(angle, 3), length=(end - start) * PT, oblique=round(lean, 1),
                        color=color, layer=layer))
    return out
