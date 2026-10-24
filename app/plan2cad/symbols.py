"""Replaces recognised symbols (valves, pumps and instruments on P&IDs, fixtures on floor plans) with blocks of the
symbol libraries (see library.py).

Hypotheses come from the drawing: two line ends that two ports of a symbol fit (a valve between two pipe ends, a safety
valve between its riser and outlet; this fixes position, turn and size), rectangles whose sides lie on linework and
that a rectangle of the symbol fits (a washbasin against a wall, a hob in a counter), circles (pumps, heat
exchangers, instruments, scaled to the radius) and, where no symbol was found yet, single line ends at a port and
groups of small strokes. Each is fitted by moving it a little and scored on points along the centre lines of the ink
(or the drawing's lines), like an oriented chamfer match: a point of the symbol's outline counts as found if ink runs
close by in the same direction. The score rewards the ink explained and penalises the outline missing and the ink
inside the symbol it does not explain; lines leaving through a port, such as pipes, are expected there. Of overlapping
hits the best one wins. The linework a symbol explains is removed (a wall only where a symbol covers it), line ends are
snapped onto its ports and the texts at its attribute positions become its attributes; a symmetric symbol (a valve
that looks the same turned half way) takes the pose in which most of them stand there, then the one with its back on
the wall."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree
from skimage.morphology import skeletonize

from .library import Template, place, transform
from .model import Arc, Circle, Drawing, Line, Polyline, Symbol

STEP = 0.25  # mm between sample points
TOL = 0.5  # mm a recognised line may lie off the symbol's outline
TURN = 0.25  # mm that a direction differing by 90 degrees adds to the distance (with half of it at 30 degrees)
COVERAGE = 0.8  # share of the outline that must be found
PART = 0.5  # share of each part of the symbol (a polyline, circle, ...) that must be found
MISSING = 0.5  # weight of a millimetre of missing outline against one of linework explained
CLUTTER = 0.25  # unexplained linework inside the symbol, relative to the length of its outline
SCALES = (0.6, 1.6)  # symbols may be drawn smaller or larger than in the library
SIZES = (1.0, 0.87, 1.15)  # sizes tried around a size estimated from line ends (the estimate first)
SMALL = 25.0  # mm: groups of strokes up to this size (bounding box diagonal) can be a symbol
STUB = 4.0  # mm: straight lines of the drawing longer than this are pipes, not parts of a symbol
SNAP = 1.5  # mm: line ends this close to a port are moved onto it
SNAP_ASIDE = 0.5  # mm: ... if the port lies on the line's course; further aside, the line would be turned askew
SNAP_PARALLEL = math.radians(15)  # ... but of parallel lines arriving at one port, only the one pointing at it
BLOCK = (200.0, 120.0)  # mm: largest title block (width, height)
ROUND_END = 1.0  # mm: the smallest radius of the round end of a box that a symbol without curves must not lie on
REACH = 4.0  # mm: a text this close to an attribute position becomes the attribute
SAME = 1.5 * TOL  # mm: a symbol turned or mirrored with its outline this close to the original fits the same strokes
TAKEN = 0.3  # a hit is dropped if this share of the linework it explains belongs to a better one
GONE = 0.6  # a piece is removed if this share of it is explained by symbols (of a wall only if inside a symbol)
AXIS = 0.4  # mm a line may lean over its length and still be a side of an axis-parallel rectangle
LEAN = 0.06  # or this share of its length: tracing may join a side with a wall end it continues
BEND = 0.5  # mm an arc may bend and still be a side of a rectangle (photos trace short sides with a slight bend)
LETTER = 6.0  # mm: closed shapes up to this size may be read as a letter (a hotplate as "0", a small bowl as "O")
CORNER = 0.75  # mm the sides of a rectangle may stop short of its corners (gaps in traced lines)
LETTERS = 0.3  # share of a symbol's outline that may be found on words only (see ocr.worded)
ONE_SIZE = 1.4  # symbols drawn at the printed size are at most this factor larger or smaller than most on the sheet
ONE_SIZE_LEAST = 12  # symbols that must be found before their size counts as the sheet's
FRAMED = 0.3  # a symbol not holding text may explain at most this share of its linework on a frame around a text
ASPECT = 0.15  # a rectangle may be this much more stretched than the one of the symbol it fits (real objects vary)
BOX = (2.0, 15.0)  # mm: sides of the closed boxes of a drawing, such as the frame of a line size or a tag ...
BOXED = 0.5  # ... on which at most this share of the linework of a symbol without a rectangle may lie
CONTINUED = 1.0  # mm: a line that runs through a symbol (the pipe under a flow arrow) goes on at least this far
BARB = 1.5  # mm: strokes this close beside a pipe may be the barbs of a flow arrow drawn on it
PIPE = 10.0  # mm: ... a pipe at least this long; the letters of a word traced as a line are shorter
SMALL_CIRCLE = 1.5  # mm: a circle up to this radius may be filled, by the drawing or by blur (the ball of a valve)
BLOB = 1.5  # ... and then its ink is at least this many times as thick as the usual stroke of the sheet
SURE = 0.9  # a symbol with less of its outline found is one to check
RIVAL = 0.9  # ... and so is one that a symbol of another kind fits with at least this share of its score


@dataclass
class _Shape:
    """A library symbol prepared for matching, in its own coordinates."""

    t: Template
    pts: np.ndarray  # points along the outline
    heading: np.ndarray  # direction of the outline at these points (unit vectors)
    group: np.ndarray  # the part of the symbol each point belongs to
    rim: np.ndarray  # where text may stand in for a stroke: the outer outline (convex hull), circles, small closed shapes
    through: np.ndarray  # points (and directions, oriented form) along lines that must run through the symbol
    beyond: list[np.ndarray]  # ... and along their continuation past each end
    hull: np.ndarray  # half-planes (nx, ny, offset) of the convex hull: inside where nx * x + ny * y + offset <= 0
    at: np.ndarray  # the ports
    ports: np.ndarray  # unit vectors from the origin towards the ports
    radius: float
    smallest: float  # scale below which separate parts of the symbol (two bars) would melt into one line
    variants: list[tuple[int, bool, np.ndarray, np.ndarray]]  # distinct (quarter turns, mirror, box low, box high)
    ring: tuple[np.ndarray, float] | None  # centre and radius of the largest circle
    rects: np.ndarray  # axis-parallel rectangles of the outline: centre x, y, width, height
    holds_text: bool  # an attribute stands inside the outline, as the tag of an instrument bubble


@dataclass
class _Hit:
    shape: _Shape
    pose: tuple[float, float, float, bool, float]  # x, y, angle, mirror, scale
    score: float
    claim: np.ndarray  # indices of the ink points it explains
    found: float = 1.0  # share of the outline found

    def explains(self, points: np.ndarray, oriented: np.ndarray, tol: float = 2 * TOL) -> np.ndarray:
        """Which of the points (with their oriented form) lie on the symbol's outline (within ``tol``; the symbol is
        confirmed, so a little more than for matching), inside it, not on a line leaving through a port."""
        s, (x, y, angle, mirror, scale) = self.shape, self.pose
        m = _matrix(angle, mirror, scale)
        q = (points - (x, y)) @ np.linalg.inv(m).T  # in the symbol's own coordinates
        inside = np.all(q @ s.hull[:, :2].T + s.hull[:, 2] <= tol / scale, axis=1)
        outline = _oriented(s.pts @ m.T + (x, y), s.heading @ m.T / scale)
        on = cKDTree(outline).query(oriented, distance_upper_bound=tol)[0] < np.inf
        return inside & on & ~_in_port(s, q, m, oriented)


class _Ink:
    """The sampled linework of the drawing: positions, and positions with directions for oriented matching. Text
    points may complete a symbol's outline (a misread arrow tip) but never count against it (letters in a bubble)."""

    def __init__(self, points: np.ndarray, heading: np.ndarray, text: np.ndarray, straight: np.ndarray,
                 frame: np.ndarray | None = None, thick: np.ndarray | None = None):
        self.points, self.heading, self.oriented = points, heading, _oriented(points, heading)
        self.thick = np.zeros(len(points)) if thick is None else thick  # half the width of the ink there (mm)
        self.usual = float(np.median(self.thick[self.thick > 0])) if (self.thick > 0).any() else 0.0  # ... mostly
        self.text, self.straight = text > 0, straight  # on text, on long straight lines (pipes)
        self.words = text > 1  # on text that reads as words or a code (see ocr.worded)
        self.frame = np.zeros(len(points), bool) if frame is None else frame  # on the frame around a text
        self.tree, self.oriented_tree = cKDTree(points), cKDTree(self.oriented)
        self.strokes = cKDTree(self.oriented[~self.text]) if self.text.any() else self.oriented_tree  # without text
        self.unworded = cKDTree(self.oriented[~self.words]) if self.words.any() else self.oriented_tree
        self.lines = cKDTree(self.oriented[self.straight])


def recognise(d: Drawing, library: dict[str, Template], ink: tuple | None = None) -> None:
    """Finds the symbols of ``library`` in ``d`` and replaces their linework by placed symbols (in place). ``ink`` are
    the points, stroke directions and text flags to match on (see ink_samples); without it, points along the
    drawing's lines."""
    pieces = _pieces(d)
    if not pieces:
        return
    if ink is None:  # the drawing's own lines
        straight = np.concatenate([np.full(len(p), _stub(p)) for _, p, _ in pieces])
        ink = np.vstack([p for _, p, _ in pieces]), np.vstack([h for _, _, h in pieces]), np.zeros(len(straight), bool), \
            straight
    ink = _Ink(*ink)
    if not len(ink.points):
        return
    shapes = [_shape(t) for t in library.values()]
    # Symbols between line ends, at rectangles and at circles first, then the rest only where no symbol was found yet
    hits = _fitted(_anchored(d, shapes), ink) + _fitted(_on_pipes(d, shapes, ink), ink)  # apart: not merged as one
    hits += _fitted(_leftover(d, shapes, ink, _select(hits, len(ink.points))), ink)
    hits = _one_size(hits, len(ink.points))
    block = _title_block(d)  # small unread text there looks like symbols
    arcs = d.arcs + [q for pl in d.polylines for q in pl.parts() if isinstance(q, Arc)]
    boxed = _boxed(d, ink)
    twins = _twins(shapes)
    chosen = [h for h in _select(hits, len(ink.points)) if not (block and _within(h.pose[:2], block))
              and not _rounded(h, arcs, ink) and (len(h.shape.rects) or boxed[h.claim].mean() <= BOXED)]
    found = [_circled(_centred(h, d.circles), twins, ink) for h in chosen]
    _remove(d, pieces, found)
    placed = [place(h.shape.t, *h.pose) for h in found]
    for s, h, first in zip(placed, found, chosen):
        s.confidence, s.doubt = round(100 * h.found, 1), _doubt(h, first, hits)
    d.symbols += placed
    _snap(d)
    _attributes(d, placed)
    d.texts = [t for t in d.texts if not any(_contains(h, (t.x, t.y)) and _misread(h, ink) for h in found)]


def ink_samples(mask: np.ndarray, text: np.ndarray, ppm: float, lines: list[Line],
                frame: np.ndarray | None = None) -> tuple:
    """Points about every quarter millimetre along the centre lines of an ink mask (0/1, ``ppm`` pixels per mm), the
    stroke direction there (from the neighbouring points) and whether they lie on text (``text`` mask: 1, or 2 on text
    that reads as words, see ocr.worded), in sheet mm, plus points along ``lines`` (flagged as straight), and whether
    each lies on the frame around a text (``frame`` mask, see ocr.frames), and half the width of the ink there (a
    filled circle is as thick as its radius). Symbols are matched on the image like this,
    not on the traced vectors, which may have rounded off small shapes such as arrowheads and springs. The mask should not hold the long
    straight lines (given as ``lines``): where a stroke meets a pipe at a sharp angle, as an arrowhead does, both would
    merge into one thick stroke with a bent centre line."""
    sampled = [_sample(q) for q in lines]
    exact = [np.vstack([p for p, _ in sampled]), np.vstack([h for _, h in sampled])] if sampled else \
        [np.zeros((0, 2)), np.zeros((0, 2))]
    ys, xs = np.nonzero(skeletonize(mask.astype(bool)))
    _, first = np.unique(np.column_stack([xs, ys]) // (STEP * ppm), axis=0, return_index=True)
    xs, ys = xs[first], ys[first]
    points = (np.column_stack([xs, ys]) + 0.5) / ppm
    if len(points) < 5:
        return *exact, np.zeros(len(exact[0]), bool), np.ones(len(exact[0]), bool), _on(frame, exact[0], ppm), \
            np.zeros(len(exact[0]))
    near = points[cKDTree(points).query(points, k=5)[1]]  # the point and its four neighbours
    off = near - near.mean(axis=1, keepdims=True)
    cxx, cyy, cxy = (off[..., 0] ** 2).sum(1), (off[..., 1] ** 2).sum(1), (off[..., 0] * off[..., 1]).sum(1)
    along = 0.5 * np.arctan2(2 * cxy, cxx - cyy)  # main axis of the neighbourhood
    every = np.vstack([points, exact[0]])
    return (every, np.vstack([np.column_stack([np.cos(along), np.sin(along)]), exact[1]]),
            np.concatenate([text[ys, xs], np.zeros(len(exact[0]), np.uint8)]),
            np.concatenate([np.zeros(len(points), bool), np.ones(len(exact[0]), bool)]), _on(frame, every, ppm),
            np.concatenate([cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)[ys, xs] / ppm,
                            np.zeros(len(exact[0]))]))


def _on(mask: np.ndarray | None, points: np.ndarray, ppm: float) -> np.ndarray:
    """Whether points (sheet mm) lie on a mask (``ppm`` pixels per mm)."""
    if mask is None or not len(points):
        return np.zeros(len(points), bool)
    xy = np.clip((points * ppm).astype(int), 0, [mask.shape[1] - 1, mask.shape[0] - 1])
    return mask[xy[:, 1], xy[:, 0]] > 0


def _fitted(hypotheses, ink: _Ink) -> list[_Hit]:
    return [h for h in (_fit(shape, pose, r, ink) for shape, pose, r in _screened(list(hypotheses), ink)) if h]


def _select(hits: list[_Hit], n: int) -> list[_Hit]:
    """The best hits that neither share much linework with a better one nor lie inside it."""
    claimed, found = np.zeros(n, bool), []
    for h in sorted(hits, key=lambda h: -h.score):
        if len(h.claim) and claimed[h.claim].mean() <= TAKEN and not any(_overlap(h, g) for g in found):
            claimed[h.claim] = True
            found.append(h)
    return found


def connections(d: Drawing) -> list[dict]:
    """Lines that join two symbol ports, as {"kind", "from": [symbol, port], "to": [symbol, port]} (indices)."""
    ports = {(round(x, 3), round(y, 3)): [i, k] for i, s in enumerate(d.symbols) for k, (x, y) in enumerate(s.ports)}
    ends = [(pl.kind, pl.points[0], pl.points[-1]) for pl in d.polylines if not pl.closed]
    ends += [("line", (ln.x1, ln.y1), (ln.x2, ln.y2)) for ln in d.lines]
    out = []
    for kind, a, b in ends:
        pa, pb = ports.get((round(a[0], 3), round(a[1], 3))), ports.get((round(b[0], 3), round(b[1], 3)))
        if pa and pb:
            out.append({"kind": kind, "from": pa, "to": pb})
    return out


# --------------------------------------------------------------------------- preparation

def _sample(q: Line | Arc | Circle) -> tuple[np.ndarray, np.ndarray]:
    """Points every STEP mm along a line, arc or circle and the direction of the stroke there."""
    if isinstance(q, Line):
        length = math.hypot(q.x2 - q.x1, q.y2 - q.y1)
        pts = np.linspace((q.x1, q.y1), (q.x2, q.y2), max(2, int(length / STEP) + 1))
        return pts, np.tile(np.subtract((q.x2, q.y2), (q.x1, q.y1)) / max(length, 1e-9), (len(pts), 1))
    start, sweep = (0.0, 360.0) if isinstance(q, Circle) else (q.start, (q.end - q.start) % 360 or 360)
    a = np.radians(start + np.linspace(0, sweep, max(3, int(math.radians(sweep) * q.r / STEP) + 1)))
    return np.column_stack([q.cx + q.r * np.cos(a), q.cy + q.r * np.sin(a)]), np.column_stack([-np.sin(a), np.cos(a)])


def _oriented(points: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Positions with the doubled direction angle as two more coordinates, so that a stroke and its reverse match."""
    c, s = heading[:, 0], heading[:, 1]
    return np.column_stack([points, TURN * (c * c - s * s), TURN * 2 * c * s])


def _matrix(angle: float, mirror: bool, scale: float) -> np.ndarray:
    c, s, m = math.cos(math.radians(angle)) * scale, math.sin(math.radians(angle)) * scale, -1 if mirror else 1
    return np.array([[c * m, -s], [s * m, c]])


def _shape(t: Template) -> _Shape:
    sampled = [_sample(q) for q in t.parts]
    pts, heading = np.vstack([p for p, _ in sampled]), np.vstack([h for _, h in sampled])
    group = np.repeat(t.groups, [len(p) for p, _ in sampled])
    try:
        hull = ConvexHull(pts).equations
    except QhullError:  # a single straight stroke: its bounding box
        lo, hi = pts.min(0) - STEP, pts.max(0) + STEP
        hull = np.array([[-1, 0, lo[0]], [1, 0, -hi[0]], [0, -1, lo[1]], [0, 1, -hi[1]]])
    circle = np.repeat([isinstance(q, Circle) for q in t.parts], [len(p) for p, _ in sampled])
    rim = ((pts @ hull[:, :2].T + hull[:, 2]).max(axis=1) > -2 * STEP) | circle | np.isin(group, _letters(t, sampled))
    lines = [_sample(q) for q in t.through]
    through = np.vstack([np.hstack(q) for q in lines]) if lines else np.zeros((0, 4))
    beyond = [np.hstack(_sample(q)) for ln in t.through for q in _continued(ln)]
    ports = np.array(t.ports, float).reshape(-1, 2)
    units = ports / np.maximum(np.linalg.norm(ports, axis=1, keepdims=True), 1e-9)
    variants, seen = [], []
    for mirror in (False, True):
        for k in range(4):
            p = pts @ _matrix(90 * k, mirror, 1).T
            if not any(cKDTree(q).query(p)[0].max() < 0.2 and cKDTree(p).query(q)[0].max() < 0.2 for q in seen):
                seen.append(p)
                variants.append((k, mirror, p.min(0), p.max(0)))
    ring = max(((np.array([q.cx, q.cy]), q.r) for q in t.parts if isinstance(q, Circle)), key=lambda c: c[1],
               default=None)
    gaps = [cKDTree(pts[group == a]).query(pts[group == b])[0].min() for a in set(group) for b in set(group) if a < b]
    smallest = TOL / min([g for g in gaps if g > STEP] or [np.inf])
    holds_text = any((np.array(p, float) @ hull[:, :2].T + hull[:, 2]).max() <= 0 for p in t.tags.values())
    return _Shape(t, pts, heading, group, rim, through, beyond, hull, ports, units,
                  float(np.linalg.norm(pts, axis=1).max()), smallest, variants, ring, rectangles(t), holds_text)


def _continued(ln: Line) -> tuple[Line, Line]:
    """The continuation of a line CONTINUED mm past each of its ends."""
    length = math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1) or 1.0
    ux, uy = CONTINUED * (ln.x2 - ln.x1) / length, CONTINUED * (ln.y2 - ln.y1) / length
    return Line(ln.x1 - ux, ln.y1 - uy, ln.x1, ln.y1), Line(ln.x2, ln.y2, ln.x2 + ux, ln.y2 + uy)


def _letters(t: Template, sampled: list) -> list[int]:
    """The parts (groups) of the symbol that are closed and small enough for OCR to read as a letter."""
    out = []
    for g in set(t.groups):
        mine = [p for (p, _), h in zip(sampled, t.groups) if h == g]
        ends = np.array([p[i] for p in mine for i in (0, -1)])
        if np.ptp(np.vstack(mine), 0).max() <= LETTER and cKDTree(ends).query(ends, k=2)[0][:, 1].max() < STEP:
            out.append(g)  # every end meets another one: closed
    return out


def rectangles(t: Template) -> np.ndarray:
    """The axis-parallel rectangles of the symbol's outline (centre x, y, width, height): anchors for finding it."""
    segments = np.array([(q.x1, q.y1, q.x2, q.y2) for q in t.parts if isinstance(q, Line)]).reshape(-1, 4)
    return _rectangles(segments, 1.0, np.inf)


def _pieces(d: Drawing) -> list[tuple[tuple, np.ndarray, np.ndarray]]:
    """Every piece of linework that may belong to a symbol: (owner, sample points, directions)."""
    out = [(("pl", i, j), *_sample(q)) for i, pl in enumerate(d.polylines) if pl.kind != "frame"
           for j, q in enumerate(pl.parts())]
    for kind in ("line", "arc", "circle"):
        out += [((kind, i), *_sample(q)) for i, q in enumerate(getattr(d, kind + "s"))]
    return out


# --------------------------------------------------------------------------- hypotheses

def _anchored(d: Drawing, shapes: list[_Shape]):
    """(shape, pose, search radius in mm, sizes to try) where two ports fit two line ends, and at circles."""
    ends = _ends(d, bend=10)
    shape, mirror, pa, pb = _port_choices(shapes, 2)
    if len(ends) > 1 and len(shape):
        points, out = np.array([p for p, _ in ends]), np.array([e for _, e in ends])
        w = pb - pa
        length, base, ua, ub = np.hypot(w[:, 0], w[:, 1]), np.arctan2(w[:, 1], w[:, 0]), _unit(pa), _unit(pb)
        i, j = cKDTree(points).query_pairs(length.max() * SCALES[1] + TOL, output_type="ndarray").T
        v = points[j] - points[i]
        scale = np.hypot(v[:, 0], v[:, 1])[:, None] / length  # end pairs x port choices
        turn = np.arctan2(v[:, 1], v[:, 0])[:, None] - base
        c, s = np.cos(turn), np.sin(turn)

        def facing(u, e):  # the line end points into the symbol: against the direction from its origin to the port
            return (c * u[:, 0] - s * u[:, 1]) * e[:, :1] + (s * u[:, 0] + c * u[:, 1]) * e[:, 1:]

        ok = (scale >= SCALES[0]) & (scale <= SCALES[1]) & (facing(ua, out[i]) < -0.94) & (facing(ub, out[j]) < -0.94)
        for n, k in np.argwhere(ok):
            cn, sn = c[n, k], s[n, k]
            origin = points[i[n]] - scale[n, k] * np.array([cn * pa[k, 0] - sn * pa[k, 1], sn * pa[k, 0] + cn * pa[k, 1]])
            yield shape[k], (*origin, math.degrees(turn[n, k]), bool(mirror[k]), float(scale[n, k])), 0.5, SIZES
    yield from _framed(d, [s for s in shapes if len(s.rects)])
    for c in d.circles:
        for s in shapes:
            if s.ring and SCALES[0] <= c.r / s.ring[1] <= SCALES[1]:
                for k, mirror, _, _ in s.variants:
                    m = _matrix(90 * k, mirror, c.r / s.ring[1])
                    yield s, (*(np.array([c.cx, c.cy]) - m @ s.ring[0]), 90 * k, mirror, c.r / s.ring[1]), 0.5, (1.0,)


def _on_pipes(d: Drawing, shapes: list[_Shape], ink: _Ink):
    """(shape, pose, search radius in mm, sizes to try) for the symbols drawn on a pipe that goes on (a flow arrow, an
    orifice): where strokes stand beside a straight line, as the barbs of an arrowhead do, the symbol along it, both
    ways. The pipe joins such strokes into one large group, so they are not found as a group of their own."""
    on = [s for s in shapes if len(s.through)]
    beside = ink.points[~ink.text & ~ink.straight]
    if not on or not len(beside):
        return
    tree = cKDTree(beside)
    pipes = [q for pl in d.polylines if pl.kind != "frame" and not pl.dashed for q in pl.parts() if isinstance(q, Line)]
    for ln in [q for q in d.lines if not q.dashed] + pipes:  # not the frame or signal lines
        a, b = np.array([ln.x1, ln.y1]), np.array([ln.x2, ln.y2])
        length = float(np.hypot(*(b - a)))
        if length < PIPE:
            continue
        u = (b - a) / length
        near = beside[tree.query_ball_point((a + b) / 2, length / 2 + BARB)]
        along, across = (near - a) @ u, np.abs((near - a) @ np.array([-u[1], u[0]]))
        along = np.sort(along[(along > 0) & (along < length) & (across > TOL) & (across < BARB)])
        for group in np.split(along, np.flatnonzero(np.diff(along) > 1.0) + 1):
            if len(group) < 4:
                continue  # a speck
            centre = a + u * (group[0] + group[-1]) / 2
            for s in on:
                mid = s.through[:, :2].mean(0)
                for turn in (math.atan2(u[1], u[0]), math.atan2(-u[1], -u[0])):
                    c, sn = math.cos(turn), math.sin(turn)
                    origin = centre - np.array([c * mid[0] - sn * mid[1], sn * mid[0] + c * mid[1]])
                    yield s, (*origin, math.degrees(turn), False, 1.0), 1.0, SIZES


def _leftover(d: Drawing, shapes: list[_Shape], ink: _Ink, found: list[_Hit]):
    """(shape, pose, search radius in mm, sizes to try) where no symbol was found yet: a port at a single line end (an
    off-page connector at the end of a pipe), at the library size, and groups of small strokes."""

    def free(p):
        return not any(_contains(h, p) for h in found)  # not inside a symbol found already

    ports = [transform(*h.pose)(*q) for h in found if not len(h.shape.through) for q in h.shape.t.ports]
    connected = cKDTree(ports) if ports else None
    shape, mirror, pa, _ = _port_choices(shapes, 1)
    for p, e in _ends(d, bend=45, shortest=STUB):  # pipe ends
        if connected and connected.query(p, distance_upper_bound=SNAP)[0] < np.inf:
            continue  # the line ends at a symbol found already
        if len(shape) and free(p):
            turn = math.atan2(-e[1], -e[0]) - np.arctan2(pa[:, 1], pa[:, 0])  # port direction against the line end
            c, s = np.cos(turn), np.sin(turn)
            origins = p - np.column_stack([c * pa[:, 0] - s * pa[:, 1], s * pa[:, 0] + c * pa[:, 1]])
            for k in range(len(shape)):
                yield shape[k], (*origins[k], math.degrees(turn[k]), bool(mirror[k]), 1.0), 1.0, (1.0,)
    # Symbols without ports or circles are only found as a group of strokes, which must be allowed their size
    loose = [np.hypot(*(s.pts.max(0) - s.pts.min(0))) for s in shapes if not len(s.at) and not s.ring]
    for lo, hi in _clusters(ink, max([SMALL] + [1.25 * size for size in loose])):
        size, centre = hi - lo, (lo + hi) / 2
        if not any(map(free, (lo, hi, (lo[0], hi[1]), (hi[0], lo[1])))):
            continue  # the whole group lies inside a symbol found already
        for s in shapes:
            for k, mirror_, tlo, thi in s.variants:
                scale = float(np.clip(max(size / np.maximum(thi - tlo, STEP)), *SCALES))
                if np.all(np.abs(size - scale * (thi - tlo)) <= 0.25 * scale * (thi - tlo) + 0.5):
                    yield s, (*(centre - scale * (tlo + thi) / 2), 90 * k, mirror_, scale), 2.0, (1.0,)


def _framed(d: Drawing, shapes: list[_Shape]):
    """(shape, pose, search radius in mm, sizes to try) where a rectangle of the drawing has the proportions of a
    rectangle of the symbol, turned and mirrored like the symbol may be: this fixes position and size."""
    if not shapes:
        return
    sides = np.concatenate([s.rects[:, 2:] for s in shapes])
    found = _rectangles(_segments(d), sides.min() * SCALES[0], sides.max() * SCALES[1], pairs=True)
    if not len(found):
        return
    for s in shapes:
        for tx, ty, tw, th in s.rects:
            for k, mirror, _, _ in s.variants:
                w, h = (th, tw) if k % 2 else (tw, th)
                sx, sy = found[:, 2] / w, found[:, 3] / h
                scale = np.sqrt(sx * sy)
                ok = (np.abs(sx - sy) <= ASPECT * scale) & (scale >= SCALES[0]) & (scale <= SCALES[1])
                for cx, cy, f in zip(found[ok, 0], found[ok, 1], scale[ok]):
                    m = _matrix(90 * k, mirror, f)
                    yield s, (*(np.array([cx, cy]) - m @ (tx, ty)), 90 * k, mirror, float(f)), 0.5, (1.0,)


def _segments(d: Drawing) -> np.ndarray:
    """The straight pieces of the drawing (x1, y1, x2, y2), walls included, the frame not. An arc that bends less
    than BEND counts with its chord: on photos a short side of a rectangle is often traced with a slight bend."""
    parts = [q for pl in d.polylines if pl.kind != "frame" for q in pl.parts()] + d.lines + d.arcs
    segments = [(q.x1, q.y1, q.x2, q.y2) if isinstance(q, Line) else _chord(q) for q in parts]
    return np.array([q for q in segments if q], float).reshape(-1, 4)


def _chord(a: Arc) -> tuple[float, float, float, float] | None:
    half = math.radians((a.end - a.start) % 360) / 2
    if a.r * (1 - math.cos(half)) > BEND:
        return None
    s, e = math.radians(a.start), math.radians(a.end)
    return a.cx + a.r * math.cos(s), a.cy + a.r * math.sin(s), a.cx + a.r * math.cos(e), a.cy + a.r * math.sin(e)


def _rectangles(segments: np.ndarray, lo: float, hi: float, pairs: bool = False) -> np.ndarray:
    """Axis-parallel rectangles (centre x, y, width, height), each side lo to hi mm long, whose four sides lie on the
    ``segments`` (x1, y1, x2, y2). A side may be part of a longer line: the back of a washbasin is the face of the
    wall it hangs on, the front of a hob the edge of a counter. With ``pairs``, also two opposite sides alone, the
    longer one perhaps part of a longer line: on a small fixture, OCR may have read the other two as letters."""
    x1, y1, x2, y2 = segments.T
    lean = np.maximum(AXIS, LEAN * np.hypot(x2 - x1, y2 - y1))
    flat, upright = np.abs(y2 - y1) <= lean, np.abs(x2 - x1) <= lean
    shortest = max(lo - 2 * CORNER, 0.5)
    hor = np.column_stack([(y1 + y2) / 2, np.minimum(x1, x2), np.maximum(x1, x2)])[flat & (np.abs(x2 - x1) >= shortest)]
    ver = np.column_stack([(x1 + x2) / 2, np.minimum(y1, y2), np.maximum(y1, y2)])[upright & (np.abs(y2 - y1) >= shortest)]
    out = set()
    for a, b in zip(*np.nonzero((ver[None, :, 0] - ver[:, None, 0] >= lo) & (ver[None, :, 0] - ver[:, None, 0] <= hi))):
        (xl, la, lb), (xr, ra, rb) = ver[a], ver[b]  # left and right side
        top, bottom = max(la, ra) - CORNER, min(lb, rb) + CORNER  # where both sides run
        across = hor[(hor[:, 1] <= xl + CORNER) & (hor[:, 2] >= xr - CORNER) & (hor[:, 0] >= top) & (hor[:, 0] <= bottom)]
        for ya, yb in ((ya, yb) for ya in across[:, 0] for yb in across[:, 0] if lo <= yb - ya <= hi):
            out.add((round((xl + xr) / 2, 2), round((ya + yb) / 2, 2), round(xr - xl, 2), round(yb - ya, 2)))
    for sides, upright in ((hor, False), (ver, True)) if pairs else ():
        at, a, b = sides.T
        apart, start, end = at[None] - at[:, None], np.maximum(a[:, None], a[None]), np.minimum(b[:, None], b[None])
        run, shorter = end - start, np.minimum((b - a)[:, None], (b - a)[None])
        i, j = np.nonzero((apart >= lo) & (apart <= hi) & (run >= lo) & (run <= hi) & (run >= shorter - 2 * CORNER))
        for c, m, r, w in zip((at[i] + at[j]) / 2, (start + end)[i, j] / 2, run[i, j], apart[i, j]):
            out.add(tuple(round(float(v), 2) for v in ((c, m, w, r) if upright else (m, c, r, w))))
    return np.array(sorted(out), float).reshape(-1, 4)


def _port_choices(shapes: list[_Shape], n: int):
    """Every way to pick ``n`` (1 or 2, in order) ports of a symbol, mirrored or not: shapes, mirror flags and the
    port positions in symbol coordinates (mirrored)."""
    rows = [(s, mirror, s.at[a] * (-1 if mirror else 1, 1), s.at[b] * (-1 if mirror else 1, 1))
            for s in shapes for mirror in (False, True) for a in range(len(s.at)) for b in range(len(s.at))
            if (a != b) == (n == 2) and (n == 2 or a == b)]
    if not rows:
        return [], np.zeros(0, bool), np.zeros((0, 2)), np.zeros((0, 2))
    return [r[0] for r in rows], np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), \
        np.array([r[3] for r in rows])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)[:, None]


def _ends(d: Drawing, bend: float, shortest: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Line ends with the unit vector pointing out of the line, also where a straight run of a polyline bends by more
    than ``bend`` degrees or turns into an arc (a pipe that was joined with a stroke of a symbol); only of straight
    pieces at least ``shortest`` mm long."""
    out = []
    for pl in d.polylines:
        if pl.kind == "frame" or pl.dashed:  # signal lines carry no symbols
            continue
        pts = np.array(pl.points + pl.points[:1] if pl.closed else pl.points, float)
        straight = np.array(pl.bulges) == 0
        step = np.diff(pts, axis=0)
        heading, long = _unit(step), np.hypot(step[:, 0], step[:, 1]) >= shortest
        ends = np.ones(len(heading) + 1, bool)  # a run ends at point k
        ends[1:-1] = ~straight[1:] | ~straight[:-1] | (np.sum(heading[1:] * heading[:-1], 1) < math.cos(math.radians(bend)))
        for k in np.flatnonzero(ends):
            if k > 0 and straight[k - 1] and long[k - 1]:
                out.append((pts[k], heading[k - 1]))
            if k < len(heading) and straight[k] and long[k]:
                out.append((pts[k], -heading[k]))
    for ln in d.lines:
        a, b = np.array([ln.x1, ln.y1]), np.array([ln.x2, ln.y2])
        if not ln.dashed and (n := np.hypot(*(b - a))) > 0 and n >= shortest:
            out += [(b, (b - a) / n), (a, (a - b) / n)]
    return out


def _title_block(d: Drawing) -> tuple[float, float, float, float] | None:
    """The title block in the bottom right corner of the frame (ISO 5457): its left edge runs up from the bottom of the
    frame, its top edge from there to the right of the frame. Where a line runs across the whole frame above it, the
    strip below that line instead: notes, revision table and title block, whose grids look like symbols as well.
    (left, top, right, bottom), None if there is none."""
    frame = [pl.points for pl in d.polylines if pl.kind == "frame"]
    rights = [max(x for x, _ in pts) for pts in frame if max(x for x, _ in pts) > d.width / 2]
    bottoms = [max(y for _, y in pts) for pts in frame if max(y for _, y in pts) > d.height / 2]
    if not rights or not bottoms:
        return None
    right, bottom = min(rights), min(bottoms)  # the inner line of a double frame
    segments = [(q.x1, q.y1, q.x2, q.y2) for pl in d.polylines if pl.kind != "frame" for q in pl.parts()
                if isinstance(q, Line)] + [(q.x1, q.y1, q.x2, q.y2) for q in d.lines]
    sides = [min(x for x, _ in pts) for pts in frame if min(x for x, _ in pts) < d.width / 2]
    across = [y1 for x1, y1, x2, y2 in segments if sides and abs(y2 - y1) < TOL and abs(min(x1, x2) - max(sides)) < SNAP
              and abs(max(x1, x2) - right) < SNAP and bottom - BLOCK[1] < y1 < bottom - 5]
    if across:
        return max(sides), min(across), right, bottom
    lefts = [x1 for x1, y1, x2, y2 in segments if abs(x2 - x1) < TOL and abs(max(y1, y2) - bottom) < SNAP
             and min(y1, y2) > bottom - BLOCK[1] and right - BLOCK[0] < x1 < right - 5]
    for left in sorted(lefts):
        tops = [y1 for x1, y1, x2, y2 in segments if abs(y2 - y1) < TOL and abs(min(x1, x2) - left) < SNAP
                and abs(max(x1, x2) - right) < SNAP and bottom - BLOCK[1] < y1 < bottom - 5]
        if tops:
            return left, min(tops), right, bottom
    return None


def _within(p, box) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _clusters(ink: _Ink, limit: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bounding boxes (low, high corner) of groups of ink points within a millimetre of each other and up to ``limit``
    mm across, leaving out text and long straight lines: a pipe would make the group larger than the symbol on it."""
    points = ink.points[~ink.text & ~ink.straight]
    if len(points) < 2:
        return []
    i, j = cKDTree(points).query_pairs(1.0, output_type="ndarray").T
    _, label = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(points),) * 2), directed=False)
    order = np.argsort(label, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(label[order]) > 0])
    lo, hi = np.minimum.reduceat(points[order], starts), np.maximum.reduceat(points[order], starts)
    big = np.diff(np.r_[starts, len(order)]) >= 4  # more than a speck
    return [(a, b) for a, b in zip(lo[big], hi[big]) if np.hypot(*(b - a)) <= limit]


def _misread(h: _Hit, ink: _Ink) -> bool:
    """Whether most of the text ink inside the symbol lies on its outline: its strokes read as letters (two hotplates
    as "00", a small washbasin as "Ol"), not a label written inside it."""
    near = np.array(ink.tree.query_ball_point(h.pose[:2], h.shape.radius * h.pose[4] + TOL), int)
    near = near[ink.text[near]] if len(near) else near
    if not len(near):
        return False
    q = (ink.points[near] - h.pose[:2]) @ np.linalg.inv(_matrix(*h.pose[2:])).T
    inside = near[np.all(q @ h.shape.hull[:, :2].T + h.shape.hull[:, 2] <= TOL / h.pose[4], axis=1)]
    return len(inside) > 0 and np.isin(inside, h.claim).mean() >= 0.5


def _stub(p: np.ndarray) -> bool:
    """A straight piece longer than STUB mm."""
    chord = p[-1] - p[0]
    length = float(np.hypot(*chord))
    if length <= STUB:
        return False
    off = p - p[0]
    return float(np.abs(off[:, 0] * chord[1] - off[:, 1] * chord[0]).max()) / length < STEP


# --------------------------------------------------------------------------- matching

def _fit(s: _Shape, pose, radius: float, ink: _Ink) -> _Hit | None:
    """Moves the hypothesis to where most of its outline lies on linework and scores it there."""
    x, y, angle, mirror, scale = pose
    if scale < s.smallest:
        return None
    m = _matrix(angle, mirror, scale)
    outline = _oriented(s.pts @ m.T, s.heading @ m.T / scale)
    xy = np.array([x, y, 0, 0])
    for reach, step, every in ((radius, 0.5, 3), (0.25, 0.25, 1)):  # every third point is enough to find the place
        k = np.arange(-reach, reach + 1e-9, step)
        shifts = np.column_stack([np.stack(np.meshgrid(k, k), -1).reshape(-1, 2), np.zeros((len(k) ** 2, 2))]) + xy
        found = _found((outline[::every][None] + shifts[:, None]).reshape(-1, 4), np.tile(s.rim[::every], len(shifts)),
                       ink, TOL).reshape(len(shifts), -1).mean(1)
        if found.max() < COVERAGE - 0.1:
            return None
        xy = shifts[int(np.argmax(found))]
    outline += xy
    found = _found(outline, s.rim, ink, TOL)
    coverage = found.mean()
    if coverage < COVERAGE or (np.bincount(s.group, found) / np.bincount(s.group)).min() < PART:
        return None
    if np.mean(ink.unworded.query(outline, distance_upper_bound=TOL)[0] <= TOL) < coverage - LETTERS:
        return None  # made of letters: words may stand in for a stroke or two (a misread arrow tip), not for a symbol
    if len(s.through):  # the line running through the symbol, e.g. the pipe under a flow arrow, and on past both ends
        def on(q):
            line = _oriented(q[:, :2] @ m.T + xy[:2], q[:, 2:] @ m.T / scale)
            return np.mean(ink.strokes.query(line, distance_upper_bound=TOL)[0] <= TOL)
        if on(s.through) < COVERAGE or min(map(on, s.beyond)) < COVERAGE:
            return None  # not on a pipe: a bend where an arc runs into a line looks like an arrowhead
    near = np.array(ink.tree.query_ball_point(xy[:2], s.radius * scale + REACH), int)
    q = (ink.points[near] - xy[:2]) @ np.linalg.inv(m).T  # in the symbol's own coordinates
    if len(s.ports) and not _attached(s, q[~ink.text[near]], scale):
        return None
    inside = np.all(q @ s.hull[:, :2].T + s.hull[:, 2] <= TOL / scale, axis=1)
    explained = cKDTree(outline).query(ink.oriented[near], distance_upper_bound=TOL)[0] < np.inf
    in_port = _in_port(s, q, m, ink.oriented[near])
    texts = np.array(list(s.t.tags.values())).reshape(-1, 1, 2)  # where the library expects text, even unread
    in_text = np.any(np.all(np.abs(q[None] - texts) <= (4.0, 1.5), axis=2), axis=0) if len(texts) else False
    clutter = np.count_nonzero(inside & ~explained & ~in_port & ~ink.text[near] & ~in_text)
    if clutter > CLUTTER * len(s.pts) * scale:
        return None
    claim = near[inside & explained & ~in_port]
    if not s.holds_text and len(claim) and ink.frame[claim].mean() > FRAMED:
        return None  # the frame around a text (a line number's box), which only a symbol holding text may be
    # linework explained, minus the outline missing and the linework inside left unexplained, in mm
    score = (len(claim) - MISSING * (1 - coverage) * len(s.pts) * scale - clutter) * STEP
    return _Hit(s, (float(xy[0]), float(xy[1]), angle % 360, mirror, scale), score, claim, float(coverage))


def _rounded(h: _Hit, arcs: list[Arc], ink: _Ink) -> bool:
    """Whether a symbol drawn without curves lies on the round end of the frame around a text, such as an off-page
    connector or a flow arrow on the end of a line-number box. Only symbols on pipes (with ports): a fixture's oval
    bowl may come from a library as short straight lines."""
    if not h.shape.t.ports or any(isinstance(q, (Arc, Circle)) for q in h.shape.t.parts):
        return False
    outline = cKDTree(h.shape.pts @ _matrix(*h.pose[2:]).T + h.pose[:2])
    for a in arcs:
        sweep = (a.end - a.start) % 360
        if sweep < 90 or a.r < max(ROUND_END, 0.15 * h.shape.radius) * h.pose[4] or not _contains(h, (a.cx, a.cy)):
            continue  # a bend, the curve of a letter or the blunt tip of an arrow, or a curve beside the symbol
        t = np.radians(a.start + np.linspace(0.2, 0.8, 4) * sweep)  # along its middle, not at its ends
        middle = np.column_stack([a.cx + a.r * np.cos(t), a.cy + a.r * np.sin(t)])
        near = [k for p in middle for k in ink.tree.query_ball_point(p, TOL)]
        if outline.query(middle)[0].min() <= 2 * TOL and near and ink.frame[near].mean() > 0.5:
            return True
    return False


def _boxed(d: Drawing, ink: _Ink) -> np.ndarray:
    """Whether each ink point lies on a closed box of the drawing (sides BOX). A box whose text was not read, such as
    a line size, otherwise passes for a symbol: its sides for the ring of a bubble, its letters for a valve."""
    rects = _rectangles(_segments(d), *BOX)
    if not len(rects):
        return np.zeros(len(ink.points), bool)
    sides = []
    for cx, cy, w, h in rects:
        c = np.array([(-w, -h), (w, -h), (w, h), (-w, h), (-w, -h)]) / 2 + (cx, cy)
        sides += [np.linspace(p, q, int(np.hypot(*(q - p)) / STEP) + 2) for p, q in zip(c, c[1:])]
    return cKDTree(np.vstack(sides)).query(ink.points, distance_upper_bound=TOL)[0] <= TOL


def _centred(h: _Hit, circles: list[Circle]) -> _Hit:
    """The hit moved and scaled so that the circle of its symbol, such as the ring of an instrument bubble, is the
    circle found in the scan: the fit moves in quarter millimetres, and the text inside pulls it aside. Only symbols
    drawn around one circle, not a hob with its hotplates or a shower tray with its drain."""
    rings = [q for q in h.shape.t.parts if isinstance(q, Circle)]
    if len(rings) != 1 or rings[0].r < 0.5 * h.shape.radius or not circles:
        return h
    ring = rings[0]
    x, y, angle, mirror, scale = h.pose
    cx, cy = transform(x, y, angle, mirror, scale)(ring.cx, ring.cy)
    r = ring.r * scale
    near = [c for c in circles if math.hypot(c.cx - cx, c.cy - cy) < max(0.5, 0.25 * r) and abs(c.r - r) < 0.15 * r]
    if not near:
        return h
    c = min(near, key=lambda c: math.hypot(c.cx - cx, c.cy - cy))
    scale *= c.r / r
    ox, oy = transform(0.0, 0.0, angle, mirror, scale)(ring.cx, ring.cy)
    return replace(h, pose=(c.cx - ox, c.cy - oy, angle, mirror, scale))


def _twins(shapes: list[_Shape]) -> dict[int, list]:
    """For each symbol, the symbols drawn the same with a small circle or two more (a gate valve and a ball valve):
    {id(shape): [(twin, circle centres, radii)]}."""
    out = defaultdict(list)
    for t in shapes:
        small = [q for q in t.t.parts if isinstance(q, Circle) and q.r <= SMALL_CIRCLE]
        rest = [q for q in t.t.parts if not (isinstance(q, Circle) and q.r <= SMALL_CIRCLE)]
        if not small or not rest:
            continue
        pts = np.vstack([_sample(q)[0] for q in rest])
        for s in shapes:
            if s is not t and len(s.pts) and cKDTree(pts).query(s.pts)[0].max() < STEP and \
                    cKDTree(s.pts).query(pts)[0].max() < STEP:
                out[id(s)].append((t, np.array([(q.cx, q.cy) for q in small]), np.array([q.r for q in small])))
    return out


def _circled(h: _Hit, twins: dict[int, list], ink: _Ink) -> _Hit:
    """The twin of a symbol found where its small circles are there (see _twins): as a ring, or as a blob as thick as
    the circle, which is what a filled circle leaves, or one that blur has filled in on a photo."""
    x, y, angle, mirror, scale = h.pose
    m = _matrix(angle, mirror, scale)
    for t, centres, radii in twins.get(id(h.shape), []):
        if all(_round_mark(m @ c + (x, y), r * scale, ink) for c, r in zip(centres, radii)):
            return replace(h, shape=t)
    return h


def _round_mark(c: np.ndarray, r: float, ink: _Ink) -> bool:
    """Whether a circle of radius r around c is drawn: half of its rim, running round it, or a blob that thick and
    much thicker than the strokes of the sheet (where two strokes cross, blur thickens the ink too)."""
    near = np.array(ink.tree.query_ball_point(c, r + TOL), int)
    if not len(near):
        return False
    off = ink.points[near] - c
    dist = np.hypot(off[:, 0], off[:, 1])
    radial = np.abs(np.einsum("ij,ij->i", off, ink.heading[near])) / np.maximum(dist, 1e-9)
    rim = (np.abs(dist - r) <= TOL) & (radial < 0.5) & ~ink.text[near]  # running round, not through the centre
    # nearly r: the middle of a filled circle; the crossing in a gate valve's middle reaches about 0.7 r
    blob = (dist <= 0.5 * r) & (ink.thick[near] >= max(0.9 * r, BLOB * ink.usual))
    return rim.sum() * STEP >= 0.5 * 2 * math.pi * r or blob.any()


def _doubt(h: _Hit, first: _Hit, hits: list[_Hit]) -> str | None:
    """Why a symbol found (``h``, found as ``first`` before its circles were looked at) should be checked: a symbol of
    another kind fits its place almost as well (not its twin, which its circles decide), or little of its outline was
    found."""
    own = {h.shape.t.name, first.shape.t.name}
    rival = max((g for g in hits if g.shape.t.name not in own and _overlap(g, first)), key=lambda g: g.score,
                default=None)
    if rival and rival.score >= RIVAL * first.score:
        return f"{rival.shape.t.description} fits almost as well"
    if h.found < SURE:
        return f"only {math.floor(100 * h.found)} % of its outline found"
    return None


def _one_size(hits: list[_Hit], n: int) -> list[_Hit]:
    """Leaves out the hits far off the size of the sheet's symbols. A sheet draws its symbols at one size, so a symbol
    much smaller or larger than the others is something else that fits it at that size, such as the x marks of a
    capillary line taken for flow arrows. Only symbols drawn at the printed size: a fixture's size is the real one's."""
    printed = [h for h in _select(hits, n) if _printed(h)]
    if len(printed) < ONE_SIZE_LEAST:
        return hits
    size = float(np.median([h.pose[4] for h in printed]))
    return [h for h in hits if not _printed(h) or abs(math.log(h.pose[4] / size)) <= math.log(ONE_SIZE)]


def _printed(h: _Hit) -> bool:
    library = h.shape.t.library
    return library is None or library.size == "printed"


def _attached(s: _Shape, q: np.ndarray, scale: float) -> bool:
    """Whether a line leaves at least one port of the symbol (points in symbol coordinates): a valve sits in a pipe."""
    along = q @ s.ports.T - np.linalg.norm(s.at, axis=1)  # beyond the port
    across = np.abs(q[:, :1] * s.ports[:, 1] - q[:, 1:] * s.ports[:, 0])
    leaving = (along > STEP / scale) & (along < 2 / scale) & (across <= TOL / scale)
    return bool(leaving.sum(axis=0).max() >= 3)


def _in_port(s: _Shape, q: np.ndarray, m: np.ndarray, oriented: np.ndarray) -> np.ndarray:
    """Points (``q`` in symbol coordinates, ``oriented`` on the sheet) on a line leaving through one of the ports:
    from the origin outwards and running that way, like a pipe (not like the ring of a ball valve crossing it)."""
    if not len(s.ports):
        return np.zeros(len(q), bool)
    scale = math.sqrt(abs(np.linalg.det(m)))
    along = q @ s.ports.T
    across = np.abs(q[:, :1] * s.ports[:, 1] - q[:, 1:] * s.ports[:, 0])
    axis = _unit(s.ports @ m.T)
    doubled = np.vstack([axis[:, 0] ** 2 - axis[:, 1] ** 2, 2 * axis[:, 0] * axis[:, 1]])
    return np.any((along > 0) & (across <= TOL / scale) & (oriented[:, 2:] / TURN @ doubled > 0.6), axis=1)


def _overlap(a: _Hit, b: _Hit) -> bool:
    """One hit lies mostly inside the other: the same symbol found twice, or an arrowhead inside a valve."""
    if math.hypot(a.pose[0] - b.pose[0], a.pose[1] - b.pose[1]) > a.shape.radius * a.pose[4] + b.shape.radius * b.pose[4]:
        return False
    return max(_inside(a, b), _inside(b, a)) >= 0.5


def _contains(h: _Hit, p) -> bool:
    """Whether the point lies inside the convex hull of the hit."""
    s, scale = h.shape, h.pose[4]
    if math.hypot(p[0] - h.pose[0], p[1] - h.pose[1]) > s.radius * scale + TOL:
        return False
    q = (np.asarray(p) - h.pose[:2]) @ np.linalg.inv(_matrix(*h.pose[2:])).T
    return bool(np.all(s.hull[:, :2] @ q + s.hull[:, 2] <= TOL / scale))


def _inside(a: _Hit, b: _Hit) -> float:
    """Share of the outline of ``a`` inside the convex hull of ``b``."""
    outline = a.shape.pts @ _matrix(*a.pose[2:]).T + a.pose[:2]
    s, scale = b.shape, b.pose[4]
    q = (outline - b.pose[:2]) @ np.linalg.inv(_matrix(*b.pose[2:])).T
    return float(np.mean(np.all(q @ s.hull[:, :2].T + s.hull[:, 2] <= TOL / scale, axis=1)))


def _screened(hypotheses: list, ink: _Ink, chunk: int = 4000) -> list:
    """The hypotheses (shape, pose, search radius, sizes to try) with most of their outline (every third point) near
    linework, checked in batches; most end here. Of the sizes tried, the one with most outline on linework stays."""
    tried = [(k, (s, (x, y, angle, mirror, scale * f), r)) for k, (s, (x, y, angle, mirror, scale), r, sizes) in
             enumerate(hypotheses) for f in sizes]
    best = {}
    for start in range(0, len(tried), chunk):
        batch = tried[start:start + chunk]
        outlines = []
        for _, (s, (x, y, angle, mirror, scale), _) in batch:
            m = _matrix(angle, mirror, scale)
            outlines.append(_oriented(s.pts[::3] @ m.T + (x, y), s.heading[::3] @ m.T / scale))
        points, rim = np.vstack(outlines), np.concatenate([s.rim[::3] for _, (s, _, _) in batch])
        starts = np.cumsum([0] + [len(o) for o in outlines[:-1]])
        dist = _distances(points, rim, ink, TOL + max(r for _, (_, _, r) in batch))
        loose = np.add.reduceat(dist <= np.repeat([TOL + r for _, (_, _, r) in batch], [len(o) for o in outlines]),
                                starts)
        exact = np.add.reduceat(dist <= TOL, starts)  # to pick the size
        for (k, h), n, near, sharp in zip(batch, [len(o) for o in outlines], loose, exact):
            precise = h[2] <= 1  # anchored at line ends or a circle: most of the outline must lie on linework already
            if near >= 0.8 * n and (not precise or sharp >= 0.3 * n) and sharp / n > best.get(k, (-1, None))[0]:
                best[k] = (sharp / n, h)
    kept, out = defaultdict(list), []
    for _, h in sorted(best.values(), key=lambda b: -b[0]):  # one symbol is often reached from several ends: fit once
        s, (x, y, angle, mirror, _), _ = h
        key = (id(s), mirror, round(angle / 5) % 72)
        if not any(math.hypot(x - a, y - b) < 0.75 for a, b in kept[key]):
            kept[key].append((x, y))
            out.append(h)
    return out


def _found(outline: np.ndarray, rim: np.ndarray, ink: _Ink, tol) -> np.ndarray:
    """Which outline points (oriented) have linework within ``tol``."""
    return _distances(outline, rim, ink, np.max(tol)) <= tol


def _distances(outline: np.ndarray, rim: np.ndarray, ink: _Ink, upper: float) -> np.ndarray:
    """The distance from each outline point (oriented) to the nearest linework, infinite beyond ``upper``. Text counts
    only on the outer outline (a misread arrow tip), on circles and on small closed shapes (two hotplates read as
    "00"), not elsewhere inside (letters in a bubble do not make a bar across it)."""
    dist = np.full(len(outline), np.inf)
    for tree, which in ((ink.oriented_tree, rim), (ink.strokes, ~rim)):
        if which.any() and tree.n:
            dist[which] = tree.query(outline[which], distance_upper_bound=upper)[0]
    return dist


# --------------------------------------------------------------------------- replacing the linework

def _remove(d: Drawing, pieces: list, found: list[_Hit]) -> None:
    """Drops the pieces mostly explained by symbols; polylines are split where pieces drop out."""
    if not found:
        return
    points = np.vstack([p for _, p, _ in pieces])
    oriented, tree = _oriented(points, np.vstack([h for _, _, h in pieces])), cKDTree(points)
    explained = np.zeros(len(points), bool)
    for h in found:
        near = np.array(tree.query_ball_point(h.pose[:2], h.shape.radius * h.pose[4] + TOL), int)
        if len(near):
            explained[near[h.explains(points[near], oriented[near])]] = True

    def wall(owner, p) -> bool:  # a piece of a wall that reaches beyond the symbols: the washbasin hangs on it
        return owner[0] == "pl" and d.polylines[owner[1]].kind == "wall" and \
            not all(any(_contains(h, q) for h in found) for q in p[::4])

    gone, start = set(), 0
    for owner, p, _ in pieces:
        if explained[start:start + len(p)].mean() >= GONE and not wall(owner, p):
            gone.add(owner)
        start += len(p)
    drop = defaultdict(set)
    for owner in gone:
        if owner[0] == "pl":
            drop[owner[1]].add(owner[2])
    d.polylines = [q for i, pl in enumerate(d.polylines) for q in _split(pl, drop[i]) if _length(q) >= 1.0]
    for kind in ("line", "arc", "circle"):
        setattr(d, kind + "s", [q for i, q in enumerate(getattr(d, kind + "s")) if (kind, i) not in gone])


def _split(pl: Polyline, drop: set[int]) -> list[Polyline]:
    if not drop:
        return [pl]
    n = len(pl.bulges)
    pts = pl.points + pl.points[:1] if pl.closed else pl.points
    order = [(min(drop) + 1 + i) % n for i in range(n)] if pl.closed else range(n)  # open a closed one at a gap
    runs, run = [], []
    for j in order:
        if j in drop:
            runs.append(run)
            run = []
        else:
            run.append(j)
    runs.append(run)
    return [Polyline([pts[j] for j in run] + [pts[run[-1] + 1]], [pl.bulges[j] for j in run], False, pl.width,
                     pl.dashed, pl.kind) for run in runs if run]


def _length(pl: Polyline) -> float:
    pts = pl.points + pl.points[:1] if pl.closed else pl.points
    return sum(math.dist(a, b) for a, b in pairwise(pts))


def _snap(d: Drawing) -> None:
    """Moves line ends close to a port onto it, so that the line is connected to the symbol, if the port lies on the
    line's course: a port beside it belongs to another line or a misplaced symbol. Of parallel lines that
    end beside one port (two pipes running past a valve), only the one pointing at it is moved, so the other is not
    turned onto it. A short line with both ends at the same symbol is left over from its strokes and dropped: snapped,
    it would join the symbol to itself."""
    _inside_dropped(d)
    ends = [(ln, k) for ln in d.lines for k in (0, 1)] + [(pl, k) for pl in d.polylines if not pl.closed for k in (0, 1)]
    runs = np.array([(q.x1, q.y1, q.x2, q.y2) for q in d.lines + [q for pl in d.polylines for q in pl.parts()
                                                                   if isinstance(q, Line)]]).reshape(-1, 4)
    if ends:
        at = cKDTree([_end(*item)[0] for item in ends])
        for sym in d.symbols:
            _onto_lines(sym, ends, at, runs)
    ports = [p for s in d.symbols for p in s.ports]
    if not ports:
        return
    owner = [i for i, s in enumerate(d.symbols) for _ in s.ports]
    tree = cKDTree(ports)

    def port(p) -> int | None:
        dist, i = tree.query(p, distance_upper_bound=SNAP)
        return int(i) if dist < np.inf else None

    def leftover(a, b, length: float) -> bool:
        i, k = port(a), port(b)
        return i is not None and k is not None and owner[i] == owner[k] \
            and length <= math.dist(ports[i], ports[k]) + 2 * SNAP

    d.lines = [ln for ln in d.lines if not leftover(a := (ln.x1, ln.y1), b := (ln.x2, ln.y2), math.dist(a, b))]
    d.polylines = [pl for pl in d.polylines if pl.closed or not leftover(pl.points[0], pl.points[-1], _length(pl))]
    offers = []
    for item in [(ln, k) for ln in d.lines for k in (0, 1)] + [(pl, k) for pl in d.polylines if not pl.closed
                                                                for k in (0, 1)]:
        p, q = _end(*item)
        if (i := port(p)) is not None and (aside := _aside(p, q, ports[i])) <= SNAP_ASIDE:
            offers.append((aside, i, math.atan2(p[1] - q[1], p[0] - q[0]), item))
    taken: dict[int, list[float]] = {}
    for _, i, heading, (obj, k) in sorted(offers, key=lambda o: o[0]):
        if any(abs(math.remainder(heading - h, 2 * math.pi)) < SNAP_PARALLEL for h in taken.get(i, [])):
            continue  # a parallel line that points at the port more closely has it
        taken.setdefault(i, []).append(heading)
        _set_end(obj, k, tuple(map(float, ports[i])))


def _inside_dropped(d: Drawing) -> None:
    """Drops the pieces of linework that lie wholly inside a symbol with ports, such as the stroke left between the
    two triangles of a gate valve: it is no pipe, and it would join the pipes on either side through the valve. A
    symbol a line runs through (a flow arrow on its pipe) keeps it."""
    hulls = []
    for sym in d.symbols:
        if len(sym.ports) < 2 or sym.template.through:
            continue
        pts = np.vstack([_sample(q)[0] for q in sym.parts])
        try:
            hulls.append(ConvexHull(pts).equations)
        except QhullError:
            continue
    if not hulls:
        return

    def inside(*points) -> bool:
        p = np.array(points, float)
        return any(np.all(p @ h[:, :2].T + h[:, 2] <= TOL, axis=1).all() for h in hulls)

    d.lines = [ln for ln in d.lines if not inside((ln.x1, ln.y1), (ln.x2, ln.y2))]
    d.polylines = [q for pl in d.polylines
                   for q in _split(pl, {j for j in range(len(pl.bulges)) if not pl.closed
                                        and inside(pl.points[j], pl.points[j + 1])})]


def _onto_lines(sym: Symbol, ends: list, at: cKDTree, runs: np.ndarray) -> None:
    """Moves a symbol across the lines that end at its ports (``ends``, located by ``at``), or that run through two of
    them (``runs``, straight lines: a flow arrow or a valve drawn on a pipe that goes on), so that the ports lie on
    their course: the fit is only good to half a millimetre or so, and a pipe end moved onto a port beside it would be
    turned askew. A symbol placed on a circle stays (see _centred)."""
    if not sym.ports or any(isinstance(q, Circle) for q in sym.parts):
        return
    normals, offsets = _through(sym.ports, runs)
    for port in sym.ports:
        best = None  # the line end pointing at the port most closely
        for j in at.query_ball_point(port, SNAP):
            p, q = _end(*ends[j])
            length = math.dist(p, q)
            if length == 0:
                continue
            u = (np.array(p) - np.array(q)) / length
            if u @ (np.array(port) - np.array(p)) < -SNAP_ASIDE:
                continue  # the port lies behind the end: the line runs past it
            n = np.array([-u[1], u[0]])
            off = float(n @ (np.array(p) - np.array(port)))  # how far the port is off the line's course
            if best is None or abs(off) < abs(best[1]):
                best = (n, off)
        if best is not None:
            normals.append(best[0])
            offsets.append(best[1])
    if not normals:
        return
    n, off = np.array(normals), np.array(offsets)
    shift = np.linalg.lstsq(n, off, rcond=None)[0]  # the smallest move that puts the ports on all those courses
    if np.hypot(*shift) > SNAP or np.abs(n @ shift - off).max() > SNAP_ASIDE:
        return  # the lines disagree: not the symbol's pipes, or a misplaced symbol
    moved = place(sym.template, sym.x + float(shift[0]), sym.y + float(shift[1]), sym.angle, sym.mirror, sym.scale)
    sym.x, sym.y, sym.parts, sym.ports = moved.x, moved.y, moved.parts, moved.ports


def _through(ports: list, runs: np.ndarray) -> tuple[list, list]:
    """The normal of a straight line that runs past two of the ports, along the line between them and within SNAP of
    each, and how far each port lies off it; nothing if there is none."""
    for a, b in ((a, b) for i, a in enumerate(ports) for b in ports[i + 1:]):
        axis = np.subtract(b, a)
        if not len(runs) or not (span := float(np.hypot(*axis))):
            continue
        u = np.c_[runs[:, 2] - runs[:, 0], runs[:, 3] - runs[:, 1]]
        length = np.hypot(u[:, 0], u[:, 1])
        u = u / np.maximum(length, 1e-9)[:, None]
        n = np.c_[-u[:, 1], u[:, 0]]
        ok = np.abs(u @ axis) >= span * math.cos(SNAP_PARALLEL)
        for p in (a, b):
            rel = np.asarray(p) - runs[:, :2]
            along = np.einsum("ij,ij->i", rel, u)
            ok &= (np.abs(np.einsum("ij,ij->i", rel, n)) <= SNAP) & (along >= 0) & (along <= length)
        if ok.any():
            k = int(np.flatnonzero(ok)[0])
            return [n[k], n[k]], [float(n[k] @ (runs[k, :2] - np.asarray(p))) for p in (a, b)]
    return [], []


def _end(obj: Line | Polyline, k: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """End k (0 or 1) of a line or polyline and the point before it."""
    if isinstance(obj, Line):
        return ((obj.x1, obj.y1), (obj.x2, obj.y2)) if k == 0 else ((obj.x2, obj.y2), (obj.x1, obj.y1))
    return (obj.points[0], obj.points[1]) if k == 0 else (obj.points[-1], obj.points[-2])


def _set_end(obj: Line | Polyline, k: int, p: tuple[float, float]) -> None:
    if isinstance(obj, Line):
        if k == 0:
            obj.x1, obj.y1 = p
        else:
            obj.x2, obj.y2 = p
    else:
        obj.points[0 if k == 0 else -1] = p


def _aside(p, q, port) -> float:
    """How far the port lies beside the straight through q and the end p (mm)."""
    dx, dy = p[0] - q[0], p[1] - q[1]
    length = math.hypot(dx, dy)
    return math.dist(p, port) if length == 0 else abs((port[0] - p[0]) * dy - (port[1] - p[1]) * dx) / length


def _attributes(d: Drawing, placed: list[Symbol]) -> None:
    """Texts at the attribute positions of a symbol (its tag, the letters in an instrument bubble) become its
    attributes. A symmetric symbol fits the linework as well in each of its symmetric poses; it takes the one with
    most texts at its attributes, then the one whose back (the side through its insertion point) lies most on the
    linework left around it and continues it: a washbasin with its back on the wall face, not its front edge."""
    readable = [t for t in d.texts if not t.uncertain]
    texts_at = cKDTree([(t.x, t.y) for t in readable]) if readable else None
    left = [p for _, p, _ in _pieces(d)]  # the linework the symbols did not take
    lines = cKDTree(np.vstack(left)) if left else None
    taken: set[int] = set()

    def texts(t: Template, pose) -> dict[str, int]:
        f, found = transform(*pose), {}
        for tag, at in t.tags.items() if texts_at else ():
            dist, i = texts_at.query(f(*at), distance_upper_bound=REACH * pose[4])
            if dist < np.inf and i not in taken and i not in found.values():
                found[tag] = i
        return found

    def on_wall(t: Template, pose) -> int:  # points of the back line, at the insertion point and beyond both ends
        back = _back(t)
        if not lines or back is None:
            return 0
        u, ends = back
        f = transform(*pose)
        points = [(0.0, 0.0)] + [(end + g * np.sign(end)) * u for end in ends for g in (0.5, 1.0, 1.5)]
        return int(np.sum(lines.query([f(*p) for p in points])[0] <= TOL))

    for sym in placed:
        t = sym.template
        pose = (sym.x, sym.y, sym.angle, sym.mirror, sym.scale)
        if not t.tags and not lines:
            continue
        m = _matrix(sym.angle, sym.mirror, sym.scale)
        poses = [pose] + [(*(np.array([sym.x, sym.y]) + m @ shift), (sym.angle + (-turn if sym.mirror else turn)) % 360,
                           sym.mirror != flip, sym.scale) for turn, flip, shift in _symmetries(t)]
        found, best = max(((texts(t, q), q) for q in poses), key=lambda o: (len(o[0]), on_wall(t, o[1])))
        if best is not pose:
            turned = place(t, float(best[0]), float(best[1]), best[2], best[3], best[4])
            sym.x, sym.y, sym.angle, sym.mirror = turned.x, turned.y, turned.angle, turned.mirror
            sym.parts, sym.ports = turned.parts, turned.ports
        taken.update(found.values())
        sym.attributes.update({tag: readable[i] for tag, i in found.items()})
    gone = {id(readable[i]) for i in taken}
    d.texts = [t for t in d.texts if id(t) not in gone]


def _back(t: Template) -> tuple[np.ndarray, tuple[float, float]] | None:
    """The direction of the straight side through the insertion point (the back of a fixture) and how far it reaches
    either way, or None if the insertion point lies on no straight side."""
    for q in t.parts:
        if isinstance(q, Line) and (length := math.hypot(q.x2 - q.x1, q.y2 - q.y1)) > 0:
            u = np.array([q.x2 - q.x1, q.y2 - q.y1]) / length
            a, b = sorted((np.dot((q.x1, q.y1), u), np.dot((q.x2, q.y2), u)))
            if abs(q.x1 * u[1] - q.y1 * u[0]) < STEP and a < 0 < b:
                return u, (float(a), float(b))
    return None


def _symmetries(t: Template) -> list[tuple[float, bool, np.ndarray]]:
    """The quarter turns and mirror flips about the middle of the symbol that leave its outline where it is, with
    the shift of the insertion point they cause (in symbol coordinates): a hob turned half way looks the same, but
    has its insertion point on the other side."""
    pts = np.vstack([_sample(q)[0] for q in t.parts])
    middle, tree, out = (pts.min(0) + pts.max(0)) / 2, cKDTree(pts), []
    for turn in (0, 90, 180, 270):
        for flip in (False, True):
            r = _matrix(turn, flip, 1)
            if (turn or flip) and tree.query((pts - middle) @ r.T + middle)[0].max() < SAME:
                out.append((float(turn), flip, middle - r @ middle))
    return out
