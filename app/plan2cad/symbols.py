"""Replaces recognised P&ID symbols with blocks of the symbol library (see library.py).

Hypotheses come from the drawing: two line ends that two ports of a symbol fit (a valve between two pipe ends, a safety
valve between its riser and outlet; this fixes position, turn and size), circles (pumps, heat exchangers, instruments,
scaled to the radius) and, where no symbol was found yet, single line ends at a port and groups of small strokes. Each
is fitted by moving it a little and scored on points along the centre lines of the ink (or the drawing's lines), like
an oriented chamfer match: a point of the symbol's outline counts as found if ink runs close by in the same direction.
The score rewards the ink explained and penalises the outline missing and the ink inside the symbol it does not
explain; lines leaving through a port, such as pipes, are expected there. Of overlapping hits the best one wins. The
linework a symbol explains is removed, line ends are snapped onto its ports and the texts at its attribute positions
become its attributes."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

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
BLOCK = (200.0, 120.0)  # mm: largest title block (width, height)
REACH = 4.0  # mm: a text this close to an attribute position becomes the attribute
TAKEN = 0.3  # a hit is dropped if this share of the linework it explains belongs to a better one
GONE = 0.6  # a piece is removed if this share of it is explained by symbols


@dataclass
class _Shape:
    """A library symbol prepared for matching, in its own coordinates."""

    t: Template
    pts: np.ndarray  # points along the outline
    heading: np.ndarray  # direction of the outline at these points (unit vectors)
    group: np.ndarray  # the part of the symbol each point belongs to
    rim: np.ndarray  # points on the outer outline (convex hull), where text may stand in for a stroke
    through: np.ndarray  # points (and directions, oriented form) along lines that must run through the symbol
    hull: np.ndarray  # half-planes (nx, ny, offset) of the convex hull: inside where nx * x + ny * y + offset <= 0
    at: np.ndarray  # the ports
    ports: np.ndarray  # unit vectors from the origin towards the ports
    radius: float
    smallest: float  # scale below which separate parts of the symbol (two bars) would melt into one line
    variants: list[tuple[int, bool, np.ndarray, np.ndarray]]  # distinct (quarter turns, mirror, box low, box high)
    ring: tuple[np.ndarray, float] | None  # centre and radius of the largest circle


@dataclass
class _Hit:
    shape: _Shape
    pose: tuple[float, float, float, bool, float]  # x, y, angle, mirror, scale
    score: float
    claim: np.ndarray  # indices of the ink points it explains

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

    def __init__(self, points: np.ndarray, heading: np.ndarray, text: np.ndarray, straight: np.ndarray):
        self.points, self.oriented = points, _oriented(points, heading)
        self.text, self.straight = text, straight  # on text, on long straight lines (pipes)
        self.tree, self.oriented_tree = cKDTree(points), cKDTree(self.oriented)
        self.strokes = cKDTree(self.oriented[~self.text]) if self.text.any() else self.oriented_tree  # without text
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
    # Symbols between line ends and at circles first, then the rest only where no symbol was found yet
    hits = _fitted(_anchored(d, shapes), ink)
    hits += _fitted(_leftover(d, shapes, ink, _select(hits, len(ink.points))), ink)
    block = _title_block(d)  # small unread text there looks like symbols
    found = [h for h in _select(hits, len(ink.points)) if not (block and _within(h.pose[:2], block))]
    _remove(d, pieces, found)
    placed = [place(h.shape.t, *h.pose) for h in found]
    d.symbols += placed
    _snap(d)
    _attributes(d, placed, library)


def ink_samples(mask: np.ndarray, text: np.ndarray, ppm: float, lines: list[Line]) -> tuple:
    """Points about every quarter millimetre along the centre lines of an ink mask (0/1, ``ppm`` pixels per mm), the
    stroke direction there (from the neighbouring points) and whether they lie on text (``text`` mask), in sheet mm,
    plus points along ``lines`` (flagged as straight). Symbols are matched on the image like this, not on the traced
    vectors, which may have rounded off small shapes such as arrowheads and springs. The mask should not hold the long
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
        return *exact, np.zeros(len(exact[0]), bool), np.ones(len(exact[0]), bool)
    near = points[cKDTree(points).query(points, k=5)[1]]  # the point and its four neighbours
    off = near - near.mean(axis=1, keepdims=True)
    cxx, cyy, cxy = (off[..., 0] ** 2).sum(1), (off[..., 1] ** 2).sum(1), (off[..., 0] * off[..., 1]).sum(1)
    along = 0.5 * np.arctan2(2 * cxy, cxx - cyy)  # main axis of the neighbourhood
    return (np.vstack([points, exact[0]]), np.vstack([np.column_stack([np.cos(along), np.sin(along)]), exact[1]]),
            np.concatenate([text[ys, xs] > 0, np.zeros(len(exact[0]), bool)]),
            np.concatenate([np.zeros(len(points), bool), np.ones(len(exact[0]), bool)]))


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
    rim = (pts @ hull[:, :2].T + hull[:, 2]).max(axis=1) > -2 * STEP
    lines = [_sample(q) for q in t.through]
    through = np.vstack([np.hstack(q) for q in lines]) if lines else np.zeros((0, 4))
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
    smallest = 2 * TOL / min([g for g in gaps if g > STEP] or [np.inf])
    return _Shape(t, pts, heading, group, rim, through, hull, ports, units, float(np.linalg.norm(pts, axis=1).max()), smallest,
                  variants, ring)


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
    for c in d.circles:
        for s in shapes:
            if s.ring and SCALES[0] <= c.r / s.ring[1] <= SCALES[1]:
                for k, mirror, _, _ in s.variants:
                    m = _matrix(90 * k, mirror, c.r / s.ring[1])
                    yield s, (*(np.array([c.cx, c.cy]) - m @ s.ring[0]), 90 * k, mirror, c.r / s.ring[1]), 0.5, (1.0,)


def _leftover(d: Drawing, shapes: list[_Shape], ink: _Ink, found: list[_Hit]):
    """(shape, pose, search radius in mm, sizes to try) where no symbol was found yet: a port at a single line end (an
    off-page connector at the end of a pipe), at the library size, and groups of small strokes."""

    def free(p):
        return not any(_contains(h, p) for h in found)  # not inside a symbol found already

    ports = [transform(*h.pose)(*q) for h in found for q in h.shape.t.ports]
    connected = cKDTree(ports) if ports else None
    shape, mirror, pa, _ = _port_choices(shapes, 1)
    for p, e in _ends(d, bend=45):
        if connected and connected.query(p, distance_upper_bound=SNAP)[0] < np.inf:
            continue  # the line ends at a symbol found already
        if len(shape) and free(p):
            turn = math.atan2(-e[1], -e[0]) - np.arctan2(pa[:, 1], pa[:, 0])  # port direction against the line end
            c, s = np.cos(turn), np.sin(turn)
            origins = p - np.column_stack([c * pa[:, 0] - s * pa[:, 1], s * pa[:, 0] + c * pa[:, 1]])
            for k in range(len(shape)):
                yield shape[k], (*origins[k], math.degrees(turn[k]), bool(mirror[k]), 1.0), 1.0, (1.0,)
    for lo, hi in _clusters(ink):
        size, centre = hi - lo, (lo + hi) / 2
        if not any(map(free, (lo, hi, (lo[0], hi[1]), (hi[0], lo[1])))):
            continue  # the whole group lies inside a symbol found already
        for s in shapes:
            for k, mirror_, tlo, thi in s.variants:
                scale = float(np.clip(max(size / np.maximum(thi - tlo, STEP)), *SCALES))
                if np.all(np.abs(size - scale * (thi - tlo)) <= 0.25 * scale * (thi - tlo) + 0.5):
                    yield s, (*(centre - scale * (tlo + thi) / 2), 90 * k, mirror_, scale), 2.0, (1.0,)


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


def _ends(d: Drawing, bend: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Line ends with the unit vector pointing out of the line, also where a straight run of a polyline bends by more
    than ``bend`` degrees or turns into an arc (a pipe that was joined with a stroke of a symbol)."""
    out = []
    for pl in d.polylines:
        if pl.kind == "frame" or pl.dashed:  # signal lines carry no symbols
            continue
        pts = np.array(pl.points + pl.points[:1] if pl.closed else pl.points, float)
        straight = np.array(pl.bulges) == 0
        heading = _unit(np.diff(pts, axis=0))
        ends = np.ones(len(heading) + 1, bool)  # a run ends at point k
        ends[1:-1] = ~straight[1:] | ~straight[:-1] | (np.sum(heading[1:] * heading[:-1], 1) < math.cos(math.radians(bend)))
        for k in np.flatnonzero(ends):
            if k > 0 and straight[k - 1]:
                out.append((pts[k], heading[k - 1]))
            if k < len(heading) and straight[k]:
                out.append((pts[k], -heading[k]))
    for ln in d.lines:
        a, b = np.array([ln.x1, ln.y1]), np.array([ln.x2, ln.y2])
        if not ln.dashed and (n := np.hypot(*(b - a))) > 0:
            out += [(b, (b - a) / n), (a, (a - b) / n)]
    return out


def _title_block(d: Drawing) -> tuple[float, float, float, float] | None:
    """The title block in the bottom right corner of the frame (ISO 5457): its left edge runs up from the bottom of the
    frame, its top edge from there to the right of the frame. (left, top, right, bottom), None if there is none."""
    frame = [pl.points for pl in d.polylines if pl.kind == "frame"]
    rights = [max(x for x, _ in pts) for pts in frame if max(x for x, _ in pts) > d.width / 2]
    bottoms = [max(y for _, y in pts) for pts in frame if max(y for _, y in pts) > d.height / 2]
    if not rights or not bottoms:
        return None
    right, bottom = min(rights), min(bottoms)  # the inner line of a double frame
    segments = [(q.x1, q.y1, q.x2, q.y2) for pl in d.polylines if pl.kind != "frame" for q in pl.parts()
                if isinstance(q, Line)] + [(q.x1, q.y1, q.x2, q.y2) for q in d.lines]
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


def _clusters(ink: _Ink) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bounding boxes (low, high corner) of groups of ink points within a millimetre of each other, leaving out text
    and long straight lines: a pipe would make the group larger than the symbol on it."""
    points = ink.points[~ink.text & ~ink.straight]
    if len(points) < 2:
        return []
    i, j = cKDTree(points).query_pairs(1.0, output_type="ndarray").T
    _, label = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(points),) * 2), directed=False)
    order = np.argsort(label, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(label[order]) > 0])
    lo, hi = np.minimum.reduceat(points[order], starts), np.maximum.reduceat(points[order], starts)
    big = np.diff(np.r_[starts, len(order)]) >= 4  # more than a speck
    return [(a, b) for a, b in zip(lo[big], hi[big]) if np.hypot(*(b - a)) <= SMALL]


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
    if len(s.through):  # the straight line running through the symbol, e.g. the pipe under a flow arrow
        line = _oriented(s.through[:, :2] @ m.T + xy[:2], s.through[:, 2:] @ m.T / scale)
        if not ink.lines.n or np.mean(ink.lines.query(line, distance_upper_bound=TOL)[0] <= TOL) < COVERAGE:
            return None
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
    # linework explained, minus the outline missing and the linework inside left unexplained, in mm
    score = (len(claim) - MISSING * (1 - coverage) * len(s.pts) * scale - clutter) * STEP
    return _Hit(s, (float(xy[0]), float(xy[1]), angle % 360, mirror, scale), score, claim)


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
        loose = np.add.reduceat(_found(points, rim, ink, np.repeat([TOL + r for _, (_, _, r) in batch],
                                                                    [len(o) for o in outlines])), starts)
        exact = np.add.reduceat(_found(points, rim, ink, TOL), starts)  # to pick the size
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
    """Which outline points (oriented) have linework within ``tol``; text counts only on the outer outline (a
    misread arrow tip), not inside (letters in a bubble do not make a bar across it)."""
    tol = np.broadcast_to(tol, len(outline))
    found = np.zeros(len(outline), bool)
    for tree, which in ((ink.oriented_tree, rim), (ink.strokes, ~rim)):
        if which.any() and tree.n:
            found[which] = tree.query(outline[which], distance_upper_bound=tol.max())[0] <= tol[which]
    return found


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
    gone, start = set(), 0
    for owner, p, _ in pieces:
        if explained[start:start + len(p)].mean() >= GONE:
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
    """Moves line ends close to a port onto it, so that the line is connected to the symbol. A short line with both
    ends at the same symbol is left over from its strokes and dropped: snapped, it would join the symbol to itself."""
    ports = [p for s in d.symbols for p in s.ports]
    if not ports:
        return
    owner = [i for i, s in enumerate(d.symbols) for _ in s.ports]
    tree = cKDTree(ports)

    def port(p) -> int | None:
        dist, i = tree.query(p, distance_upper_bound=SNAP)
        return int(i) if dist < np.inf else None

    def snapped(p):
        i = port(p)
        return p if i is None else tuple(map(float, ports[i]))

    def leftover(a, b, length: float) -> bool:
        i, k = port(a), port(b)
        return i is not None and k is not None and owner[i] == owner[k] \
            and length <= math.dist(ports[i], ports[k]) + 2 * SNAP

    d.lines = [ln for ln in d.lines if not leftover(a := (ln.x1, ln.y1), b := (ln.x2, ln.y2), math.dist(a, b))]
    d.polylines = [pl for pl in d.polylines if pl.closed or not leftover(pl.points[0], pl.points[-1], _length(pl))]
    for pl in d.polylines:
        if not pl.closed:
            pl.points[0], pl.points[-1] = snapped(pl.points[0]), snapped(pl.points[-1])
    for ln in d.lines:
        (ln.x1, ln.y1), (ln.x2, ln.y2) = snapped((ln.x1, ln.y1)), snapped((ln.x2, ln.y2))


def _attributes(d: Drawing, placed: list[Symbol], library: dict[str, Template]) -> None:
    """Texts at the attribute positions of a symbol (its tag, the letters in an instrument bubble) become its
    attributes."""
    readable = [t for t in d.texts if not t.uncertain]
    if not readable:
        return
    tree, taken = cKDTree([(t.x, t.y) for t in readable]), set()
    for sym in placed:
        f = transform(sym.x, sym.y, sym.angle, sym.mirror, sym.scale)
        for tag, at in library[sym.name].tags.items():
            dist, i = tree.query(f(*at), distance_upper_bound=REACH * sym.scale)
            if dist < np.inf and i not in taken:
                taken.add(i)
                sym.attributes[tag] = readable[i]
    gone = {id(readable[i]) for i in taken}
    d.texts = [t for t in d.texts if id(t) not in gone]
