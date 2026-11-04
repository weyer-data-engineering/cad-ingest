"""Turns joined linework into the objects a CAD user works with: doors, walls, pipes, equipment and
the drawing frame, each of which the export puts on its own layer."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import groupby

import cv2
import numpy as np
from ezdxf.math import fit_points_to_cad_cv
from scipy.spatial import cKDTree

from .model import Arc, Circle, Drawing, Line, Polygon, Polyline
from .topology import TOUCH, _Linework
from .vectorize import _increasing, _sweep


def doors(lines: list[Line], arcs: list[Arc], tol: float = 0.8) -> tuple[list[Polyline], list[Line], list[Arc]]:
    """Door swings: a quarter arc around the hinge plus the leaf, a line from the hinge to one end
    of the arc. Each becomes one polyline; returns them and the remaining pieces. A leaf merged
    with the wall end it continues (the jamb) is split off at the hinge."""
    if not lines or not arcs:
        return [], lines, arcs
    lines = list(lines)
    n = len(lines)
    ends = np.array([(ln.x1, ln.y1) for ln in lines] + [(ln.x2, ln.y2) for ln in lines])
    tree = cKDTree(ends)
    used: set[int] = set()
    out, rest = [], []
    for a in arcs:
        sweep = (a.end - a.start) % 360
        tips = [(a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t))) for t in (a.start, a.end)]
        leaf = None
        if 70 <= sweep <= 110 and a.r >= 4:
            for k, tip in enumerate(tips):
                for i in tree.query_ball_point(tip, max(tol, 0.08 * a.r)):
                    far = ends[(i + n) % (2 * n)]  # the leaf runs from the tip through the hinge
                    if i % n not in used and _through(tip, far, (a.cx, a.cy), tol):
                        leaf = (i % n, k, tuple(map(float, ends[i])), tuple(map(float, far)))
        if leaf is None:
            rest.append(a)
            continue
        j, k, tip, far = leaf
        used.add(j)
        length = math.dist(tip, far)
        t = ((a.cx - tip[0]) * (far[0] - tip[0]) + (a.cy - tip[1]) * (far[1] - tip[1])) / length ** 2
        hinge = (float(tip[0] + t * (far[0] - tip[0])), float(tip[1] + t * (far[1] - tip[1])))  # on the leaf line
        if math.dist(far, hinge) > tol:  # keep the jamb beyond the hinge
            lines.append(replace(lines[j], x1=far[0], y1=far[1], x2=hinge[0], y2=hinge[1]))
        turn = 1 if k == 0 else -1  # +1: from the leaf tip clockwise on screen
        dx, dy = tip[0] - hinge[0], tip[1] - hinge[1]
        closed_at = (hinge[0] - turn * dy, hinge[1] + turn * dx)  # door swings are drawn as quarter circles
        out.append(Polyline([hinge, tip, closed_at], [0.0, turn * math.tan(math.pi / 8)], width=lines[j].width, kind="door"))
    return out, [ln for i, ln in enumerate(lines) if i not in used], rest


def _through(tip, far, hinge, tol: float) -> bool:
    """True if the segment from ``tip`` to ``far`` passes within ``tol`` of ``hinge``."""
    (x1, y1), (x2, y2), (hx, hy) = tip, far, hinge
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return False
    along = ((hx - x1) * (x2 - x1) + (hy - y1) * (y2 - y1)) / length
    return 0 < along <= length + tol and abs((x2 - x1) * (hy - y1) - (y2 - y1) * (hx - x1)) / length <= tol


def classify(d: Drawing, kind: str) -> None:
    """Marks the frame; in a floor plan ("building") the walls: wall outlines (closed, as thin as a
    wall) and lines of the heavy pen; in a P&ID ("pid") equipment (large closed outlines), signal
    lines (dashed or ending at an instrument bubble) and pipes (all other solid lines from 5 mm).
    Single lines that get a kind become two-point polylines."""
    singles = [Polyline([(ln.x1, ln.y1), (ln.x2, ln.y2)], [0.0], False, ln.width, ln.dashed, color=ln.color,
                        linetype=ln.linetype, pattern=ln.pattern) for ln in d.lines]
    items = d.polylines + singles
    for pl in items:
        if _on_frame(pl, d.width, d.height):
            pl.kind = "frame"
    lines = [pl for pl in items if pl.kind == "line"]
    if kind == "building":
        thick = _thick(lines + [q for q in _strokes(d) if isinstance(q, Line)])
        for pl in lines:
            band = pl.closed and 0.8 <= 2 * _area(pl) / _length(pl) <= 10
            if not pl.dashed and (band or thick is not None and pl.width >= thick):
                pl.kind = "wall"
    elif kind == "pid":
        bubbles = [c for c in d.circles if 3 <= c.r <= 8 and any(math.hypot(t.x - c.cx, t.y - c.cy) < c.r for t in d.texts)]
        for pl in lines:
            ends = pl.points if pl.closed else [pl.points[0], pl.points[-1]]
            if pl.dashed or any(math.hypot(x - c.cx, y - c.cy) <= c.r + 1 for x, y in ends for c in bubbles):
                pl.kind = "signal"
            elif pl.closed and _area(pl) >= 100:
                pl.kind = "equipment"
            elif not pl.closed and _length(pl) >= 5:
                pl.kind = "pipe"
    d.polylines += [pl for pl in singles if pl.kind != "line"]
    d.lines = [ln for ln, pl in zip(d.lines, singles) if pl.kind == "line"]


def _on_frame(pl: Polyline, width: float, height: float, margin: float = 25.0) -> bool:
    """The drawing frame: long straight lines along the sheet edges."""
    xs, ys = [p[0] for p in pl.points], [p[1] for p in pl.points]
    if max(xs) - min(xs) >= 0.6 * width:
        return all(y <= margin or y >= height - margin for y in ys) or (pl.closed and max(ys) - min(ys) >= 0.6 * height)
    if max(ys) - min(ys) >= 0.6 * height:
        return all(x <= margin or x >= width - margin for x in xs)
    return False


PEN = 0.08  # spread of the log of the widths measured for one pen (about 8 %)
PEN_SHARE = 0.03  # a pen draws at least this share of the linework's length


def pens(d: Drawing) -> None:
    """Gives the linework a few pens, as CAD draws it: the widths measured, weighted by length, gather around one value
    per pen, as many as there are (the modes of their distribution, each with PEN_SHARE of the length at least); every
    line, arc, circle, polyline and dimension gets the pen nearest to its width, at that pen's median width. Marks by
    hand keep theirs."""
    items = [q for q in d.lines + d.arcs + d.circles + d.polylines + _strokes(d)
             if q.width > 0 and getattr(q, "kind", "") != "markup"]
    if len(items) < 2:
        return
    x = np.log([q.width for q in items])
    w = np.array([_extent(q) for q in items])
    grid = np.arange(x.min() - 3 * PEN, x.max() + 3 * PEN, PEN / 8)
    density = (w * np.exp(-0.5 * ((grid[:, None] - x) / PEN) ** 2)).sum(axis=1)
    peaks = grid[1:-1][(density[1:-1] >= density[:-2]) & (density[1:-1] > density[2:])]
    owner = np.abs(x[:, None] - peaks).argmin(axis=1)
    kept = [k for k in range(len(peaks)) if w[owner == k].sum() >= PEN_SHARE * w.sum()]
    owner = np.abs(x[:, None] - peaks[kept]).argmin(axis=1)
    for k in range(len(kept)):
        mine = np.flatnonzero(owner == k)
        order = mine[np.argsort(x[mine])]
        middle = order[np.searchsorted(np.cumsum(w[order]), w[order].sum() / 2)]  # the length-weighted median
        for i in mine:
            items[i].width = items[middle].width
    widths = {q.width for q in items}
    for m in d.dimensions:
        m.width = min(widths, key=lambda v: abs(v - m.width))


def _extent(q) -> float:
    if isinstance(q, Line):
        return math.hypot(q.x2 - q.x1, q.y2 - q.y1)
    if isinstance(q, Arc):
        return math.radians((q.end - q.start) % 360 or 360) * q.r
    if isinstance(q, Circle):
        return 2 * math.pi * q.r
    return _length(q)


def _strokes(d: Drawing) -> list[Line | Arc | Circle]:
    """The strokes of the dimensions: no longer loose linework, but drawn with the pens of the drawing all the same."""
    return [q for m in d.dimensions for q in m.parts if isinstance(q, (Line, Arc, Circle))]


def _thick(items: list[Polyline | Line]) -> float | None:
    """Width separating the heavy pen (walls, process lines) from the rest: the split of the
    length-weighted widths with the largest between-class variance (Otsu), if the two classes
    differ clearly. None if all lines look alike."""
    rows = sorted((pl.width, _extent(pl)) for pl in items if not pl.dashed)
    if len(rows) < 4:
        return None
    w, length = np.array(rows).T
    share = np.cumsum(length) / length.sum()
    w, length = w[(share > 0.05) & (share < 0.95)], length[(share > 0.05) & (share < 0.95)]  # without outliers
    if len(w) < 4:
        return None
    c = np.cumsum(length)
    m = np.cumsum(length * w)
    p0, m0 = c[:-1], m[:-1] / c[:-1]
    m1 = (m[-1] - m[:-1]) / (c[-1] - c[:-1])
    k = int(np.argmax(p0 * (c[-1] - p0) * (m1 - m0) ** 2))
    return float((w[k] + w[k + 1]) / 2) if m1[k] >= 1.25 * m0[k] else None


def _length(pl: Polyline) -> float:
    pts = pl.points + pl.points[:1] if pl.closed else pl.points
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def _area(pl: Polyline) -> float:
    pts = np.array(pl.points)
    return float(abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))) / 2)


SMOOTH_TURN = math.radians(20)  # the sharpest turn between two pieces of a free curve
CURVE_FIT = 0.15  # mm: the farthest an arc drawn through a run of pieces may lie off them
SPLINE_FIT = 0.3  # mm: the same for a spline: the pieces of the tracing lie up to 0.15 mm off the stroke, as chords,
# and a smooth curve through their ends follows the stroke, so it may lie as far again beside them
SPLINE_PIECE = 10.0  # mm: the longest piece of a free curve; a longer straight piece is a line of its own
LARGEST = 1500.0  # mm: the largest radius of an arc on paper (a street at 1:500)
FLAT = math.radians(8)  # an arc sweeping less than this is a slightly bent line
KINK = math.radians(2)  # a turn this small between two pieces may go either way: the tracing's own wobble
CHORD_STEP = 0.03  # mm: how far the chords between the points of a spline may lie off it (about 0.05 mm in the DXF)


def curves(d: Drawing) -> None:
    """Draws smooth runs of pieces as the curves they are. An open polyline of plain solid linework (kind "line") of
    at least three pieces that turns smoothly (see _smooth) becomes one arc where it turns by FLAT or more in all and
    lies within CURVE_FIT of one circle (a street of large radius that the tracing cut into an arc, a line and an arc);
    the arc keeps the ends at which the polyline is joined to other linework (see _one_arc). A straight line traced
    with a jog, as where another line crosses it, turns to and fro and so hardly at all: it is no arc. Otherwise the
    polyline becomes a spline where it is a free curve traced as chords (see _free) and the spline stays within
    SPLINE_FIT of it (a contour line). True arcs and long straight pieces, such as a straight road running into a
    bend, stay as they are."""
    keep, linework = [], None
    for i, pl in enumerate(d.polylines):
        steps = _steps(pl)
        if pl.kind != "line" or pl.dashed or pl.closed or len(steps) < 3 or not _smooth(steps):
            keep.append(pl)
            continue
        if abs(_turning(steps)) >= FLAT:
            linework = linework or _Linework(d.polylines, _rest(d), [p for s in d.symbols for p in s.ports])
            if (arc := _one_arc(pl, _joined(linework, i))) is not None:
                d.arcs.append(arc)
                continue
        keep.append(_spline(pl, steps) or pl)
    d.polylines = keep


def _rest(d: Drawing) -> list:
    """The linework of a drawing but for its polylines: what the end of a polyline can be joined to besides them."""
    return d.lines + d.arcs + d.circles + [q for s in d.symbols for q in s.parts] + \
        [q for m in d.dimensions for q in m.parts if not isinstance(q, Polygon)]


def _joined(linework: _Linework, i: int) -> list[bool]:
    """Whether each end of polyline ``i`` of the linework touches other linework (within its stroke): a corner, a
    junction or a symbol it is joined to."""
    pl = linework.items[i]
    return [linework.distance(p, (i, k), whole=True)[0] <= pl.width / 2 + TOUCH
            for k, p in ((0, pl.points[0]), (1, pl.points[-1]))]


def _steps(pl: Polyline) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """The pieces of an open polyline as (start, end, bulge), without those of no length, which have no direction."""
    return [(p, q, b) for p, q, b in zip(pl.points, pl.points[1:], pl.bulges) if math.dist(p, q) > 0.01]


def _turns(steps: list) -> tuple[list[float], list[float]]:
    """The signed turns where two pieces meet, and those along each piece (0 for a straight one), in radians."""
    chords = [math.atan2(q[1] - p[1], q[0] - p[0]) for p, q, _ in steps]
    half = [2 * math.atan(b) for _, _, b in steps]  # an arc leaves its start and reaches its end this far off its chord
    ends = [(c - h, c + h) for c, h in zip(chords, half)]  # the direction at the start and at the end of each piece
    return [math.remainder(b[0] - a[1], 2 * math.pi) for a, b in zip(ends, ends[1:])], [2 * h for h in half]


def _smooth(steps: list) -> bool:
    """No turn sharper than SMOOTH_TURN where two of its pieces meet, and some turn in all (not a straight run)."""
    joints, along = _turns(steps)
    return max(map(abs, joints)) <= SMOOTH_TURN and sum(map(abs, joints)) + sum(map(abs, along)) >= FLAT


def _turning(steps: list) -> float:
    """How far the pieces turn in all from the start of the first to the end of the last, where they meet and along
    their arcs (radians, counter-clockwise positive)."""
    joints, along = _turns(steps)
    return sum(joints) + sum(along)


def _free(steps: list) -> bool:
    """Whether the pieces are the chords of a free curve: straight and no longer than SPLINE_PIECE, bent away from the
    straight line through their ends by more than SPLINE_FIT, and turning one way along each bend. A straight line
    traced with a jog turns to and fro at each joint: no turn larger than KINK has the same sense as the next one."""
    if any(b or math.dist(p, q) > SPLINE_PIECE for p, q, b in steps):
        return False
    pts = np.array([steps[0][0]] + [q for _, q, _ in steps]) - steps[0][0]
    v = pts[-1] / max(float(np.hypot(*pts[-1])), 1e-9)
    if np.abs(pts[:, 0] * v[1] - pts[:, 1] * v[0]).max() <= SPLINE_FIT:
        return False
    signs = [math.copysign(1, t) for t in _turns(steps)[0] if abs(t) > KINK]
    runs = [len(list(g)) for _, g in groupby(signs)]
    return not runs or min(runs) >= 2


def _one_arc(pl: Polyline, joined: list[bool]) -> Arc | None:
    """The arc along the polyline if all of it lies within CURVE_FIT of one circle of at most LARGEST radius, or
    within its stroke on a wide pen (a straight piece cut from the flat middle of a long arc lies on its stroke, not on
    its centre line), sweeping FLAT or more. The circle passes through the ends that are ``joined`` (one flag for the
    first and one for the last) where one through them fits as well, so that the arc meets what the polyline met
    there; otherwise it is the circle that fits best, whose ends lie within the polyline's stroke."""
    pts = _path(pl)
    for fixed in [pts[[0, -1]][joined]] + ([pts[:0]] if any(joined) else []):
        fit = _circle_through(pts, fixed)
        if fit is None or not fit[2] <= LARGEST:
            continue
        cx, cy, r = fit
        if np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r).max() > max(CURVE_FIT, pl.width / 2 + 0.1):
            continue
        a = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
        if not math.radians(_sweep(a)) >= FLAT:
            continue
        start, end = (a[0], a[-1]) if _increasing(a) else (a[-1], a[0])
        return Arc(float(cx), float(cy), float(r), float(start % 360), float(end % 360), pl.width, pl.color, pl.layer)
    return None


def _circle_through(pts: np.ndarray, fixed: np.ndarray) -> tuple[float, float, float] | None:
    """Centre and radius of the circle that fits the points best and passes through the ``fixed`` ones (none, one or
    two; rows x, y): least squares of x² + y² + Dx + Ey + F, the fixed points as constraints on D, E and F. None where
    the squares give no circle."""
    o = pts.mean(axis=0)  # about the middle of the points, for well-conditioned squares
    p, q = pts - o, fixed - o
    A, b = np.column_stack([p, np.ones(len(p))]), -(p ** 2).sum(axis=1)
    if len(q):
        C = np.column_stack([q, np.ones(len(q))])
        base = np.linalg.lstsq(C, -(q ** 2).sum(axis=1), rcond=None)[0]
        free = np.linalg.svd(C)[2][len(q):].T  # the directions in which D, E, F keep the circle through them
    else:
        base, free = np.zeros(3), np.eye(3)
    D, E, F = base + free @ np.linalg.lstsq(A @ free, b - A @ base, rcond=None)[0]
    r2 = (D * D + E * E) / 4 - F
    return (float(o[0] - D / 2), float(o[1] - E / 2), math.sqrt(r2)) if r2 > 0 else None


def _path(pl: Polyline, step: float = 0.25) -> np.ndarray:
    """Points about ``step`` mm apart along a polyline, in its order (its parts lose the direction of its arcs)."""
    out = [np.array(pl.points[:1], float)]
    for p, q, b in zip(pl.points, pl.points[1:], pl.bulges):
        if not b or p == q:
            t = np.linspace(0, 1, max(2, int(math.dist(p, q) / step) + 1))[1:, None]
            out.append(np.array(p) + t * np.subtract(q, p))
            continue
        arc = Polyline([p, q], [b]).parts()[0]
        sweep = math.copysign(4 * math.atan(abs(b)), b)  # turning clockwise on screen, as the angles grow
        n = max(2, int(abs(sweep) * arc.r / step) + 1)
        a = math.atan2(p[1] - arc.cy, p[0] - arc.cx) + sweep * np.linspace(0, 1, n)[1:]
        out.append(np.column_stack([arc.cx + arc.r * np.cos(a), arc.cy + arc.r * np.sin(a)]))
    return np.vstack(out)


def _spline(pl: Polyline, steps: list) -> Polyline | None:
    """The polyline as a free curve, if it is one (see _free) and that stays within SPLINE_FIT of it: a spline through
    its points and the middles of its longer pieces (through its points alone, a spline would bulge out beside a long
    straight piece). Its points are then taken along the spline so densely that the chords between them, which the
    preview and the benchmark draw, stay within CHORD_STEP of it."""
    if not _free(steps):
        return None
    fit = [steps[0][0]]
    for p, q, _ in steps:
        if math.dist(p, q) > 3.0:
            fit.append(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2))
        fit.append(q)
    try:
        curve = np.array([(v.x, v.y) for v in fit_points_to_cad_cv(fit).approximate(8 * len(fit))])
    except (np.linalg.LinAlgError, ValueError):  # points the fit cannot pass through: keep the polyline
        return None
    if cKDTree(_path(pl)).query(curve)[0].max() > SPLINE_FIT:
        return None
    points = cv2.approxPolyDP(curve.astype(np.float32).reshape(-1, 1, 2), CHORD_STEP, False).reshape(-1, 2)
    points = [(float(x), float(y)) for x, y in points[1:-1]]
    return replace(pl, points=[fit[0]] + points + [fit[-1]], bulges=[0.0] * (len(points) + 1), smooth=True)
