"""Turns joined linework into the objects a CAD user works with: doors, walls, pipes, equipment and
the drawing frame, each of which the export puts on its own layer."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .model import Arc, Circle, Drawing, Line, Polyline


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
            lines.append(Line(far[0], far[1], hinge[0], hinge[1], lines[j].width, lines[j].dashed))
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
    singles = [Polyline([(ln.x1, ln.y1), (ln.x2, ln.y2)], [0.0], False, ln.width, ln.dashed, color=ln.color)
               for ln in d.lines]
    items = d.polylines + singles
    for pl in items:
        if _on_frame(pl, d.width, d.height):
            pl.kind = "frame"
    lines = [pl for pl in items if pl.kind == "line"]
    if kind == "building":
        thick = _thick(lines)
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
    line, arc, circle and polyline gets the pen nearest to its width, at that pen's median width. Marks by hand keep
    theirs."""
    items = [q for q in d.lines + d.arcs + d.circles + d.polylines if q.width > 0 and getattr(q, "kind", "") != "markup"]
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


def _extent(q) -> float:
    if isinstance(q, Line):
        return math.hypot(q.x2 - q.x1, q.y2 - q.y1)
    if isinstance(q, Arc):
        return math.radians((q.end - q.start) % 360 or 360) * q.r
    if isinstance(q, Circle):
        return 2 * math.pi * q.r
    return _length(q)


def _thick(items: list[Polyline]) -> float | None:
    """Width separating the heavy pen (walls, process lines) from the rest: the split of the
    length-weighted widths with the largest between-class variance (Otsu), if the two classes
    differ clearly. None if all lines look alike."""
    rows = sorted((pl.width, _length(pl)) for pl in items if not pl.dashed)
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
