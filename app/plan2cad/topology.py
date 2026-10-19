"""Joins the recognised pieces into CAD objects. Lines and arcs that meet end to end become one
polyline (a wall outline, a pipe run, a valve symbol); slivers between two junctions are dissolved
and loose specks dropped, so that the drawing does not fall apart into many short strokes."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .model import Arc, Line, Polyline

TOL = 0.9  # mm: ends closer than this meet (parallel lines of a window are 1.4 mm apart)
DASH_TOL = 2.0  # dashed lines may stop up to a gap short of their corner
SLIVER = 1.2  # mm: shorter pieces between two junctions or hanging off one are skeleton artefacts
SPECK = 1.0  # mm: shorter loose pieces are noise
CRUMB = 2.5  # mm: connected groups of pieces with less linework than this are noise as well
STRAIGHT = math.radians(20)  # passing straight through a junction
SAME_PEN = 0.6  # pieces join only if the thinner one is at least this share of the thicker one


def join(lines: list[Line], arcs: list[Arc], crumb: float = CRUMB) -> tuple[list[Polyline], list[Line], list[Arc]]:
    """Returns the polylines of pieces joined end to end and the pieces that stay on their own.
    Connected groups with less linework than ``crumb`` mm are dropped as noise."""
    _tangents(lines, arcs)
    ends, bulge, width, dashed = [], [], [], []
    for ln in lines:
        ends.append(((ln.x1, ln.y1), (ln.x2, ln.y2)))
        bulge.append(0.0)
        width.append(ln.width)
        dashed.append(ln.dashed)
    for a in arcs:
        ends.append(tuple((a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t)))
                          for t in (a.start, a.end)))
        bulge.append(math.tan(math.radians((a.end - a.start) % 360) / 4))
        width.append(a.width)
        dashed.append(False)
    if not ends:
        return [], [], []
    g = _Graph(np.array(ends, np.float64), np.array(bulge), np.array(width), np.array(dashed), lines + arcs)
    g.dissolve(crumb)
    return g.chains()


def _tangents(lines: list[Line], arcs: list[Arc], reach: float = 3.0) -> None:
    """Where a line runs on into an arc along its tangent (vessel heads, rounded corners), both
    usually overshoot or stop short of the tangent point; moves both ends onto it."""
    if not lines or not arcs:
        return
    ends = np.array([(ln.x1, ln.y1) for ln in lines] + [(ln.x2, ln.y2) for ln in lines])
    tree = cKDTree(ends)
    for a in arcs:
        for attr in ("start", "end"):
            t = math.radians(getattr(a, attr))
            e = (a.cx + a.r * math.cos(t), a.cy + a.r * math.sin(t))
            for i in tree.query_ball_point(e, reach):
                ln = lines[i % len(lines)]
                dx, dy = ln.x2 - ln.x1, ln.y2 - ln.y1
                length = math.hypot(dx, dy)
                if length < 1.0 or abs(math.sin(math.atan2(dy, dx) - t - math.pi / 2)) > math.sin(math.radians(15)):
                    continue  # not running along the arc's tangent at this end
                ux, uy = dx / length, dy / length
                along = (a.cx - ln.x1) * ux + (a.cy - ln.y1) * uy
                fx, fy = ln.x1 + along * ux, ln.y1 + along * uy  # foot of the centre on the line
                if abs(math.hypot(fx - a.cx, fy - a.cy) - a.r) > max(0.35, 0.04 * a.r) or math.dist((fx, fy), e) > reach:
                    continue
                k = "1" if i < len(lines) else "2"
                setattr(ln, "x" + k, fx)
                setattr(ln, "y" + k, fy)
                setattr(a, attr, math.degrees(math.atan2(fy - a.cy, fx - a.cx)) % 360)
                break


class _Graph:
    def __init__(self, ends: np.ndarray, bulge: np.ndarray, width: np.ndarray, dashed: np.ndarray, source: list):
        self.ends, self.bulge, self.width, self.dashed, self.source = ends, bulge, width, dashed, source
        self.alive = np.ones(len(ends), bool)
        pts = ends.reshape(-1, 2)
        self.parent = list(range(len(pts)))  # union-find over the 2n end points
        members = {i: [i] for i in range(len(pts))}
        pairs = sorted((math.dist(pts[i], pts[j]), i, j) for i, j in cKDTree(pts).query_pairs(DASH_TOL))
        for dist, i, j in pairs:  # closest first; a node never grows wider than the tolerance
            limit = DASH_TOL if dashed[i // 2] and dashed[j // 2] else TOL
            a, b = self._root(i), self._root(j)
            if dist > limit or a == b or any(math.dist(pts[m], pts[n]) > limit for m in members[a] for n in members[b]):
                continue
            self.parent[a] = b
            members[b] += members.pop(a)

    def _root(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def _union(self, i: int, j: int) -> None:
        self.parent[self._root(i)] = self._root(j)

    def node(self, e: int, k: int) -> int:
        return self._root(2 * e + k)

    def length(self, e: int) -> float:
        chord = math.dist(*self.ends[e])
        b = abs(self.bulge[e])
        if not b:
            return chord
        sweep = 4 * math.atan(b)
        return chord * sweep / (2 * math.sin(sweep / 2))

    def dissolve(self, crumb: float) -> None:
        """Drops loose specks and spurs (short stubs off a longer piece) and contracts slivers, short
        connectors between longer pieces (skeleton artefacts at corners and junctions). Chains of
        short pieces, such as small symbols at low resolution, stay as they are."""
        inc: dict[int, set[int]] = {}
        for e in np.flatnonzero(self.alive):
            for k in (0, 1):
                inc.setdefault(self.node(e, k), set()).add(int(e))
        for e in sorted(np.flatnonzero(self.alive), key=self.length):
            length = self.length(e)
            if length >= SLIVER:
                break
            a, b = self.node(e, 0), self.node(e, 1)
            others = [self.length(f) for f in (inc[a] | inc[b]) - {e}]
            loose = (len(inc[a]) == 1) + (len(inc[b]) == 1)
            if loose == 2 and length >= SPECK or loose == 1 and max(others) < 2 * SLIVER \
                    or not loose and min(others) < 2 * SLIVER:
                continue
            self.alive[e] = False
            inc[a].discard(e)
            inc[b].discard(e)
            if not loose and a != b:
                self._union(2 * e, 2 * e + 1)
                inc[self._root(a)] = inc.pop(a) | inc.pop(b)
        # loose crumbs: groups of connected pieces that are all together shorter than a few millimetres
        group = {n: n for n in inc}

        def top(n: int) -> int:
            while group[n] != n:
                group[n] = group[group[n]]
                n = group[n]
            return n

        alive = np.flatnonzero(self.alive)
        for e in alive:
            group[top(self.node(e, 0))] = top(self.node(e, 1))
        total: dict[int, float] = {}
        for e in alive:
            total[top(self.node(e, 0))] = total.get(top(self.node(e, 0)), 0.0) + self.length(e)
        for e in alive:
            if total[top(self.node(e, 0))] < crumb:
                self.alive[e] = False

    def _positions(self) -> dict[int, tuple[float, float]]:
        """Node position: the mean of its end points, but on the exact level of any horizontal or
        vertical line ending there, so that those stay straight."""
        xs, ys, axis_x, axis_y = {}, {}, {}, {}
        for e in np.flatnonzero(self.alive):
            (x1, y1), (x2, y2) = self.ends[e]
            straight = not self.bulge[e]
            for k, (x, y) in enumerate(self.ends[e]):
                n = self.node(e, k)
                xs.setdefault(n, []).append(x)
                ys.setdefault(n, []).append(y)
                if straight and x1 == x2:
                    axis_x.setdefault(n, []).append(x)
                if straight and y1 == y2:
                    axis_y.setdefault(n, []).append(y)
        return {n: (float(np.mean(axis_x.get(n, xs[n]))), float(np.mean(axis_y.get(n, ys[n])))) for n in xs}

    def _direction(self, e: int, k: int, pos) -> float:
        """Direction in which edge ``e`` leaves its end ``k`` (radians); arcs leave along their tangent."""
        p, q = pos[self.node(e, 0)], pos[self.node(e, 1)]
        chord = math.atan2(q[1] - p[1], q[0] - p[0])
        half = 2 * math.atan(self.bulge[e])  # half the sweep, signed
        return chord - half if k == 0 else chord + half + math.pi

    def _pairs(self, pos) -> dict[tuple[int, int], tuple[int, int]]:
        """Which edge continues which at each node: through degree-2 nodes always (corners and bends),
        at junctions the pieces that go straight on; a lone compatible pair left over forms a corner."""
        at: dict[int, list[tuple[int, int]]] = {}
        for e in np.flatnonzero(self.alive):
            for k in (0, 1):
                at.setdefault(self.node(e, k), []).append((int(e), k))
        pair = {}
        for ends in at.values():
            if len(ends) < 2:
                continue
            dirs = [self._direction(e, k, pos) for e, k in ends]
            free = set(range(len(ends)))
            if len(ends) > 2:
                straight = []
                for i, j in combinations(range(len(ends)), 2):
                    dev = abs(math.pi - abs((dirs[i] - dirs[j]) % (2 * math.pi)))
                    if dev <= STRAIGHT and self._fits(ends[i][0], ends[j][0]):
                        straight.append((dev, i, j))
                for _, i, j in sorted(straight):
                    if i in free and j in free:
                        pair[ends[i]], pair[ends[j]] = ends[j], ends[i]
                        free -= {i, j}
            left = [(i, j) for i, j in combinations(sorted(free), 2) if self._fits(ends[i][0], ends[j][0])]
            if len(left) == 1:
                i, j = left[0]
                pair[ends[i]], pair[ends[j]] = ends[j], ends[i]
        return pair

    def _fits(self, e: int, f: int) -> bool:
        """Two different pieces of the same pen: both dashed or both solid, similar width."""
        w = sorted((self.width[e], self.width[f]))
        return e != f and self.dashed[e] == self.dashed[f] and w[0] >= SAME_PEN * w[1]

    def chains(self) -> tuple[list[Polyline], list[Line], list[Arc]]:
        polylines, lines, arcs = [], [], []
        for e in np.flatnonzero(self.alive):
            if self.node(e, 0) == self.node(e, 1):  # both ends on one node: a dot or an almost full arc
                self.alive[e] = False
                if self.bulge[e]:
                    arcs.append(self.source[e])
        pos = self._positions()
        pair = self._pairs(pos)
        done = np.zeros(len(self.ends), bool)
        for e0 in np.flatnonzero(self.alive):
            if done[e0]:
                continue
            path, closed = self._walk(int(e0), pair, done)
            pts = [pos[self.node(e, 0 if fwd else 1)] for e, fwd in path]
            if not closed:
                e, fwd = path[-1]
                pts.append(pos[self.node(e, 1 if fwd else 0)])
            bulges = [float(self.bulge[e] if fwd else -self.bulge[e]) for e, fwd in path]
            weights = [self.length(e) for e, _ in path]
            w = float(np.average([self.width[e] for e, _ in path], weights=np.add(weights, 1e-6)))
            pl = _straighten(Polyline(pts, bulges, closed, w, bool(self.dashed[path[0][0]])))
            if len(pl.bulges) == 1 and not closed:
                part = pl.parts()[0]
                (lines if isinstance(part, Line) else arcs).append(part)
            elif len(pl.points) > 1:
                polylines.append(pl)
        return polylines, lines, arcs

    def _walk(self, e0: int, pair, done) -> tuple[list[tuple[int, bool]], bool]:
        """Follows the pairings from edge ``e0`` in both directions; returns (edge, forwards) pairs."""
        done[e0] = True
        path = [(e0, True)]
        e, k = e0, 1
        while (e, k) in pair:
            f, kf = pair[(e, k)]
            if f == e0:
                return path, True
            done[f] = True
            path.append((f, kf == 0))
            e, k = f, 1 - kf
        e, k = e0, 0
        while (e, k) in pair:
            f, kf = pair[(e, k)]
            done[f] = True
            path.insert(0, (f, kf == 1))
            e, k = f, 1 - kf
        return path, False


def _straighten(pl: Polyline) -> Polyline:
    """Removes vertices between two straight segments that continue in the same direction."""
    pts, bulges = list(pl.points), list(pl.bulges)
    i = 1
    while i < len(pts) - (0 if pl.closed else 1) and len(pts) > 2:
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        if not bulges[i - 1] and not bulges[i % len(bulges)] and _collinear(a, b, c):
            del pts[i]
            del bulges[i]
        else:
            i += 1
    return Polyline(pts, bulges, pl.closed, pl.width, pl.dashed, pl.kind)


def _collinear(a, b, c) -> bool:
    u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
    cross = abs(u[0] * v[1] - u[1] * v[0])
    return u[0] * v[0] + u[1] * v[1] > 0 and cross <= math.sin(math.radians(1)) * math.hypot(*u) * math.hypot(*v)
