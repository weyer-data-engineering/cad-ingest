"""Joins the recognised pieces into CAD objects. Lines and arcs that meet end to end become one
polyline (a wall outline, a pipe run, a valve symbol); slivers between two junctions are dissolved
and loose specks dropped, so that the drawing does not fall apart into many short strokes. Once the
symbols are placed, line ends that stop just short of other linework are joined to it (connect)."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .model import Arc, Circle, Drawing, Line, Polyline, Symbol

TOL = 0.9  # mm: ends closer than this meet (parallel lines of a window are 1.4 mm apart)
DASH_TOL = 2.0  # dashed lines may stop up to a gap short of their corner
SLIVER = 1.2  # mm: shorter pieces between two junctions or hanging off one are skeleton artefacts
SPECK = 1.0  # mm: shorter loose pieces are noise
CRUMB = 2.5  # mm: connected groups of pieces with less linework than this are noise as well
STRAIGHT = math.radians(20)  # passing straight through a junction
SAME_PEN = 0.6  # pieces join only if the thinner one is at least this share of the thicker one
REACH = 2.0  # mm: a line end this short of other linework is joined to it: tracing stops short of a junction by
# about the width of the line it meets, and taking out the strokes of a symbol leaves its pipes a little short of it
ASIDE = 0.4  # mm: an end this far beside other linework, without pointing at it, is moved onto it
OVER = 0.6  # mm: an end that runs this far past the line it meets is cut back to it
TOUCH = 0.05  # mm: an end this close to other linework touches it
FACING = math.radians(10)  # two loose ends that point at each other along one line within this angle
CORNER = math.radians(30)  # two loose ends whose lines meet at least at this angle form a corner where they meet
LOOSE = 2.0  # mm: a shorter piece that touches no other linework is left over from strokes not recognised
STUB = 6.0  # mm: a port of a symbol that faces a pipe or the port of the next symbol this close is piped to it
FIXED = ("frame", "door")  # kinds of polylines whose ends stay as they are


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
            if total[top(self.node(e, 0))] < crumb and not self.dashed[e]:  # a short dashed link is drawn on purpose
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
    """Removes vertices between two straight segments that continue in the same direction, and between two arcs of
    one circle (a curve that a crossing line had cut in two)."""
    pts, bulges = list(pl.points), list(pl.bulges)
    i = 1
    while i < len(pts) - (0 if pl.closed else 1) and len(pts) > 2:
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        b1, b2 = bulges[i - 1], bulges[i % len(bulges)]
        if not b1 and not b2 and _collinear(a, b, c):
            del pts[i]
            del bulges[i]
        elif b1 and b2 and (merged := _one_circle(a, b, c, b1, b2)) is not None:
            bulges[i - 1] = merged
            del pts[i]
            del bulges[i]
        else:
            i += 1
    return replace(pl, points=pts, bulges=bulges)


def _circle(p, q, b: float) -> tuple[float, float, float]:
    """Centre and radius of the arc from p to q with bulge b (sign as in Polyline)."""
    chord = math.dist(p, q)
    sweep = 4 * math.atan(abs(b))
    r = chord / (2 * math.sin(sweep / 2))
    h = r * math.cos(sweep / 2) * (1 if b > 0 else -1) / chord
    return (p[0] + q[0]) / 2 - (q[1] - p[1]) * h, (p[1] + q[1]) / 2 + (q[0] - p[0]) * h, r


def _one_circle(a, b, c, b1: float, b2: float) -> float | None:
    """The bulge of a single arc from a to c if the arcs a-b and b-c turn the same way on (nearly) one circle and
    together sweep less than a full turn; None otherwise."""
    if (b1 > 0) != (b2 > 0):
        return None
    (x1, y1, r1), (x2, y2, r2) = _circle(a, b, b1), _circle(b, c, b2)
    if math.hypot(x2 - x1, y2 - y1) > max(0.3, 0.04 * max(r1, r2)) or abs(r1 - r2) > max(0.3, 0.04 * max(r1, r2)):
        return None
    sweep = 4 * math.atan(abs(b1)) + 4 * math.atan(abs(b2))
    if sweep >= 2 * math.pi - 1e-6 or math.dist(a, c) < 1e-6:
        return None
    return math.copysign(math.tan(sweep / 4), b1)


def _collinear(a, b, c) -> bool:
    u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
    cross = abs(u[0] * v[1] - u[1] * v[0])
    return u[0] * v[0] + u[1] * v[1] > 0 and cross <= math.sin(math.radians(1)) * math.hypot(*u) * math.hypot(*v)


def connect(d: Drawing) -> None:
    """Joins the ends of lines and open polylines that stop just short of other linework (REACH), so that pipes, signal
    lines and the lines of a drawing meet the junctions, corners, circles and symbols they end at, as drawn in CAD:
    two loose ends whose lines meet nearby form a corner there, two that point at each other along one line meet
    halfway, and any other end is moved along its line onto the first linework ahead (or cut back onto one it runs
    past, or moved onto one just beside it). Pieces that then meet end to end become one polyline (through a crossing,
    the pieces that go straight on), and pieces that touch nothing and are shorter than LOOSE are dropped."""
    singles = [Polyline([(ln.x1, ln.y1), (ln.x2, ln.y2)], [0.0], False, ln.width, ln.dashed, color=ln.color,
                        layer=ln.layer) for ln in d.lines]
    fixed = [pl for pl in d.polylines if pl.closed or pl.kind in FIXED]
    items = [pl for pl in d.polylines if not pl.closed and pl.kind not in FIXED] + singles
    if not items:
        return
    rest = [q for pl in fixed for q in pl.parts()] + d.arcs + d.circles + [q for s in d.symbols for q in s.parts]
    ports = [p for s in d.symbols for p in s.ports]
    g = _Linework(items, rest, ports)
    loose = [(i, k) for i in range(len(items)) for k in (0, 1) if g.distance(_end(items[i], k), (i, k))[0] > TOUCH]
    hits = {end: g.ray(*end) for end in loose}
    for (i, k), (j, m), p in _pairs(items, loose, hits):  # corners first: both ends move
        _move(items[i], k, p)
        _move(items[j], m, p)
        hits.pop((i, k))
        hits.pop((j, m))
    g = _Linework(items, rest, ports)
    moves = []
    for i, k in hits:
        end, u = _heading(items[i], k)
        if (t := g.ray(i, k)) is not None:
            moves.append((i, k, (end[0] + t * u[0], end[1] + t * u[1])))
            continue
        dist, at = g.distance(end, (i, k))
        if dist <= ASIDE:
            moves.append((i, k, at))
    for i, k, p in moves:
        _move(items[i], k, p)
    items += _stubs(d.symbols, items)
    items = [_straighten(pl) for pl in _merged(items)]
    g = _Linework(items, rest, ports)
    items = [pl for i, pl in enumerate(items) if _length(pl) >= LOOSE or g.touched(i)]
    single = [pl for pl in items if pl.kind == "line" and len(pl.bulges) == 1 and not pl.bulges[0]]
    d.lines = [Line(*pl.points[0], *pl.points[1], pl.width, pl.dashed, pl.color, pl.layer) for pl in single]
    d.polylines = fixed + [pl for pl in items if not any(pl is q for q in single)]


def _length(pl: Polyline) -> float:
    return sum(math.dist(a, b) for a, b in zip(pl.points, pl.points[1:] + (pl.points[:1] if pl.closed else [])))


def _end(pl: Polyline, k: int) -> tuple[float, float]:
    return pl.points[0 if k == 0 else -1]


def _heading(pl: Polyline, k: int) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """End k (0 or 1) of an open polyline and the unit direction in which it leaves; None if it ends in an arc."""
    end, before, bulge = (pl.points[0], pl.points[1], pl.bulges[0]) if k == 0 else \
        (pl.points[-1], pl.points[-2], pl.bulges[-1])
    length = math.dist(end, before)
    if bulge or length < 1e-9:
        return end, None
    return end, ((end[0] - before[0]) / length, (end[1] - before[1]) / length)


def _move(pl: Polyline, k: int, p: tuple[float, float]) -> None:
    pl.points[0 if k == 0 else -1] = (float(p[0]), float(p[1]))


class _Linework:
    """The linework that line ends are joined to: the straight segments, the curves and the symbol ports, with the
    item (an open polyline) and the end each segment belongs to, so that an end is not joined to its own last
    segment."""

    def __init__(self, items: list[Polyline], rest: list, ports: list):
        segs, owner, curves, self.items = [], [], [], items
        for i, pl in enumerate(items):
            parts = pl.parts()
            for n, q in enumerate(parts):
                at_end = (0 if n == 0 else -1) if len(parts) > 1 and n in (0, len(parts) - 1) else -2
                self._add(q, segs, curves, owner, (i, at_end) if len(parts) > 1 else (i, 2))
        for q in rest:
            self._add(q, segs, curves, owner, (-1, -2))
        self.segs = np.array(segs, float).reshape(-1, 4)
        self.seg_owner = np.array([o[:2] for o in owner if o[2] == "seg"], int).reshape(-1, 2)
        self.curve_owner = [o[:2] for o in owner if o[2] == "curve"]
        self.curves = curves
        self.ports = np.array(ports, float).reshape(-1, 2)

    @staticmethod
    def _add(q, segs, curves, owner, who) -> None:
        if isinstance(q, Line):
            segs.append((q.x1, q.y1, q.x2, q.y2))
            owner.append((*who, "seg"))
        elif isinstance(q, Circle):
            curves.append((q.cx, q.cy, q.r, None, None))
            owner.append((*who, "curve"))
        else:
            curves.append((q.cx, q.cy, q.r, q.start, q.end))
            owner.append((*who, "curve"))

    def _mask(self, owners: np.ndarray | list, end: tuple[int, int] | None, whole: bool = False) -> np.ndarray:
        """Which segments (or curves) count for ``end`` (item, k): all but its own last one (or its whole item)."""
        o = np.asarray(owners, int).reshape(-1, 2)
        if end is None:
            return np.ones(len(o), bool)
        mine = o[:, 0] == end[0]
        if whole:
            return ~mine
        last = 0 if end[1] == 0 else -1
        return ~(mine & ((o[:, 1] == last) | (o[:, 1] == 2)))

    def distance(self, p, end: tuple[int, int] | None, whole: bool = False):
        """The distance from p to the nearest linework (not counting the last segment of ``end``) and that point."""
        best, at = math.inf, None
        keep = self._mask(self.seg_owner, end, whole)
        segs = self.segs[keep]
        if len(segs):
            a, v = segs[:, :2], segs[:, 2:] - segs[:, :2]
            t = np.clip(np.einsum("ij,ij->i", np.subtract(p, a), v) / np.maximum(np.einsum("ij,ij->i", v, v), 1e-12),
                        0, 1)
            foot = a + t[:, None] * v
            dist = np.hypot(*(foot - p).T)
            k = int(np.argmin(dist))
            best, at = float(dist[k]), (float(foot[k, 0]), float(foot[k, 1]))
        for (cx, cy, r, start, end_), ok in zip(self.curves, self._mask(self.curve_owner, end, whole)):
            if not ok or abs(math.hypot(p[0] - cx, p[1] - cy) - r) >= best:
                continue
            angle = math.degrees(math.atan2(p[1] - cy, p[0] - cx)) % 360
            if start is None or _within(angle, start, end_):
                f = r / max(math.hypot(p[0] - cx, p[1] - cy), 1e-9)
                best, at = abs(math.hypot(p[0] - cx, p[1] - cy) - r), (cx + (p[0] - cx) * f, cy + (p[1] - cy) * f)
        if len(self.ports):
            dist = np.hypot(*(self.ports - p).T)
            k = int(np.argmin(dist))
            if dist[k] < best:
                best, at = float(dist[k]), (float(self.ports[k, 0]), float(self.ports[k, 1]))
        return best, at

    def ray(self, i: int, k: int) -> float | None:
        """How far end k of item i goes on along its line until it meets other linework, within REACH; negative if it
        ran past linework by up to OVER (the nearest either way)."""
        end, u = _heading(self.items[i], k)
        if u is None:
            return None
        found = self.cast(end, u, (i, k), -OVER, REACH)
        found = [t for t in found if t > -(_length(self.items[i]) - 0.1)]  # not cut back to nothing
        return min(found, key=abs) if found else None

    def cast(self, end, u, skip: tuple[int, int] | None, low: float, high: float) -> list[float]:
        """How far along the ray from ``end`` in the direction ``u`` it meets linework, between ``low`` and ``high``
        (not counting the last segment of ``skip``)."""
        found = []
        segs = self.segs[self._mask(self.seg_owner, skip)]
        if len(segs):
            a, v = segs[:, :2], segs[:, 2:] - segs[:, :2]
            w = a - end
            den = u[0] * v[:, 1] - u[1] * v[:, 0]
            ok = np.abs(den) > 1e-9
            den = np.where(ok, den, 1)
            t = (w[:, 0] * v[:, 1] - w[:, 1] * v[:, 0]) / den
            s = (w[:, 0] * u[1] - w[:, 1] * u[0]) / den
            slack = TOUCH / np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
            found += t[ok & (s >= -slack) & (s <= 1 + slack) & (t >= low) & (t <= high)].tolist()
        for (cx, cy, r, start, stop), ok in zip(self.curves, self._mask(self.curve_owner, skip)):
            wx, wy = end[0] - cx, end[1] - cy
            b = wx * u[0] + wy * u[1]
            disc = b * b - (wx * wx + wy * wy - r * r)
            if not ok or disc < 0:
                continue
            for t in (-b - math.sqrt(disc), -b + math.sqrt(disc)):
                angle = math.degrees(math.atan2(end[1] + t * u[1] - cy, end[0] + t * u[0] - cx)) % 360
                if low <= t <= high and (start is None or _within(angle, start, stop)):
                    found.append(t)
        return found

    def touched(self, i: int) -> bool:
        """Whether any point along item i touches other linework (a flange mark across its pipe does)."""
        for q in self.items[i].parts():
            ends = [(q.x1, q.y1), (q.x2, q.y2)] if isinstance(q, Line) else \
                [(q.cx + q.r * math.cos(math.radians(a)), q.cy + q.r * math.sin(math.radians(a)))
                 for a in (q.start, q.end)]
            for f in np.linspace(0, 1, max(2, round(math.dist(*ends) / 0.25) + 1)):
                p = (ends[0][0] + f * (ends[1][0] - ends[0][0]), ends[0][1] + f * (ends[1][1] - ends[0][1]))
                if self.distance(p, (i, 0), whole=True)[0] <= 2 * TOUCH + q.width / 2:
                    return True
        return False


def _stubs(symbols: list[Symbol], items: list[Polyline]) -> list[Polyline]:
    """Pipes from the ports of symbols that no line reaches to the pipe or the port of the next symbol that they face
    within STUB: the short pipe between a valve and its junction, or between two valves, with its flanges, is often
    lost with the strokes of the symbols around it. A port facing a signal line, or nothing near, stays as it is."""
    pen = float(np.median([pl.width for pl in items if pl.kind == "pipe"] or [0.35]))
    kind = "pipe" if any(pl.kind == "pipe" for pl in items) else "line"
    g = _Linework(items, [], [])
    ends = [(p, sym) for sym in symbols if not sym.template.through for p in sym.ports]
    out = []
    for p, sym in ends:
        u = np.subtract(p, (sym.x, sym.y))
        if not (length := float(np.hypot(*u))) or g.distance(p, None)[0] <= TOUCH:
            continue
        u = u / length
        best, to = math.inf, None
        for q, other in ends:  # the port of the next symbol, facing back
            gap = np.subtract(q, p)
            back = np.subtract((other.x, other.y), q)
            if other is not sym and 0 < gap @ u <= STUB and abs(gap[0] * u[1] - gap[1] * u[0]) <= ASIDE \
                    and back @ u > 0 and gap @ u < best:
                best, to = float(gap @ u), tuple(map(float, q))
        hits = g.cast(p, u, None, TOUCH, STUB)
        if hits and min(hits) < best:
            at = (p[0] + min(hits) * u[0], p[1] + min(hits) * u[1])
            if not any(pl.kind == "signal" and _Linework([pl], [], []).distance(at, None)[0] <= TOUCH for pl in items):
                best, to = min(hits), at
        if to is not None:
            out.append(Polyline([tuple(map(float, p)), to], [0.0], False, pen, False, kind))
    return out


def _within(angle: float, start: float, end: float) -> bool:
    start, end = start % 360, end % 360
    return start <= angle <= end if start <= end else angle >= start or angle <= end


def _pairs(items: list[Polyline], loose: list[tuple[int, int]], hits: dict):
    """Loose ends to bring together, closest first: ((item, end), (item, end), meeting point). Two ends of the same pen
    meet where their lines cross if that is near both (a corner), or halfway if they point at each other along one
    line; not beyond other linework that either one meets first."""
    if len(loose) < 2:
        return []
    ends = {e: _heading(items[e[0]], e[1]) for e in loose}
    tree = cKDTree([ends[e][0] for e in loose])
    offers = []
    for a, b in tree.query_pairs(2 * REACH):
        ea, eb = loose[a], loose[b]
        (p, u), (q, v) = ends[ea], ends[eb]
        if u is None or v is None or ea[0] == eb[0] or items[ea[0]].dashed != items[eb[0]].dashed:
            continue
        cross = u[0] * v[1] - u[1] * v[0]
        gap = (q[0] - p[0], q[1] - p[1])
        if abs(cross) >= math.sin(CORNER):
            ta = (gap[0] * v[1] - gap[1] * v[0]) / cross
            tb = (gap[0] * u[1] - gap[1] * u[0]) / cross
            meet = (p[0] + ta * u[0], p[1] + ta * u[1])
        elif u[0] * v[0] + u[1] * v[1] <= -math.cos(FACING) and gap[0] * u[0] + gap[1] * u[1] > 0 \
                and abs(gap[0] * u[1] - gap[1] * u[0]) <= ASIDE:
            ta = tb = math.hypot(*gap) / 2
            meet = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        else:
            continue
        if not (-OVER <= ta <= REACH and -OVER <= tb <= REACH):
            continue
        if any(hits[e] is not None and hits[e] < t - TOUCH for e, t in ((ea, ta), (eb, tb))):
            continue  # one of them meets other linework before the corner
        offers.append((abs(ta) + abs(tb), ea, eb, meet))
    taken, out = set(), []
    for _, ea, eb, meet in sorted(offers, key=lambda o: o[0]):
        if ea not in taken and eb not in taken:
            taken |= {ea, eb}
            out.append((ea, eb, meet))
    return out
def _merged(items: list[Polyline]) -> list[Polyline]:
    """The polylines with those that meet end to end joined into one: where two meet alone, and at a node where more
    meet, the pairs that go straight on. Only polylines of one pen join (dashed or not, the same kind or a plain
    line)."""
    at: dict[tuple[float, float], list[tuple[int, int]]] = {}
    for i, pl in enumerate(items):
        for k in (0, 1):
            p = pl.points[0 if k == 0 else -1]
            at.setdefault((round(p[0], 6), round(p[1], 6)), []).append((i, k))

    def fits(a: int, b: int) -> bool:
        p, q = items[a], items[b]
        return a != b and p.dashed == q.dashed and (p.kind == q.kind or "line" in (p.kind, q.kind))

    def leaving(i: int, k: int) -> float:
        pl = items[i]
        end, before = (pl.points[0], pl.points[1]) if k == 0 else (pl.points[-1], pl.points[-2])
        return math.atan2(before[1] - end[1], before[0] - end[0])

    pair: dict[tuple[int, int], tuple[int, int]] = {}
    for ends in at.values():
        if len(ends) == 2:
            if fits(ends[0][0], ends[1][0]):
                pair[ends[0]], pair[ends[1]] = ends[1], ends[0]
            continue
        straight = sorted((abs(math.pi - abs(math.remainder(leaving(*a) - leaving(*b), 2 * math.pi))), a, b)
                          for n, a in enumerate(ends) for b in ends[n + 1:] if fits(a[0], b[0]))
        for dev, a, b in straight:
            if dev <= STRAIGHT and a not in pair and b not in pair:
                pair[a], pair[b] = b, a
    done, out = set(), []
    for i in range(len(items)):
        if i in done:
            continue
        chain, closed = _chain(i, pair, done)
        pts, bulges = [], []
        for j, forwards in chain:
            pl = items[j]
            p, b = (pl.points, pl.bulges) if forwards else (pl.points[::-1], [-x for x in pl.bulges[::-1]])
            pts += p if not pts else p[1:]
            bulges += b
        if closed:
            pts = pts[:-1]
        longest = max((items[j] for j, _ in chain), key=_length)
        kind = next((items[j].kind for j, _ in chain if items[j].kind != "line"), "line")
        out.append(Polyline(pts, bulges, closed, longest.width, longest.dashed, kind, longest.color, longest.layer))
    return out


def _chain(i: int, pair: dict, done: set) -> tuple[list[tuple[int, bool]], bool]:
    """The polylines joined to item i, in order, each with whether it runs forwards; and whether they close."""
    done.add(i)
    chain, (e, k) = [(i, True)], (i, 1)
    while (e, k) in pair:
        f, kf = pair[(e, k)]
        if f == i:
            return chain, True
        done.add(f)
        chain.append((f, kf == 0))
        e, k = f, 1 - kf
    e, k = i, 0
    while (e, k) in pair:
        f, kf = pair[(e, k)]
        done.add(f)
        chain.insert(0, (f, kf == 1))
        e, k = f, 1 - kf
    return chain, False
