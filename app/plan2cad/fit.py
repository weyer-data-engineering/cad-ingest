"""Dimensions that dimension the drawing (step ``dimension_fit``). A dimension recognised on a sheet measures from the
starts of its extension lines, which leave a gap to the object they dimension (ISO 129: about 1 to 2 mm), and its
length differs from its value by what reading and tracing are off: a tenth or two of a millimetre on a scan, up to
about a millimetre on a photo. So in CAD, measuring the drawing did not give the written values, and the definition
points of the dimensions lay in the air.

Here each measured point is first put on the object its extension line points at: the first linework ahead of the
extension line within REACH, a wall face, a corner, an axis or the jamb of an opening, exactly on it; the gap is kept
as the dimension's offset, so that the DXF draws it as printed. Then the objects are moved, direction by direction, so
that the dimensions along each measure their values exactly: the positions along it that the values say, as near as
they can be to the positions drawn, a small least-squares problem with the values as constraints. The objects are the
lines a dimension measures the position of (those at right angles to it at its measured point, with the lines that
continue them); the lines that end on them follow, so the drawing stays connected, and the dimensions move with them.

A value that does not fit its length within what reading and tracing explain (dimensions.tolerance, the tolerance the
dimensions are checked with) moves nothing and is marked to check. So is a value that does not add up with the others
along its direction, such as an overall dimension that is not the sum of the chain it spans."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import dimensions
from .model import Arc, Circle, Dimension, Drawing, Line, Polygon

REACH = 4.0  # mm: an extension line starts at most this far from the object it dimensions ...
BACK = 0.3  # mm: ... or this far inside it, where tracing ran it on into the object
ASIDE = 0.4  # mm: the corner or edge of an object lies this close to the line of its extension line
CROSSING = math.sin(math.radians(30))  # a line crosses an extension line at this angle at least, else runs along it
CORNER = 0.15  # mm: a corner this little further than an edge crossed is where the extension line meets the object
WALL = 10.0  # mm: a line that continues an extension line beyond the face it meets starts this far behind it at most
SQUARE = math.sin(math.radians(5))  # a line measured along a dimension stands at right angles to it within this
ON = 0.05  # mm: points this close are one, and a point this close to a line lies on it
DIRECTION = math.radians(1.0)  # dimensions within this of one another run along one direction
SAME_PLACE = 0.3  # mm: ends of dimensions without an object this close along their direction measure one place
SAME_LINE = 0.3  # mm: dimensions whose dimension lines lie this close to one line are one chain
AGREE = 0.005  # mm: values along one direction agree where all of them can be measured exactly at once
DISAGREE = "its value does not add up with the other dimensions along it"


@dataclass
class _End:
    """An end of a dimension along a direction: its measured point at ``s`` along it and the vertices of the object it
    measures (none where its measured point is on no linework, or on a line along the direction)."""

    dim: int
    k: int
    s: float
    obj: set[int] = field(default_factory=set)


class Linework:
    """The vertices and straight segments of the drawing's lines and polylines, moved in place and written back."""

    def __init__(self, d: Drawing):
        self.d = d
        points, self.owners, segs = [], [], []
        for i, q in enumerate(d.lines):
            segs.append((len(points), len(points) + 1))
            points += [(q.x1, q.y1), (q.x2, q.y2)]
            self.owners += [(0, i, 0), (0, i, 1)]
        for i, pl in enumerate(d.polylines):
            first, n = len(points), len(pl.points)
            segs += [(first + k, first + (k + 1) % n) for k, b in enumerate(pl.bulges) if not b]
            points += pl.points
            self.owners += [(1, i, k) for k in range(n)]
        self.xy = np.array(points, float).reshape(-1, 2)
        self.seg = np.array(segs, int).reshape(-1, 2)
        self.moved = np.zeros(len(self.xy), bool)

    def write(self) -> None:
        for v in np.flatnonzero(self.moved):
            kind, i, k = self.owners[v]
            x, y = map(float, self.xy[v])
            if kind == 0:
                q = self.d.lines[i]
                q.x1, q.y1, q.x2, q.y2 = (x, y, q.x2, q.y2) if k == 0 else (q.x1, q.y1, x, y)
            else:
                self.d.polylines[i].points[k] = (x, y)

    def _frame(self, p: np.ndarray, v: np.ndarray) -> tuple:
        a, b = self.xy[self.seg[:, 0]], self.xy[self.seg[:, 1]]
        w = np.array([-v[1], v[0]])
        return a, b, (a - p) @ v, (a - p) @ w, (b - p) @ v, (b - p) @ w

    def ahead(self, p: np.ndarray, v: np.ndarray) -> np.ndarray | None:
        """Where the line from ``p`` along ``v`` first meets the linework, from BACK behind ``p`` to REACH ahead of it:
        the corner or end of a line within ASIDE of it (a jamb, the corner of a wall drawn a little beside the extension
        line), else the point where it crosses a line; None where it meets nothing."""
        if not len(self.seg):
            return None
        a, b, at, aw, bt, bw = self._frame(p, v)
        length = np.hypot(*(b - a).T)
        along = np.abs(aw - bw) < CROSSING * np.maximum(length, 1e-9)
        found = []  # (how far along, the point, how far beside the line for a vertex, else None)
        for t, w, x, other in ((at, aw, a, bt), (bt, bw, b, at)):  # ends of lines beside the line
            # not one that runs back along it, such as the extension line itself where the reference draws it
            near = (np.abs(w) <= ASIDE) & (t >= -BACK) & (t <= REACH) & ~(along & (other < t))
            for j in np.flatnonzero(near):
                found.append((float(t[j]), x[j], abs(float(w[j]))))
        crosses = ~along & (aw * bw <= 0) & (length > 0)
        for j in np.flatnonzero(crosses):
            lam = aw[j] / (aw[j] - bw[j])
            t = at[j] + lam * (bt[j] - at[j])
            if -BACK <= t <= REACH:
                found.append((float(t), a[j] + lam * (b[j] - a[j]), None))
        if not found:
            return None
        first = min(t for t, _, _ in found)
        corners = [(aside, x) for t, x, aside in found if aside is not None and t <= first + CORNER]
        return min(corners, key=lambda c: c[0])[1].copy() if corners else \
            next(x for t, x, aside in found if t == first).copy()

    def beyond(self, p: np.ndarray, v: np.ndarray) -> np.ndarray | None:
        """The nearest start of a line that runs on along the line from ``p`` along ``v``, within ASIDE of it and
        WALL ahead of ``p``: the face of an inner wall that a chain dimensions through the outer wall. None if none."""
        if not len(self.seg):
            return None
        a, b, at, aw, bt, bw = self._frame(p, v)
        along = np.abs(aw - bw) < SQUARE * np.maximum(np.hypot(*(b - a).T), 1e-9)
        best = None
        for t, w, x, other in ((at, aw, a, bt), (bt, bw, b, at)):
            near = along & (np.abs(w) <= ASIDE) & (t > ON) & (t <= WALL) & (other > t)
            for j in np.flatnonzero(near):
                if best is None or t[j] < best[0]:
                    best = (float(t[j]), x[j].copy())
        return None if best is None else best[1]

    def across(self, p: np.ndarray, u: np.ndarray) -> np.ndarray | None:
        """Where a line crosses the line through ``p`` along ``u`` within ASIDE of ``p``, nearest first: the drawing's
        linework a dimension line ends on. None where none does."""
        if not len(self.seg):
            return None
        a, b, at, aw, bt, bw = self._frame(p, u)
        length = np.hypot(*(b - a).T)
        crosses = (np.abs(aw - bw) >= CROSSING * np.maximum(length, 1e-9)) & (aw * bw <= 0) & (length > 0)
        best = None
        for j in np.flatnonzero(crosses):
            lam = aw[j] / (aw[j] - bw[j])
            t = at[j] + lam * (bt[j] - at[j])
            if abs(t) <= ASIDE and (best is None or abs(t) < best[0]):
                best = (abs(t), a[j] + lam * (b[j] - a[j]))
        return None if best is None else best[1].copy()

    def _square(self, u: np.ndarray) -> np.ndarray:
        d = self.xy[self.seg[:, 1]] - self.xy[self.seg[:, 0]]
        length = np.hypot(*d.T)
        return (length > 0) & (np.abs(d @ u) <= SQUARE * np.maximum(length, 1e-9))

    def on(self, p: np.ndarray, segs: np.ndarray, tol: float = ON, xy: np.ndarray | None = None,
           beyond: float = 0.0) -> np.ndarray:
        """Which of the segments ``segs`` (indices) pass within ``tol`` of ``p`` (with the vertices at ``xy``, each
        segment taken on ``beyond`` past its ends)."""
        xy = self.xy if xy is None else xy
        a, b = xy[self.seg[segs, 0]], xy[self.seg[segs, 1]]
        d = b - a
        length = np.maximum(np.hypot(*d.T), 1e-9)
        t = np.clip(((p - a) * d).sum(axis=1) / length ** 2, -beyond / length, 1 + beyond / length)
        return np.hypot(*(a + t[:, None] * d - p).T) <= tol

    def measured(self, p: np.ndarray, u: np.ndarray) -> tuple[set[int], list[int]]:
        """What a dimension along ``u`` measures at ``p``: the vertices of the lines through ``p`` at right angles to
        ``u``, with the lines at right angles that continue them end to end (a wall face drawn in pieces), and the
        vertices at ``p`` (a corner); and those segments."""
        if not len(self.seg):
            return set(), []
        square = np.flatnonzero(self._square(u))
        segs = set(square[self.on(p, square)].tolist())
        todo = list(segs)
        while todo:  # the lines at right angles that continue them
            ends = self.xy[self.seg[todo].ravel()]
            todo = []
            for e in ends:
                near = square[np.minimum(*(np.hypot(*(self.xy[self.seg[square, k]] - e).T) for k in (0, 1))) <= ON]
                todo += [int(j) for j in near if j not in segs]
                segs.update(todo)
        vertices = {int(v) for j in segs for v in self.seg[j]}
        at = [p] + [self.xy[v] for v in vertices]
        for x in at:  # and every vertex where they are, such as the end of a line drawn to a corner
            vertices.update(np.flatnonzero(np.hypot(*(self.xy - x).T) <= ON).tolist())
        return vertices, sorted(segs)

    def move(self, vertices: set[int], segs: list[int], by: np.ndarray, done: np.ndarray) -> None:
        """Moves an object (its ``vertices``, its ``segs`` the lines through them) by ``by``, and with it the ends of
        lines that end on those lines; the vertices ``done`` are not moved again."""
        follow = set()
        if segs:
            others = np.flatnonzero(~done)
            for j in segs:
                a, b = self.xy[self.seg[j, 0]], self.xy[self.seg[j, 1]]
                d = b - a
                t = np.clip((self.xy[others] - a) @ d / max(float(d @ d), 1e-12), 0, 1)
                follow.update(others[np.hypot(*(a + t[:, None] * d - self.xy[others]).T) <= ON].tolist())
        for v in (vertices | follow) - set(np.flatnonzero(done).tolist()):
            self.xy[v] += by
            self.moved[v] = done[v] = True


def to_values(d: Drawing) -> None:
    """Puts the measured points of the drawing's dimensions on the objects they dimension and moves the objects, and
    the dimensions with them, to where the values of the dimensions say (see the module's docstring). A value that does
    not fit its length, or does not add up with the others along it, moves nothing and is marked to check."""
    if not d.dimensions:
        return
    work = Linework(d)
    _shared(d.dimensions)
    on = [_on_object(work, m) for m in d.dimensions]
    says, why = dimensions.written(d.dimensions, d.scale)
    for group in _directions(d.dimensions):
        _fit(work, d.dimensions, group, on, says, why)
    work.write()


def _shared(dims: list[Dimension]) -> None:
    """A dimension line that ends on the extension line of another dimension, such as that of an overall dimension
    running on past a chain to its own dimension line, measures what that one does: its measured point there, which
    is its end itself, becomes the other's."""
    lines = [(np.array(p, float), np.array(e, float), m) for m in dims
             for p, e in ((m.p1, (m.x1, m.y1)), (m.p2, (m.x2, m.y2))) if math.dist(p, e) > 1e-6]
    for m in dims:
        u = np.array((m.x2 - m.x1, m.y2 - m.y1)) / max(math.dist((m.x1, m.y1), (m.x2, m.y2)), 1e-12)
        for k, (p, e) in enumerate(((m.p1, (m.x1, m.y1)), (m.p2, (m.x2, m.y2)))):
            if math.dist(p, e) > 1e-6:
                continue
            for start, end, other in lines:
                v = (end - start) / float(np.hypot(*(end - start)))
                past = end + other.extension * v
                t = float(np.clip((np.array(e) - start) @ (past - start) / float((past - start) @ (past - start)),
                                  0, 1))
                if other is not m and abs(float(v @ u)) <= SQUARE and \
                        math.dist(e, start + t * (past - start)) <= SAME_LINE:
                    m.p1, m.p2 = ((tuple(map(float, start)), m.p2) if k == 0 else (m.p1, tuple(map(float, start))))
                    break


def _on_object(work: Linework, m: Dimension) -> tuple[bool, bool]:
    """Puts each measured point of ``m`` on the object its extension line meets (Linework.ahead), or, where the
    dimension line ends on the drawing's linework, on the line it ends on; keeps the gap as the offset. Each end of
    the dimension line moves along it with its measured point. Whether each point is now on linework."""
    ends = [np.array((m.x1, m.y1), float), np.array((m.x2, m.y2), float)]
    u = ends[1] - ends[0]
    u /= max(float(np.hypot(*u)), 1e-12)
    points, gaps, placed = [], [], []
    for p, e in zip((np.array(m.p1, float), np.array(m.p2, float)), ends):
        out = p - e
        far = float(np.hypot(*out))
        hit = work.ahead(p, out / far) if far > 1e-6 else work.across(p, u)
        placed.append(hit is not None)
        q = p if hit is None else hit
        if far > 1e-6:  # an end without an extension line has no gap: CAD draws none there
            gaps.append(max(0.0, float((q - p) @ out / far)))
        points.append(q)
    m.x1, m.y1 = map(float, ends[0] + float((points[0] - ends[0]) @ u) * u)
    m.x2, m.y2 = map(float, ends[1] + float((points[1] - ends[1]) @ u) * u)
    m.p1, m.p2 = (tuple(map(float, q)) for q in points)
    m.offset = min(gaps, default=0.0)  # DXF has one for both: the smaller gap, so neither is drawn shorter than printed
    return placed[0], placed[1]


def _directions(dims: list[Dimension]) -> list[tuple[np.ndarray, list[int]]]:
    """The dimensions by direction, each direction with its unit vector (that of its longest dimension)."""
    groups: list[tuple[float, list[int]]] = []
    angle = [math.atan2(m.y2 - m.y1, m.x2 - m.x1) % math.pi for m in dims]
    for i in sorted(range(len(dims)), key=lambda i: -math.dist((dims[i].x1, dims[i].y1), (dims[i].x2, dims[i].y2))):
        g = next((g for g in groups if abs((angle[i] - g[0] + math.pi / 2) % math.pi - math.pi / 2) <= DIRECTION),
                 None)
        if g is None:
            groups.append((angle[i], [i]))
        else:
            g[1].append(i)
    return [(np.array([math.cos(a), math.sin(a)]), members) for a, members in groups]


def _fit(work: Linework, dims: list[Dimension], group: tuple[np.ndarray, list[int]], on: list[tuple[bool, bool]],
         says: dict[int, tuple[float, float]], why: str) -> None:
    """Fits the dimensions along one direction to their values: solves for the positions of the places they measure
    (_places), with the values that fit their lengths as constraints (_constraints, _solve), and moves the objects and
    the dimensions there."""
    u, members = group
    ends = []
    for i in members:
        for k in (0, 1):
            obj, segs = _measured(work, dims[i], k, u) if on[i][k] else (set(), [])
            ends.append((_End(i, k, float(np.array((dims[i].p1, dims[i].p2)[k], float) @ u), obj), segs))
    place, drawn = _places([e for e, _ in ends])
    rows = _constraints(dims, members, [e for e, _ in ends], place, drawn, says, why)
    at = _solve(dims, rows, drawn)
    if at is None:
        return
    done = np.zeros(len(work.xy), bool)
    moves = at - drawn
    others = [m for j, m in enumerate(dims) if j not in set(members)]
    for g in np.flatnonzero(np.abs(moves) > 1e-9):  # the objects, and the lines that end on them
        own = [(e, segs) for e, segs in ends if place[id(e)] == g]
        vertices = set().union(*(e.obj for e, _ in own))
        segs = sorted({j for _, s in own for j in s})
        before = work.xy.copy()
        work.move(vertices, segs, moves[g] * u, done)
        _follow(others, before, work.xy, segs, work)
    _shift(dims, members, [e for e, _ in ends], place, moves, u)


def _measured(work: Linework, m: Dimension, k: int, u: np.ndarray) -> tuple[set[int], list[int]]:
    """What end ``k`` of ``m`` measures along ``u`` (Linework.measured): the object at its measured point, or, where
    that point lies on a line along the dimension, such as the outer face of a wall where a chain dimensions the inner
    walls behind it, the line that continues its extension line beyond the face (Linework.beyond, the face of an inner
    wall). The end and its measured point then move along ``u`` onto that line, which is what it measures."""
    p = np.array((m.p1, m.p2)[k], float)
    obj, segs = work.measured(p, u)
    e = np.array(((m.x1, m.y1), (m.x2, m.y2))[k], float)
    far = float(np.hypot(*(p - e)))
    if obj or far <= 1e-6 or (at := work.beyond(p, (p - e) / far)) is None:
        return obj, segs
    obj, segs = work.measured(at, u)
    if obj:
        shift = float((at - p) @ u) * u
        q, f = tuple(map(float, p + shift)), tuple(map(float, e + shift))
        m.p1, m.x1, m.y1 = (q, *f) if k == 0 else (m.p1, m.x1, m.y1)
        m.p2, m.x2, m.y2 = (q, *f) if k == 1 else (m.p2, m.x2, m.y2)
    return obj, segs


def _places(ends: list[_End]) -> tuple[dict[int, int], np.ndarray]:
    """The places the ends measure, each an unknown of the fit: ends on one object are at one place, and so are ends
    without an object that lie at one place along the direction (SAME_PLACE), such as the ends of an overall dimension
    and of the chain it spans on extension lines of their own. Their positions as drawn."""
    parent = list(range(len(ends)))

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, ea in enumerate(ends):
        for b in range(a):
            eb = ends[b]
            if ea.dim == eb.dim:
                continue
            if (ea.obj & eb.obj) or ((not ea.obj or not eb.obj) and abs(ea.s - eb.s) <= SAME_PLACE):
                parent[root(a)] = root(b)
    roots = sorted({root(a) for a in range(len(ends))})
    index = {r: n for n, r in enumerate(roots)}
    place = {id(e): index[root(a)] for a, e in enumerate(ends)}
    drawn = np.zeros(len(roots))
    count = np.zeros(len(roots))
    for e in ends:
        drawn[place[id(e)]] += e.s
        count[place[id(e)]] += 1
    return place, drawn / count


def _constraints(dims: list[Dimension], members: list[int], ends: list[_End], place: dict[int, int],
                 drawn: np.ndarray, says: dict[int, tuple[float, float]], why: str
                 ) -> list[tuple[int, int, int, float, float]]:
    """The values to fit, as (dimension, first place, second place, signed length, tolerance): those that fit their
    length as drawn within dimensions.tolerance. One that does not is marked to check."""
    by_dim = {(e.dim, e.k): place[id(e)] for e in ends}
    rows = []
    for i in members:
        if i not in says or dims[i].doubt:
            continue
        a, b = by_dim[(i, 0)], by_dim[(i, 1)]
        length, step = says[i]
        span = drawn[b] - drawn[a]
        tol = dimensions.tolerance(abs(span), step)
        if a == b or abs(abs(span) - length) > tol:
            dims[i].doubt = why
            continue
        rows.append((i, a, b, math.copysign(length, span), tol))
    return rows


def _solve(dims: list[Dimension], rows: list[tuple[int, int, int, float, float]], drawn: np.ndarray
           ) -> np.ndarray | None:
    """The positions of the places that make every value exact and move the places least from where they are drawn:
    the drawn positions plus the least correction that solves the values (a pseudo-inverse). Where the values do not
    add up, so that no positions make all of them exact, the value that fits its drawn length worst (for its
    tolerance) of those that do not add up is marked to check and left out, until they do. None where none is left."""
    rows = list(rows)
    while rows:
        a = np.zeros((len(rows), len(drawn)))
        for r, (_, i, j, _, _) in enumerate(rows):
            a[r, i], a[r, j] = -1.0, 1.0
        want, tol = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
        at = drawn + np.linalg.pinv(a, rcond=1e-9) @ (want - a @ drawn)
        if np.abs(a @ at - want).max() <= AGREE:
            return at
        miss = np.abs(a @ drawn - want) / tol
        worst = int(np.argmax(np.where(np.abs(a @ at - want) > AGREE, miss, -1.0)))
        dims[rows[worst][0]].doubt = DISAGREE
        rows.pop(worst)
    return None


def _follow(others: list[Dimension], before: np.ndarray, after: np.ndarray, segs: list[int], work: Linework) -> None:
    """The measured points of dimensions of other directions that lie on a moved object, or at a vertex that moved,
    move with it."""
    by = after - before
    moved = np.flatnonzero(np.abs(by).max(axis=1) > 0)
    for m in others:
        for k, p in enumerate((m.p1, m.p2)):
            at = moved[np.hypot(*(before[moved] - p).T) <= ON]
            hit = work.on(np.array(p, float), np.array(segs, int), ON, before) if segs else np.zeros(0, bool)
            if len(at) or hit.any():
                shift = by[at[0]] if len(at) else by[work.seg[segs[int(np.argmax(hit))], 0]]
                q = (float(p[0] + shift[0]), float(p[1] + shift[1]))
                if k == 0:
                    if math.dist(p, (m.x1, m.y1)) <= 1e-6:
                        m.x1, m.y1 = q
                    m.p1 = q
                else:
                    if math.dist(p, (m.x2, m.y2)) <= 1e-6:
                        m.x2, m.y2 = q
                    m.p2 = q


def _shift(dims: list[Dimension], members: list[int], ends: list[_End], place: dict[int, int], moves: np.ndarray,
           u: np.ndarray) -> None:
    """Moves the dimensions of one direction with the places they measure: each end and measured point by its place's
    move, and the strokes and the value of each chain by the moves of its ends, in between linearly."""
    by_end = {(e.dim, e.k): float(moves[place[id(e)]]) for e in ends}
    n = np.array([-u[1], u[0]])
    chains: list[list[int]] = []
    for i in members:  # dimensions on one dimension line make a chain
        o = float(np.array((dims[i].x1, dims[i].y1)) @ n)
        chain = next((c for c in chains if abs(float(np.array((dims[c[0]].x1, dims[c[0]].y1)) @ n) - o) <= SAME_LINE),
                     None)
        if chain is None:
            chains.append([i])
        else:
            chain.append(i)
    for chain in chains:
        stations = sorted({(float(np.array(p, float) @ u), by_end[(i, k)])
                           for i in chain for k, p in enumerate(((dims[i].x1, dims[i].y1), (dims[i].x2, dims[i].y2)))})
        s, by = np.array([a for a, _ in stations]), np.array([b for _, b in stations])

        def moved(x: float, y: float) -> tuple[float, float]:
            t = float(np.interp(x * u[0] + y * u[1], s, by))
            return float(x + t * u[0]), float(y + t * u[1])

        for i in chain:
            m = dims[i]
            m.parts = [_moved(q, moved) for q in m.parts]
            if m.text:
                m.text.x, m.text.y = moved(m.text.x, m.text.y)
            (x1, y1), (x2, y2) = ((x + by_end[(i, k)] * u[0], y + by_end[(i, k)] * u[1])
                                  for k, (x, y) in enumerate(((m.x1, m.y1), (m.x2, m.y2))))
            p1, p2 = ((float(x + by_end[(i, k)] * u[0]), float(y + by_end[(i, k)] * u[1]))
                      for k, (x, y) in enumerate((m.p1, m.p2)))
            m.x1, m.y1, m.x2, m.y2, m.p1, m.p2 = float(x1), float(y1), float(x2), float(y2), p1, p2


def _moved(q, moved):
    """A stroke of a dimension moved point by point."""
    if isinstance(q, Line):
        (x1, y1), (x2, y2) = moved(q.x1, q.y1), moved(q.x2, q.y2)
        q.x1, q.y1, q.x2, q.y2 = x1, y1, x2, y2
    elif isinstance(q, Polygon):
        q.points = [moved(x, y) for x, y in q.points]
    elif isinstance(q, (Arc, Circle)):
        q.cx, q.cy = moved(q.cx, q.cy)
    return q
