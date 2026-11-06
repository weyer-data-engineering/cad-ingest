"""Lines drawn parallel or at right angles come out exactly so (step ``square_lines``). A traced line leans by the
noise of its ends: a wall at 0.03 degrees, a short pipe between a valve and a junction at 0.3. A draftsman expects
the walls of a plan, the pipes of a P&ID and the sides of a building turned on a site plan to be exactly parallel and
at right angles to each other, as they were drawn.

The sheet's directions are the axes and the directions that many or long straight segments share (a building turned
on a site plan, a 45 degree hatching), each with its right angle. A straight segment whose direction lies within what
the noise of its ends allows of one of them (more for a short segment than for a long one) is turned to exactly it.
Where segments meet (a corner, a T, a line ending on another, a pipe ending at a symbol's port or on a ring), the
meeting point is placed anew where the turned lines meet, so the linework stays connected: corners stay closed, Ts
stay on their line, ports stay where their symbol puts them. Pieces of one direction whose offsets differ by a
fraction of a millimetre are put on one line. Where that would move a point further than MOVE, the segments stay as
drawn. A lone oblique line (a roof pitch, a stair line, a leader) shares its direction with nothing and stays as
drawn, as do arcs, circles, texts and symbols."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from . import dimensions
from .model import Arc, Circle, Dimension, Drawing, Line, Polyline, Symbol

NOISE = 0.3  # mm: how far the ends of a traced line lie off the line as drawn
FLOOR, CEILING = math.radians(0.5), math.radians(5.0)  # the turn a segment may be given: atan(2 NOISE / length) within
SHORTEST = 1.0  # mm: a shorter segment has no direction to speak of; it follows its ends
MOVE = 0.5  # mm: no point of the linework moves further than this
SUPPORT = (3, 30.0)  # segments and mm of their length that a direction of the sheet besides the axes needs
COUNTED = 3.0  # mm: the segments that count for a direction of the sheet are at least this long
APART = math.radians(1.0)  # two directions of the sheet are at least this far apart (and from the axes)
WINDOW = math.radians(0.5)  # half the width of the window over which the angles of a direction are averaged
BEND = math.radians(25.0)  # a turn between two segments up to this may be a curve's
COLLINEAR = 0.15  # mm: pieces of one direction whose offsets differ by less are one line
TOUCH = 0.05  # mm: ends this close are one meeting point (topology.TOUCH) ...
NEAR = 0.15  # mm: ... and one this close to other linework lies on it: moved off it a little, it would stop short
EXACT = 1e-6  # mm: a point this close to a line lies on it
SHALLOW = math.sin(math.radians(5.0))  # a line a node lies on that runs this close to one of its own is
# no place to meet: moving the one a little moves where they cross a lot
ZERO = np.zeros(2)
ROUNDS = 6  # rounds of leaving segments as drawn where a meeting point would move too far


def tolerance(length: float) -> float:
    """The turn (radians) a straight segment of ``length`` mm may be given: what the noise of its ends allows."""
    return min(max(math.atan2(2 * NOISE, max(length, 1e-9)), FLOOR), CEILING)


def directions(angles: np.ndarray, lengths: np.ndarray) -> list[float]:
    """The directions of a sheet (radians, from 0 to pi/2; each stands for its right angle too): the axes (0), which
    take the segments within their tolerance of them, and, densest first, the angles (mod 90 degrees) of the rest that
    SUPPORT segments within their tolerance back, at the length-weighted mean angle of those (a hatching traced a
    degree or two off here and there is one direction, not several). A single oblique line makes none."""
    keep = lengths >= COUNTED
    a, w = np.mod(angles[keep], math.pi / 2), lengths[keep]
    tol = np.array([tolerance(v) for v in w])
    found = [0.0]
    free = np.abs(_turn(a, 0.0)) > tol
    while free.any():
        centre = _densest(a[free], w[free])
        for _ in range(3):  # to the mean of what backs it
            backed = free & (np.abs(_turn(a, centre)) <= tol)
            centre = _mean(a[backed], w[backed], centre) if backed.any() else centre
        backed = free & (np.abs(_turn(a, centre)) <= tol)
        if backed.sum() < SUPPORT[0] or w[backed].sum() < SUPPORT[1]:
            break
        if all(abs(_turn(centre, f)) >= APART for f in found):
            found.append(centre)
        free &= ~backed & (np.abs(_turn(a, centre)) > WINDOW)
    return found


def _densest(a: np.ndarray, w: np.ndarray) -> float:
    """The angle (mod 90 degrees) with the most length within WINDOW of it."""
    order = np.argsort(a)
    a, w = a[order], w[order]
    ext = np.r_[a - math.pi / 2, a, a + math.pi / 2]
    total = np.r_[0.0, np.cumsum(np.r_[w, w, w])]
    within = total[np.searchsorted(ext, a + WINDOW, "right")] - total[np.searchsorted(ext, a - WINDOW, "left")]
    return float(a[int(np.argmax(within))])


def _turn(a, b):
    """The signed difference of two angles mod 90 degrees, from -45 to 45."""
    return np.mod(np.asarray(a) - b + math.pi / 4, math.pi / 2) - math.pi / 4


def _mean(a: np.ndarray, w: np.ndarray, around: float) -> float:
    """The weighted mean of angles mod 90 degrees near ``around``."""
    return float(np.mod(around + np.average(_turn(a, around), weights=w), math.pi / 2))


def _unit(theta: float) -> tuple[float, float]:
    """The unit vector at ``theta``, exact on the axes, so that a line turned to them has two equal coordinates."""
    if theta == 0.0:
        return 1.0, 0.0
    if theta == math.pi / 2:
        return 0.0, 1.0
    return math.cos(theta), math.sin(theta)


class _Net:
    """The straight segments of the linework and the points where they meet. A member is a place that holds a point
    (an end of a line, a vertex of a polyline, an end of a dimension line or one of its measured points); members
    within TOUCH make one node. Nodes keep what they lie on besides their own segments: other segments (a T), the lines
    and rings of symbols, the lines of dimensions, circles and arcs; and the port of a symbol they are at."""

    def __init__(self, d: Drawing):
        self.members: list[tuple[object, int]] = []
        pts: list[tuple[float, float]] = []
        seg_members: list[tuple[int, int]] = []

        def add(obj, k: int, p) -> int:
            self.members.append((obj, k))
            pts.append((float(p[0]), float(p[1])))
            return len(pts) - 1

        for ln in d.lines + [q for m in d.dimensions for q in m.parts if isinstance(q, Line)]:
            seg_members.append((add(ln, 1, (ln.x1, ln.y1)), add(ln, 2, (ln.x2, ln.y2))))
        for pl in d.polylines:
            first = len(pts)
            for k, p in enumerate(pl.points):
                add(pl, k, p)
            if pl.smooth:  # a spline: its ends meet other linework, its chords are no lines
                continue
            ring = list(range(first, len(pts))) + ([first] if pl.closed else [])
            seg_members += [(i, j) for (i, j), b in zip(zip(ring, ring[1:]), pl.bulges) if not b]
        for m in d.dimensions:  # its dimension line, and its measured points on the linework
            seg_members.append((add(m, 1, (m.x1, m.y1)), add(m, 2, (m.x2, m.y2))))
            add(m, 3, m.p1)
            add(m, 4, m.p2)
        for a in d.arcs:  # an arc's ends move with what they meet, along its circle
            for k, t in ((5, a.start), (6, a.end)):
                add(a, k, (a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t))))
        self.p0 = np.array(pts, float).reshape(-1, 2)
        self.node = self._cluster()
        n = int(self.node.max()) + 1 if len(self.node) else 0
        self.pos0 = np.zeros((n, 2))
        np.add.at(self.pos0, self.node, self.p0)
        self.pos0 /= np.bincount(self.node, minlength=n)[:, None]
        sm = np.array(seg_members, int).reshape(-1, 2)
        self.ends = self.node[sm]  # nodes of each segment
        v = self.p0[sm[:, 1]] - self.p0[sm[:, 0]]
        self.length = np.hypot(v[:, 0], v[:, 1])
        self.angle = np.mod(np.arctan2(v[:, 1], v[:, 0]), math.pi)
        self.counts = np.ones(len(sm), bool)  # dimension lines are the sheet's, but no direction counts on them
        self.counts[len(sm) - len(d.dimensions):] = False
        self.incident: dict[int, list[int]] = defaultdict(list)
        for s, (a, b) in enumerate(self.ends):
            self.incident[a].append(s)
            self.incident[b].append(s)
        self.counts &= ~self._bends()
        self.on = self._on_segments()
        self.fixed, self.rings, self.port, self.shared = self._others(d)

    def _bends(self) -> np.ndarray:
        """Which segments bend into the one other segment they meet at a node, by more than the noise of both allows
        and less than BEND: the chords of a curve traced in straight pieces, such as a contour line, whose directions
        are the curve's, not the sheet's."""
        out = np.zeros(len(self.length), bool)
        for i, segs in self.incident.items():
            if len(segs) != 2 or segs[0] == segs[1]:
                continue
            heading = [self.pos0[self.ends[s, 1] if self.ends[s, 0] == i else self.ends[s, 0]] - self.pos0[i]
                       for s in segs]
            cos = heading[0] @ heading[1] / max(float(np.hypot(*heading[0]) * np.hypot(*heading[1])), 1e-12)
            turn = math.pi - math.acos(min(max(cos, -1.0), 1.0))
            if max(tolerance(self.length[s]) for s in segs) < turn < BEND:
                out[segs] = True
        return out

    def _cluster(self) -> np.ndarray:
        """The node of each member: members within TOUCH, joined closest first, a node no wider than TOUCH."""
        parent = list(range(len(self.p0)))
        members = {i: [i] for i in range(len(self.p0))}

        def root(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        pairs = cKDTree(self.p0).query_pairs(TOUCH, output_type="ndarray") if len(self.p0) else np.zeros((0, 2), int)
        gaps = np.hypot(*(self.p0[pairs[:, 0]] - self.p0[pairs[:, 1]]).T) if len(pairs) else np.zeros(0)
        for i, j in pairs[np.argsort(gaps, kind="stable")]:
            a, b = root(int(i)), root(int(j))
            if a != b and all(math.dist(self.p0[m], self.p0[n]) <= TOUCH for m in members[a] for n in members[b]):
                parent[a] = b
                members[b] += members.pop(a)
        roots = np.array([root(i) for i in range(len(self.p0))], int)
        return np.unique(roots, return_inverse=True)[1].reshape(-1)

    def _on_segments(self) -> dict[int, list[int]]:
        """The segments each node lies on between their ends, within NEAR (a line ending on another: a T)."""
        out: dict[int, list[int]] = defaultdict(list)
        if not len(self.pos0):
            return out
        tree = cKDTree(self.pos0)
        a, b = self.pos0[self.ends[:, 0]], self.pos0[self.ends[:, 1]]
        for s in range(len(self.ends)):
            if self.length[s] < 2 * TOUCH:
                continue
            for i in tree.query_ball_point((a[s] + b[s]) / 2, self.length[s] / 2 + NEAR):
                if i in self.ends[s]:
                    continue
                if _off(self.pos0[i], a[s], b[s]) <= NEAR:
                    out[i].append(s)
        return out

    def _others(self, d: Drawing):
        """For each node, the lines it lies on that are no segments of the net (the lines of symbols and the dimension
        and extension lines CAD draws, as (normal, offset, symbol or -1)), the circles it lies on (centre, radius,
        symbol or -1) and the port of a symbol it is at (symbol, port); and the symbols with a port at another's."""
        fixed, rings, port, shared = defaultdict(list), defaultdict(list), {}, set()
        if not len(self.pos0):
            return fixed, rings, port, shared
        tree = cKDTree(self.pos0)
        straight = [(q, k) for k, s in enumerate(d.symbols) for q in s.parts if isinstance(q, Line)]
        straight += [(q, -1) for m in d.dimensions for q in dimensions.linework(m)]
        straight += [(Line(*a, *b), -1) for g in d.polygons for a, b in zip(g.points, g.points[1:] + g.points[:1])]
        bends = [q for pl in d.polylines if not pl.smooth for q in pl.parts() if isinstance(q, Arc)]
        straight += [(q, -1) for pl in d.polylines if pl.smooth for q in pl.parts()]  # a spline's chords
        for q, k in straight:
            a, b = np.array([q.x1, q.y1]), np.array([q.x2, q.y2])
            length = float(np.hypot(*(b - a)))
            if length < 2 * TOUCH:
                continue
            u = (b - a) / length
            n = np.array([-u[1], u[0]])
            for i in tree.query_ball_point((a + b) / 2, length / 2 + NEAR):
                if _off(self.pos0[i], a, b) <= NEAR:
                    fixed[i].append((n, float(n @ a), k))
        curves = [(q, -1) for q in d.circles + d.arcs + bends]
        curves += [(q, k) for k, s in enumerate(d.symbols) for q in s.parts if isinstance(q, (Arc, Circle))]
        for q, k in curves:
            for i in tree.query_ball_point((q.cx, q.cy), q.r + NEAR):
                p = self.pos0[i]
                if abs(math.hypot(p[0] - q.cx, p[1] - q.cy) - q.r) <= NEAR and _on_arc(q, p):
                    rings[i].append((q.cx, q.cy, q.r, k))
        for k, s in enumerate(d.symbols):
            for p in s.ports:
                if (hit := tree.query(p, distance_upper_bound=TOUCH))[0] <= TOUCH:
                    i = int(hit[1])
                    if i in port and port[i][0] != k:  # two symbols joined port to port move only together
                        shared |= {k, port[i][0]}
                    port.setdefault(i, (k, np.array(p, float)))
        return fixed, rings, port, shared

    def targets(self, found: list[float]) -> np.ndarray:
        """The direction (radians, mod pi) each segment is turned to, NaN for one that stays as drawn."""
        dirs = np.array(sorted({v for f in found for v in (f, f + math.pi / 2)}))
        out = np.full(len(self.angle), np.nan)
        for s, (phi, length) in enumerate(zip(self.angle, self.length)):
            if length < SHORTEST:
                continue
            diff = np.abs(np.mod(phi - dirs + math.pi / 2, math.pi) - math.pi / 2)
            k = int(np.argmin(diff))
            if diff[k] <= tolerance(length):
                out[s] = dirs[k]
        return out

    def lines(self, target: np.ndarray, anchor: dict[int, np.ndarray]) -> tuple[dict[int, tuple], set[int]]:
        """The line each turned segment lies on, as (unit normal, offset, line id), and the segments that cannot be
        turned: those whose line would have to pass through two points held (``anchor``: the ports of symbols) that do
        not lie on one line of its direction. Turned segments of one direction that meet at a node are one line; so are
        lines of one direction whose offsets differ by less than COLLINEAR, unless points held tell them apart."""
        out, stuck, parts = {}, set(), []  # parts: direction, offset, held, segments, normal
        for segs in self._collinear(target):
            theta = float(target[segs[0]])
            n = np.array([-_unit(theta)[1], _unit(theta)[0]])
            held = [float(n @ anchor[i]) for s in segs for i in self.ends[s] if i in anchor]
            if held and max(held) - min(held) > EXACT:
                stuck.update(segs)
                continue
            mids = (self.pos0[self.ends[segs, 0]] + self.pos0[self.ends[segs, 1]]) / 2
            rho = float(np.average(mids @ n, weights=self.length[segs] + 1e-9))
            parts.append((theta, held[0] if held else rho, bool(held), segs, n))
        parts.sort(key=lambda q: (q[0], q[1]))
        i = line = 0
        while i < len(parts):  # pieces of one line a little apart: one offset, that of a point held if one is
            j = i + 1
            while j < len(parts) and parts[j][0] == parts[i][0] and parts[j][1] - parts[i][1] < COLLINEAR:
                j += 1
            run = parts[i:j]
            held = [q[1] for q in run if q[2]]
            runs = [[q] for q in run] if held and max(held) - min(held) > EXACT else [run]
            for r in runs:
                rho = held[0] if held and len(runs) == 1 else \
                    float(np.average([q[1] for q in r], weights=[self.length[q[3]].sum() + 1e-9 for q in r]))
                for seg in (seg for q in r for seg in q[3]):
                    out[seg] = (r[0][4], rho, line)
                line += 1
            i = j
        return out, stuck

    def shifts(self, lines: dict[int, tuple], fixed: set[int]) -> dict[int, np.ndarray | None]:
        """How far each symbol not ``fixed`` must move (without turning) so that its ports lie on the turned lines of
        the segments that end at them; None for one whose ports cannot all lie on them or would move further than
        MOVE (a symbol placed a little turned)."""
        rows: dict[int, dict[int, tuple]] = defaultdict(dict)
        for i, (k, p) in self.port.items():
            for s in self.incident.get(i, []):
                if k not in fixed and s in lines:
                    n, rho, line = lines[s]
                    rows[k][line] = (n, rho - float(n @ p))
        out = {}
        for k, by_line in rows.items():
            A = np.array([n for n, _ in by_line.values()])
            b = np.array([v for _, v in by_line.values()])
            t = np.linalg.lstsq(A, b, rcond=None)[0]
            out[k] = t if np.all(np.abs(A @ t - b) <= EXACT) and np.hypot(*t) <= MOVE else None
        return out

    def _collinear(self, target: np.ndarray) -> list[list[int]]:
        """The turned segments grouped by the nodes they share with segments turned to the same direction."""
        turned = np.flatnonzero(~np.isnan(target))
        parent = {int(s): int(s) for s in turned}

        def root(s: int) -> int:
            while parent[s] != s:
                parent[s] = parent[parent[s]]
                s = parent[s]
            return s

        for segs in self.incident.values():
            by_dir = defaultdict(list)
            for s in segs:
                if not np.isnan(target[s]):
                    by_dir[float(target[s])].append(s)
            for same in by_dir.values():
                for s in same[1:]:
                    parent[root(s)] = root(same[0])
        groups = defaultdict(list)
        for s in turned:
            groups[root(int(s))].append(int(s))
        return list(groups.values())

    def place(self, lines: dict[int, tuple], moved: dict[int, np.ndarray], anchor: dict[int, np.ndarray],
              shift: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], set[int]]:
        """The new place of each node on a turned line or on a segment that moved (``moved``: the nodes placed so far),
        and the nodes that cannot be placed within MOVE of where they were. A node at a port is where the port is
        (``anchor``); a node on a segment that stays as drawn stays on it where its ends are now; a node on a symbol's
        line or ring stays on it where the symbol is moved to (``shift``)."""
        out, bad = {}, set()
        for i in set(self.incident) | set(self.on):
            own = {lines[s][2]: lines[s][:2] for s in self.incident.get(i, []) if s in lines}
            under = [s for s in self.on.get(i, []) if s in lines or any(int(e) in moved for e in self.ends[s])]
            if not own and not under and i not in anchor:
                continue
            ls = list(own.values())
            if i not in anchor:  # a port stays where its symbol puts it, on the pipe it lies on or not
                for s in sorted(self.on.get(i, []), key=lambda s: -self.length[s]):  # the longest line first
                    if s in lines:
                        ls += [lines[s][:2]] if lines[s][2] not in own else []
                    else:
                        ls.append(_through(*(moved.get(int(e), self.pos0[e]) for e in self.ends[s])))
                ls += [(n, rho + float(n @ shift.get(k, ZERO))) for n, rho, k in self.fixed.get(i, [])]
            rings = [(*(np.array(c) + shift.get(k, ZERO)), r) for *c, r, k in self.rings.get(i, [])]
            if i in anchor:
                p = anchor[i]
                for n, rho in own.values():  # on a line turned to an axis exactly, not a rounding error off it
                    p = _exact(n, rho, p)
            else:
                p = _point(self.pos0[i], ls, len(own), rings)
            off = p is None or any(abs(n @ p - rho) > EXACT for n, rho in own.values())
            if off or math.dist(p, self.pos0[i]) > MOVE:
                bad.add(i)
            else:
                out[i] = p
        return out, bad

    def write(self, places: dict[int, np.ndarray]) -> None:
        """Moves the members of each node placed anew."""
        for m, (obj, k) in enumerate(self.members):
            if (p := places.get(int(self.node[m]))) is None:
                continue
            x, y = float(p[0]), float(p[1])
            if isinstance(obj, Line):
                setattr(obj, f"x{k}", x)
                setattr(obj, f"y{k}", y)
            elif isinstance(obj, Polyline):
                obj.points[k] = (x, y)
            elif isinstance(obj, Arc):
                setattr(obj, "start" if k == 5 else "end", math.degrees(math.atan2(y - obj.cy, x - obj.cx)) % 360)
            elif k in (1, 2):
                setattr(obj, f"x{k}", x)
                setattr(obj, f"y{k}", y)
            else:
                setattr(obj, f"p{k - 2}", (x, y))


def _point(old: np.ndarray, ls: list[tuple], own: int, rings: list[tuple]) -> np.ndarray | None:
    """The point on the lines ``ls`` nearest to where a node was (``old``): on its own ``own`` first lines exactly,
    and, while they leave it free to move along a line, on the next other line (one it lies on) that does not run
    within SHALLOW of one before it; on the ring it lies on where a single line leaves it free to slide. None where no
    point lies on all its own lines."""
    kept = ls[:own]
    for n, rho in ls[own:]:  # two lines meeting at a shallow angle meet far off: the node stays on the first
        if _free(kept) and all(abs(n[0] * m[1] - n[1] * m[0]) >= SHALLOW for m, _ in kept):
            kept.append((n, rho))
    if not kept:
        return old
    if _free(kept):
        n, rho = kept[0]
        p = _on_ring(n, rho, old, rings)
        if p is None:
            p = _exact(n, rho, old - (n @ old - rho) * n)
    else:
        p = np.linalg.lstsq(np.array([n for n, _ in kept]), np.array([rho for _, rho in kept]), rcond=None)[0]
        for n, rho in kept:  # on a line turned to an axis, exactly on it
            p = _exact(n, rho, p)
    return p if all(abs(n @ p - rho) <= EXACT for n, rho in kept) else None


def _free(ls: list[tuple]) -> bool:
    """Whether lines through one point leave it free to move: they all run in one direction, or there are none."""
    return all(abs(n[0] * ls[0][0][1] - n[1] * ls[0][0][0]) < 1e-9 for n, _ in ls)


def _off(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """How far p lies off the segment from a to b, measured only where it lies between its ends (else inf)."""
    v = b - a
    t = float((p - a) @ v / max(v @ v, 1e-12))
    if t <= 0 or t >= 1:
        return math.inf
    return float(np.hypot(*(a + t * v - p)))


def _on_arc(q, p) -> bool:
    if isinstance(q, Circle):
        return True
    angle = math.degrees(math.atan2(p[1] - q.cy, p[0] - q.cx)) % 360
    start, end = q.start % 360, q.end % 360
    slack = math.degrees(TOUCH / max(q.r, 1e-9))
    return start - slack <= angle <= end + slack if start <= end else angle >= start - slack or angle <= end + slack


def _through(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """The line through a and b as (unit normal, offset)."""
    v = b - a
    n = np.array([-v[1], v[0]]) / max(float(np.hypot(*v)), 1e-12)
    return n, float(n @ a)


def _exact(n: np.ndarray, rho: float, p: np.ndarray) -> np.ndarray:
    """p with the coordinate a line along an axis fixes set to exactly its offset."""
    if n[0] == 0.0:
        return np.array([p[0], rho / n[1]])
    if n[1] == 0.0:
        return np.array([rho / n[0], p[1]])
    return p


def _on_ring(n: np.ndarray, rho: float, old: np.ndarray, rings: list[tuple]) -> np.ndarray | None:
    """Where the line meets the ring that the point lies on, nearest the point; None if it does not or there is none."""
    best = None
    for cx, cy, r in rings:
        c = np.array([cx, cy], float)
        off = n @ c - rho
        if abs(off) > r:
            continue
        foot = c - off * n
        u = np.array([n[1], -n[0]])
        h = math.sqrt(r * r - off * off)
        for p in (foot + h * u, foot - h * u):
            if best is None or math.dist(p, old) < math.dist(best, old):
                best = p
    return None if best is None else _exact(n, rho, best)


def square(d: Drawing) -> None:
    """Turns the straight segments of the lines, polylines and dimensions of ``d`` that run nearly along a direction
    of the sheet (see directions) to exactly it, places their meeting points anew where the turned lines meet and
    moves the symbols whose ports they end at onto them."""
    net = _Net(d)
    if not len(net.length):
        return
    target = net.targets(directions(net.angle[net.counts], net.length[net.counts]))
    fixed = set(net.shared)  # symbols that stay where they are
    for last in range(ROUNDS, -1, -1):
        lines, anchor, shift = _settled(net, target, fixed)
        places, bad = net.place(lines, {}, anchor, shift)
        for _ in range(2):  # again: a segment left as drawn runs where its ends are now
            places, bad = net.place(lines, places, anchor, shift)
        if not bad or not last:
            break
        for i in bad:  # a symbol that cannot move with its ports stays, else the shortest turned segment ending
            if i in net.port and net.port[i][0] in shift:  # there does (not the line it ends on)
                fixed.add(net.port[i][0])
                continue
            turned = [s for s in net.incident.get(i, []) if not np.isnan(target[s])]
            if turned:
                target[min(turned, key=lambda s: net.length[s])] = np.nan
    stay = {k for i, (k, _) in net.port.items() if k in shift and i not in places}
    if stay:  # a port that cannot be placed: its symbol stays, and with it what ends at its ports
        places = {i: p for i, p in places.items() if not (i in net.port and net.port[i][0] in stay)}
    net.write(places)  # a node still not placed stays where it is, with all that meets there
    for k, t in shift.items():
        if k not in stay:
            _move(d.symbols[k], float(t[0]), float(t[1]))
    for m in d.dimensions:
        _perpendicular(m)


def _settled(net: _Net, target: np.ndarray, fixed: set[int]):
    """The turned lines, the ports they pass through and how far each symbol moves to put its ports on them: a symbol
    whose ports cannot all lie on the lines that end at them stays where it is (``fixed`` grows by it), and so its
    ports hold those lines; a segment whose line would have to pass through two ports held apart stays as drawn
    (``target`` set to NaN)."""
    while True:
        held = {i: p for i, (k, p) in net.port.items() if k in fixed}
        lines, stuck = net.lines(target, held)
        target[list(stuck)] = np.nan
        shift = net.shifts(lines, fixed)
        if all(t is not None for t in shift.values()):
            break
        fixed |= {k for k, t in shift.items() if t is None}
    anchor = {i: p + shift[k] if k in shift else p for i, (k, p) in net.port.items()}
    return lines, anchor, shift


def _move(s: Symbol, dx: float, dy: float) -> None:
    """Moves a symbol, its strokes, ports and attributes by (dx, dy) mm."""
    if not dx and not dy:
        return
    s.x, s.y = s.x + dx, s.y + dy
    s.ports = [(x + dx, y + dy) for x, y in s.ports]
    s.parts = [replace(q, x1=q.x1 + dx, y1=q.y1 + dy, x2=q.x2 + dx, y2=q.y2 + dy) if isinstance(q, Line)
               else replace(q, cx=q.cx + dx, cy=q.cy + dy) for q in s.parts]
    s.attributes = {k: replace(t, x=t.x + dx, y=t.y + dy) for k, t in s.attributes.items()}


def _perpendicular(m: Dimension) -> None:
    """Puts each end of a dimension line that has an extension line where that line meets it at right angles, so that
    CAD draws the extension line as recognised."""
    v = np.array([m.x2 - m.x1, m.y2 - m.y1])
    length = float(np.hypot(*v))
    if length < 1e-9:
        return
    u = v / length
    a = np.array([m.x1, m.y1])
    for k, p in ((1, m.p1), (2, m.p2)):
        end = np.array([getattr(m, f"x{k}"), getattr(m, f"y{k}")])
        if math.dist(p, end) <= TOUCH:
            continue
        foot = a + ((np.array(p) - a) @ u) * u
        if math.dist(foot, end) <= MOVE:
            if u[0] == 0.0 or u[1] == 0.0:
                foot = np.where(u == 0.0, end, foot)  # along an axis: the other coordinate exactly the line's
            setattr(m, f"x{k}", float(foot[0]))
            setattr(m, f"y{k}", float(foot[1]))
