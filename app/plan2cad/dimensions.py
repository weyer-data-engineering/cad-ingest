"""Dimensions: a thin dimension line with a terminator at each end (an oblique tick, an arrowhead or a dot), extension
lines at right angles to it and its value written along it near the middle. They are found among the recognised lines
and texts, without assuming a layout, and become dimension objects (model.Dimension) that keep the strokes and the
text they were found from. Dimensions that follow one another on one dimension line make a chain and share their
extension lines. A value that does not fit the length of its dimension at the drawing's scale, or that is read with
another decimal separator than the other numbers of the sheet, is marked to check."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .model import Dimension, Line, Polygon, Text

PARALLEL = math.sin(math.radians(3))  # a value and its dimension line, and the pieces of one line, are this parallel
SQUARE = math.sin(math.radians(5))  # an extension line stands at right angles to its dimension line within this
NEAR = (0.2, 2.5)  # text heights: the middle of a value is this far from its dimension line
SAME_LINE = 0.4  # mm: pieces of one dimension line lie this close to one line ...
GAP = 2.0  # mm: ... and follow one another with at most this gap
TOUCH = 0.6  # mm: an extension line stops at most this short of its dimension line ...
CROSS = 0.4  # mm: ... and runs past it where it goes on at least this far (ISO 129: about 2 mm) ...
OVER = 4.0  # mm: ... and its end at most this far: a line running on further on both sides is the drawing's own
SAME_STATION = 0.5  # mm: extension lines meeting the dimension line this close mark one end of a dimension
TICK = (0.3, 3.0)  # mm: an arm of an oblique tick shows at least this much off the lines and ends within this
SLIM = 0.5  # off the lines, a tick shows less ink across itself than this share of what it shows along itself
DOT = (0.35, 1.3)  # mm: the radius of a dot
ARROW = (1.5, 5.0)  # mm: the length of an arrowhead
FIT = (0.6, 0.005)  # a value fits its length within this many mm plus this share of the length
UNITS = {"mm": 1.0, "cm": 10.0, "m": 1000.0}  # what a value may be written in; feet and inches are read as inches
INCH = 25.4
COMMAS = 2  # numbers read with a decimal comma that show a sheet writes them so: one may be a speck under a point
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_DECIMAL = re.compile(r"[Ø⌀]?\s?(\d+)(?:[.,](\d+))?\s?(mm|cm|m)?")
# a number with a decimal separator: not a date, nor one with three digits after it, which may group thousands (1,250)
_SEPARATED = re.compile(r"(?<![\d.,])\d+([.,])(?:\d{1,2}|\d{4,})(?![\d.,])")
_IMPERIAL = re.compile(r"(?:(\d+)'\s?-?\s?)?(?:(\d+)(?:\s(\d+)/(\d+))?\")?")


def value(s: str) -> tuple[float, float, float | None] | None:
    """The number a dimension's value says, the step it is written to and its unit in mm where it names one (feet
    and inches as inches); None for a text that is no value of a dimension (a room area, a level, a word, nought)."""
    s = s.strip()
    if re.search(r"m\s?[²³]", s):  # an area or a volume
        return None
    s = s.translate(_SUPERSCRIPT)  # a raised last digit: half centimetres (1,01⁵)
    if m := _DECIMAL.fullmatch(s):
        number = float(m.group(1) + "." + (m.group(2) or "0"))
        found = number, 10.0 ** -len(m.group(2) or ""), UNITS.get(m.group(3) or "")
    elif (m := _IMPERIAL.fullmatch(s)) and (m.group(1) or m.group(2)) and m.group(4) != "0":
        step = 1.0 / int(m.group(4)) if m.group(4) else 1.0
        inches = 12 * int(m.group(1) or 0) + int(m.group(2) or 0) + (int(m.group(3)) * step if m.group(3) else 0.0)
        found = inches, step, INCH
    else:
        return None
    return found if found[0] > 0 else None


@dataclass
class _Rail:
    """A dimension line: through ``o`` along ``u``, from ``lo`` to ``hi`` (positions along u), made of ``members``."""

    o: np.ndarray
    u: np.ndarray
    lo: float
    hi: float
    members: list[int]
    width: float

    @property
    def n(self) -> np.ndarray:
        return np.array([-self.u[1], self.u[0]])

    def at(self, s: float) -> np.ndarray:
        return self.o + s * self.u


@dataclass
class _Station:
    """Where an extension line meets a dimension line: at ``s`` along it, with the lines of the drawing it found there
    (``own``: extension lines ending near it; ``through``: lines running on far past it on both sides)."""

    s: float
    own: list[int] = field(default_factory=list)
    through: list[int] = field(default_factory=list)
    unseen: float = 0.0  # the side (+1 or -1 along the normal) where no line was traced past it, 0 if one was
    kind: str | None = None  # the terminator
    size: float = 0.0
    parts: list = field(default_factory=list)
    at: float = 0.0  # where the terminator crosses the dimension line on the image


def find(lines: list[Line], texts: list[Text], ink: np.ndarray, ppm: float, mm: float = 1.0,
         scale: int | None = None) -> tuple[list[Dimension], list[Line], list[Text]]:
    """The dimensions among ``lines`` and ``texts`` (``ppm`` pixels of ``ink`` each; ``mm`` of them make a millimetre
    of the drawing as drawn) and the lines and texts that are not part of one. A value belongs to the nearer of the
    two nearest lines along it that has ends of dimensions on both sides of it, as a wall may run between a value and
    its dimension line; the values of a dimension line make a chain (see _chain). A dimension's value is checked
    against its length at the drawing ``scale`` (see check) and its decimal separator against the other numbers of
    the sheet (see separators)."""
    values = [k for k, t in enumerate(texts) if value(t.text) is not None]
    if not values or not lines:
        return [], lines, texts
    seg = np.array([(q.x1, q.y1, q.x2, q.y2) for q in lines], float)
    vec = seg[:, 2:] - seg[:, :2]
    length = np.hypot(vec[:, 0], vec[:, 1])
    unit = vec / np.maximum(length, 1e-9)[:, None]
    solid = np.array([not q.dashed for q in lines]) & (length > 0)
    rails: dict[tuple[int, ...], tuple[_Rail, list[_Station]]] = {}  # each line tried as a dimension line, its ends
    chains: dict[tuple[int, ...], dict[int, int]] = {}  # the values on each, with the number of lines nearer to them
    for k in values:
        for nearer, members in enumerate(_beside(texts[k], seg, unit, solid, mm)):
            if members not in rails:
                rail = _rail(list(members), seg, unit, lines)
                rails[members] = rail, _ends(rail, ink, ppm, seg, unit, solid, lines, mm)
            rail, ends = rails[members]
            if _interval(ends, _along(texts[k], rail), texts[k].height) is not None:
                chains.setdefault(members, {})[k] = nearer
                break
    used_lines: set[int] = set()
    used_texts: set[int] = set()
    dims: list[Dimension] = []
    rest: list[Line] = []  # what of a dimension line runs on past its chain
    for members, ks in chains.items():
        if not used_lines.intersection(members):
            dims += _chain(*rails[members], ks, texts, lines, seg, used_lines, used_texts, rest, mm)
    check(dims, scale, mm)
    separators(dims, texts)
    return (dims, [q for i, q in enumerate(lines) if i not in used_lines] + rest,
            [t for k, t in enumerate(texts) if k not in used_texts])


def _direction(t: Text) -> np.ndarray:
    a = math.radians(t.angle)  # counter-clockwise on the sheet, y pointing down
    return np.array([math.cos(a), -math.sin(a)])


def _under(t: Text, seg: np.ndarray, unit: np.ndarray, solid: np.ndarray) -> list[int]:
    """The solid lines along the text, nearest first, with its middle beside the line (NEAR) and over its length."""
    d = _direction(t)
    c = np.array([t.x, t.y])
    rel = c - seg[:, :2]
    across = np.abs(rel[:, 0] * unit[:, 1] - rel[:, 1] * unit[:, 0])
    along = rel[:, 0] * unit[:, 0] + rel[:, 1] * unit[:, 1]
    span = np.hypot(*(seg[:, 2:] - seg[:, :2]).T)
    ok = solid & (np.abs(unit[:, 0] * d[1] - unit[:, 1] * d[0]) <= PARALLEL)
    ok &= (across >= NEAR[0] * t.height) & (across <= NEAR[1] * t.height) & (along >= 0) & (along <= span)
    near = np.flatnonzero(ok)
    return near[np.argsort(across[near])].tolist()


def _beside(t: Text, seg: np.ndarray, unit: np.ndarray, solid: np.ndarray, mm: float) -> Iterator[tuple[int, ...]]:
    """The two nearest lines along a text that may be its dimension line, nearest first, each with the pieces it is
    drawn in."""
    seen: list[tuple[int, ...]] = []
    for i in _under(t, seg, unit, solid):
        if len(seen) == 2:
            return
        if not any(i in members for members in seen):
            seen.append(members := tuple(_pieces(i, seg, unit, solid, mm)))
            yield members


def _interval(ends: list[_Station], s: float, height: float) -> int | None:
    """The interval between two ends of a dimension line that a value at ``s`` along it is written in, with half its
    height to spare at either end; None where it is in none."""
    return next((i for i, (a, b) in enumerate(zip(ends, ends[1:])) if a.s + height / 2 <= s <= b.s - height / 2),
                None)


def _pieces(i: int, seg: np.ndarray, unit: np.ndarray, solid: np.ndarray, mm: float) -> list[int]:
    """Line ``i`` with the solid lines that continue it on one line (a dimension line drawn in pieces), sorted."""
    o, u = seg[i, :2], unit[i]
    n = np.array([-u[1], u[0]])
    a, b = (seg[:, :2] - o) @ u, (seg[:, 2:] - o) @ u
    off = np.maximum(np.abs((seg[:, :2] - o) @ n), np.abs((seg[:, 2:] - o) @ n))
    ok = solid & (np.abs(unit @ n) <= PARALLEL) & (off <= SAME_LINE * mm)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    members, start, end = {i}, lo[i], hi[i]
    grew = True
    while grew:  # take pieces that overlap or follow within GAP, from the line outwards
        near = ok & (hi >= start - GAP * mm) & (lo <= end + GAP * mm)
        near[list(members)] = False
        grew = bool(near.any())
        members |= set(np.flatnonzero(near).tolist())
        start, end = min(start, lo[near].min(initial=start)), max(end, hi[near].max(initial=end))
    return sorted(members)


def _rail(members: list[int], seg: np.ndarray, unit: np.ndarray, lines: list[Line]) -> _Rail:
    """The dimension line made of ``members``, along the longest of them, pointing right or down."""
    longest = max(members, key=lambda i: np.hypot(*(seg[i, 2:] - seg[i, :2])))
    u = unit[longest]
    if u[0] < -1e-9 or (abs(u[0]) <= 1e-9 and u[1] < 0):
        u = -u
    o = seg[longest, :2]
    ends = np.concatenate([(seg[members, :2] - o) @ u, (seg[members, 2:] - o) @ u])
    return _Rail(o, u, float(ends.min()), float(ends.max()), members,
                 float(np.median([lines[i].width for i in members])))


def _stations(rail: _Rail, seg: np.ndarray, unit: np.ndarray, solid: np.ndarray, mm: float) -> list[_Station]:
    """Where lines at right angles meet the dimension line: an extension line of its own ends within OVER past it (or
    at it, where a style stops it there), a line running on further on both sides is the drawing's own (the dimension
    line ends on it). Where no line was traced past the dimension line, as where a wire branches off, only a tick or
    an arrowhead makes it an end (see _terminator)."""
    n = rail.n
    ea, eb = (seg[:, :2] - rail.o) @ n, (seg[:, 2:] - rail.o) @ n
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    across = unit @ n
    ok = solid & (np.abs(unit @ rail.u) <= SQUARE) & (lo <= TOUCH * mm) & (hi >= -TOUCH * mm)
    ok &= np.maximum(-lo, hi) >= CROSS * mm
    ok[rail.members] = False
    out: list[_Station] = []
    for j in np.flatnonzero(ok):
        t = -ea[j] / across[j]  # where line j meets the dimension line, along it from its first end
        s = float((seg[j, :2] + t * unit[j] - rail.o) @ rail.u)
        if not rail.lo - SAME_STATION * mm <= s <= rail.hi + SAME_STATION * mm:
            continue
        st = next((st for st in out if abs(st.s - s) <= SAME_STATION * mm), None)
        if st is None:
            out.append(st := _Station(s, unseen=math.nan))
        (st.own if min(-lo[j], hi[j]) <= OVER * mm else st.through).append(int(j))
        unseen = 0.0 if lo[j] <= -CROSS * mm and hi[j] >= CROSS * mm else -1.0 if hi[j] >= CROSS * mm else 1.0
        st.unseen = unseen if math.isnan(st.unseen) or st.unseen == unseen else 0.0
    return sorted(out, key=lambda st: st.s)


def _ends(rail: _Rail, ink: np.ndarray, ppm: float, seg: np.ndarray, unit: np.ndarray, solid: np.ndarray,
          lines: list[Line], mm: float) -> list[_Station]:
    """The ends of dimensions on a dimension line: where extension lines meet it (_stations) and the image shows a
    terminator there (_terminator). The line's stroke is measured on the image away from them first."""
    stations = _stations(rail, seg, unit, solid, mm)
    s = np.linspace(rail.lo, rail.hi, 41)[1:-1]
    s = s[np.abs(s[:, None] - np.array([st.s for st in stations] + [math.inf])).min(axis=1) > 2.0 * mm]
    rail.width = (_stroke(ink, ppm, [rail.at(v) for v in s], rail.n, mm) or (rail.width,))[0]
    at = [-math.inf] + [st.s for st in stations] + [math.inf]
    for k, st in enumerate(stations, 1):  # with the room there is along the line for a head, each way
        room = (at[k + 1] - st.s, st.s - at[k - 1])
        st.kind, st.size, st.parts, st.at = _terminator(ink, ppm, rail, st, lines, room, mm) or (None, 0.0, [], st.s)
    return _apart([st for st in stations if st.kind], mm)


def _apart(ends: list[_Station], mm: float) -> list[_Station]:
    """One end per terminator: where two ends found theirs at one place on the image, tracing broke a piece of it off
    as a short line across the dimension line, and only the end nearer the terminator is one."""
    out: list[_Station] = []
    for st in ends:
        if out and abs(st.at - out[-1].at) <= SAME_STATION * mm:
            if abs(st.s - st.at) < abs(out[-1].s - out[-1].at):
                out[-1] = st
        else:
            out.append(st)
    return out


def _chain(rail: _Rail, stations: list[_Station], ks: dict[int, int], texts: list[Text], lines: list[Line],
           seg: np.ndarray, used_lines: set[int], used_texts: set[int], rest: list[Line], mm: float
           ) -> list[Dimension]:
    """The dimensions between the ends with terminators on one dimension line: each with the value written over its
    middle (of the values ``ks``, one with fewer lines nearer to it first), and one without a value where the chain
    has none written there (a thin wall or slab), as long as a third of them have one at least: a line with many such
    ends and few values is something else. The dimension line, as one line, goes with the first dimension whose value
    was read reliably, an extension line or terminator with the first dimension that ends at it; where the dimension
    line runs on more than OVER past an end of the chain, it goes on as other linework from there (to ``rest``)."""
    if len(stations) < 2:
        return []
    best: dict[int, tuple[float, float, int]] = {}  # the value in each interval: on its nearest line, by the middle
    for k, nearer in ks.items():
        s = _along(texts[k], rail)
        if (i := _interval(stations, s, texts[k].height)) is not None:
            middle = (stations[i].s + stations[i + 1].s) / 2
            best[i] = min(best.get(i, (math.inf, math.inf, k)), (nearer, abs(s - middle), k))
    values = {i: k for i, (_, _, k) in best.items()}
    if not values or 3 * len(values) < len(stations) - 1:
        return []
    side = _object_side(rail, stations, seg)
    extension = [_overshoot(rail, j, seg, side) for st in stations for j in st.own]
    over = float(np.median(extension)) if extension else 0.0
    kinds = [st.kind for st in stations]
    kind = max(dict.fromkeys(kinds), key=kinds.count)  # on a tie the first along the line, the same in every process
    size = float(np.median([st.size for st in stations if st.kind == kind]))
    reach = (size if kind == "arrow" else size / 2) + 0.5 * mm  # an arrowhead lies outside an end too short for two
    line = _clipped(rail, [lines[j] for j in rail.members], stations[0].s, stations[-1].s, OVER * mm, reach, rest)
    owner = next((i for i in sorted(values) if not texts[values[i]].uncertain), 0)  # one sure of its value, if any
    dims = []
    for i, (a, b) in enumerate(zip(stations, stations[1:])):
        t = texts[values[i]] if i in values else None
        parts = line if i == owner else []
        for st in (a, b):
            if st.parts:  # an end two dimensions share goes with the first
                own = [j for j in st.own if j not in used_lines]
                parts += [lines[j] for j in own] + st.parts  # the terminator as on the image, not as traced
                used_lines.update(own + _fragments(st, rail.at(st.s), lines, seg, used_lines, mm))
                st.parts = []
        doubt = None if t and not t.uncertain else "uncertain reading" if t else "no value found"
        ends = [float(v) for v in (*rail.at(a.s), *rail.at(b.s))]
        dims.append(Dimension(*ends, t.text if t else "", _measured(rail, a, seg, side), _measured(rail, b, seg, side),
                              kind, size, over, rail.width, t, parts, doubt))
        if t:  # and the same value read a second time at its place
            used_texts.update(k for k in ks if _same(texts[k], t))
    used_lines.update(rail.members)
    return dims


def _clipped(rail: _Rail, members: list[Line], first: float, last: float, over: float, reach: float,
             rest: list[Line]) -> list[Line]:
    """The dimension line made of ``members`` as one line (tracing breaks it at its terminators), from its ``first`` to
    its ``last`` end, and on past an end where it runs on further than the terminator there ``reach``es (tracing runs
    it into them) but no further than ``over``. Where it runs on further, it goes on as other linework from that end:
    those pieces go to ``rest``."""
    along = [tuple(float((np.array(p) - rail.o) @ rail.u) for p in ((q.x1, q.y1), (q.x2, q.y2))) for q in members]
    lo, hi = min(min(a) for a in along), max(max(a) for a in along)
    start = lo if reach < first - lo <= over else max(lo, first)
    end = hi if reach < hi - last <= over else min(hi, last)
    inside, before, after = [], first - lo > over, hi - last > over  # other linework before and after the chain
    for q, (a, b) in zip(members, along):
        for x0, x1, other in ((min(a, b), start, before), (start, end, None), (end, max(a, b), after)):
            x0, x1 = max(x0, min(a, b)), min(x1, max(a, b))
            if x1 - x0 > 1e-6:
                piece = q if (x0, x1) == (min(a, b), max(a, b)) else _between(q, a, b, x0, x1)
                if other is None:
                    inside.append(piece)
                elif other:  # else it is what tracing ran into a terminator
                    rest.append(piece)
    if not inside:
        return []
    (x1, y1), (x2, y2) = rail.at(start), rail.at(end)
    return [replace(inside[0], x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), width=rail.width)]


def _between(q: Line, a: float, b: float, start: float, end: float) -> Line:
    """The piece of line ``q`` (from position ``a`` to ``b`` along a dimension line) between ``start`` and ``end``."""
    def at(s: float) -> tuple[float, float]:
        t = (s - a) / (b - a)
        return q.x1 + t * (q.x2 - q.x1), q.y1 + t * (q.y2 - q.y1)

    (x1, y1), (x2, y2) = at(start), at(end)
    return replace(q, x1=x1, y1=y1, x2=x2, y2=y2)


def _same(a: Text, b: Text) -> bool:
    return "".join(a.text.split()) == "".join(b.text.split()) and math.dist((a.x, a.y), (b.x, b.y)) <= max(
        a.height, b.height)


def _along(t: Text, rail: _Rail) -> float:
    return float((np.array([t.x, t.y]) - rail.o) @ rail.u)


def _object_side(rail: _Rail, stations: list[_Station], seg: np.ndarray) -> float:
    """The side of the dimension line the measured object is on (+1 or -1 along its normal): where its own extension
    lines reach further."""
    reach = sum(float((seg[j, :2] - rail.o) @ rail.n) + float((seg[j, 2:] - rail.o) @ rail.n)
                for st in stations for j in st.own)
    return 1.0 if reach >= 0 else -1.0


def _overshoot(rail: _Rail, j: int, seg: np.ndarray, side: float) -> float:
    """How far extension line ``j`` runs past the dimension line, away from the object."""
    e = ((seg[j, :2] - rail.o) @ rail.n, (seg[j, 2:] - rail.o) @ rail.n)
    return max(0.0, -side * min(e, key=lambda v: side * v))


def _measured(rail: _Rail, st: _Station, seg: np.ndarray, side: float) -> tuple[float, float]:
    """Where the extension line of a dimension end starts: the far end of its own extension line on the object's
    side, or the end of the dimension line itself where it ends on the drawing's linework."""
    point = rail.at(st.s)
    ends = [seg[j, :2] if side * ((seg[j, :2] - rail.o) @ rail.n) > side * ((seg[j, 2:] - rail.o) @ rail.n)
            else seg[j, 2:] for j in st.own]
    if ends:
        far = max(ends, key=lambda p: side * float((p - rail.o) @ rail.n))
        point = point + float((far - point) @ rail.n) * rail.n  # straight out from the dimension line
    return float(point[0]), float(point[1])


def _fragments(st: _Station, x: np.ndarray, lines: list[Line], seg: np.ndarray, used: set[int], mm: float
               ) -> list[int]:
    """Short pieces traced from the terminator at a dimension end (the arms of a tick, the sides of an arrowhead): the
    lines that lie wholly on the terminator as found on the image, which takes their place."""
    tol = 0.4 * mm
    reach = st.size + 1.0 * mm
    near = np.flatnonzero((np.hypot(*(seg[:, :2] - x).T) <= reach) & (np.hypot(*(seg[:, 2:] - x).T) <= reach))
    out = []
    for j in near:
        if j in used or j in st.own or j in st.through:
            continue
        q = lines[j]
        if all(_off(p, st.parts) <= tol + q.width / 2 for p in np.linspace((q.x1, q.y1), (q.x2, q.y2), 5)):
            out.append(int(j))
    return out


def _off(p: np.ndarray, parts: list) -> float:
    """The distance from ``p`` to the nearest of ``parts`` (lines, or filled polygons, inside which it is 0)."""
    best = math.inf
    for q in parts:
        if isinstance(q, Line):
            a, b = np.array([q.x1, q.y1]), np.array([q.x2, q.y2])
            t = float(np.clip((p - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-12), 0, 1))
            best = min(best, float(np.hypot(*(p - a - t * (b - a)))))
        else:
            inside = cv2.pointPolygonTest(np.float32(q.points).reshape(-1, 1, 2), tuple(map(float, p)), True)
            best = min(best, max(0.0, -inside))
    return best


def _terminator(ink: np.ndarray, ppm: float, rail: _Rail, st: _Station, lines: list[Line],
                room: tuple[float, float], mm: float) -> tuple[str, float, list, float] | None:
    """The terminator drawn at a dimension end on the image: an arrowhead, a dot or an oblique tick, with its size, its
    strokes as printed and where it crosses the dimension line; None where there is none (as on a grid or a frame)."""
    x = rail.at(st.s)
    along = np.concatenate([np.linspace(1.2, 3.0, 7), -np.linspace(1.2, 3.0, 7)]) * mm  # away from the terminator
    cross = _stroke(ink, ppm, [x + t * rail.n for t in along], rail.u, mm)
    line = _stroke(ink, ppm, [x + t * rail.u for t in along], rail.n, mm)
    x = x + (cross[1] if cross else 0.0) * rail.u + (line[1] if line else 0.0) * rail.n  # where the strokes cross
    at = st.s + (cross[1] if cross else 0.0)
    width = cross[0] if cross else float(np.median([lines[j].width for j in st.own + st.through]))
    clear = max(rail.width, width) / 2 + 1.5 / ppm  # off the lines' strokes (sampling looks a pixel around)
    if found := _arrow(ink, ppm, x, rail.u, rail.n, clear, rail.width, room, mm):
        return *found, at  # first: two heads tip to tip look like a dot or a tick
    # a dot where the lines run on far past it, or only meet the dimension line there, is the junction of wires: only
    # a dimension's own extension line crossing the dimension line makes it an end
    for test in (_dot, _tick) if st.own and not st.unseen else (_tick,):
        if found := test(ink, ppm, x, rail.u, rail.n, clear, rail.width, mm):
            return *found, at
    return None


def _inked(ink: np.ndarray, ppm: float, points: np.ndarray, around: bool = True) -> np.ndarray:
    """Whether there is ink at each of ``points`` (rows x, y), or next to it if ``around``."""
    px = np.rint(np.asarray(points, float).reshape(-1, 2) * ppm).astype(np.int64)
    out = np.zeros(len(px), bool)
    h, w = ink.shape
    near = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)) if around else ((0, 0),)
    for dx, dy in near:
        x, y = px[:, 0] + dx, px[:, 1] + dy
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        out[ok] |= ink[y[ok], x[ok]] > 0
    return out


def _stroke(ink: np.ndarray, ppm: float, points: list[np.ndarray], across: np.ndarray, mm: float
            ) -> tuple[float, float] | None:
    """How wide a line is drawn and how far it lies off ``points`` (along ``across``), measured on the image across
    it at those points: its traced width takes in what touches it (a line through dots comes out as thick as they
    are), and a traced line may lie a pixel beside the ink, or more where a photo bends it. The stroke is the run of
    ink nearest each point. None where it is not seen three times."""
    offsets = np.arange(-1.5 * mm * ppm, 1.5 * mm * ppm + 1) / ppm
    runs = []
    for x in points:
        hit = _inked(ink, ppm, x + np.outer(offsets, across), around=False)
        near = np.flatnonzero(hit & (np.abs(offsets) <= 0.6 * mm))
        if not len(near):
            continue
        k = near[np.argmin(np.abs(offsets[near]))]
        gaps = np.flatnonzero(~hit)
        lo, hi = gaps[gaps < k], gaps[gaps > k]
        if len(lo) and len(hi):  # a stroke of its own, not one along another
            runs.append(((hi[0] - lo[-1] - 1) / ppm, (offsets[lo[-1] + 1] + offsets[hi[0] - 1]) / 2))
    return tuple(map(float, np.median(runs, axis=0))) if len(runs) >= 3 else None


def _arm(ink: np.ndarray, ppm: float, x: np.ndarray, d: np.ndarray, start: float, stop: float) -> float:
    """How far from ``x`` along ``d`` the ink runs on, from ``start`` (0 if there is none there) up to ``stop``."""
    r = np.arange(start, stop, 0.5 / ppm)
    hit = _inked(ink, ppm, x + r[:, None] * d)
    if not len(r) or not hit[0]:
        return 0.0
    miss = np.flatnonzero(~hit[:-1] & ~hit[1:])  # two samples without ink: the stroke ended
    return float(r[miss[0]]) if len(miss) else stop


def _dot(ink, ppm, x, u, n, clear, width, mm):
    """A filled dot: ink off the lines on all four diagonals between them, out to about one radius and none beyond it
    (a tick has ink on two of them, a cross runs on further), and all around inside that radius where it reaches out
    beyond the lines' strokes (two arrowheads meeting tip to tip, tested before, have it only near the line)."""
    start = clear * math.sqrt(2)
    diagonals = [(i * u + j * n) / math.sqrt(2) for i in (1, -1) for j in (1, -1)]
    arms = [_arm(ink, ppm, x, d, start, DOT[1] * mm + 3 / ppm) for d in diagonals]
    if min(arms) - start < 1 / ppm or min(arms) < 0.6 * max(arms) or max(arms) > DOT[1] * mm:
        return None
    radius = float(np.mean(arms)) - 1 / ppm  # the samples look a pixel around
    if radius < DOT[0] * mm:
        return None
    for a in np.tile(np.radians([35, 55]), 4) + np.repeat(np.arange(4) * math.pi / 2, 2):
        r = np.arange(clear / min(abs(math.sin(a)), abs(math.cos(a))), 0.9 * radius, 0.5 / ppm)  # may be none
        if not _inked(ink, ppm, x + np.outer(r, math.cos(a) * u + math.sin(a) * n)).all():
            return None
    ring = [(float(x[0] + radius * math.cos(a)), float(x[1] + radius * math.sin(a)))
            for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    return "dot", 2 * radius, [Polygon(ring)]


def _tick(ink, ppm, x, u, n, clear, width, mm):
    """An oblique tick through the end: a short stroke at 25 to 65 degrees to the line that ends on both sides, with
    no stroke across it (that would be a cross) and slim: the ink reaches out across it less than SLIM of how far it
    runs along it. A blob as wide as it is long, such as the dot where wires join on a circuit diagram, is no tick."""
    best = None
    for deg in range(25, 66, 5):
        for sign in (1, -1):
            a = math.radians(deg)
            d = math.cos(a) * u + sign * math.sin(a) * n
            start = clear / min(math.sin(a), math.cos(a))
            arms = [_arm(ink, ppm, x, k * d, start, TICK[1] * mm) for k in (1, -1)]
            if min(arms) - start < TICK[0] * mm or max(arms) >= TICK[1] * mm:
                continue
            if best is None or min(arms) > min(best[1]):
                best = (d, arms, a, start)
    if best is None:
        return None
    d, arms, a, start = best
    across = np.array([-d[1], d[0]])
    wide = [max(0.0, _arm(ink, ppm, x, k * across, start, TICK[1] * mm) - start) for k in (1, -1)]
    if min(wide) >= TICK[0] * mm or sum(wide) >= SLIM * (arms[0] + arms[1] - 2 * start):
        return None
    back = 1 / ppm + width / 2  # the samples look a pixel around and the stroke ends round
    p, q = x + (arms[0] - back) * d, x - (arms[1] - back) * d
    tick = Line(float(q[0]), float(q[1]), float(p[0]), float(p[1]), max(width, 0.25 * mm))
    return "tick", float((arms[0] + arms[1] - 2 * back) * math.cos(a)), [tick]


def _arrow(ink, ppm, x, u, n, clear, width, room, mm):
    """The arrowheads with their tip at the end, pointing at it along the line from either side, or from both where
    two dimensions of a chain meet, within the ``room`` there is along the line each way before the next end (see
    _head); the length of the first."""
    heads = [h for sign, space in zip((1, -1), room) if (h := _head(ink, ppm, x, sign * u, n, clear, width, space, mm))]
    return ("arrow", heads[0][0], [q for _, parts in heads for q in parts]) if heads else None


def _head(ink, ppm, x, v, n, clear, width, space, mm) -> tuple[float, list] | None:
    """An arrowhead with its tip at ``x`` and its back along ``v``, filled or open, with its length and its strokes as
    printed: across the line the ink widens about evenly on both sides from the tip to the back of the head, and right
    behind it is no wider than the line's stroke. Measured as a profile, as a slim head (CAD draws them at about 9
    degrees) is barely wider than the line."""
    off = np.arange(-1.5 * mm * ppm, 1.5 * mm * ppm + 1) / ppm  # across the line
    r = np.arange(clear, min(ARROW[1] * mm + 1.0 * mm, space - 1.0 * mm), 0.5 / ppm)  # from the tip
    if not len(r) or r[-1] < ARROW[0] * mm:
        return None
    lo, hi = _extent(ink, ppm, x, v, n, r, off)
    wide = hi - lo
    out = (wide > np.percentile(wide, 25) + 3 / ppm) & (np.abs(hi + lo) <= wide / 2 + 2 / ppm)
    back = _back(out, r, mm)
    if back is None:
        return None
    k = int(np.searchsorted(r, back))
    half = float(wide[k] / 2 / back)
    head = (r <= back) & out
    if not 0.04 < half < 0.5 or np.polyfit(r[head], wide[head], 1)[0] <= 0:
        return None  # neither a bar along the line nor a blob, and widening towards its back
    base = x + back * v + (hi[k] + lo[k]) / 2 * n  # the middle of its back
    points = [tuple(map(float, p)) for p in (x, base + back * half * n, base - back * half * n)]
    if _inked(ink, ppm, x + 0.8 * (base - x) + np.outer((1, -1), 0.4 * back * half * n)).all():
        return float(back), [Polygon(points)]
    return float(back), [Line(*points[0], *points[1], width), Line(*points[0], *points[2], width)]


def _extent(ink, ppm, x, v, n, r, off) -> tuple[np.ndarray, np.ndarray]:
    """The outermost ink across the line (offsets ``off`` along ``n``) at each distance ``r`` along ``v`` from ``x``:
    its two edges (both 0 where there is none)."""
    points = x + r[:, None, None] * v + off[None, :, None] * n
    hit = _inked(ink, ppm, points.reshape(-1, 2), around=False).reshape(len(r), len(off))
    first, last = np.argmax(hit, axis=1), len(off) - 1 - np.argmax(hit[:, ::-1], axis=1)
    seen = hit.any(axis=1)
    return np.where(seen, off[first], 0.0), np.where(seen, off[last], 0.0)


def _back(out: np.ndarray, r: np.ndarray, mm: float) -> float | None:
    """Where the back of an arrowhead is, along the line from its tip: the end of the first stretch where the ink
    reaches out beyond the line's stroke, if that stretch starts in the first three quarters of it (a slim head is no
    wider than the line near its tip), ends between ARROW[0] and ARROW[1] and is followed by half a millimetre of bare
    line."""
    if not out.any():
        return None
    first = int(np.argmax(out))
    gaps = np.flatnonzero(~out[first:])
    end = first + (gaps[0] if len(gaps) else len(out) - first) - 1
    behind = (r > r[end]) & (r <= r[end] + 0.5 * mm)
    if not ARROW[0] * mm <= r[end] <= ARROW[1] * mm or r[first] > 0.75 * r[end] or out[behind].any():
        return None
    return float(r[end])


def linework(m: Dimension) -> list[Line]:
    """The lines a dimension is drawn with in CAD, where other lines of the drawing end: its dimension line from end to
    end, and each extension line from its measured point to ``extension`` past the dimension line (none where the
    dimension line ends on the drawing's linework); not its terminators, nor what of its strokes CAD does not draw."""
    out = [Line(m.x1, m.y1, m.x2, m.y2, m.width)]
    for p, end in ((m.p1, (m.x1, m.y1)), (m.p2, (m.x2, m.y2))):
        if (length := math.dist(p, end)) > 1e-6:
            k = 1 + m.extension / length
            out.append(Line(p[0], p[1], p[0] + k * (end[0] - p[0]), p[1] + k * (end[1] - p[1]), m.width))
    return out


def check(dims: list[Dimension], scale: int | None, mm: float = 1.0) -> None:
    """Marks the dimensions whose value does not fit their length (on the drawing, in ``mm`` units per millimetre):
    at the drawing ``scale``, in the unit most of the sheet's dimensions are written in; without a scale, against the
    ratio of value to length most of them share (from three on)."""
    rows = [(m, v, math.dist((m.x1, m.y1), (m.x2, m.y2)) / mm) for m in dims
            if not m.doubt and m.value and (v := value(m.value))]
    if not rows:
        return
    if scale:
        def fit(unit: float) -> list[bool]:
            return [abs(number * (own or unit) / scale - length) <= _tolerance(length, step * (own or unit) / scale)
                    for _, (number, step, own), length in rows]

        best = max((*UNITS.values(), INCH), key=lambda unit: (sum(fit(unit)), -unit))
        why = f"its value does not fit its length at 1:{scale}"
        fits = fit(best)
    elif len(rows) >= 3:
        ratio = float(np.median([length / (number * (own or 1.0)) for _, (number, _, own), length in rows]))
        why = "its value does not fit its length as the other dimensions of the sheet do"
        fits = [abs(number * (own or 1.0) * ratio - length) <= _tolerance(length, step * (own or 1.0) * ratio)
                for _, (number, step, own), length in rows]
    else:
        return
    for (m, _, _), ok in zip(rows, fits):
        if not ok:
            m.doubt = why


def separators(dims: list[Dimension], texts: list[Text]) -> None:
    """Marks the dimensions whose value is read with another decimal separator than the other numbers of the sheet
    (those in ``texts`` read reliably) are written with: a drawing writes its decimals one way, and the value is the
    same number either way, so its length does not tell. Text recognition reads a comma whose tail it misses as a
    point, but hardly ever a point as a comma: the other numbers are written with commas where at least COMMAS of them
    are read with one, else with points where more are read with a point than with a comma."""
    for m in dims:
        own = _SEPARATED.search(m.text.text) if m.text and not m.doubt else None
        if own is None:
            continue
        others = [s for t in texts if not t.uncertain and t is not m.text and not _same(t, m.text)
                  for s in _SEPARATED.findall(t.text)]
        commas, points = others.count(","), others.count(".")
        written = "," if commas >= COMMAS else "." if points > commas else None
        if written and own.group(1) != written:
            m.doubt = "its decimal separator is not the one the other numbers of the sheet are written with"


def _tolerance(length: float, step: float) -> float:
    """How far a value may be off its length on the drawing: what a scan and the drawing's own accuracy allow, plus
    half the step the value is written to."""
    return FIT[0] + FIT[1] * length + step / 2
