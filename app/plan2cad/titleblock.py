"""The title block as a table: the grid of ruled cells in the corner of the drawing frame's right and bottom line, where
ISO 5457 puts it, with the texts that stand in each cell. No layout is assumed: the rows, the columns and the cells that
span several of them are those of the rulings on the sheet."""

from __future__ import annotations

import math
from statistics import median

from .model import Arc, Cell, Circle, Drawing, Line, Text, TitleBlock

LAYER = "P2C-TITLEBLOCK"
SNAP = 2.0  # mm: a ruling that ends this close to another ends on it (connect closes gaps up to 2 mm)
SAME = 1.0  # mm: parallel rulings closer than this are one (a line traced twice)
LEAN = 0.005  # a ruling may lean by 0.3 mm and this share of its length
FRAME = 0.5  # share of the sheet's width or height that a line of the drawing frame runs at least
MARGIN = 40.0  # mm: the drawing frame runs at most this far inside the edge of the sheet
ROW = 50.0  # mm: no row of a title block is higher than this (a logo, a box of notes)
COVER = 0.6  # share of the side between two cells that rulings cover where they divide the two
SMALL = 1.5  # mm: no cell is narrower or lower than this

Segment = tuple[float, float, float]  # position across, start and end along it, in mm


def find(d: Drawing) -> TitleBlock | None:
    """The title block of the drawing as a table, or None if it has none: the rows that run to the right line of the
    drawing frame, stacked up from its bottom line (see _box), cut into the cells their rulings enclose; cells that no
    ruling divides are one, spanning several rows or columns. The texts are those whose centre is in a cell. A table
    needs two cells with text."""
    across, down = _rulings(d)
    for right, bottom in _corners(d, across, down):
        found = _box(across, down, right, bottom)
        if found is None:
            continue
        box = _open_top(*found, down, d.texts)
        cells = _cells(box, across, down)
        for c in cells:
            mine = _reading_order([t for t in d.texts if c.box[0] <= t.x < c.box[2] and c.box[1] <= t.y < c.box[3]])
            c.text, c.uncertain = " ".join(t.text for t in mine), any(t.uncertain for t in mine)
        if len(cells) >= 2 and sum(bool(c.text) for c in cells) >= 2:
            return TitleBlock(box, cells)
    return None


def _corners(d: Drawing, across: list[Segment], down: list[Segment]) -> list[tuple[float, float]]:
    """The bottom right corners of the drawing frame, the inner one of a double frame first: where a long ruling near
    the right edge of the sheet ends at the bottom on a long one near its bottom edge that ends there too. The rows of
    a title block wider than half the sheet are long as well, but the right line runs on below them."""
    out = []
    for x, _, end in sorted(r for r in down if r[0] > d.width - MARGIN and r[2] - r[1] >= FRAME * d.height):
        out += [(x, y) for y, a, b in across if y > d.height - MARGIN and b - a >= FRAME * d.width
                and abs(y - end) <= SNAP and abs(b - x) <= SNAP]
    return out


def lay(d: Drawing, block: TitleBlock) -> None:
    """Puts the linework of the title block on LAYER: the lines, polylines, arcs and circles inside its box (its
    rulings, a logo), but not those of the frame's lines along its right and bottom side."""
    x0, y0, x1, y1 = block.box

    def mine(points: list[tuple[float, float]]) -> bool:
        inside = all(x0 - SNAP <= x <= x1 + SNAP and y0 - SNAP <= y <= y1 + SNAP for x, y in points)
        frame = all(abs(x - x1) <= SNAP for x, _ in points) or all(abs(y - y1) <= SNAP for _, y in points)
        return inside and not frame

    for pl in d.polylines:
        if pl.kind != "frame" and mine(pl.points):
            pl.layer = LAYER
    for ln in d.lines:
        if mine([(ln.x1, ln.y1), (ln.x2, ln.y2)]):
            ln.layer = LAYER
    for q in d.arcs + d.circles:  # also the arcs that objects.curves made of polylines in it
        if mine(_along(q)):
            q.layer = LAYER


def _along(q: Arc | Circle) -> list[tuple[float, float]]:
    """Points along an arc or a circle, its ends included, at most 15 degrees apart."""
    start, sweep = (0.0, 360.0) if isinstance(q, Circle) else (q.start, (q.end - q.start) % 360 or 360.0)
    angles = [math.radians(start + sweep * k / 24) for k in range(25)]
    return [(q.cx + q.r * math.cos(a), q.cy + q.r * math.sin(a)) for a in angles]


def _rulings(d: Drawing) -> tuple[list[Segment], list[Segment]]:
    """The solid horizontal (y, x0, x1) and vertical (x, y0, y1) rulings of the drawing, each joined from the straight
    pieces of its lines and polylines that run along it. The chords of a spline count for none: they draw a curve, such
    as a logo, and the flattest of them would cut its cell into rows and columns (objects.curves makes the chords of a
    curve an arc or a spline first)."""
    pieces = [q for pl in d.polylines if not pl.dashed and not pl.smooth for q in pl.parts() if isinstance(q, Line)]
    across, down = [], []
    for q in pieces + [q for q in d.lines if not q.dashed]:
        dx, dy = abs(q.x2 - q.x1), abs(q.y2 - q.y1)
        if dy <= 0.3 + LEAN * dx < dx:
            across.append(((q.y1 + q.y2) / 2, min(q.x1, q.x2), max(q.x1, q.x2)))
        elif dx <= 0.3 + LEAN * dy < dy:
            down.append(((q.x1 + q.x2) / 2, min(q.y1, q.y2), max(q.y1, q.y2)))
    return _joined(across), _joined(down)


def _joined(segments: list[Segment]) -> list[Segment]:
    """Segments along one line (within SAME of each other across it) that overlap or nearly meet (SNAP) as one, at
    their position weighted by length."""
    lines: list[list[Segment]] = []
    for s in sorted(segments):
        if lines and s[0] - lines[-1][-1][0] <= SAME:
            lines[-1].append(s)
        else:
            lines.append([s])
    out = []
    for line in lines:
        run: list[Segment] = []
        for s in sorted(line, key=lambda s: s[1]):
            if run and s[1] > max(r[2] for r in run) + SNAP:
                out.append(_merged(run))
                run = []
            run.append(s)
        out.append(_merged(run))
    return out


def _merged(run: list[Segment]) -> Segment:
    weight = [max(b - a, 1e-6) for _, a, b in run]
    return (sum(p * w for (p, _, _), w in zip(run, weight)) / sum(weight), min(a for _, a, _ in run),
            max(b for _, _, b in run))


def _box(across: list[Segment], down: list[Segment], right: float, bottom: float
         ) -> tuple[tuple[float, float, float, float], list[float]] | None:
    """The box of the title block in the corner of the frame's ``right`` and ``bottom`` line, and the heights of its
    rows: the rulings that run from one left side to the right line, stacked up from the bottom line with no row
    higher than ROW. The rows of a table share its left side: two of them, or a ruling down that side from the top to
    the bottom line, make it one; the frame's left line, which runs on above, closes no single row (a line across the
    whole frame, such as a ground line, is no title block). Of several left sides, the one furthest left."""
    reach = [(y, a) for y, a, b in across if abs(b - right) <= SNAP and y < bottom - SMALL]
    for left in sorted({a for _, a in reach}):
        top, heights = bottom, []
        for y in sorted((y for y, a in reach if abs(a - left) <= SNAP), reverse=True):
            if top - y > ROW:
                break
            if top - y >= SMALL:
                heights.append(top - y)
                top = y
        closed = any(abs(x - left) <= SNAP and abs(a - top) <= SNAP and b >= bottom - SNAP for x, a, b in down)
        if heights and (len(heights) >= 2 or closed):
            return (left, top, right, bottom), heights
    return None


def _open_top(box: tuple[float, float, float, float], heights: list[float], down: list[Segment], texts: list[Text]
              ) -> tuple[float, float, float, float]:
    """The box with the row above its topmost ruling where the block is open at the top (no ruling runs down its left
    side from there) and texts stand in that row: a row as high as the middle one of the others."""
    left, top, right, bottom = box
    if any(abs(x - left) <= SNAP and a <= top + SNAP and b >= top + SMALL for x, a, b in down):
        return box
    row = median(heights)
    if not any(left < t.x < right and top - row < t.y < top for t in texts):
        return box
    return left, top - row, right, bottom


def _cells(box: tuple[float, float, float, float], across: list[Segment], down: list[Segment]) -> list[Cell]:
    """The cells of the box: the grid of all its rulings, with the pieces of it that no ruling divides joined; a piece
    that is not a rectangle then (a ruling broken on the scan) gives a cell per row."""
    x0, y0, x1, y1 = box
    rows = [s for s in across if y0 + SMALL <= s[0] <= y1 - SMALL and min(s[2], x1) - max(s[1], x0) > SMALL]
    cols = [s for s in down if x0 + SMALL <= s[0] <= x1 - SMALL and min(s[2], y1) - max(s[1], y0) > SMALL]
    ys, xs = [y0] + _positions([s[0] for s in rows]) + [y1], [x0] + _positions([s[0] for s in cols]) + [x1]
    n, m = len(ys) - 1, len(xs) - 1
    parent = list(range(n * m))

    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i in range(n):
        for j in range(m):
            if j + 1 < m and _covered(cols, xs[j + 1], ys[i], ys[i + 1]) < COVER:
                parent[root(i * m + j + 1)] = root(i * m + j)
            if i + 1 < n and _covered(rows, ys[i + 1], xs[j], xs[j + 1]) < COVER:
                parent[root((i + 1) * m + j)] = root(i * m + j)
    groups: dict[int, list[tuple[int, int]]] = {}
    for k in range(n * m):
        groups.setdefault(root(k), []).append(divmod(k, m))
    cells = []
    for members in groups.values():
        r0, r1 = min(i for i, _ in members), max(i for i, _ in members)
        c0, c1 = min(j for _, j in members), max(j for _, j in members)
        if len(members) == (r1 - r0 + 1) * (c1 - c0 + 1):
            cells.append(Cell(r0, c0, (xs[c0], ys[r0], xs[c1 + 1], ys[r1 + 1]), rows=r1 - r0 + 1, cols=c1 - c0 + 1))
            continue
        for i in range(r0, r1 + 1):
            js = sorted(j for r, j in members if r == i)
            for a, b in _runs(js):
                cells.append(Cell(i, a, (xs[a], ys[i], xs[b + 1], ys[i + 1]), cols=b - a + 1))
    return sorted(cells, key=lambda c: (c.row, c.col))


def _positions(values: list[float]) -> list[float]:
    """The places of the rulings: values closer than SMALL to the one before are the same place."""
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] < SMALL:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(g) / len(g) for g in out]


def _covered(segments: list[Segment], at: float, a: float, b: float) -> float:
    """The share of the stretch from ``a`` to ``b`` along the place ``at`` that the segments there cover."""
    spans = sorted((max(s[1], a), min(s[2], b)) for s in segments if abs(s[0] - at) <= SAME and s[2] > a and s[1] < b)
    covered, end = 0.0, a
    for lo, hi in spans:
        covered += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return covered / (b - a)


def _runs(values: list[int]) -> list[tuple[int, int]]:
    """Consecutive integers as (first, last) runs."""
    out: list[tuple[int, int]] = []
    for v in values:
        if out and v == out[-1][1] + 1:
            out[-1] = (out[-1][0], v)
        else:
            out.append((v, v))
    return out


def _reading_order(texts: list[Text]) -> list[Text]:
    """Texts line by line from the top, each line from the left: a text starts a new line where it stands more than
    half a text height lower than the line's first."""
    out: list[Text] = []
    line: list[Text] = []
    for t in sorted(texts, key=lambda t: t.y):
        if line and t.y - line[0].y > 0.5 * max(t.height, line[0].height):
            out += sorted(line, key=lambda t: t.x)
            line = []
        line.append(t)
    return out + sorted(line, key=lambda t: t.x)
