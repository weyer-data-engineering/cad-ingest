"""Linework that looks like a symbol, but that was recognised as no symbol of the libraries: a valve, a fixture or a
device the libraries lack, or one of theirs that recognition missed (drawn too unlike its block, or too faint). Nothing
is guessed about what it is: it stays the linework it was traced as, on the layer LAYER, and is listed with its box and
why it looks like a symbol, so that a person finds it and checks it, or adds the symbol to a library.

A cluster looks like a symbol where it is compact and closed: small pieces of linework that touch one another,
together the size of a symbol and not much longer than wide, with several strokes and at least one closed shape (a
ring, a box, a triangle) that is wide for the cluster, as a symbol is drawn around its shape. Not a pattern: the loops
that hatching, freehand strokes crossing one another or a letter enclose are narrow for the cluster they are part of.
Not text (larger than the sheet's letters, no lower than one across, as a row of letters that was not read is only as
high as one letter, and not covered by a text, the values of the dimensions being texts too), not the frame around a
text (one closed shape of at most four strokes with a text inside: a box, a triangle), not walls, doors, equipment, the
frame or signal lines (objects of their own), nothing in the title block (its rulings, a logo) and not a symbol found
already. Pipes and wires are left out as long lines, and so are the short pipes that run from one thing to another,
ending on its strokes rather than meeting them at a corner (a pipe between two valves), so that each valve stands
alone."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .model import Arc, Circle, Drawing, Line, Polyline, Unclassified
from .render import text_length
from .titleblock import SNAP

LAYER = "P2C-SYMBOL-UNCLASSIFIED"
TOUCH = 0.5  # mm: pieces this close to each other are one cluster
STEP = 0.25  # mm between the points a piece is sampled at
STROKES = 3  # strokes a cluster has at least: a ring alone, or with a bar across, is as likely a frame or a mark
HOLE = 1.0  # mm²: the opening of a closed shape at least; a dot (a junction) or a letter encloses less
RASTER = 8.0  # px per mm at which the pieces of a cluster are drawn to find the openings they enclose
LETTERS = 1.5  # a cluster is at least this many times as large as the sheet's usual text: its letters are smaller
# and no lower than its usual text across: a row of letters that was not read, such as a device tag, is as long as it
# has letters but only as high as one; a symbol lower than the letters cannot be told from such a row
OPENING = 0.2  # a cluster's widest opening is at least this share of its width across (the largest circle inside it
# against the shorter side of its box): a symbol is drawn around its ring, box or triangle, while the loops in a corner
# of hatching, between freehand strokes crossing one another or in a letter are narrow for the cluster they are part of
SIZES = (3.0, 25.0)  # mm (box diagonal): the smallest and largest symbol where the libraries tell nothing larger
LARGER = 1.25  # a cluster may be this much larger than the largest symbol of the libraries (real objects vary)
COMPACT = 3.0  # a cluster's box is at most this many times as long as it is wide: a level mark or a window is longer
KINDS = ("wall", "door", "equipment", "frame", "signal")  # polylines of these kinds are objects, not part of a symbol
FRAME = 4  # strokes of a text's frame at most: a closed box or triangle with a text inside is the text's, no symbol
COVERED = 0.5  # share of a cluster's box that a text's box covers where the cluster is that text, not read


def sizes(templates: dict) -> float | None:
    """The size (box diagonal, mm) of the largest of the symbols ``templates`` (library.Template by name), if any."""
    out = []
    for t in templates.values():
        pts = np.vstack([_sampled(q) for q in t.parts])
        out.append(float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0)))))
    return max(out) if out else None


def find(d: Drawing, largest: float | None = None) -> None:
    """Puts the linework of ``d`` that looks like a symbol on LAYER and lists each cluster in ``d.unclassified``.
    ``largest``: the size of the largest symbol of the libraries on the sheet (see sizes). The texts are those of the
    sheet with the values of its dimensions, as pipeline._texts_and_values gives them; what lies in the title block
    (``d.title_block``, found before) is the block's."""
    texts = d.texts + [m.text for m in d.dimensions if m.text]
    heights = [t.height for t in texts if not t.uncertain]
    letter = float(np.median(heights)) if heights else 0.0  # the sheet's usual text height
    lo = max(SIZES[0], LETTERS * letter)
    hi = max(SIZES[1], LARGER * largest) if largest else SIZES[1]
    every = d.lines + d.arcs + d.circles + d.polylines
    sampled = [_sampled(q) for q in every]
    block = d.title_block.box if d.title_block else None
    small = [n for n, (q, p) in enumerate(zip(every, sampled)) if not q.layer and not getattr(q, "dashed", False)
             and getattr(q, "kind", "") not in KINDS and np.hypot(*(p.max(axis=0) - p.min(axis=0))) <= hi
             and not (block and _within(p, block))]
    if not small:
        return
    placed = [np.vstack([_sampled(q) for q in s.parts]) for s in d.symbols if s.parts]  # pipes end on them too
    pipes = _connectors(every, sampled + placed, small)  # and the long lines of pipes and wires are left out
    small = [n for n in small if n not in pipes]
    if not small:
        return
    points = np.vstack([sampled[n] for n in small])
    owner = np.repeat(np.arange(len(small)), [len(sampled[n]) for n in small])
    i, j = cKDTree(points).query_pairs(TOUCH, output_type="ndarray").T
    _, label = connected_components(coo_matrix((np.ones(len(i)), (owner[i], owner[j])), shape=(len(small),) * 2),
                                    directed=False)
    of = label[owner]  # the cluster of each point
    order = np.argsort(of, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(of[order]) > 0])
    lows, highs = np.minimum.reduceat(points[order], starts), np.maximum.reduceat(points[order], starts)
    symbols = [_box(np.vstack([_sampled(q) for q in s.parts])) for s in d.symbols if s.parts]
    for k, low, high in zip(of[order][starts], lows, highs):
        box, (w, h) = (*low, *high), high - low
        if not lo <= math.hypot(w, h) <= hi or max(w, h) > COMPACT * min(w, h) or min(w, h) < letter:
            continue
        members = [every[small[n]] for n in np.flatnonzero(label == k)]
        strokes, (closed, widest) = sum(len(_parts(q)) for q in members), _closed(members, box)
        if strokes < STROKES or not closed or widest < OPENING * min(w, h) or \
                any(_share(box, b) >= 0.5 for b in symbols) or \
                any(_share(box, _text_box(t)) >= COVERED for t in texts) or \
                closed == 1 and strokes <= FRAME and any(_inside(box, t) for t in texts):
            continue
        for q in members:
            q.layer = LAYER
        # the box gives the size, in sheet mm once the drawing is brought to the size of the sheet (Drawing.scaled)
        d.unclassified.append(Unclassified(tuple(round(float(v), 2) for v in box),
                                           f"{strokes} strokes with {closed} closed shape{'s' * (closed > 1)}, "
                                           f"recognised as no symbol of the libraries"))


def _connectors(every: list, sampled: list[np.ndarray], small: list[int]) -> set[int]:
    """The items among ``small`` that run from one thing to another like a pipe or a wire: straight, with each end on
    other linework or a placed symbol (``sampled``: the points of each item of ``every``, then of each symbol) but at
    no corner with it, and the two ends on different things (a pipe from a valve's side to the next valve, or from the
    tip of an off-page connector to a vessel). At a corner a stroke of a symbol meets one other stroke, which ends there
    too; a bar or a ring's diameter ends on one thing at both ends."""
    tips = [_tips(q) for q in every]
    corner, corner_of = cKDTree(np.vstack(tips)), np.repeat(np.arange(len(every)), [len(t) for t in tips])
    body, body_of = cKDTree(np.vstack(sampled)), np.repeat(np.arange(len(sampled)), [len(p) for p in sampled])
    out = set()
    for n in small:
        q = every[n]
        if not (isinstance(q, Line) or isinstance(q, Polyline) and not q.closed and not any(q.bulges)):
            continue
        touched = []
        for end in tips[n]:
            at = set(corner_of[corner.query_ball_point(end, TOUCH)].tolist()) - {n}
            on = set(body_of[body.query_ball_point(end, TOUCH)].tolist()) - {n}
            if len(at) == 1 and on == at:
                break  # a corner
            touched.append(on)
        if len(touched) == 2 and all(touched) and not touched[0] & touched[1]:
            out.add(n)
    return out


def _tips(q) -> np.ndarray:
    """The two ends of an open item (none of a circle or a closed polyline)."""
    if isinstance(q, Circle) or isinstance(q, Polyline) and q.closed:
        return np.zeros((0, 2))
    return np.array([q.points[0], q.points[-1]] if isinstance(q, Polyline) else _ends(q), float)


def _parts(q) -> list:
    return q.parts() if isinstance(q, Polyline) else [q]


def _ends(q: Line | Arc | Circle) -> list[tuple[float, float]]:
    if isinstance(q, Circle):
        return []
    if isinstance(q, Line):
        return [(q.x1, q.y1), (q.x2, q.y2)]
    return [(q.cx + q.r * math.cos(math.radians(t)), q.cy + q.r * math.sin(math.radians(t))) for t in (q.start, q.end)]


def _sampled(q) -> np.ndarray:
    """Points about every STEP mm along a line, arc, circle or polyline."""
    out = []
    for p in _parts(q):
        if isinstance(p, Line):
            n = max(2, int(math.hypot(p.x2 - p.x1, p.y2 - p.y1) / STEP) + 1)
            out.append(np.linspace((p.x1, p.y1), (p.x2, p.y2), n))
            continue
        start, sweep = (0.0, 360.0) if isinstance(p, Circle) else (p.start, (p.end - p.start) % 360 or 360)
        a = np.radians(start + np.linspace(0, sweep, max(3, int(math.radians(sweep) * p.r / STEP) + 1)))
        out.append(np.column_stack([p.cx + p.r * np.cos(a), p.cy + p.r * np.sin(a)]))
    return np.vstack(out)


def _closed(items: list, box: tuple) -> tuple[int, float]:
    """The closed shapes of the pieces: how many openings they enclose of at least HOLE mm², drawn with strokes TOUCH
    wide (so that pieces meeting with a small gap, or one ending on the side of another, close a shape as well), and
    how wide the widest of them is (mm: the diameter of the largest circle inside it). A dot, such as a junction,
    encloses less."""
    x0, y0 = box[0] - TOUCH, box[1] - TOUCH
    img = np.zeros((math.ceil((box[3] - y0 + TOUCH) * RASTER) + 1, math.ceil((box[2] - x0 + TOUCH) * RASTER) + 1),
                   np.uint8)
    for q in items:
        for p in _parts(q):
            cv2.polylines(img, [np.round((_sampled(p) - (x0, y0)) * RASTER).astype(np.int32)], False, 1,
                          max(1, round(TOUCH * RASTER)))
    contours, tree = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # no contours, no tree
    holes = [c for k, c in enumerate(contours) if tree[0][k][3] >= 0 and cv2.contourArea(c) >= HOLE * RASTER ** 2]
    if not holes:
        return 0, 0.0
    inside = np.zeros_like(img)
    cv2.drawContours(inside, holes, -1, 1, -1)
    return len(holes), 2 * float(cv2.distanceTransform(inside, cv2.DIST_L2, 5).max()) / RASTER


def _box(points: np.ndarray) -> tuple[float, float, float, float]:
    return (*points.min(axis=0), *points.max(axis=0))


def _text_box(t) -> tuple[float, float, float, float]:
    half_l, half_h = (t.length or text_length(t)) / 2, t.height / 2
    return (t.x - half_h, t.y - half_l, t.x + half_h, t.y + half_l) if round(t.angle) % 180 == 90 else \
        (t.x - half_l, t.y - half_h, t.x + half_l, t.y + half_h)


def _inside(box: tuple, t) -> bool:
    return box[0] <= t.x <= box[2] and box[1] <= t.y <= box[3]


def _within(points: np.ndarray, box: tuple) -> bool:
    """Whether all the points lie in the box (x0, y0, x1, y1), give or take SNAP, as titleblock.lay takes a line for
    one of the title block's."""
    return bool((points.min(axis=0) >= np.subtract(box[:2], SNAP)).all()
                and (points.max(axis=0) <= np.add(box[2:], SNAP)).all())


def _share(a: tuple, b: tuple) -> float:
    """The share of box ``a`` (x0, y0, x1, y1) that box ``b`` covers."""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(w, 0) * max(h, 0) / area if area > 0 else 0.0
