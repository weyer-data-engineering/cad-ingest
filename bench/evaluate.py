"""Compares a recognised drawing with its ground truth (all in sheet millimetres)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from dataclasses import replace

import cv2
import numpy as np
from scipy.spatial import cKDTree

from app.plan2cad.model import Arc, Circle, Drawing, Line, Polygon, Text
from app.plan2cad.simulate import MARKUP
from app.plan2cad.render import text_length

STEP = 0.25  # mm between sample points along the linework


def _line(x1, y1, x2, y2) -> np.ndarray:
    n = max(2, int(math.hypot(x2 - x1, y2 - y1) / STEP) + 1)
    t = np.linspace(0, 1, n)[:, None]
    return np.array([x1, y1]) + t * np.array([x2 - x1, y2 - y1])


def _arc(cx, cy, r, start, end) -> np.ndarray:
    sweep = (end - start) % 360 or 360
    n = max(3, int(math.radians(sweep) * r / STEP) + 1)
    a = np.radians(start + np.linspace(0, sweep, n))
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def samples(d: Drawing, dashed_only: bool = False) -> np.ndarray:
    """Points every STEP mm along all linework (dashed lines count as continuous)."""
    pts = [_line(ln.x1, ln.y1, ln.x2, ln.y2) for ln in d.lines if ln.dashed or not dashed_only]
    if not dashed_only:
        pts += [_arc(a.cx, a.cy, a.r, a.start, a.end) for a in d.arcs]
        pts += [_arc(c.cx, c.cy, c.r, 0, 360) for c in d.circles]
        for g in d.polygons:
            ring = g.points + g.points[:1]
            pts += [_line(*a, *b) for a, b in zip(ring, ring[1:])]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def geometry(gt: Drawing, pred: Drawing) -> dict:
    g, p = samples(gt), samples(pred)
    if not len(p):
        return {"recall_1mm": 0.0, "precision_1mm": 0.0, "f1_1mm": 0.0, "recall_05mm": 0.0, "precision_05mm": 0.0}
    dg, dp = cKDTree(p).query(g)[0], cKDTree(g).query(p)[0]
    out = {}
    for tol, key in ((1.0, "1mm"), (0.5, "05mm")):
        r, pr = float(np.mean(dg <= tol)), float(np.mean(dp <= tol))
        out[f"recall_{key}"], out[f"precision_{key}"] = r, pr
        out[f"f1_{key}"] = 2 * r * pr / (r + pr) if r + pr else 0.0
    out["deviation_mm"] = float(np.mean(dp[dp <= 1.0])) if np.any(dp <= 1.0) else None
    gd, pd = samples(gt, True), samples(pred, True)
    if len(gd):
        out["dashed_recall_1mm"] = float(np.mean(cKDTree(pd).query(gd)[0] <= 1.0)) if len(pd) else 0.0
    return out


def _sampled(parts: list) -> tuple[np.ndarray, np.ndarray]:
    """Sample points of lines, arcs, circles and polygon outlines with their direction (mod pi)."""
    pts, dirs = [], []
    for q in parts:
        if isinstance(q, Line):
            pts.append(_line(q.x1, q.y1, q.x2, q.y2))
            dirs.append(np.full(len(pts[-1]), math.atan2(q.y2 - q.y1, q.x2 - q.x1)))
        elif isinstance(q, (Arc, Circle)):
            pts.append(_arc(q.cx, q.cy, q.r, *((q.start, q.end) if isinstance(q, Arc) else (0, 360))))
            dirs.append(np.arctan2(pts[-1][:, 1] - q.cy, pts[-1][:, 0] - q.cx) + np.pi / 2)
        elif isinstance(q, Polygon):
            ring = q.points + q.points[:1]
            pts.append(np.vstack([_line(*a, *b) for a, b in zip(ring, ring[1:])]))
            dirs.append(np.zeros(len(pts[-1])))  # filled areas match in any direction
    return (np.vstack(pts), np.concatenate(dirs) % np.pi) if pts else (np.zeros((0, 2)), np.zeros(0))


def pens(gt: Drawing, pred: Drawing) -> dict:
    """The share of the reference linework drawn with the right pen: the ISO pen of the output's linework within 1 mm
    (the pen of the DXF) is the pen of the reference."""
    from app.plan2cad.export import lineweight

    def widths(d: Drawing) -> tuple[np.ndarray, np.ndarray]:
        parts = [q for q in d.lines + d.arcs + d.circles if q.width > 0]
        sampled = [_sampled([q])[0] for q in parts]
        if not parts:
            return np.zeros((0, 2)), np.zeros(0)
        return np.vstack(sampled), np.concatenate([np.full(len(s), lineweight(q.width)) for q, s in zip(parts, sampled)])

    (g, gw), (p, pw) = widths(gt), widths(pred)
    if not len(g) or not len(p):
        return {}
    dist, idx = cKDTree(p).query(g)
    near = dist <= 1.0
    return {"pens_correct": float(np.mean(gw[near] == pw[idx[near]])) if near.any() else None}


def _reference_objects(d: Drawing) -> list[list]:
    """What a CAD user would draw as one object: lines and arcs of the same pen that meet end to end
    with nothing else at that point form one polyline; circles and filled areas count singly."""
    pieces = [(q, (q.x1, q.y1), (q.x2, q.y2)) for q in d.lines]
    for a in d.arcs:
        ends = [(a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t))) for t in (a.start, a.end)]
        pieces.append((a, *ends))
    at = defaultdict(list)
    for i, (_, p, q) in enumerate(pieces):
        for x, y in (p, q):
            at[(round(x, 3), round(y, 3))].append(i)
    parent = list(range(len(pieces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def pen(q):
        return q.width, getattr(q, "dashed", False)

    for ids in at.values():
        if len(ids) == 2 and pen(pieces[ids[0]][0]) == pen(pieces[ids[1]][0]):
            parent[root(ids[0])] = root(ids[1])
    groups = defaultdict(list)
    for i, (q, _, _) in enumerate(pieces):
        groups[root(i)].append(q)
    return list(groups.values()) + [[c] for c in d.circles] + [[g] for g in d.polygons] + [s.parts for s in d.symbols]


def _entities(d: Drawing) -> list[list]:
    """The pieces of each output entity; a polyline and a symbol (a block reference) are one entity each."""
    return [[q] for q in d.lines + d.arcs + d.circles + d.polygons] + [pl.parts() for pl in d.polylines] + \
        [s.parts for s in d.symbols]


def _length(parts: list) -> float:
    return sum(math.hypot(q.x2 - q.x1, q.y2 - q.y1) if isinstance(q, Line)
               else math.radians((q.end - q.start) % 360 or 360) * q.r for q in parts if isinstance(q, (Line, Arc)))


def objects(gt: Drawing, pred: Drawing) -> dict:
    """How many output entities make up each reference object (1 = in one piece, as drawn in CAD),
    and how many tiny pieces (under 1.5 mm) the output contains."""
    ents = _entities(pred)
    sampled = [_sampled(e) for e in ents]
    if not ents or not sum(len(p) for p, _ in sampled):
        return {}
    P = np.vstack([p for p, _ in sampled])
    D = np.concatenate([a for _, a in sampled])
    E = np.concatenate([np.full(len(p), i) for i, (p, _) in enumerate(sampled)])
    tree = cKDTree(P)
    pieces, whole = [], 0
    for obj in _reference_objects(gt):
        pts, dirs = _sampled(obj)
        dist, idx = tree.query(pts, k=8, distance_upper_bound=1.0)
        hit = idx < len(P)
        diff = np.abs(D[np.minimum(idx, len(P) - 1)] - dirs[:, None]) % np.pi
        ok = hit & ((np.minimum(diff, np.pi - diff) < math.radians(20)) | isinstance(obj[0], Polygon))
        first = np.argmax(ok, axis=1)
        found = E[idx[np.arange(len(pts)), first][ok.any(axis=1)]]
        if len(found) < 0.5 * len(pts):
            continue  # mostly missing: a recognition error, not a question of pieces
        counts = Counter(found.tolist())
        pieces.append(sum(k >= 2 for k in counts.values()))
        whole += max(counts.values()) >= 0.9 * len(found)
    lengths = [_length(e) for e in ents if not isinstance(e[0], (Circle, Polygon))]
    return {"pieces_per_object": float(np.mean(pieces)), "objects_whole": whole / len(pieces),
            "entities": len(ents), "reference_objects": len(_reference_objects(gt)),
            "short_pieces": sum(x < 1.5 for x in lengths)}


def _nodes(d: Drawing) -> np.ndarray:
    """Where reference linework ends or branches: line and arc ends that are not the inner vertex of a run
    (exactly two pieces of the same pen meeting)."""
    pieces = [(q, ((q.x1, q.y1), (q.x2, q.y2))) for q in d.lines]
    pieces += [(a, tuple((a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t)))
                         for t in (a.start, a.end))) for a in d.arcs]
    at = defaultdict(list)
    for q, ends in pieces:
        for x, y in ends:
            at[(round(x, 3), round(y, 3))].append(q)
    pen = {id(q): (q.width, getattr(q, "dashed", False)) for q, _ in pieces}
    nodes = [p for p, qs in at.items() if len(qs) != 2 or pen[id(qs[0])] != pen[id(qs[1])]]
    return np.array(nodes, np.float64).reshape(-1, 2)


def continuity(gt: Drawing, pred: Drawing) -> dict:
    """What makes reworking a conversion tedious. ``breaks_per_m``: places per metre of reference linework where an
    output entity stops although the reference runs on (a gap or two pieces where one is drawn), counted from the
    output's open ends that lie on the reference and run along it, away from its ends and branches. ``curve_pieces``:
    output entities per reference arc (1 = in one piece; circles are scored by ``circles``). ``curves_as_arcs``: share
    of the reference arcs that the output draws as arcs or circles, not as chains of short straight segments."""
    flat_gt = gt.flat()
    ents = _entities(pred)
    ref = samples(flat_gt)
    if not ents or not len(ref):
        return {}
    pts, dirs, ent, curved = [], [], [], []
    for i, e in enumerate(ents):
        for q in e:
            p, a = _sampled([q])
            pts.append(p)
            dirs.append(a)
            ent.append(np.full(len(p), i))
            curved.append(np.full(len(p), isinstance(q, (Arc, Circle))))
    out = {}
    ends, heading = _open_ends(pred)
    lw, lw_dir = _sampled(flat_gt.lines + flat_gt.arcs + flat_gt.circles)
    if len(ends) and len(lw):
        dist, near = cKDTree(lw).query(ends)
        turn = np.abs(heading - lw_dir[near]) % np.pi
        on_ref = (dist <= 0.5) & (np.minimum(turn, np.pi - turn) < math.radians(25))  # running along it
        nodes = _nodes(flat_gt)
        free = cKDTree(nodes).query(ends)[0] > 1.5 if len(nodes) else np.ones(len(ends), bool)
        length = len(ref) * STEP / 1000
        out["breaks_per_m"] = float(np.sum(on_ref & free) / 2 / length)
    P, D, E, C = np.vstack(pts), np.concatenate(dirs), np.concatenate(ent), np.concatenate(curved)
    tree = cKDTree(P)
    pieces, as_arc, total = [], 0, 0
    for q in flat_gt.arcs:
        if q.r < 1.0:
            continue
        s, a = _sampled([q])
        dist, idx = tree.query(s, k=8, distance_upper_bound=0.5)
        hit = idx < len(P)
        diff = np.abs(D[np.minimum(idx, len(P) - 1)] - a[:, None]) % np.pi
        ok = hit & (np.minimum(diff, np.pi - diff) < math.radians(25))
        if ok.any(axis=1).mean() < 0.5:
            continue  # mostly missing: a recognition error, not a question of pieces
        first = idx[np.arange(len(s)), np.argmax(ok, axis=1)][ok.any(axis=1)]
        counts = Counter(E[first].tolist())
        pieces.append(sum(k >= 2 for k in counts.values()))
        as_arc += int(C[first].sum())
        total += len(first)
    if pieces:
        out["curve_pieces"] = float(np.mean(pieces))
        out["curves_as_arcs"] = as_arc / total
    return out


def _open_ends(d: Drawing) -> tuple[np.ndarray, np.ndarray]:
    """The two ends of every line, arc and open polyline (symbols are blocks and have none) with the direction of the
    entity there (mod pi)."""
    at = [(q, (q.x1, q.y1)) for q in d.lines] + [(q, (q.x2, q.y2)) for q in d.lines]
    at += [(a, (a.cx + a.r * math.cos(math.radians(t)), a.cy + a.r * math.sin(math.radians(t))))
           for a in d.arcs for t in (a.start, a.end)]
    for pl in d.polylines:
        if not pl.closed:
            parts = pl.parts()
            at += [(parts[0], pl.points[0]), (parts[-1], pl.points[-1])]
    dirs = [math.atan2(q.y2 - q.y1, q.x2 - q.x1) if isinstance(q, Line)
            else math.atan2(y - q.cy, x - q.cx) + math.pi / 2 for q, (x, y) in at]
    return np.array([p for _, p in at], np.float64).reshape(-1, 2), np.array(dirs) % np.pi


def circles(gt: Drawing, pred: Drawing) -> dict:
    def match(a, b):
        return math.hypot(a.cx - b.cx, a.cy - b.cy) < max(0.5, 0.15 * a.r) and abs(a.r - b.r) < max(0.4, 0.15 * a.r)

    reference = [c for c in gt.circles if c.r >= 1.5]  # smaller dots are outside the detection range
    if not reference:
        return {}
    hit = sum(any(match(c, q) for q in pred.circles) for c in reference)
    ok = sum(any(match(c, q) for c in gt.circles) for q in pred.circles)
    return {"circle_recall": hit / len(reference), "circle_precision": ok / max(1, len(pred.circles))}


def _cer(a: str, b: str) -> float:
    """Character error rate: Levenshtein distance of b to the reference a, divided by len(a)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def _box(t: Text, margin: float) -> tuple[float, float, float, float]:
    half_l, half_h = text_length(t) / 2 + margin, t.height / 2 + margin
    return (t.x - half_h, t.y - half_l, t.x + half_h, t.y + half_l) if t.angle == 90 else \
        (t.x - half_l, t.y - half_h, t.x + half_l, t.y + half_h)


def _in(t: Text, q: Text, margin: float = 1.0) -> bool:
    """Whether the centre of q lies in the box of t (turned with t) and q reads the same way."""
    if abs((q.angle - t.angle + 180) % 360 - 180) > 5:
        return False
    a = math.radians(t.angle)
    dx, dy = q.x - t.x, q.y - t.y
    along, across = dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)
    return abs(along) <= text_length(t) / 2 + margin and abs(across) <= t.height / 2 + margin


def _kind(s: str) -> str:
    """codes (tags, line numbers: letters and digits), numbers (sizes, dimensions) or words (notes, names)."""
    digits, letters = any(c.isdigit() for c in s), any(c.isalpha() for c in s)
    return "codes" if digits and letters else "numbers" if digits else "words"


def texts(gt: Drawing, pred: Drawing) -> dict:
    """Per ground-truth text, all recognised texts whose centre lies in its box are joined in
    reading order and compared. Recognised texts that match nothing count as false positives. A text read wrong is
    flagged if a text marked for review lies in its box; the others are errors nobody is pointed to."""
    if not gt.texts:
        return {}
    review = sum(t.uncertain for t in pred.texts)  # marked as unread: not counted as read text
    marked = [t for t in pred.texts if t.uncertain]
    pred = Drawing(pred.width, pred.height, texts=[t for t in pred.texts if not t.uncertain])
    used, cers, exact, unflagged, tokens, found = set(), [], 0, 0, 0, 0
    kinds = defaultdict(lambda: [0, 0])  # kind: [exact, all]
    for t in gt.texts:
        inside = [i for i, q in enumerate(pred.texts) if _in(t, q)]
        used.update(inside)
        a = math.radians(t.angle)
        read = " ".join(pred.texts[i].text for i in sorted(  # in reading order: along the text's direction
            inside, key=lambda i: pred.texts[i].x * math.cos(a) - pred.texts[i].y * math.sin(a)))
        cers.append(min(1.0, _cer(t.text, read)))
        exact += read == t.text
        kinds[_kind(t.text)][0] += read == t.text
        kinds[_kind(t.text)][1] += 1
        unflagged += read != t.text and not any(_in(t, q) for q in marked)
        want, got = Counter(t.text.split()), Counter(read.split())
        tokens += sum(want.values())
        found += sum((want & got).values())
    return {"text_cer": float(np.mean(cers)), "text_exact": exact / len(gt.texts),
            "token_exact": found / max(1, tokens),
            "text_false_positives": len(pred.texts) - len(used), "text_review": review,
            "text_unflagged": unflagged / len(gt.texts)} | \
        {f"text_exact_{k}": kinds[k][0] / kinds[k][1] if kinds[k][1] else None for k in ("codes", "numbers", "words")}


def symbols(gt: Drawing, pred: Drawing) -> dict:
    """Symbols placed as library blocks: the right one within 1.5 mm of the reference, and its attributes read. And
    the symbols marked to check, and the wrong ones left unmarked (share of the reference symbols): nobody is pointed
    to those."""
    if not gt.symbols:
        return {"symbol_false": len(pred.symbols)} if pred.symbols else {}

    def match(a, b):
        return a.name == b.name and math.hypot(a.x - b.x, a.y - b.y) <= 1.5

    hit = sum(any(match(s, q) for q in pred.symbols) for s in gt.symbols)
    ok = sum(any(match(s, q) for s in gt.symbols) for q in pred.symbols)
    attributes = [(s, tag, t.text) for s in gt.symbols for tag, t in s.attributes.items()]
    read = sum(any(match(s, q) and getattr(q.attributes.get(tag), "text", None) == text for q in pred.symbols)
               for s, tag, text in attributes)
    unflagged = sum(not q.doubt and not any(match(s, q) for s in gt.symbols) for q in pred.symbols)
    return {"symbol_recall": hit / len(gt.symbols), "symbol_precision": ok / max(1, len(pred.symbols)),
            "symbol_attributes": read / len(attributes) if attributes else None,
            "symbol_review": sum(bool(q.doubt) for q in pred.symbols), "symbol_unflagged": unflagged / len(gt.symbols)}


def _linked(d: Drawing, tol: float = 0.5) -> set[frozenset]:
    """Pairs of symbols joined by linework. Lines, arcs and polylines that meet end to end, or with an end on another
    one (a branch), form one net; a symbol belongs to every net that reaches one of its ports."""
    nets = [[q] for q in d.lines + d.arcs] + [pl.parts() for pl in d.polylines if pl.kind != "frame"]
    sampled = [_sampled(parts)[0] for parts in nets]
    if not d.symbols or not any(len(p) for p in sampled):
        return set()
    points = np.vstack([p for p in sampled if len(p)])
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(sampled) if len(p)])
    tree = cKDTree(points)
    parent = list(range(len(nets)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(sampled):
        for end in (p[:1], p[-1:]) if len(p) else ():
            for j in tree.query_ball_point(end[0], tol):
                parent[root(owner[j])] = root(i)
    members = defaultdict(set)
    for k, s in enumerate(d.symbols):
        for port in s.ports:
            for j in tree.query_ball_point(port, tol):
                members[root(owner[j])].add(k)
    return {frozenset((a, b)) for m in members.values() for a in m for b in m if a < b}


def loose(d: Drawing) -> dict:
    """What has to be joined or cleaned up by hand in CAD, without a reference: ``gap_ends``, ends of lines, arcs and
    open polylines that stop short of other linework by up to 2 mm without touching it (a pipe that does not reach its
    junction, valve or vessel); ``crumbs``, pieces under 2 mm that touch no other linework."""
    ents = [e for e in _entities(d) if e]
    sampled = [_sampled(e)[0] for e in ents]
    if not sampled:
        return {"gap_ends": 0, "crumbs": 0}
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(sampled)])
    tree = cKDTree(np.vstack(sampled))

    def nearest_other(points: np.ndarray, i: int) -> np.ndarray:
        dist, idx = tree.query(points, k=12, distance_upper_bound=2.0)
        dist[(idx < len(owner)) & (owner[np.minimum(idx, len(owner) - 1)] == i)] = np.inf
        return dist.min(axis=1)

    touch = STEP / 2 + 0.05
    gaps = crumbs = 0
    for i, e in enumerate(ents[:len(ents) - len(d.symbols)]):  # the symbols are blocks
        if isinstance(e[0], (Circle, Polygon)) or _closed(e):
            continue
        ends = np.array([_ends(e[0])[0], _ends(e[-1])[1]])
        dist = nearest_other(ends, i)
        gaps += int(np.sum((dist > touch) & (dist <= 2.0)))
        crumbs += _length(e) < 2.0 and bool(nearest_other(sampled[i], i).min() > touch)
    return {"gap_ends": gaps, "crumbs": crumbs}


def _ends(q) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(q, Line):
        return (q.x1, q.y1), (q.x2, q.y2)
    return tuple((q.cx + q.r * math.cos(math.radians(t)), q.cy + q.r * math.sin(math.radians(t)))
                 for t in (q.start, q.end))


def _closed(parts: list) -> bool:
    return len(parts) > 1 and math.dist(_ends(parts[0])[0], _ends(parts[-1])[1]) < 1e-6


def _segments(d: Drawing) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The straight pieces of the solid lines and polylines of a drawing (not of its symbols) as rows x1, y1, x2, y2,
    the number of the line or polyline each belongs to, and the ends of the open ones (rows x, y)."""
    rows, owner, ends = [], [], []
    for i, q in enumerate(d.lines + d.polylines):
        parts = [p for p in (q.parts() if hasattr(q, "parts") else [q]) if isinstance(p, Line)]
        if q.dashed or not parts:
            continue
        rows += [(p.x1, p.y1, p.x2, p.y2) for p in parts]
        owner += [i] * len(parts)
        if not getattr(q, "closed", False):
            ends += [(parts[0].x1, parts[0].y1), (parts[-1].x2, parts[-1].y2)]
    return np.array(rows, float).reshape(-1, 4), np.array(owner, int), np.array(ends, float).reshape(-1, 2)


def _meets(seg: np.ndarray, owner: np.ndarray, ends: np.ndarray, dots: list[Circle]
           ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Where two lines cross, both running on (no end of a line within 1.5 mm, no dot), and where they join: where
    they meet at an end (within 0.5 mm), or at a dot. Places with an end in between are neither."""
    crossings, junctions = [], [(c.cx, c.cy) for c in dots]
    tree = cKDTree(ends) if len(ends) else None
    a, d = seg[:, :2], seg[:, 2:] - seg[:, :2]
    for i in range(len(seg)):
        p, r = a[i], d[i]
        denom = r[0] * d[:, 1] - r[1] * d[:, 0]
        ok = (np.abs(denom) > 1e-12) & (owner > owner[i])
        safe = np.where(ok, denom, 1.0)
        t = ((a[:, 0] - p[0]) * d[:, 1] - (a[:, 1] - p[1]) * d[:, 0]) / safe
        u = ((a[:, 0] - p[0]) * r[1] - (a[:, 1] - p[1]) * r[0]) / safe
        for j in np.flatnonzero(ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)):
            x = (float(p[0] + t[j] * r[0]), float(p[1] + t[j] * r[1]))
            near = len(tree.query_ball_point(x, 1.5)) if tree is not None else 0
            if tree is not None and tree.query_ball_point(x, 0.5) or any(math.dist(x, (c.cx, c.cy)) <= 1.0 for c in dots):
                junctions.append(x)
            elif not near:
                crossings.append(x)
    return crossings, junctions


def crossings(gt: Drawing, pred: Drawing) -> dict:
    """Lines that cross without joining, and lines that join. Of the places where two lines of the reference cross
    and run on (no dot), the share where two lines of the output run through as well, neither ending within 1.5 mm
    (crossings kept); of the places where they join (at an end, at a dot), the share where a line of the output ends,
    turns (a corner of a closed outline) or a dot is within 1 mm (junctions kept). A crossing closed into a junction
    loses the one, a junction left with a gap the other."""
    seg, owner, ends = _segments(gt)
    if len(seg) < 2:
        return {}
    across, joins = _meets(seg, owner, ends, [c for c in gt.circles if c.r <= 1.0])
    pseg, powner, pends = _segments(pred)
    if not len(pseg) or not len(pends):
        return {"crossings_kept": 0.0 if across else None, "junctions_kept": 0.0 if joins else None}
    tree = cKDTree(pends)
    end_owner = np.repeat([o for o in dict.fromkeys(powner.tolist())
                           if not getattr((pred.lines + pred.polylines)[o], "closed", False)], 2)
    a, d = pseg[:, :2], pseg[:, 2:] - pseg[:, :2]
    length2 = np.maximum((d ** 2).sum(axis=1), 1e-12)

    def through(x) -> int:  # output lines passing within 0.75 mm of x without an end within 1.5 mm of it
        t = np.clip(((x[0] - a[:, 0]) * d[:, 0] + (x[1] - a[:, 1]) * d[:, 1]) / length2, 0, 1)
        near = set(powner[np.hypot(a[:, 0] + t * d[:, 0] - x[0], a[:, 1] + t * d[:, 1] - x[1]) <= 0.75].tolist())
        ending = set(end_owner[tree.query_ball_point(x, 1.5)].tolist())
        return len(near - ending)

    kept = sum(through(x) >= 2 for x in across)
    corners = cKDTree(np.vstack([pseg[:, :2], pseg[:, 2:]]))  # the ends of every straight piece
    met = sum(bool(corners.query_ball_point(x, 1.0)) or any(math.dist(x, (c.cx, c.cy)) <= 1.0 for c in pred.circles)
              for x in joins)
    return {"crossings_kept": kept / len(across) if across else None,
            "junctions_kept": met / len(joins) if joins else None}


def connectivity(gt: Drawing, pred: Drawing) -> dict:
    """Symbols joined by linework in the reference and in the output; output symbols count as the reference symbol of
    the same name within 1.5 mm. A broken pipe loses a connection, a wrong junction adds one."""
    ref = _linked(gt)
    if not ref:
        return {}
    same = {}
    for k, q in enumerate(pred.symbols):
        near = [(math.hypot(s.x - q.x, s.y - q.y), i) for i, s in enumerate(gt.symbols) if s.name == q.name]
        near = [n for n in near if n[0] <= 1.5]
        same[k] = min(near)[1] if near else -1 - k
    found = {frozenset(same[k] for k in pair) for pair in _linked(pred)}
    found = {pair for pair in found if len(pair) == 2}
    hit = len(ref & found)
    return {"connection_recall": hit / len(ref), "connection_precision": hit / max(1, len(found))}


MARKUP_LAYERS = {"P2C-MARKUP", "P2C-MARKUP-TEXT", "P2C-HIGHLIGHT"}  # where the pipeline puts what was added by hand


def _is_markup(q) -> bool:
    return getattr(q, "kind", None) == MARKUP or q.layer in MARKUP_LAYERS


def content(d: Drawing) -> Drawing:
    """The drawing without what was added by hand (markups are measured on their own)."""
    return replace(d, polylines=[pl for pl in d.polylines if not _is_markup(pl)],
                   lines=[q for q in d.lines if not _is_markup(q)], arcs=[q for q in d.arcs if not _is_markup(q)],
                   polygons=[g for g in d.polygons if not _is_markup(g)], texts=[t for t in d.texts if not _is_markup(t)])


def markups(gt: Drawing, pred: Drawing) -> dict:
    """How well what was added by hand is kept apart from the drawing: of the markup strokes of the reference, the
    share the pipeline put on a markup layer (recall) and the share it took for the drawing (leaked; where no line
    of the drawing lies under them); of what it put on a markup layer, the share that is markup (precision); and the
    texts of the drawing it read on a markup (phantoms) and the fills it made there."""
    marks = [pl for pl in gt.polylines if pl.kind == MARKUP]
    if not marks:
        return {}
    g = np.vstack([_sampled(pl.parts())[0] for pl in marks])
    drawn = samples(content(gt).flat())
    clear = g[cKDTree(drawn).query(g)[0] > 1.0] if len(drawn) else g  # markup away from the print
    found = Drawing(pred.width, pred.height, polylines=[pl for pl in pred.polylines if _is_markup(pl)],
                    lines=[q for q in pred.lines if _is_markup(q)])
    m = samples(found.flat())
    rest = content(pred).flat()
    c = samples(rest)
    tree = cKDTree(g)
    out = {"markup_recall": float(np.mean(cKDTree(m).query(g)[0] <= 1.5)) if len(m) else 0.0,
           "markup_precision": float(np.mean(tree.query(m)[0] <= 1.5)) if len(m) else None,
           "markup_leaked": float(np.mean(cKDTree(c).query(clear)[0] <= 0.5)) if len(c) and len(clear) else 0.0}
    known = content(gt).flat().texts
    out["markup_phantom_texts"] = sum(1 for t in rest.texts if tree.query((t.x, t.y))[0] <= 2.0
                                      and not any(_in(k, t) for k in known))
    out["markup_fills"] = sum(1 for p in rest.polygons if p.filled
                              and tree.query(np.mean(p.points, axis=0))[0] <= 3.0)
    return out


def uncertain(pred: Drawing) -> dict:
    """The share of the texts and symbols the pipeline marked to check."""
    n = len(pred.texts) + len(pred.symbols)
    flagged = sum(t.uncertain for t in pred.texts) + sum(s.doubt is not None for s in pred.symbols)
    return {"uncertain_share": flagged / n if n else None}


def deviations(gt: Drawing, pred: Drawing, ppm: float = 4.0) -> dict:
    """Areas where the recognised linework and the reference differ, as on an overlay of the two: ink of either
    that has none of the other within 1 mm, in patches of at least 2 mm² (texts left out)."""
    from app.plan2cad.render import render

    a = render(gt, ppm, texts=False) < 128
    b = render(replace(pred, width=gt.width, height=gt.height), ppm, texts=False) < 128  # on the same canvas
    k = np.ones((2 * round(ppm) + 1,) * 2, np.uint8)
    off = (a & ~(cv2.dilate(b.astype(np.uint8), k) > 0)) | (b & ~(cv2.dilate(a.astype(np.uint8), k) > 0))
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(off.astype(np.uint8), k), connectivity=8)
    return {"deviation_areas": int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= 2 * ppm * ppm))}


def evaluate(gt: Drawing, pred: Drawing) -> dict:
    marks = markups(gt, pred)
    gt, pred = content(gt), content(pred)
    flat_gt, flat_pred = gt.flat(), pred.flat()
    out = geometry(flat_gt, flat_pred) | circles(flat_gt, flat_pred) | texts(flat_gt, flat_pred) | objects(gt, pred)
    out |= continuity(gt, pred) | symbols(gt, pred) | connectivity(gt, pred) | crossings(gt, pred) | loose(pred) | marks
    out |= uncertain(pred) | deviations(flat_gt, flat_pred) | pens(flat_gt, flat_pred)
    out["scale_ok"] = pred.scale == gt.scale if gt.scale else None
    return out
